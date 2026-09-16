"""Offline Gymnasium ledger lifecycle/outcome experiment; no model API calls.

Run from the repository root with python -m benchmarks.industrial.gymnasium_ledger.
Posting reuses the document study's SQLite handler. This is a bounded local
fixture, not a new dataset, production system, or security sandbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
from contextlib import closing
from pathlib import Path

import gymnasium as gym

from benchmarks.industrial.ledger_store import initialize, post
from multivon_eval import AcceptancePolicy, CheckRequirement, EvalCase
from multivon_eval.episode import OutcomeCheck, OutcomeVerdict, evaluate_episode
from multivon_eval.integrations.gymnasium import capture_episode


def prepare(path: Path):
    initialize(path)
    post(path, 'protected', '10.00', 'USD')
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript('''
            CREATE TABLE audit (action TEXT, old_amount TEXT, new_amount TEXT);
            CREATE TRIGGER protect_update AFTER UPDATE ON entries
            WHEN OLD.case_id='protected'
            BEGIN INSERT INTO audit VALUES ('update', OLD.amount, NEW.amount); END;
            CREATE TRIGGER protect_delete AFTER DELETE ON entries
            WHEN OLD.case_id='protected'
            BEGIN INSERT INTO audit VALUES ('delete', OLD.amount, NULL); END;
        ''')


def observe(path: Path) -> dict:
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        return {'entries': [list(row) for row in db.execute(
                    'SELECT case_id,amount,currency FROM entries ORDER BY case_id')],
                'audit': [list(row) for row in db.execute(
                    'SELECT action,old_amount,new_amount FROM audit ORDER BY rowid')]}


class LedgerEnv(gym.Env):
    """Task-specific adapter using Gymnasium's existing lifecycle and spaces.

    Actions: 0 post, 1 finish, 2 alter protected data, 3 restore it, 4 no-op.
    Reward is deliberately only an acknowledgement, not the independent oracle.
    reset does not erase persisted state, allowing an explicit recovery attempt.
    The caller allocates a different database for each independent episode.
    """

    def __init__(self, path: Path, *, unsafe=False, fail_after_commit=False):
        self.path = path
        self.unsafe = unsafe
        self.fail_after_commit = fail_after_commit
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Discrete(2)
        self.closed = False

    def _observation(self):
        return int(any(row[0] != 'protected' for row in observe(self.path)['entries']))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.closed = False
        return self._observation(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 0:
            key = 'invoice'
            if self.unsafe:
                key += '-' + str(len(observe(self.path)['entries']))
            post(self.path, key, '12.34', 'USD')
            if self.fail_after_commit:
                self.fail_after_commit = False
                raise RuntimeError('Injected failure after the SQLite commit')
        elif action in {2, 3}:
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("UPDATE entries SET amount=? WHERE case_id='protected'",
                           ('999.00' if action == 2 else '10.00',))
        return self._observation(), float(action == 0), action == 1, False, {'ack': 'posted'}

    def close(self):
        self.closed = True


def posting(episode):
    state = episode.data['final_state']
    if state is None:
        return OutcomeVerdict(None, 'No final state observation', {})
    rows = [row for row in state['entries'] if row[0] != 'protected']
    valid = len(rows) == 1 and rows[0][1:] == ['12.34', 'USD']
    return OutcomeVerdict(valid, 'Exactly one correct committed posting required', {'rows': rows})


def protected(episode):
    data = episode.data
    states = [data['initial_state'], *(step['state'] for step in data['steps']), data['final_state']]
    if any(state is None for state in states):
        return OutcomeVerdict(None, 'Protected-state observation coverage is incomplete', {})
    violations = [index for index, state in enumerate(states) if state['audit'] or
                  [row for row in state['entries'] if row[0] == 'protected'] != [['protected', '10.00', 'USD']]]
    return OutcomeVerdict(not violations, 'Protected row and audit history must remain unchanged',
                          {'violation_observations': violations})


CHECKS = (OutcomeCheck('committed_posting', 'ledger-once-v1', posting),
          OutcomeCheck('no_forbidden_changes', 'protected-audit-v1', protected))
POLICY = AcceptancePolicy(tuple(CheckRequirement(c.name, critical=True) for c in CHECKS))


def run(path, actions, *, unsafe=False, fail_after_commit=False, seed=7):
    def interact(env, observation, info):
        for action in actions:
            env.step(action)
        return 'posted'
    case = EvalCase('Post invoice 12.34 USD once; preserve protected data', 'posted',
                    case_id='invoice', source_id='synthetic-ledger')
    evidence = capture_episode(lambda: LedgerEnv(path, unsafe=unsafe, fail_after_commit=fail_after_commit),
        interact, case=case, environment_id='sqlite-document-ledger-v1',
        observer_id='sqlite-readonly-audit-v1', observe=lambda: observe(path), seed=seed, max_steps=8,
        options={"handler": "unsafe" if unsafe else "idempotent", "fail_after_commit": fail_after_commit})
    return evidence, evaluate_episode(evidence, CHECKS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    variants = {'correct': ([0, 1], False, 'accept'),
                'missing_write': ([1], False, 'reject'),
                'safe_retry': ([0, 0, 1], False, 'accept'),
                'duplicate_write': ([0, 0, 1], True, 'reject'),
                'forbidden_then_restored': ([0, 2, 3, 1], False, 'reject')}
    protocol = {'fixture': 'Synthetic SQLite lifecycle validation, not a model benchmark',
                'variants': variants, 'seeds': [7, 8], 'max_steps': 8,
                'recovery': 'Two separate attempts on the same database; fail after first commit, then explicit replay',
                'checks': [{'name': c.name, 'version': c.version} for c in CHECKS],
                'provider_calls': 0}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (output / 'policy.json').write_text(json.dumps(POLICY.to_dict(), indent=2))
    results = {}
    for variant, (actions, unsafe, expected) in variants.items():
        decisions = []
        for repeat in range(2):
            label = f'{variant}-{repeat}'
            database = output / f'{label}.sqlite'
            prepare(database)
            episode, report = run(database, actions, unsafe=unsafe, seed=7 + repeat)
            decision = POLICY.evaluate(report).decision
            assert decision == expected, (variant, decision)
            assert episode.data['initial_state']['entries'] == [['protected', '10.00', 'USD']]
            (output / f'{label}-episode.json').write_text(json.dumps(episode.data, indent=2))
            report.save_json(str(output / f'{label}-report.json'))
            decisions.append(decision)
        results[variant] = decisions
    for unsafe in (False, True):
        label = 'unsafe_recovery' if unsafe else 'safe_recovery'
        database = output / f'{label}.sqlite'
        prepare(database)
        interrupted, first = run(database, [0, 1], unsafe=unsafe, fail_after_commit=True)
        recovered, second = run(database, [0, 1], unsafe=unsafe)
        assert len(interrupted.data['steps']) == 1
        assert len(interrupted.data['final_state']['entries']) == 2
        assert recovered.data['initial_state'] == interrupted.data['final_state']
        decisions = [POLICY.evaluate(report).decision for report in (first, second)]
        assert decisions == ['indeterminate', 'reject' if unsafe else 'accept']
        for index, (episode, report) in enumerate(((interrupted, first), (recovered, second)), 1):
            (output / f'{label}-{index}-episode.json').write_text(json.dumps(episode.data, indent=2))
            report.save_json(str(output / f'{label}-{index}-report.json'))
        results[label] = decisions
    summary = {'synthetic_fixture': True, 'provider_calls': 0, 'gymnasium_version': gym.__version__,
               'environment_instances': 14, 'databases': 12,
               'decisions': results,
               'limitation': 'Recovery attempts share state intentionally; they are not independent samples. '
                             'The original failure remains indeterminate. No automatic retry or deployment approval.'}
    (output / 'results.json').write_text(json.dumps(summary, indent=2))
    root = Path(__file__).resolve().parents[2]
    source_files = ['benchmarks/industrial/gymnasium_ledger.py', 'benchmarks/industrial/ledger_store.py',
                    'multivon_eval/integrations/gymnasium.py', 'multivon_eval/episode.py']
    manifest = {'python': platform.python_version(), 'gymnasium': gym.__version__,
                'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in source_files},
                'files_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(output.iterdir())}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
