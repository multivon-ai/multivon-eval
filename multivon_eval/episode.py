"""Immutable episode evidence and explicit task-outcome checks, independent of rewards."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .case_manifest import canonical_json, case_from_dict, digest
from .result import CaseResult, EvalReport, EvalResult
from .trials import TrialRecord, attach_trial, capture_case


@dataclass(frozen=True)
class EpisodeEvidence:
    """Detached evidence from one environment lifecycle; hashes are not signatures."""
    _json: str

    def __post_init__(self):
        data = json.loads(self._json)
        claimed = data.pop('digest', None)
        if data.get('schema') != 'multivon.episode/v1' or digest(data) != claimed:
            raise ValueError('Invalid episode schema or digest')
        required = {'schema', 'case', 'case_id', 'case_digest', 'environment_id', 'observer_id',
                    'gymnasium_version', 'seed', 'max_steps', 'options', 'recorded_at', 'spaces',
                    'initial', 'initial_state', 'steps', 'final_state', 'output', 'errors',
                    'issues', 'cleanup', 'duration_ms', 'termination'}
        if set(data) != required:
            raise ValueError('Episode fields do not match this schema version')
        case = case_from_dict(data['case'])
        if case.identity() != (data['case_id'], data['case_digest']):
            raise ValueError('Episode case identity does not match retained inputs')
        if type(data['max_steps']) is not int or data['max_steps'] < 1:
            raise ValueError('Episode max_steps must be a positive integer')
        if len(data['steps']) > data['max_steps']:
            raise ValueError('Episode exceeds its recorded step bound')
        ended = False
        for index, step in enumerate(data['steps'], 1):
            if ended:
                raise ValueError('Episode contains actions after termination or failure')
            if step['index'] != index:
                raise ValueError('Episode step indices must be consecutive')
            if step['transition'] is not None:
                transition = step['transition']
                if any(type(transition[k]) is not bool for k in ('terminated', 'truncated')):
                    raise ValueError('Episode termination flags must be booleans')
                ended = transition['terminated'] or transition['truncated']
            ended = ended or step['error'] is not None
        last = data['steps'][-1]['transition'] if data['steps'] else None
        if data['termination'] != {'terminated': bool(last and last['terminated']),
                                   'truncated': bool(last and last['truncated'])}:
            raise ValueError('Episode termination does not match its final transition')
        if not isinstance(data['issues'], list) or any(not isinstance(i, str) for i in data['issues']):
            raise ValueError('Episode issues must be strings')

    @classmethod
    def from_dict(cls, data: dict) -> EpisodeEvidence:
        return cls(canonical_json(data))

    @property
    def data(self) -> dict:
        return json.loads(self._json)

    @property
    def digest(self) -> str:
        return self.data['digest']

    @property
    def coverage_issues(self) -> list[str]:
        data = self.data
        issues = list(data['issues'])
        if data['errors']:
            issues.append('Episode has recorded execution or observation errors')
        if not data['termination']['terminated'] or data['termination']['truncated']:
            issues.append('Episode lacks an untruncated natural termination')
        if not data['cleanup']['completed']:
            issues.append('Environment cleanup was not confirmed')
        if data['initial_state'] is None or data['final_state'] is None or any(
                step['state'] is None for step in data['steps']):
            issues.append('Independent state observations are incomplete')
        return sorted(set(issues))


@dataclass(frozen=True)
class OutcomeVerdict:
    """True/False are measured outcomes; None explicitly means unmeasured."""
    passed: bool | None
    reason: str
    evidence: dict

    def __post_init__(self):
        if self.passed is not None and type(self.passed) is not bool:
            raise ValueError('Outcome verdict must be boolean or None')
        if not isinstance(self.reason, str) or not isinstance(self.evidence, dict):
            raise TypeError('Outcome reason must be text and evidence must be an object')
        canonical_json(self.evidence)


@dataclass(frozen=True)
class OutcomeCheck:
    """A versioned, caller-supplied assertion over saved evidence, not a live env.

    The version identifies the task contract. It cannot authenticate the check's
    implementation or establish that its state observer was independent.
    """
    name: str
    version: str
    evaluate: Callable[[EpisodeEvidence], OutcomeVerdict]

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.name, self.version)):
            raise ValueError('Outcome checks require nonempty name and version')
        if not callable(self.evaluate):
            raise TypeError('Outcome check must be callable')


def evaluate_episode(episode: EpisodeEvidence, checks: Iterable[OutcomeCheck], *,
                     name: str = 'environment outcomes') -> EvalReport:
    """Evaluate retained state with no target calls and preserve incomplete coverage.

    Checks receive only a detached episode. They may inspect before/after state,
    every transition, and partial-failure observations. Reward and termination
    are evidence, never automatic success. Exceptions are evaluator errors.
    """
    checks = tuple(checks)
    if not checks or len({c.name for c in checks}) != len(checks):
        raise ValueError('Supply at least one uniquely named outcome check')
    data = episode.data
    case = case_from_dict(data['case'])
    results, errors = [], []
    for check in checks:
        metadata = {'outcome_contract': check.version, 'episode_digest': episode.digest}
        try:
            verdict = check.evaluate(episode)
            if not isinstance(verdict, OutcomeVerdict):
                raise TypeError('Outcome check must return OutcomeVerdict')
            metadata['outcome_evidence'] = json.loads(canonical_json(verdict.evidence))
            if verdict.passed is None:
                metadata['skipped'] = True
            results.append(EvalResult(check.name, float(verdict.passed is True),
                                      verdict.passed is True, verdict.reason, metadata))
        except Exception as exc:  # noqa: BLE001 - a failed user check is an evaluator error
            reason = f'{type(exc).__name__}: {exc}'
            errors.append(f'{check.name}: {reason}')
            results.append(EvalResult(check.name, 0.0, False, reason,
                                      {**metadata, 'error_kind': 'outcome_check_error'}))
    cr = CaseResult(case.input, data['output'] or '', results, tags=case.tags,
                    evaluator_error='; '.join(errors) if errors else None)
    attach_trial(cr, capture_case(case), origin='gymnasium', latency_known=False)
    if not cr.trials:
        raise ValueError('Episode report requires portable case and outcome evidence')
    trial = cr.trials[0].data
    trial.pop('digest')
    trial['upstream'] = {'format': 'gymnasium', 'evidence': data,
                         'outcome_checks': [{'name': c.name, 'version': c.version} for c in checks]}
    trial['evidence_gaps'] = [
        'State observers, environment isolation and contract versions are caller assertions',
        'This adapter is not an execution sandbox or durable scheduler',
        'Provider requests and usage require separate upstream instrumentation',
    ]
    cr.trials = (TrialRecord.from_dict({**trial, 'digest': digest(trial)}),)
    from . import __version__
    from .lockfile import EvaluatorFingerprint, SuiteLock
    contract = {k: data[k] for k in ('environment_id', 'observer_id', 'gymnasium_version',
                                     'seed', 'max_steps', 'options', 'spaces')}
    lock = SuiteLock(__version__, name, digest({'checks': trial['upstream']['outcome_checks'],
                                               'environment': contract, 'case': data['case_digest']}),
        evaluators=[EvaluatorFingerprint(c.name, 'multivon_eval.episode.OutcomeCheck', 1.0,
                                         version=c.version, extra=contract) for c in checks],
        case_count=1, cases_hash=data['case_digest'])
    return EvalReport(name, [cr], evidence_issues=episode.coverage_issues, suite_lock=lock)
