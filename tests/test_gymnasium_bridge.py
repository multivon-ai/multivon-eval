"""Real Gymnasium lifecycle and independent SQLite outcome failures."""
import json
from pathlib import Path

import pytest

gym = pytest.importorskip('gymnasium')

from multivon_eval import AcceptancePolicy, CheckRequirement, EvalCase, EvalReport
from multivon_eval.compare import compare_reports
from multivon_eval.episode import EpisodeEvidence, OutcomeCheck, OutcomeVerdict, evaluate_episode
from multivon_eval.integrations.gymnasium import capture_episode
from multivon_eval.trials import trial_integrity_issues


def lake(actions=(2, 2, 1, 1, 1, 2), *, max_steps=8, time_limit=20, observer=None, interact=None):
    holder = {}
    def factory():
        holder['env'] = gym.make('FrozenLake-v1', is_slippery=False, max_episode_steps=time_limit)
        return holder['env']
    def act(env, observation, info):
        for action in actions:
            env.step(action)
        return 'done'
    evidence = capture_episode(factory, interact or act, case=EvalCase('Reach the goal', case_id='lake'),
        environment_id='FrozenLake-v1:4x4:deterministic', observer_id='state-index-v1',
        observe=observer or (lambda: {'state': int(holder['env'].unwrapped.s)}), seed=11,
        max_steps=max_steps)
    return evidence


def goal(episode):
    data = episode.data
    state = data['final_state']
    if not data['termination']['terminated'] or data['termination']['truncated']:
        return OutcomeVerdict(None, 'Goal status requires a completed episode', {'state': state})
    return OutcomeVerdict(None if state is None else state['state'] == 15,
                          'Read final simulator state', {'state': state})


GOAL = OutcomeCheck('goal', 'state-15-v1', goal)
POLICY = AcceptancePolicy((CheckRequirement('goal'),))


def test_real_gymnasium_episode_roundtrip_and_outcome():
    episode = lake()
    data = episode.data
    assert data['termination'] == {'terminated': True, 'truncated': False}
    assert data['initial_state'] == {'state': 0}
    assert data['final_state'] == {'state': 15}
    assert data['cleanup'] == {'attempted': True, 'completed': True, 'error': None}
    assert data['steps'][0]['action'] == [2]  # Native Space.to_jsonable uses a batch.
    restored = EpisodeEvidence.from_dict(json.loads(json.dumps(data)))
    assert restored.digest == episode.digest
    report = evaluate_episode(restored, [GOAL])
    assert POLICY.evaluate(report).decision == 'accept'
    assert not trial_integrity_issues(report.case_results[0])
    saved = EvalReport.from_dict(json.loads(report.to_json()))
    assert saved.case_results[0].trials[0].data['upstream']['evidence'] == data
    assert POLICY.evaluate(saved).decision == 'accept'
    data['final_state']['state'] = 0
    assert episode.data['final_state']['state'] == 15
    with pytest.raises(ValueError, match='digest'):
        EpisodeEvidence.from_dict(data)


def test_natural_termination_is_not_task_success():
    # Right then down lands in a hole: naturally terminal, but not the goal.
    episode = lake((2, 1))
    assert episode.data['termination']['terminated']
    assert not episode.coverage_issues
    assert POLICY.evaluate(evaluate_episode(episode, [GOAL])).decision == 'reject'


@pytest.mark.parametrize('options', [{'time_limit': 1}, {'max_steps': 1}, {'actions': ()}])
def test_truncation_step_limits_and_early_return_cannot_approve(options):
    episode = lake(**options)
    assert episode.coverage_issues
    assert POLICY.evaluate(evaluate_episode(episode, [GOAL])).decision == 'indeterminate'
    if options.get('max_steps') == 1:
        assert len(episode.data['steps']) == 1
        assert episode.data['final_state']['state'] == 1


def test_step_after_done_and_reset_cannot_erase_episode_history():
    def reset_again(env, *_):
        env.step(2)
        env.reset(seed=2)
    episode = lake(interact=reset_again)
    assert len(episode.data['steps']) == 1
    assert episode.data['initial_state'] == {'state': 0}
    assert any(e['phase'] == 'lifecycle' for e in episode.data['errors'])
    after_done = lake((2, 1, 2))
    assert len(after_done.data['steps']) == 2
    assert after_done.coverage_issues
    # The observed terminal hole still proves failure, despite the extra API misuse.
    assert POLICY.evaluate(evaluate_episode(after_done, [GOAL])).decision == 'reject'


def test_observer_failure_is_unmeasured_not_false_or_pass():
    def unavailable():
        raise ConnectionError('State API unavailable')
    episode = lake(observer=unavailable)
    report = evaluate_episode(episode, [GOAL])
    assert report.case_results[0].results[0].metadata['skipped']
    assert POLICY.evaluate(report).decision == 'indeterminate'


def test_check_exception_and_contract_drift():
    episode = lake()
    def broken(_):
        raise RuntimeError('Bad oracle')
    report = evaluate_episode(episode, [OutcomeCheck('goal', 'broken-v1', broken)])
    assert report.case_results[0].evaluator_error
    assert report.errors == 1
    first = evaluate_episode(episode, [GOAL])
    changed = evaluate_episode(episode, [OutcomeCheck('goal', 'state-15-v2', goal)])
    comparison = compare_reports(first, changed)
    assert comparison.mcnemar_p is None
    assert any('configuration changed' in issue for issue in comparison.identity_issues)


def test_real_box_space_uses_upstream_json_conversion():
    env = gym.make('CartPole-v1', max_episode_steps=1)
    def act(wrapped, *_):
        wrapped.step(0)
    episode = capture_episode(lambda: env, act, case=EvalCase('Balance'),
        environment_id='CartPole-v1:limit1', observer_id='cartpole-state-v1',
        observe=lambda: {'state': env.unwrapped.state.tolist()}, seed=7, max_steps=2)
    assert len(episode.data['initial']['observation'][0]) == 4
    assert len(episode.data['steps'][0]['transition']['observation'][0]) == 4
    assert episode.data['termination']['truncated']


def test_cleanup_runs_after_reset_failure_and_is_recorded_if_it_fails():
    class Broken(gym.Env):
        def __init__(self):
            self.action_space = self.observation_space = gym.spaces.Discrete(2)
            self.close_count = 0
        def reset(self, **kwargs):
            raise RuntimeError('reset failed')
        def close(self):
            self.close_count += 1
            raise OSError('close failed')
    env = Broken()
    episode = capture_episode(lambda: env, lambda *_: 'unused', case=EvalCase('broken'),
        environment_id='broken-v1', observer_id='fixture-v1', observe=dict, seed=1, max_steps=2)
    assert env.close_count == 1
    assert {e['phase'] for e in episode.data['errors']} == {'reset', 'close'}
    assert not episode.data['cleanup']['completed']


def test_cancellation_propagates_after_cleanup():
    env = gym.make('FrozenLake-v1')
    closed = []
    original_close = env.close
    def close():
        closed.append(True)
        original_close()
    env.close = close
    def cancel(*_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        capture_episode(lambda: env, cancel, case=EvalCase('cancel'),
            environment_id='lake', observer_id='fixture', observe=dict, seed=1, max_steps=1)
    assert closed == [True]


def test_final_observer_cancellation_still_closes_environment():
    class Closing(gym.Env):
        def __init__(self):
            self.action_space = self.observation_space = gym.spaces.Discrete(2)
            self.closed = False
        def reset(self, **kwargs):
            return 0, {}
        def close(self):
            self.closed = True
    env = Closing()
    observations = []
    def observer():
        observations.append(True)
        if len(observations) > 1:
            raise KeyboardInterrupt
        return {}
    with pytest.raises(KeyboardInterrupt):
        capture_episode(lambda: env, lambda *_: None, case=EvalCase('cancel-final'),
            environment_id='fixture', observer_id='fixture', observe=observer, seed=1, max_steps=1)
    assert env.closed


def test_factory_failure_and_false_success_without_observation():
    def broken_factory():
        raise OSError('Cannot create environment')
    episode = capture_episode(broken_factory, lambda *_: 'unused', case=EvalCase('setup'),
        environment_id='fixture', observer_id='fixture', observe=dict, seed=1, max_steps=1)
    report = evaluate_episode(episode, [OutcomeCheck('goal', 'bad-oracle',
        lambda _: OutcomeVerdict(True, 'Unjustified assertion', {}))])
    assert not episode.data['cleanup']['attempted']
    assert POLICY.evaluate(report).decision == 'indeterminate'


@pytest.mark.parametrize('reward', ['1', float('nan')])
def test_invalid_transition_after_side_effect_preserves_observed_state(reward):
    class BadReward(gym.Env):
        def __init__(self):
            self.action_space = self.observation_space = gym.spaces.Discrete(2)
            self.state = 0
        def reset(self, **kwargs):
            return 0, {}
        def step(self, action):
            self.state = 1
            return 1, reward, True, False, {}
    env = BadReward()
    episode = capture_episode(lambda: env, lambda wrapped, *_: wrapped.step(0), case=EvalCase('write'),
        environment_id='bad-reward-v1', observer_id='state-v1', observe=lambda: {'state': env.state},
        seed=1, max_steps=1)
    assert episode.data['steps'][0]['transition'] is None
    assert episode.data['final_state'] == {'state': 1}
    assert episode.data['steps'][0]['error']['phase'] == 'step'


def test_environment_contract_change_blocks_paired_significance():
    from multivon_eval.case_manifest import digest
    first = lake()
    changed = first.data
    changed.pop('digest')
    changed['observer_id'] = 'different-observation-contract'
    second = EpisodeEvidence.from_dict({**changed, 'digest': digest(changed)})
    comparison = compare_reports(evaluate_episode(first, [GOAL]), evaluate_episode(second, [GOAL]))
    assert comparison.mcnemar_p is None
    assert any('configuration changed' in issue for issue in comparison.identity_issues)


def ledger_module():
    # Benchmark modules are repository examples, not shipped package imports.
    import sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from benchmarks.industrial import gymnasium_ledger
    return gymnasium_ledger


def test_sqlite_partial_commit_recovery_and_forbidden_history(tmp_path):
    ledger = ledger_module()
    database = tmp_path / 'ledger.sqlite'
    ledger.prepare(database)
    partial, report = ledger.run(database, [0, 1], fail_after_commit=True)
    assert partial.data['steps'][0]['transition'] is None
    assert partial.data['steps'][0]['state']['entries'][0] == ['invoice', '12.34', 'USD']
    assert ledger.POLICY.evaluate(report).decision == 'indeterminate'
    recovered, report = ledger.run(database, [0, 1])
    assert recovered.data['initial_state'] == partial.data['final_state']
    assert ledger.POLICY.evaluate(report).decision == 'accept'
    # Correct final values do not erase the forbidden intermediate write.
    episode, report = ledger.run(database, [2, 3, 1])
    assert episode.data['final_state']['entries'] == recovered.data['final_state']['entries']
    assert ledger.POLICY.evaluate(report).decision == 'reject'


def test_fresh_sqlite_episodes_and_duplicate_control(tmp_path):
    ledger = ledger_module()
    initials = []
    for index, unsafe in enumerate((False, True)):
        database = tmp_path / f'{index}.sqlite'
        ledger.prepare(database)
        episode, report = ledger.run(database, [0, 0, 1], unsafe=unsafe)
        initials.append(episode.data['initial_state'])
        assert ledger.POLICY.evaluate(report).decision == ('reject' if unsafe else 'accept')
    assert initials[0] == initials[1]
