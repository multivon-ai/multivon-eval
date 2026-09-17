"""Agent-grader measurement failures must remain missing evidence, never votes."""
import asyncio
import json
from unittest.mock import patch

import pytest

from multivon_eval import (
    AgentMemoryEval,
    AgentStep,
    EvalCase,
    EvalReport,
    EvalSuite,
    JudgeConfig,
    JudgeRetry,
    PlanQuality,
    StepFaithfulness,
    TaskCompletion,
    ToolArgumentAccuracy,
    ToolCall,
    ToolCallAccuracy,
    ToolCallNecessity,
    TrajectoryEfficiency,
    regrade,
)
from multivon_eval.exceptions import JudgeUnavailable

ITEMWISE = [ToolArgumentAccuracy, ToolCallNecessity, StepFaithfulness]
ALL = ITEMWISE + [PlanQuality, TaskCompletion, TrajectoryEfficiency, AgentMemoryEval]


def case(n=2, name='task'):
    return EvalCase(name, context='prior context', agent_trace=[
        AgentStep(output=f'step-{i}', tool_calls=[ToolCall('lookup', {'id': i}, 'found')])
        for i in range(n)])


@pytest.mark.parametrize('cls', ALL)
@pytest.mark.parametrize('reply', ['', 'Maybe', 'Yes, but no', 'Yes\nNo', None, True])
def test_incomplete_judgments_never_form_a_quality_score(cls, reply):
    with (
        patch('multivon_eval.evaluators.agent._judge_call_with', side_effect=['Yes', reply]),
        pytest.raises(JudgeUnavailable) as caught,
    ):
        cls().evaluate(case(), 'done')
    records = caught.value.evaluation_evidence['judgments']
    assert [r['status'] for r in records] == ['measured', 'error']
    assert records[0]['verdict'] is True and records[1]['verdict'] is None


@pytest.mark.parametrize('cls', ITEMWISE)
def test_errors_never_become_quality_votes(cls):
    with (
        patch('multivon_eval.evaluators.agent._judge_call_with', side_effect=RuntimeError('broken grader')),
        pytest.raises(RuntimeError) as caught,
    ):
        cls().evaluate(case(), 'done')
    assert caught.value.evaluation_evidence['judgments'][0]['error_type'] == 'RuntimeError'


@pytest.mark.parametrize('cls', ITEMWISE)
def test_long_trace_is_not_silently_truncated(cls):
    with patch('multivon_eval.evaluators.agent._judge_call_with', return_value='Yes') as judge:
        skipped = cls().evaluate(case(9), 'done')
        assert skipped.metadata['skipped'] and not skipped.passed
        assert skipped.metadata['requested_items'] == 9
        judge.assert_not_called()
        full = cls(max_items=9).evaluate(case(9), 'done')
        assert full.score == 1 and judge.call_count == 9
        assert len(full.metadata['evaluation_evidence']['judgments']) == 9


@pytest.mark.parametrize('cls', ITEMWISE)
@pytest.mark.parametrize('limit', [0, -1, True, 1.5, None])
def test_invalid_limits_rejected(cls, limit):
    with pytest.raises(ValueError, match='max_items'):
        cls(max_items=limit)


def test_observed_no_tools_and_missing_trace_are_distinct():
    for trace in (None, [], [AgentStep(output='direct answer')]):
        c = EvalCase('task', agent_trace=trace, expected_tool_calls=[])
        for cls in (ToolArgumentAccuracy, ToolCallNecessity):
            result = cls().evaluate(c, 'done')
            assert result.metadata['skipped'] and not result.passed
        accuracy = ToolCallAccuracy().evaluate(c, 'done')
        assert accuracy.passed is (trace is not None)


@pytest.mark.parametrize('cls', ALL)
def test_explicit_judge_and_protocol_are_recorded(cls):
    config = JudgeConfig(provider='anthropic', model='test-model')
    evaluator = cls(judge=config)
    with patch('multivon_eval.evaluators.agent._judge_call_with', return_value='Yes') as judge:
        evaluator.evaluate(case(), 'done')
    assert all(call.args[1].model == 'test-model' for call in judge.call_args_list)
    lock = EvalSuite('x').add_evaluator(evaluator).lock().to_dict()
    assert lock['evaluators'][0]['extra']['config']['protocol'] == 'agent-judgments/v2'


def test_full_result_and_memory_reference_reach_judge():
    c = case(1)
    c.agent_trace[0].tool_calls[0].result = 'prefix' * 100 + 'CRITICAL TAIL'
    c.expected_output = 'reference' * 100 + 'REFERENCE TAIL'
    with patch('multivon_eval.evaluators.agent._judge_call_with', return_value='Yes') as judge:
        StepFaithfulness().evaluate(c, 'done')
        assert 'CRITICAL TAIL' in judge.call_args.args[0]
        AgentMemoryEval().evaluate(c, 'done')
        assert 'REFERENCE TAIL' in judge.call_args.args[0]


def test_ambiguous_recovery_invalidates_whole_measurement():
    c = case(1)
    c.agent_trace[0].tool_calls[0].result = 'error: failed'
    with (
        patch('multivon_eval.evaluators.agent._judge_call_with', side_effect=['Yes'] * 3 + ['unclear']),
        pytest.raises(JudgeUnavailable) as caught,
    ):
        TrajectoryEfficiency().evaluate(c, 'done')
    assert len(caught.value.evaluation_evidence['judgments']) == 4


@pytest.mark.parametrize('mode', ['sync', 'parallel', 'async', 'saved'])
@pytest.mark.parametrize('failure', ['ambiguous', 'exception'])
def test_all_execution_paths_preserve_errors_and_concurrent_evidence(mode, failure):
    cases = [case(name=f'task-{i}') for i in range(4)]
    suite = EvalSuite('integrity').add_cases(cases).add_evaluator(ToolCallNecessity())
    def judge(prompt, config, **kwargs):
        if '"id": 1' in prompt.split('Current tool call:')[1]:
            if failure == 'exception':
                raise RuntimeError('broken')
            return 'unknown'
        return 'Yes'
    async def model(_):
        return 'done'
    with patch('multivon_eval.evaluators.agent._judge_call_with', side_effect=judge):
        if mode == 'async':
            report = asyncio.run(suite.run_async(model, verbose=False))
        elif mode == 'saved':
            report = suite.run_on_cases([(c, 'done') for c in cases], verbose=False)
        else:
            report = suite.run(lambda _: 'done', workers=4 if mode == 'parallel' else 1, verbose=False)
    loaded = EvalReport.from_dict(json.loads(report.to_json()))
    for c, result in zip(cases, loaded.case_results):
        assert result.status.value == ('judge_error' if failure == 'ambiguous' else 'evaluator_error')
        records = result.trials[0].data['evaluators'][0]['metadata']['evaluation_evidence']['judgments']
        assert len(records) == 2 and records[1]['status'] == 'error'
        assert all(f'Task: {c.input}\n' in r['prompt'] for r in records)
    with patch('multivon_eval.evaluators.agent._judge_call_with', return_value='Yes'):
        reviewed = regrade(loaded, suite)
    assert all(r.passed for r in reviewed.case_results)
    assert all(len(r.trials[0].data['evaluators'][0]['metadata']['evaluation_evidence']['judgments']) == 2
               for r in reviewed.case_results)


def test_retry_retains_failed_and_successful_judgments():
    suite = EvalSuite('retry').add_case(case(1)).add_evaluator(ToolCallNecessity())
    with patch('multivon_eval.evaluators.agent._judge_call_with', side_effect=['unclear', 'Yes']):
        report = suite.run(lambda _: 'done', verbose=False,
                           judge_retry=JudgeRetry(max_attempts=2, base_backoff=0))
    trials = report.case_results[0].trials
    assert len(trials) == 2
    assert [t.data['evaluators'][0]['metadata']['evaluation_evidence']['judgments'][0]['status']
            for t in trials] == ['error', 'measured']


def test_negative_criterion_keeps_expected_verdict_for_replay():
    with patch('multivon_eval.evaluators.agent._judge_call_with', side_effect=['Yes'] * 3 + ['No']):
        result = TaskCompletion().evaluate(case(), 'done')
    records = result.metadata['evaluation_evidence']['judgments']
    assert result.score == 1
    assert records[-1]['verdict'] is False and records[-1]['expected_verdict'] is False


def test_prior_tool_result_is_available_to_necessity_judge():
    c = case()
    c.agent_trace[0].tool_calls[0].result = 'prefix' * 100 + 'ALREADY FOUND'
    with patch('multivon_eval.evaluators.agent._judge_call_with', return_value='Yes') as judge:
        ToolCallNecessity().evaluate(c, 'done')
    assert 'ALREADY FOUND' in judge.call_args.args[0]


def test_hardness_retains_grader_failure_evidence():
    from multivon_eval.hardness import validate_adversarial_cases
    c = case(1)
    c.metadata['stress_tests'] = ['ToolCallNecessity']
    with patch('multivon_eval.evaluators.agent._judge_call_with', return_value='unknown'):
        kept, reports = validate_adversarial_cases([c], lambda _: 'done', n_shots=1)
    assert not kept and reports[0].failure_rate is None
    assert reports[0].shots[0]['evaluation_evidence']['judgments'][0]['response'] == 'unknown'
