"""Release regressions from the September 2026 measurement audit (offline)."""
import asyncio
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multivon_eval import (
    AgentStep,
    EvalCase,
    EvalSuite,
    Faithfulness,
    Hallucination,
    JudgeConfig,
    NotEmpty,
    ToolCall,
    ToolCallAccuracy,
)
from multivon_eval.compare import compare_reports
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.evaluators.llm_judge import _parse_yes_no
from multivon_eval.result import CaseResult, EvalGateFailure, EvalReport, EvalResult, EvalStatus


class HalfScore(Evaluator):
    name = 'half'

    def evaluate(self, case, output):
        return self._result(0.5)


@pytest.mark.parametrize('runs', [1, 3])
def test_missing_context_is_not_evidence_of_success(runs):
    suite = EvalSuite('missing').add_case(EvalCase('q')).add_evaluator(Faithfulness())
    report = suite.run(lambda _: 'unsupported', runs=runs, verbose=False)
    assert report.evaluated == 0
    assert report.skipped == 1
    assert report.pass_rate == 0
    assert report.scores_by_evaluator() == {}
    assert report.pass_at_k(1).value is None
    assert report.case_results[0].status == EvalStatus.SKIPPED
    assert report.case_results[0].results[0].passed is False
    restored = EvalReport.from_dict(json.loads(report.to_json()))
    assert restored.skipped == 1
    assert ET.fromstring(report.to_junit_xml()).find('.//skipped') is not None
    with pytest.raises(EvalGateFailure, match='INDETERMINATE'):
        suite.run(lambda _: 'unsupported', runs=runs, verbose=False, fail_threshold=0)


@pytest.mark.parametrize('runs', [1, 3])
def test_optional_skipped_metric_does_not_inflate_or_fail_measured_case(runs):
    suite = EvalSuite('mixed').add_case(EvalCase('q')).add_evaluators(HalfScore(), Faithfulness())
    report = suite.run(lambda _: 'answer', runs=runs, verbose=False, fail_threshold=1)
    assert report.pass_rate == 1
    assert report.avg_score == 0.5
    assert report.scores_by_evaluator() == {'half': 0.5}
    assert report.passed_by_evaluator() == {'half': 1}
    root = ET.fromstring(report.to_junit_xml())
    assert root.find('.//testsuite').get('skipped') == '1'


@pytest.mark.parametrize('evaluator,replies', [
    (Faithfulness, ['["The capital of France is Berlin."]', 'No']),
    (Hallucination, ['No', 'Yes', 'Yes', 'No']),
])
def test_apology_prefix_cannot_bypass_factual_grading(evaluator, replies):
    case = EvalCase('Capital?', context='The capital of France is Paris.')
    with patch('multivon_eval.evaluators.llm_judge.make_judge_call', side_effect=replies) as call:
        result = evaluator(threshold=0.7).evaluate(case, 'Sorry, the capital of France is Berlin.')
    assert call.called
    assert result.score == 0
    assert not result.passed
    assert not result.metadata.get('skipped')


def test_claimless_refusal_still_can_be_faithful():
    with patch('multivon_eval.evaluators.llm_judge.make_judge_call', return_value='[]'):
        result = Faithfulness(threshold=0.7).evaluate(
            EvalCase('Unknown?', context='No relevant facts.'), 'I cannot answer that.')
    assert result.passed


@pytest.mark.parametrize('text', [
    'I cannot say yes with confidence.', 'There is no way to determine this.',
    'Yes or no, I cannot tell.', 'No or yes, I cannot tell.',
    'The word yes appears in the response.', 'Insufficient evidence; no verdict.',
])
def test_explanation_mentions_are_unknown(text):
    assert _parse_yes_no(text) is None


@pytest.mark.parametrize('error_field', ['judge_error', 'model_error', 'evaluator_error'])
def test_outage_recovery_is_not_a_significant_model_improvement(error_field):
    baseline, proposal = [], []
    for i in range(20):
        baseline.append(CaseResult(str(i), '', [], **{error_field: 'outage'}))
        proposal.append(CaseResult(str(i), 'ok', [EvalResult('e', 1, True)]))
    for a, b in [(baseline, proposal), (proposal, baseline)]:
        diff = compare_reports(EvalReport('a', a), EvalReport('b', b))
        assert not diff.improvements
        assert not diff.regressions
        assert diff.mcnemar_p is None
        assert abs(diff.errors_delta) == 20


def test_mixed_errors_do_not_change_quality_significance():
    passed = lambda i: CaseResult(i, 'ok', [EvalResult('e', 1, True)])
    failed = lambda i: CaseResult(i, 'bad', [EvalResult('e', 0, False)])
    base = [failed('real')]
    prop = [passed('real')]
    clean = compare_reports(EvalReport('a', base), EvalReport('b', prop))
    base += [CaseResult(str(i), '', [], judge_error='outage') for i in range(20)]
    prop += [passed(str(i)) for i in range(20)]
    mixed = compare_reports(EvalReport('a', base), EvalReport('b', prop))
    assert mixed.mcnemar_p == clean.mcnemar_p
    assert len(mixed.improvements) == 1


def test_gate_rejects_missing_measurements_by_default_but_accepts_explicit_error_budget():
    suite = EvalSuite('errors').add_cases([EvalCase('ok'), EvalCase('error')]).add_evaluator(NotEmpty())
    def model(prompt):
        if prompt == 'error':
            raise RuntimeError('offline outage')
        return 'ok'
    with pytest.raises(EvalGateFailure, match='error budget'):
        suite.run(model, fail_threshold=1, verbose=False)
    report = suite.run(model, fail_threshold=1, max_error_rate=0.5, verbose=False)
    assert report.error_rate == 0.5
    assert report.scores_by_evaluator() == {'not_empty': 1.0}


def test_async_skip_and_error_gates_match_sync():
    async def run():
        suite = EvalSuite('missing').add_case(EvalCase('q')).add_evaluator(Faithfulness())
        async def model(prompt):
            return 'answer'
        with pytest.raises(EvalGateFailure, match='INDETERMINATE'):
            await suite.run_async(model, fail_threshold=0, verbose=False)
    asyncio.run(run())


def test_strict_ordered_repeated_tools_stay_in_unit_interval():
    case = EvalCase('q', expected_tool_calls=['a', 'a', 'a'],
                    agent_trace=[AgentStep(tool_calls=[ToolCall(n) for n in ['a', 'a', 'a', 'b']])])
    result = ToolCallAccuracy(require_order=True, penalize_unexpected=True).evaluate(case, '')
    assert result.score == 0.75


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_provider_request_honors_configured_timeout(provider, asynchronous):
    from multivon_eval import judge
    client = MagicMock()
    response = SimpleNamespace(usage=None, content=[SimpleNamespace(text='Yes')],
                               choices=[SimpleNamespace(message=SimpleNamespace(content='Yes'))])
    create = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    if provider == 'openai':
        client.chat.completions.create = create
    else:
        client.messages.create = create
    target = provider + '.' + ('Async' if asynchronous else '') + ('OpenAI' if provider == 'openai' else 'Anthropic')
    fn = getattr(judge, ('_async_' if asynchronous else '_sync_') + provider + '_call')
    with patch(target, return_value=client) as factory:
        config = JudgeConfig(provider=provider, model='test-model', timeout=3).resolve()
        result = asyncio.run(fn('q', config)) if asynchronous else fn('q', config)
    assert result == 'Yes'
    assert factory.call_args.kwargs['timeout'] == 3


def test_judge_reliability_samples_fresh_calls_even_when_cache_enabled():
    from multivon_eval.evaluators.llm_judge import Relevance
    from multivon_eval.judge import configure, get_global_judge
    old = get_global_judge()
    configure(JudgeConfig(provider='openai', model='gpt-4o-mini', cache=True,
                          reliability_check=True, reliability_sample=1))
    suite = EvalSuite('cache').add_case(EvalCase('q')).add_evaluator(Relevance(threshold=0.7))
    try:
        with (
            patch('multivon_eval.judge._cache_get_safe', side_effect=lambda p, c:
                  (None, 'No' if 'significant content unrelated' in p else 'Yes')) as cached,
            patch('multivon_eval.judge._make_judge_call_uncached',
                  side_effect=['No', 'No', 'No', 'Yes']) as fresh,
        ):
            report = suite.run(lambda _: 'answer', verbose=False)
        assert cached.call_count == 4
        assert fresh.call_count == 4
        assert report.judge_reliability == 0
    finally:
        configure(old)


def test_comparison_ci_gate_rejects_unmeasured_pairs(tmp_path, capsys):
    from multivon_eval.compare import _cli
    baseline = EvalReport('a', [CaseResult('q', '', [], judge_error='outage')])
    proposal = EvalReport('b', [CaseResult('q', 'ok', [EvalResult('e', 1, True)])])
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    baseline.save_json(str(a))
    proposal.save_json(str(b))
    assert _cli([str(a), str(b), '--fail-on-regression']) == 2
    assert 'INDETERMINATE' in capsys.readouterr().err


def test_subgroup_scores_exclude_missing_measurements():
    report = EvalReport('groups', [
        CaseResult('ok', 'ok', [EvalResult('e', .5, True)], tags=['rag']),
        CaseResult('error', '', [EvalResult('e', 0, False)], tags=['rag'], judge_error='outage'),
        CaseResult('skip', '', [EvalResult('e', 0, False, metadata={'skipped': True})], tags=['rag']),
    ])
    assert report.scores_by_tag() == {'rag': .5}
    assert report.passed_by_tag() == {'rag': 1}
    assert report.score_percentiles() == {'p10': .5, 'p50': .5, 'p90': .5}


def test_mixed_skip_and_outage_does_not_hide_infrastructure_error():
    from multivon_eval.suite import _aggregate_runs
    skipped = CaseResult('q', '', [EvalResult('e', 0, False, metadata={'skipped': True})])
    errored = CaseResult('q', '', [], judge_error='outage')
    result = _aggregate_runs(EvalCase('q'), [skipped, errored])
    assert result.status == EvalStatus.JUDGE_ERROR
    assert EvalReport('mixed', [result]).errors == 1


def test_deterministic_only_suite_cannot_claim_judge_reliability():
    from multivon_eval.judge import configure, get_global_judge
    old = get_global_judge()
    try:
        configure(JudgeConfig(reliability_check=True))
        suite = EvalSuite('deterministic').add_case(EvalCase('q')).add_evaluator(NotEmpty())
        assert suite.run(lambda _: 'ok', verbose=False).judge_reliability is None
    finally:
        configure(old)
