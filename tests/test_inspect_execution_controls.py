"""Actual upstream limits, cancellation and concurrency; no network providers."""
import asyncio
import json

import pytest

pytest.importorskip('inspect_ai')

from inspect_ai.log import ProvenanceData, read_eval_log

from multivon_eval import AcceptancePolicy, CheckRequirement, EvalReport, EvalSuite, ExactMatch, regrade
from multivon_eval.integrations.inspect import from_inspect_log
from benchmarks.industrial.execution_controls_experiment import (
    cancellation_probe, concurrency_probe, limit_probe,
)


@pytest.mark.parametrize('kind', ['complete', 'time', 'working', 'message', 'token', 'cost', 'turn'])
def test_native_limit_stops_preserve_partial_output_and_task_outcome(tmp_path, kind):
    result = asyncio.run(limit_probe(tmp_path, kind))
    assert result['persisted_entries'] == int(kind == 'complete')
    copied = EvalReport.from_dict(json.loads((tmp_path / kind / 'strict-report.json').read_text()))
    evidence = copied.case_results[0].trials[0].data['upstream']['execution']
    assert evidence['limit'] == result['limit']
    assert evidence['accepted_limits'] == []
    # New graders cannot turn an unapproved stop into a completed execution.
    rescored = regrade(copied, EvalSuite('saved output').add_evaluator(ExactMatch()))
    assert AcceptancePolicy((CheckRequirement('exact_match'),)).evaluate(rescored).decision == result['strict_decision']
    if kind == 'token':
        # A generation already in flight can exceed a sample usage threshold.
        assert result['model_calls'] == 1
        assert result['native_usage']['mockllm/controls']['total_tokens'] == 10 > 8
    if kind == 'turn':
        # Measured upstream behavior in 0.3.263; do not claim a strict one-call cap.
        assert result['model_calls'] == 2


def test_native_sample_and_connection_concurrency(tmp_path):
    asyncio.run(concurrency_probe(tmp_path))


def test_native_cancel_retains_native_log_and_drains_active_calls(tmp_path):
    asyncio.run(cancellation_probe(tmp_path))


def test_invalidated_scores_cannot_be_accepted_or_cleared_by_limit_declaration(tmp_path):
    result = asyncio.run(limit_probe(tmp_path, 'complete'))
    log = read_eval_log(str(tmp_path / result['log']))
    log.samples[0].invalidation = ProvenanceData(author='fixture-reviewer', reason='Wrong task state')
    imported = from_inspect_log(log, accepted_limits=('time', 'cost'))
    assert imported.errors == 1 and 'invalidated' in imported.case_results[0].evaluator_error
    assert AcceptancePolicy((CheckRequirement('exact_match'),)).evaluate(imported).decision == 'indeterminate'
    assert imported.case_results[0].trials[0].data['upstream']['execution']['invalidation']['reason'] == 'Wrong task state'
    for _ in range(2):
        imported = regrade(imported, EvalSuite('saved output').add_evaluator(ExactMatch()))
        assert imported.errors == 1
        assert AcceptancePolicy((CheckRequirement('exact_match'),)).evaluate(imported).decision == 'indeterminate'


@pytest.mark.parametrize('accepted', ['time', ('typo',), (True,), ['token']])
def test_invalid_limit_policy_fails_before_import(accepted):
    with pytest.raises(ValueError, match='accepted_limits'):
        from_inspect_log(None, accepted_limits=accepted)
