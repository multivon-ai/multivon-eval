"""Missing metric inputs are skipped, never counted as passing evidence."""
from __future__ import annotations

from multivon_eval import EvalCase
from multivon_eval.evaluators.llm_judge import ContextRecall


def test_context_recall_skips_when_expected_output_missing():
    case = EvalCase(input="Q", context="some retrieved context")
    result = ContextRecall().evaluate(case, output="A")
    assert result.passed is False
    assert result.score == 0.0
    assert "[skipped]" in result.reason
    assert result.metadata.get("skipped") is True


def test_context_recall_skips_when_context_missing():
    case = EvalCase(input="Q", expected_output="A")
    result = ContextRecall().evaluate(case, output="A")
    assert result.passed is False
    assert result.score == 0.0
    assert "[skipped]" in result.reason


def test_context_recall_skips_when_both_missing():
    case = EvalCase(input="Q")
    result = ContextRecall().evaluate(case, output="A")
    assert result.passed is False
    assert "[skipped]" in result.reason
