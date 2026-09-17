"""Claim extraction/coverage cannot manufacture a passing measurement."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from multivon_eval import (
    AcceptancePolicy,
    CheckRequirement,
    EvalCase,
    EvalReport,
    EvalSuite,
    ExactMatch,
    Faithfulness,
    JudgeConfig,
)
from multivon_eval.result import EvalStatus

CALL = "multivon_eval.evaluators.llm_judge.make_judge_call"
CASE = EvalCase("What is supported?", expected_output="answer", context="Only the supported facts.")


@pytest.mark.parametrize("limit", [0, -1, True, False, None, "10", 1.5])
def test_invalid_claim_limits_fail_before_inference(limit):
    with patch(CALL) as call, pytest.raises(ValueError, match="positive integer"):
        Faithfulness(max_claims=limit)
    call.assert_not_called()


@pytest.mark.parametrize("raw", ['[null]', '[1]', '[true]', '[{}]', '[[]]', '[""]', '["  "]', '["valid", null]'])
def test_invalid_claim_types_are_not_sent_for_verification(raw):
    with patch(CALL, return_value=raw) as call:
        suite = EvalSuite("malformed").add_case(CASE).add_evaluator(Faithfulness(threshold=1))
        report = suite.run(lambda _: "answer", workers=1, verbose=False)
    assert call.call_count == 1
    assert report.case_results[0].status == EvalStatus.EVALUATOR_ERROR
    assert report.evaluated == 0


@pytest.mark.parametrize("raw", ["[]", json.dumps([f"claim {i}" for i in range(11)])])
def test_unmeasured_claims_block_required_coverage_after_roundtrip(raw):
    suite = EvalSuite("coverage").add_case(CASE).add_evaluators(ExactMatch(), Faithfulness(threshold=1))
    with patch(CALL, return_value=raw) as call:
        report = suite.run(lambda _: "answer", workers=1, verbose=False)
    assert call.call_count == 1
    result = next(r for r in report.case_results[0].results if r.evaluator == "faithfulness")
    assert result.metadata["skipped"] and not result.passed
    assert result.metadata["verified_claims"] == 0
    restored = EvalReport.from_dict(json.loads(report.to_json()))
    policy = AcceptancePolicy((CheckRequirement("faithfulness", min_cases=1),))
    assert policy.evaluate(restored).decision == "indeterminate"


def test_raised_limit_checks_the_unsupported_eleventh_claim():
    claims = [f"claim {i}" for i in range(11)]
    with patch(CALL, side_effect=[json.dumps(claims)] + ["Yes"] * 10 + ["No"]) as call:
        result = Faithfulness(threshold=1, max_claims=11).evaluate(CASE, "long answer")
    assert call.call_count == 12
    assert result.score == pytest.approx(10 / 11)
    assert not result.passed and not result.metadata.get("skipped")
    assert result.metadata["unique_claims"] == result.metadata["verified_claims"] == 11
    assert len(result.metadata["verdict_responses"]) == 11


def test_duplicate_claims_cannot_inflate_the_supported_fraction():
    with patch(CALL, side_effect=['["supported", " supported ", "unsupported"]', "Yes", "No"]) as call:
        result = Faithfulness(threshold=.6).evaluate(CASE, "answer")
    assert call.call_count == 3
    assert result.score == .5 and not result.passed
    assert result.metadata["extracted_claims"] == 3
    assert result.metadata["unique_claims"] == 2
    assert result.metadata["claims"] == ["supported", "unsupported"]


def test_one_unknown_claim_does_not_become_a_perfect_score():
    with patch(CALL, side_effect=['["a", "b"]', "Yes", "unclear"]):
        suite = EvalSuite("unknown").add_case(CASE).add_evaluator(Faithfulness(threshold=1))
        report = suite.run(lambda _: "answer", workers=1, verbose=False)
    assert report.case_results[0].status == EvalStatus.JUDGE_ERROR
    assert report.evaluated == 0
    assert AcceptancePolicy((CheckRequirement("faithfulness", min_cases=1),)).evaluate(report).decision == "indeterminate"


def test_resolved_threshold_is_local_under_concurrent_grading():
    local = threading.local()
    barrier = threading.Barrier(2)
    evaluator = Faithfulness()

    def reply(prompt, config):
        if prompt.startswith("Extract"):
            return '["a", "b"]'
        if "Claim: a" in prompt:
            barrier.wait(timeout=5)
            return "Yes"
        return "No"

    def run(model):
        local.model = model
        return evaluator.evaluate(CASE, "answer")

    with (
        patch(CALL, side_effect=reply),
        patch("multivon_eval.evaluators.llm_judge.resolve_judge",
              side_effect=lambda _: JudgeConfig(provider="anthropic", model=local.model)),
        patch.object(evaluator, "_resolve_threshold", side_effect=lambda j: .9 if j.model == "strict" else .1),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        strict, lenient = list(pool.map(run, ["strict", "lenient"]))
    assert strict.score == lenient.score == .5
    assert not strict.passed and lenient.passed
    assert strict.metadata["threshold"] == .9 and lenient.metadata["threshold"] == .1
    assert evaluator.threshold == .7


def test_claim_limit_is_part_of_grading_contract():
    first = EvalSuite("claims").add_case(CASE).add_evaluator(Faithfulness(max_claims=10))
    second = EvalSuite("claims").add_case(CASE).add_evaluator(Faithfulness(max_claims=20))
    assert first.lock().suite_hash != second.lock().suite_hash
