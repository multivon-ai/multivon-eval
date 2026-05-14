"""Tests for the 0.7.0 CaseResult.status enum + error-isolation semantics.

Three things being verified:

  1. ``EvalStatus`` correctly classifies every kind of terminal outcome.
  2. ``EvalReport.pass_rate`` excludes error cases from the denominator —
     a judge outage doesn't masquerade as a quality regression.
  3. The suite catches ``JudgeUnavailable`` from one evaluator without
     crashing the whole case (per-evaluator isolation).
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from multivon_eval import (
    EvalSuite, EvalCase, EvalReport, EvalResult, EvalStatus,
    EVALUATION_STATUSES, ERROR_STATUSES, JudgeUnavailable,
)
from multivon_eval.result import CaseResult


# ─────────────────────────────────────────────────────────────────────────────
# CaseResult.status classification
# ─────────────────────────────────────────────────────────────────────────────

def _make_case(*, results=None, model_error=None, judge_error=None,
               evaluator_error=None, skipped=False) -> CaseResult:
    return CaseResult(
        case_input="q",
        actual_output="ans",
        results=results or [],
        model_error=model_error,
        judge_error=judge_error,
        evaluator_error=evaluator_error,
        skipped=skipped,
    )


def test_status_passed_when_all_evaluators_pass():
    cr = _make_case(results=[EvalResult(evaluator="x", score=1.0, passed=True)])
    assert cr.status == EvalStatus.PASSED


def test_status_failed_quality_when_one_evaluator_fails():
    cr = _make_case(results=[
        EvalResult(evaluator="x", score=1.0, passed=True),
        EvalResult(evaluator="y", score=0.4, passed=False, reason="below threshold"),
    ])
    assert cr.status == EvalStatus.FAILED_QUALITY


def test_status_model_error_takes_precedence_over_quality():
    """Even if all evaluators returned passed=False (because they were skipped
    by the suite due to model_error), status should be MODEL_ERROR, not FAILED_QUALITY."""
    cr = _make_case(
        results=[EvalResult(evaluator="x", score=0.0, passed=False)],
        model_error="ConnectionError: refused",
    )
    assert cr.status == EvalStatus.MODEL_ERROR


def test_status_judge_error_when_judge_unavailable():
    cr = _make_case(
        results=[EvalResult(evaluator="x", score=0.0, passed=False)],
        judge_error="Judge call failed: 429 rate limit",
    )
    assert cr.status == EvalStatus.JUDGE_ERROR


def test_status_evaluator_error_when_evaluator_crashed():
    cr = _make_case(
        results=[EvalResult(evaluator="x", score=0.0, passed=False)],
        evaluator_error="ValueError: bad input",
    )
    assert cr.status == EvalStatus.EVALUATOR_ERROR


def test_status_skipped_overrides_everything():
    """If the case was explicitly skipped, that's the status — even if a
    judge error was also captured (skipping should bypass evaluators
    entirely, but the precedence rule defends against future bugs)."""
    cr = _make_case(
        results=[EvalResult(evaluator="x", score=1.0, passed=True)],
        skipped=True,
        judge_error="leftover",
    )
    assert cr.status == EvalStatus.SKIPPED


def test_eval_status_is_string_subclass():
    """JSON-serializable as-is; comparable to literal strings."""
    assert EvalStatus.PASSED == "passed"
    assert json.dumps(EvalStatus.PASSED.value) == '"passed"'


# ─────────────────────────────────────────────────────────────────────────────
# EvalReport — error cases excluded from pass_rate denominators
# ─────────────────────────────────────────────────────────────────────────────

def _report(*cases: CaseResult) -> EvalReport:
    return EvalReport(suite_name="t", case_results=list(cases))


def test_pass_rate_excludes_error_cases_from_denominator():
    """Two cases passed, three cases had judge errors. pass_rate should be
    2/2 = 1.0, not 2/5 = 0.4. Errors are surfaced separately via .errors."""
    passing = _make_case(results=[EvalResult("x", 1.0, True)])
    failing = _make_case(results=[EvalResult("x", 0.2, False)])
    judge_err = _make_case(judge_error="429")

    report = _report(passing, passing, failing, judge_err, judge_err)
    # 3 evaluated (2 pass + 1 quality fail), 2 errors.
    assert report.evaluated == 3
    assert report.passed == 2
    assert report.failed == 1
    assert report.errors == 2
    # pass_rate is 2/3 (not 2/5).
    assert abs(report.pass_rate - 2 / 3) < 1e-9


def test_avg_score_excludes_error_cases():
    """Two cases scored 1.0, one error case (score 0.0). avg_score = 1.0
    not 0.67, because the error case is excluded from the average."""
    good = _make_case(results=[EvalResult("x", 1.0, True)])
    err = _make_case(results=[EvalResult("x", 0.0, False)], model_error="boom")
    report = _report(good, good, err)
    assert report.avg_score == 1.0


def test_errors_by_kind_breakdown():
    """Surface counts per error kind so users can distinguish 1 model_error
    + 2 judge_error vs 3 of the same kind."""
    me = _make_case(model_error="boom")
    je1 = _make_case(judge_error="429")
    je2 = _make_case(judge_error="auth")
    report = _report(me, je1, je2)
    breakdown = report.errors_by_kind
    assert breakdown == {"model_error": 1, "judge_error": 2}


def test_skipped_cases_excluded_from_both_evaluated_and_errors():
    skipped = _make_case(skipped=True)
    report = _report(skipped, skipped)
    assert report.evaluated == 0
    assert report.errors == 0
    assert report.skipped == 2
    assert report.pass_rate == 0.0  # no evaluated → 0 by convention


def test_pass_rate_when_all_cases_are_errors_is_zero():
    """Edge case: every case errored. denom=0 → pass_rate=0.0 (matches the
    pre-existing convention for empty reports)."""
    err = _make_case(judge_error="boom")
    report = _report(err, err)
    assert report.pass_rate == 0.0
    assert report.errors == 2


def test_evaluation_statuses_are_the_two_real_outcomes():
    """The frozenset constants must stay in sync with the enum split."""
    assert EVALUATION_STATUSES == {EvalStatus.PASSED, EvalStatus.FAILED_QUALITY}
    assert ERROR_STATUSES == {
        EvalStatus.MODEL_ERROR, EvalStatus.JUDGE_ERROR,
        EvalStatus.EVALUATOR_ERROR, EvalStatus.TIMEOUT,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Suite-level integration: one judge-failing evaluator does NOT crash the case
# ─────────────────────────────────────────────────────────────────────────────

class _AlwaysJudgeFail:
    """Evaluator that always raises JudgeUnavailable. Lets us simulate a
    transient outage without mocking deep internals."""
    name = "always_fails"
    threshold = 0.7

    def evaluate(self, case, output):
        raise JudgeUnavailable("Judge call failed: simulated 429")


class _AlwaysPass:
    name = "always_passes"
    threshold = 0.7

    def evaluate(self, case, output):
        from multivon_eval import EvalResult
        return EvalResult(evaluator="always_passes", score=1.0, passed=True, reason="ok")


def test_judge_unavailable_does_not_crash_other_evaluators():
    """If JudgeUnavailable propagates out of one evaluator, the rest of the
    case's evaluators should still run (per-evaluator isolation). The case
    gets ``judge_error`` set; the passing evaluator still records its result."""
    suite = EvalSuite("isolation-test")
    suite.add_cases([EvalCase(input="x")])
    suite.add_evaluators(_AlwaysJudgeFail(), _AlwaysPass())

    # No fail_threshold → suite.run returns rather than raising EvalGateFailure.
    report = suite.run(lambda i: "ans", verbose=False)
    cr = report.case_results[0]

    # The passing evaluator still ran and recorded its result.
    passes = [r for r in cr.results if r.evaluator == "always_passes"]
    assert len(passes) == 1 and passes[0].passed is True

    # The failing evaluator surfaced as a judge-unavailable reason.
    fails = [r for r in cr.results if r.evaluator == "always_fails"]
    assert len(fails) == 1
    assert "judge unavailable" in fails[0].reason.lower()

    # The case is classified as JUDGE_ERROR (precedence over FAILED_QUALITY).
    assert cr.status == EvalStatus.JUDGE_ERROR
    # And the report sees 0 evaluated cases, 1 error.
    assert report.evaluated == 0
    assert report.errors == 1


def test_evaluator_error_distinct_from_judge_error():
    """A non-JudgeUnavailable exception in an evaluator becomes evaluator_error,
    not judge_error — important for downstream code that wants to retry on
    judge outages but bubble up real bugs."""
    class _BadEval:
        name = "bad"
        threshold = 0.7
        def evaluate(self, case, output):
            raise ValueError("evaluator has a bug")

    suite = EvalSuite("evaluator-bug")
    suite.add_cases([EvalCase(input="x")])
    suite.add_evaluators(_BadEval())
    report = suite.run(lambda i: "ans", verbose=False)
    cr = report.case_results[0]
    assert cr.status == EvalStatus.EVALUATOR_ERROR
    assert cr.judge_error is None
    assert "evaluator has a bug" in (cr.evaluator_error or "")


def test_serialization_round_trip_preserves_status_fields():
    """to_json + from_dict must preserve the new error fields so saved
    reports keep their status classification when reloaded."""
    me = _make_case(model_error="boom")
    je = _make_case(judge_error="429")
    good = _make_case(results=[EvalResult("x", 1.0, True)])
    report = _report(good, me, je)
    blob = json.loads(report.to_json())
    rt = EvalReport.from_dict(blob)
    statuses = [cr.status for cr in rt.case_results]
    assert statuses == [EvalStatus.PASSED, EvalStatus.MODEL_ERROR, EvalStatus.JUDGE_ERROR]


# ─────────────────────────────────────────────────────────────────────────────
# Round-2 codex review regressions
# ─────────────────────────────────────────────────────────────────────────────

def test_aggregate_runs_propagates_error_fields():
    """When runs > 1 and one of the runs errors, the aggregated CaseResult
    must surface the error (not silently downgrade to FAILED_QUALITY).
    Codex round-2 P1 finding."""
    from multivon_eval.suite import _aggregate_runs

    case = EvalCase(input="x")
    good_run = _make_case(results=[EvalResult("e", 1.0, True)])
    judge_err_run = _make_case(judge_error="429 rate limit",
                                results=[EvalResult("e", 0.0, False)])
    agg = _aggregate_runs(case, [good_run, judge_err_run, good_run])
    assert agg.judge_error is not None and "429" in agg.judge_error
    assert agg.status == EvalStatus.JUDGE_ERROR


def test_pass_rate_ci_uses_evaluated_not_total():
    """pass_rate_ci must use the same denominator as pass_rate so the
    confidence interval describes the metric being reported.
    Codex round-2 P1/P2 finding."""
    passing = _make_case(results=[EvalResult("e", 1.0, True)])
    err = _make_case(judge_error="boom")
    report = _report(passing, passing, passing, err, err)
    # 3 of 3 evaluated passed; the Wilson lower bound on 3/3 ≠ 3/5.
    lo, hi = report.pass_rate_ci()
    # Bound must be >= the lo for 3/3 (n=3), which is much higher than 3/5.
    lo_3_3, _ = (lambda: __import__("multivon_eval.experiments", fromlist=["wilson_interval"]).wilson_interval(3, 3))()
    assert lo == pytest.approx(lo_3_3, abs=1e-9)


def test_run_on_cases_isolates_judge_unavailable():
    """The imported-trace path (run_on_cases) must also catch JudgeUnavailable
    so one outage doesn't crash the whole batch. Codex round-2 P2 finding."""
    suite = EvalSuite("imported-isolation")
    suite.add_cases([EvalCase(input="x")])
    suite.add_evaluators(_AlwaysJudgeFail(), _AlwaysPass())
    report = suite.run_on_cases([(EvalCase(input="x"), "out")], verbose=False)
    cr = report.case_results[0]
    assert cr.status == EvalStatus.JUDGE_ERROR
    # Both evaluators recorded results (the passing one wasn't crashed).
    assert len(cr.results) == 2


def test_async_error_classification_uses_structured_metadata():
    """The async path should use EvalResult.metadata['error_kind'] for
    classification, not string-parse the human-readable reason. Codex
    round-2 P2 finding.

    Indirect check: an evaluator whose reason TEXT happens to start with
    '[judge unavailable:' but whose metadata says no error must NOT be
    classified as judge_error."""
    import asyncio
    from multivon_eval import EvalResult

    class _ReasonMimicry:
        name = "innocent"
        threshold = 0.7
        async def aevaluate(self, case, output):
            # Reason text mimicry; metadata says no error.
            return EvalResult(
                evaluator="innocent", score=1.0, passed=True,
                reason="[judge unavailable: but this is actually a benign note]",
                metadata={},  # no error_kind
            )
        def evaluate(self, case, output):
            import asyncio
            return asyncio.run(self.aevaluate(case, output))

    suite = EvalSuite("mimicry")
    suite.add_cases([EvalCase(input="x")])
    suite.add_evaluators(_ReasonMimicry())

    async def model(i): return "out"
    report = asyncio.run(suite.run_async(model, verbose=False))
    cr = report.case_results[0]
    # The case should be PASSED, not misclassified as JUDGE_ERROR from the
    # reason string.
    assert cr.status == EvalStatus.PASSED
    assert cr.judge_error is None


def test_passed_and_status_agree_on_empty_results():
    """A case with no evaluator results — `passed` and `status` must agree.
    P3 finding: status said FAILED_QUALITY but passed said True (empty all())."""
    cr = _make_case(results=[])
    assert cr.passed is False
    assert cr.status == EvalStatus.FAILED_QUALITY


def test_passed_false_when_case_in_error_state():
    """Even if some individual evaluator result has passed=True, if the case
    is in an error state, the case-level .passed must be False."""
    cr = _make_case(
        results=[EvalResult("e", 1.0, True)],
        judge_error="429",
    )
    assert cr.passed is False
    assert cr.status == EvalStatus.JUDGE_ERROR
