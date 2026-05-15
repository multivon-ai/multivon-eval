"""D13: per-case retry on transient judge / timeout errors.

Sarah persona: 10k-case weekend run shouldn't require Monday triage
for a single 429. Cases whose status is in ``JudgeRetry.retry_on`` are
re-evaluated end-to-end up to ``max_attempts`` times with exponential
backoff. The retry history lands on the CaseResult.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from multivon_eval import (
    EvalCase, EvalResult, EvalStatus, EvalSuite, JudgeConfig, JudgeRetry,
)
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.exceptions import JudgeUnavailable
from multivon_eval.retry import (
    async_sleep_for_attempt, should_retry, sleep_for_attempt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures: an evaluator that raises JudgeUnavailable N times then passes.
# ─────────────────────────────────────────────────────────────────────────────


class _FlakeyJudgeEvaluator(Evaluator):
    """Raises ``JudgeUnavailable`` for the first ``fail_n`` invocations,
    then returns a passing :class:`EvalResult`. Used to simulate a
    transient judge outage that recovers after a few retries."""

    name = "flakey_judge"

    def __init__(self, *, fail_n: int):
        super().__init__(threshold=0.5)
        self._fail_n = fail_n
        self.call_count = 0

    def evaluate(self, case, output: str) -> EvalResult:
        self.call_count += 1
        if self.call_count <= self._fail_n:
            raise JudgeUnavailable(f"transient 429 (call {self.call_count})")
        return EvalResult(self.name, 1.0, True, reason="recovered")

    async def aevaluate(self, case, output: str) -> EvalResult:
        # Mirror the sync logic so both paths can be exercised.
        self.call_count += 1
        if self.call_count <= self._fail_n:
            raise JudgeUnavailable(f"transient 429 (call {self.call_count})")
        return EvalResult(self.name, 1.0, True, reason="recovered")


def _basic_suite(evaluator) -> EvalSuite:
    suite = EvalSuite("retry-test")
    suite.add_case(EvalCase("ping", expected_output="pong"))
    suite.add_evaluator(evaluator)
    return suite


# ─────────────────────────────────────────────────────────────────────────────
# JudgeRetry dataclass validation
# ─────────────────────────────────────────────────────────────────────────────

def test_judge_retry_validates_max_attempts():
    with pytest.raises(ValueError, match="max_attempts"):
        JudgeRetry(max_attempts=0)


def test_judge_retry_validates_base_backoff():
    with pytest.raises(ValueError, match="base_backoff"):
        JudgeRetry(base_backoff=-1.0)


def test_judge_retry_validates_factor():
    with pytest.raises(ValueError, match="factor"):
        JudgeRetry(factor=0.5)  # would shrink backoff each retry — almost certainly a bug


def test_judge_retry_is_frozen():
    """Frozen so two suites sharing one policy can't mutate each other."""
    pol = JudgeRetry()
    with pytest.raises(Exception):
        pol.max_attempts = 99  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# backoff_for() — the deterministic + jitter math
# ─────────────────────────────────────────────────────────────────────────────

def test_backoff_for_first_attempt_is_zero():
    """No sleep before the FIRST attempt — that's the initial try, not a retry."""
    pol = JudgeRetry(base_backoff=2.0, factor=2.0, jitter=0)
    assert pol.backoff_for(1) == 0.0


def test_backoff_for_second_attempt_is_base():
    pol = JudgeRetry(base_backoff=2.0, factor=2.0, jitter=0)
    assert pol.backoff_for(2) == 2.0


def test_backoff_for_third_attempt_doubles():
    pol = JudgeRetry(base_backoff=2.0, factor=2.0, jitter=0)
    assert pol.backoff_for(3) == 4.0


def test_backoff_for_caps_at_max_backoff():
    pol = JudgeRetry(base_backoff=10.0, factor=10.0, jitter=0, max_backoff=15.0)
    # Attempt 4 deterministic = 10 * 10^2 = 1000 — capped at 15.
    assert pol.backoff_for(4) == 15.0


def test_backoff_for_jitter_stays_in_band(monkeypatch):
    """Jitter is symmetric around the deterministic value."""
    pol = JudgeRetry(base_backoff=10.0, factor=1.0, jitter=0.1, max_backoff=100.0)
    # Force the worst case at each end of the jitter window.
    monkeypatch.setattr("random.uniform", lambda a, b: a)  # returns the LOW end
    low = pol.backoff_for(2)
    monkeypatch.setattr("random.uniform", lambda a, b: b)
    high = pol.backoff_for(2)
    assert 8.9 <= low <= 9.0   # 10 - 10*0.1 ≈ 9
    assert 11.0 <= high <= 11.1  # 10 + 10*0.1 ≈ 11


# ─────────────────────────────────────────────────────────────────────────────
# should_retry / sleep_for_attempt helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_should_retry_default_policy_matches_judge_error():
    pol = JudgeRetry()
    assert should_retry(EvalStatus.JUDGE_ERROR, pol) is True
    assert should_retry(EvalStatus.TIMEOUT, pol) is True


def test_should_retry_excludes_quality_and_model_failures():
    """Quality failures and model errors are NOT retried by default —
    those are signal, not noise."""
    pol = JudgeRetry()
    assert should_retry(EvalStatus.FAILED_QUALITY, pol) is False
    assert should_retry(EvalStatus.MODEL_ERROR, pol) is False
    assert should_retry(EvalStatus.EVALUATOR_ERROR, pol) is False


def test_should_retry_accepts_enum_in_retry_on():
    """Both string values and EvalStatus members work in retry_on."""
    pol = JudgeRetry(retry_on=(EvalStatus.MODEL_ERROR,))
    assert should_retry(EvalStatus.MODEL_ERROR, pol) is True
    assert should_retry(EvalStatus.JUDGE_ERROR, pol) is False


def test_sleep_for_attempt_actually_sleeps(monkeypatch):
    """``sleep_for_attempt`` calls ``time.sleep`` with the computed backoff
    and returns the duration. The wall-clock value isn't asserted (CI is
    noisy) — we monkeypatch ``time.sleep`` to capture the request."""
    captured: list[float] = []
    monkeypatch.setattr("multivon_eval.retry.time.sleep", lambda s: captured.append(s))
    pol = JudgeRetry(base_backoff=3.0, factor=2.0, jitter=0)
    returned = sleep_for_attempt(pol, 3)  # 3.0 * 2.0^1 = 6.0
    assert returned == 6.0
    assert captured == [6.0]


def test_sleep_for_attempt_skips_when_zero(monkeypatch):
    """First attempt has no backoff — don't call ``time.sleep(0)``."""
    captured: list[float] = []
    monkeypatch.setattr("multivon_eval.retry.time.sleep", lambda s: captured.append(s))
    sleep_for_attempt(JudgeRetry(), 1)
    assert captured == []


# ─────────────────────────────────────────────────────────────────────────────
# Sync suite.run integration
# ─────────────────────────────────────────────────────────────────────────────

def test_suite_run_no_retry_by_default(monkeypatch):
    """Default behavior unchanged: a judge outage produces a single
    judge_error case with retry_attempts=0."""
    ev = _FlakeyJudgeEvaluator(fail_n=99)
    suite = _basic_suite(ev)
    monkeypatch.setattr("multivon_eval.retry.time.sleep", lambda _: None)

    report = suite.run(lambda _: "pong", verbose=False)
    cr = report.case_results[0]
    assert cr.status == EvalStatus.JUDGE_ERROR
    assert cr.retry_attempts == 0
    assert cr.retry_errors == []
    assert ev.call_count == 1


def test_suite_run_retries_judge_error_and_recovers(monkeypatch):
    """A 2-failure flake recovers on attempt 3 when max_attempts=3.
    Final CR is PASS, retry_attempts=2 (two failed prior attempts)."""
    monkeypatch.setattr("multivon_eval.retry.time.sleep", lambda _: None)
    ev = _FlakeyJudgeEvaluator(fail_n=2)
    suite = _basic_suite(ev)

    report = suite.run(
        lambda _: "pong", verbose=False,
        judge_retry=JudgeRetry(max_attempts=3, base_backoff=0, jitter=0),
    )
    cr = report.case_results[0]
    assert cr.status == EvalStatus.PASSED
    assert cr.retry_attempts == 2
    assert len(cr.retry_errors) == 2
    assert "429" in cr.retry_errors[0]
    assert ev.call_count == 3  # 2 failed + 1 successful


def test_suite_run_exhausts_retries_and_keeps_error_status(monkeypatch):
    """A persistent outage doesn't magically recover — after
    max_attempts the case is reported with JUDGE_ERROR, retry_attempts =
    max_attempts - 1, and ``retry_errors`` records every failed try."""
    monkeypatch.setattr("multivon_eval.retry.time.sleep", lambda _: None)
    ev = _FlakeyJudgeEvaluator(fail_n=99)
    suite = _basic_suite(ev)

    report = suite.run(
        lambda _: "pong", verbose=False,
        judge_retry=JudgeRetry(max_attempts=3, base_backoff=0, jitter=0),
    )
    cr = report.case_results[0]
    assert cr.status == EvalStatus.JUDGE_ERROR
    assert cr.retry_attempts == 2  # 2 retries that ALSO failed
    assert len(cr.retry_errors) == 2
    assert ev.call_count == 3  # 1 initial + 2 retries


def test_suite_run_does_not_retry_quality_failure(monkeypatch):
    """A FAILED_QUALITY case is NOT retried — quality failures are
    signal. Otherwise every CI run would 3x the cost on every failed
    case."""
    monkeypatch.setattr("multivon_eval.retry.time.sleep", lambda _: None)

    class _AlwaysFails(Evaluator):
        name = "always_fails"
        def __init__(self):
            super().__init__(threshold=0.5)
            self.call_count = 0
        def evaluate(self, case, output):
            self.call_count += 1
            return EvalResult(self.name, 0.0, False, reason="quality fail")

    ev = _AlwaysFails()
    suite = _basic_suite(ev)
    report = suite.run(
        lambda _: "pong", verbose=False,
        judge_retry=JudgeRetry(max_attempts=5, base_backoff=0, jitter=0),
    )
    assert report.case_results[0].status == EvalStatus.FAILED_QUALITY
    assert ev.call_count == 1   # NOT retried


def test_suite_run_retry_backs_off_between_attempts(monkeypatch):
    """The backoff sequence is base, base*factor, base*factor^2, …"""
    captured: list[float] = []
    monkeypatch.setattr("multivon_eval.retry.time.sleep", lambda s: captured.append(s))

    ev = _FlakeyJudgeEvaluator(fail_n=99)
    suite = _basic_suite(ev)
    suite.run(
        lambda _: "pong", verbose=False,
        judge_retry=JudgeRetry(max_attempts=4, base_backoff=1.0, factor=2.0, jitter=0),
    )
    # Sleeps before attempts 2, 3, 4 → 1.0, 2.0, 4.0.
    assert captured == [1.0, 2.0, 4.0]


# ─────────────────────────────────────────────────────────────────────────────
# Async path
# ─────────────────────────────────────────────────────────────────────────────

def test_suite_run_async_retries_judge_error(monkeypatch):
    """Async path mirrors sync: 2-failure flake recovers on attempt 3."""
    async def _no_sleep(s):
        return None
    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    ev = _FlakeyJudgeEvaluator(fail_n=2)
    suite = _basic_suite(ev)

    async def model_fn(_):
        return "pong"

    report = asyncio.run(suite.run_async(
        model_fn, verbose=False,
        judge_retry=JudgeRetry(max_attempts=3, base_backoff=0, jitter=0),
    ))
    cr = report.case_results[0]
    assert cr.status == EvalStatus.PASSED
    assert cr.retry_attempts == 2
    assert ev.call_count == 3


def test_async_sleep_for_attempt(monkeypatch):
    """asyncio variant skips the await when backoff is 0."""
    captured: list[float] = []

    async def _spy(s):
        captured.append(s)

    monkeypatch.setattr("asyncio.sleep", _spy)
    pol = JudgeRetry(base_backoff=1.5, factor=2.0, jitter=0)

    s = asyncio.run(async_sleep_for_attempt(pol, 3))
    assert s == 3.0
    assert captured == [3.0]

    captured.clear()
    asyncio.run(async_sleep_for_attempt(pol, 1))  # no sleep on first attempt
    assert captured == []


# ─────────────────────────────────────────────────────────────────────────────
# Parallel workers honor the policy
# ─────────────────────────────────────────────────────────────────────────────

def test_suite_run_workers_2_honors_retry_policy(monkeypatch):
    """Each parallel worker independently applies the retry policy to
    its case."""
    monkeypatch.setattr("multivon_eval.retry.time.sleep", lambda _: None)

    ev = _FlakeyJudgeEvaluator(fail_n=1)
    suite = EvalSuite("parallel-retry")
    suite.add_case(EvalCase("a", expected_output="A"))
    suite.add_case(EvalCase("b", expected_output="B"))
    suite.add_evaluator(ev)

    report = suite.run(
        lambda _: "ok", verbose=False, workers=2,
        judge_retry=JudgeRetry(max_attempts=3, base_backoff=0, jitter=0),
    )
    # Both cases pass after one retry each (or one case retries twice
    # depending on which thread races first — either way every case
    # ends up PASSED, since fail_n=1 across the SHARED evaluator means
    # exactly one call fails total).
    for cr in report.case_results:
        assert cr.status == EvalStatus.PASSED
