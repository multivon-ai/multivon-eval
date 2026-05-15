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
    """Default retry_on is judge_error only — no code path produces
    TIMEOUT today (the enum exists for future use). See
    ``test_default_retry_on_excludes_timeout_until_wired`` below."""
    pol = JudgeRetry()
    assert should_retry(EvalStatus.JUDGE_ERROR, pol) is True
    assert should_retry(EvalStatus.TIMEOUT, pol) is False


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

def test_suite_run_workers_2_honors_retry_policy_per_case(monkeypatch):
    """Each parallel worker independently applies the retry policy to
    its case. Codex round-1 ISSUE 4: the original test shared one
    evaluator across both cases, so a single failure across BOTH was
    enough to make the test pass — per-case retry routing wasn't
    actually verified. Use a per-case evaluator wrapper instead, so
    each case independently sees fail_n=2 and must retry exactly twice.
    """
    monkeypatch.setattr("multivon_eval.retry.time.sleep", lambda _: None)

    class _PerCaseFlake(Evaluator):
        """Same flake pattern, but counts calls keyed by case.input —
        so each input sees ``fail_n`` failures BEFORE recovering."""
        name = "per_case_flake"

        def __init__(self, fail_n: int):
            super().__init__(threshold=0.5)
            self._fail_n = fail_n
            self._counts: dict[str, int] = {}

        def evaluate(self, case, output):
            key = case.input
            n = self._counts.get(key, 0) + 1
            self._counts[key] = n
            if n <= self._fail_n:
                raise JudgeUnavailable(f"transient on {key} call {n}")
            return EvalResult(self.name, 1.0, True, reason="recovered")

    ev = _PerCaseFlake(fail_n=2)
    suite = EvalSuite("parallel-retry")
    suite.add_case(EvalCase("a", expected_output="A"))
    suite.add_case(EvalCase("b", expected_output="B"))
    suite.add_evaluator(ev)

    report = suite.run(
        lambda _: "ok", verbose=False, workers=2,
        judge_retry=JudgeRetry(max_attempts=3, base_backoff=0, jitter=0),
    )
    # Each case independently saw 2 failed retries → final attempt passed.
    assert len(report.case_results) == 2
    for cr in report.case_results:
        assert cr.status == EvalStatus.PASSED
        assert cr.retry_attempts == 2, (
            f"each case should retry exactly twice; got {cr.retry_attempts} on {cr.case_input!r}"
        )
        assert len(cr.retry_errors) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Codex round-1 regressions
# ─────────────────────────────────────────────────────────────────────────────

def test_backoff_jitter_cannot_exceed_max_backoff(monkeypatch):
    """Codex ISSUE 1: max_backoff was applied BEFORE jitter, so a
    high-jitter draw could push actual sleep above the cap. Now jitter
    is applied first and the total is clamped."""
    pol = JudgeRetry(base_backoff=10.0, factor=1.0, jitter=0.5, max_backoff=12.0)
    # Force the HIGH end of the jitter window (10 + 5 = 15 deterministic).
    monkeypatch.setattr("random.uniform", lambda a, b: b)
    val = pol.backoff_for(2)
    assert val <= 12.0, f"jitter pushed actual sleep above max_backoff: {val}"
    # And the LOW end stays non-negative.
    monkeypatch.setattr("random.uniform", lambda a, b: a)
    val = pol.backoff_for(2)
    assert val >= 0.0


def test_default_retry_on_excludes_timeout_until_wired():
    """Codex ISSUE 2: there's no code path that classifies a case as
    EvalStatus.TIMEOUT today. Including ``timeout`` in the default
    ``retry_on`` would be dead config — at best confusing, at worst
    promising behavior that never fires. Default is judge_error only.
    Callers can opt in to timeout retry once they wire it themselves."""
    pol = JudgeRetry()
    assert "timeout" not in pol.normalized_retry_on()
    assert "judge_error" in pol.normalized_retry_on()

    # ...but the policy still ACCEPTS timeout if a caller opts in:
    pol = JudgeRetry(retry_on=("judge_error", "timeout"))
    assert "timeout" in pol.normalized_retry_on()


def test_retry_errors_length_matches_retry_attempts(monkeypatch):
    """Codex ISSUE 3: the docstring claimed retry_errors held every
    failed attempt, but exhausted retries omit the FINAL failure.
    Lock in the contract: len(retry_errors) == retry_attempts, ALWAYS.
    The final failure (when retries are exhausted) lives on the
    case's ``judge_error`` / ``status`` fields, not duplicated here."""
    monkeypatch.setattr("multivon_eval.retry.time.sleep", lambda _: None)

    for fail_n in (0, 1, 2, 5, 99):
        ev = _FlakeyJudgeEvaluator(fail_n=fail_n)
        suite = _basic_suite(ev)
        report = suite.run(
            lambda _: "pong", verbose=False,
            judge_retry=JudgeRetry(max_attempts=3, base_backoff=0, jitter=0),
        )
        cr = report.case_results[0]
        assert len(cr.retry_errors) == cr.retry_attempts, (
            f"contract violated for fail_n={fail_n}: "
            f"len(retry_errors)={len(cr.retry_errors)} retry_attempts={cr.retry_attempts}"
        )
