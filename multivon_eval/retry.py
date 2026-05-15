"""Per-case retry on transient judge / timeout errors.

Sarah persona (ML platform engineer running 10k cases on weekend cron)
needs the eval suite to self-heal through transient judge outages — a
single 429 from OpenAI shouldn't require Monday-morning triage. The
existing :class:`EvalStatus.JUDGE_ERROR` / :class:`EvalStatus.TIMEOUT`
classification lets us route exactly those cases to a retry policy
without disturbing real quality failures or evaluator bugs.

Usage::

    from multivon_eval import EvalSuite, JudgeRetry

    suite.run(
        model_fn,
        judge_retry=JudgeRetry(max_attempts=3, base_backoff=2.0),
    )

The retry happens at the case level (after :meth:`EvalSuite._run_case`
returns). If the case's status is in ``retry_on``, the case is
re-evaluated end-to-end up to ``max_attempts`` times. Each retry sleeps
``base_backoff * factor ** (attempt - 1)`` seconds, capped at
``max_backoff``, with optional symmetric jitter.

Final result records:
  - ``CaseResult.retry_attempts`` — number of retries performed
    (0 = no retry; max_attempts - 1 = exhausted)
  - ``CaseResult.retry_errors`` — error string per failed attempt

The retry policy is OPT-IN (default ``None``) so existing CI gates that
treat judge errors as actionable signal continue to fire immediately.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from .result import EvalStatus


# Which EvalStatus values are retriable by default. Quality failures
# (FAILED_QUALITY), model errors (MODEL_ERROR) and evaluator bugs
# (EVALUATOR_ERROR) are NOT retried — they are signals, not noise.
_DEFAULT_RETRY_ON: tuple[str, ...] = (
    EvalStatus.JUDGE_ERROR.value,
    EvalStatus.TIMEOUT.value,
)


@dataclass(frozen=True)
class JudgeRetry:
    """Retry policy for transient judge / timeout errors on a per-case basis.

    Attributes:
        max_attempts: Total attempts INCLUDING the first run. ``3`` means
            "try once, then retry up to twice." Must be ``>= 1``.
        base_backoff: Seconds to sleep before the FIRST retry. Subsequent
            retries multiply by ``factor``. Must be ``>= 0``.
        factor: Exponential multiplier per retry. ``2.0`` doubles the
            backoff each time. Must be ``>= 1.0``.
        jitter: Fraction of the computed backoff to add as symmetric
            random noise (``backoff * U(-jitter, +jitter)``). Avoids
            thundering-herd retries across parallel workers. ``0`` for
            no jitter.
        max_backoff: Upper bound on per-retry sleep (seconds). Keeps a
            long retry chain from spending 10 minutes on a single case.
        retry_on: Tuple of :class:`EvalStatus` string values to retry on.
            Defaults to ``("judge_error", "timeout")``. Setting this to
            ``()`` disables retry entirely; ``EvalStatus`` enum members
            (or their string values) are both accepted.

    Frozen so two suites sharing one policy instance can't accidentally
    mutate each other's config mid-run.
    """

    max_attempts: int = 3
    base_backoff: float = 1.0
    factor: float = 2.0
    jitter: float = 0.1
    max_backoff: float = 60.0
    retry_on: tuple[str, ...] = _DEFAULT_RETRY_ON

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                f"JudgeRetry.max_attempts must be >= 1, got {self.max_attempts}"
            )
        if self.base_backoff < 0:
            raise ValueError(
                f"JudgeRetry.base_backoff must be >= 0, got {self.base_backoff}"
            )
        if self.factor < 1.0:
            raise ValueError(
                f"JudgeRetry.factor must be >= 1.0, got {self.factor}"
            )
        if self.jitter < 0:
            raise ValueError(
                f"JudgeRetry.jitter must be >= 0, got {self.jitter}"
            )
        if self.max_backoff < 0:
            raise ValueError(
                f"JudgeRetry.max_backoff must be >= 0, got {self.max_backoff}"
            )

    def normalized_retry_on(self) -> frozenset[str]:
        """``retry_on`` coerced to a set of string values."""
        out: set[str] = set()
        for v in self.retry_on:
            out.add(v.value if isinstance(v, EvalStatus) else str(v))
        return frozenset(out)

    def backoff_for(self, attempt_index: int) -> float:
        """Seconds to sleep BEFORE attempt ``attempt_index`` (1-indexed).

        ``attempt_index=1`` returns 0 (no sleep before the first attempt).
        ``attempt_index=2`` returns ``base_backoff`` plus jitter.
        ``attempt_index=N`` returns ``base_backoff * factor ** (N - 2)``
        (clipped to ``max_backoff``) plus jitter.

        Jitter is symmetric around the deterministic value; callers can
        treat the return value as the actual sleep duration.
        """
        if attempt_index <= 1:
            return 0.0
        deterministic = self.base_backoff * (self.factor ** (attempt_index - 2))
        deterministic = min(deterministic, self.max_backoff)
        if self.jitter > 0 and deterministic > 0:
            spread = deterministic * self.jitter
            return max(0.0, deterministic + random.uniform(-spread, spread))
        return deterministic


def should_retry(status: EvalStatus, policy: JudgeRetry) -> bool:
    """True iff a case in ``status`` should be retried under ``policy``."""
    return status.value in policy.normalized_retry_on()


def sleep_for_attempt(policy: JudgeRetry, attempt_index: int) -> float:
    """Sleep ``policy.backoff_for(attempt_index)`` seconds; return the duration."""
    s = policy.backoff_for(attempt_index)
    if s > 0:
        time.sleep(s)
    return s


async def async_sleep_for_attempt(policy: JudgeRetry, attempt_index: int) -> float:
    """Async variant of :func:`sleep_for_attempt`."""
    import asyncio
    s = policy.backoff_for(attempt_index)
    if s > 0:
        await asyncio.sleep(s)
    return s


__all__ = ["JudgeRetry", "should_retry", "sleep_for_attempt", "async_sleep_for_attempt"]
