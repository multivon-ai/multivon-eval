"""
Structured exception hierarchy for multivon-eval.

CI and automation can branch on exception class instead of parsing error
strings:

    from multivon_eval import (
        MultivonError, JudgeUnavailable, CalibrationMissing, EvalGateFailure,
    )

    try:
        report = suite.run(model_fn, fail_threshold=0.85)
    except JudgeUnavailable as exc:
        # 3xx/5xx/rate-limit issue with the judge model — retry later
        ...
    except CalibrationMissing as exc:
        # Threshold table doesn't cover the configured judge model
        ...
    except EvalGateFailure as exc:
        # pass_rate fell below fail_threshold — non-zero CI exit
        ...
    except MultivonError as exc:
        # Catch-all for any library error
        ...

All library-raised errors derive from MultivonError. External (anthropic,
openai, litellm) exceptions raised inside make_judge_call() are wrapped in
JudgeUnavailable so callers don't need to know about provider SDKs.
"""
from __future__ import annotations


class MultivonError(Exception):
    """Root of every error multivon-eval raises directly."""


class JudgeUnavailable(MultivonError):
    """Raised when the configured judge model can't be reached or refused
    the request after retries (4xx/5xx/rate limit/timeout). The wrapped
    provider exception is preserved via __cause__."""

    def __init__(self, message: str, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class CalibrationMissing(MultivonError):
    """Raised when a strict caller asks for a calibrated threshold and the
    (evaluator, judge_model) pair has no entry in the calibration table.

    As of 0.7.3 the default fallback policy is ``"warn"`` — a UserWarning is
    emitted once per (evaluator, judge_model) pair and the call returns
    ``0.7``. Use ``calibrated_threshold(..., strict=True)`` or
    ``set_calibration_fallback_policy("strict")`` to raise this instead.
    Pre-0.7.3 silent behaviour can be restored with
    ``set_calibration_fallback_policy("silent")``.
    """

    def __init__(self, evaluator: str, judge_model: str) -> None:
        super().__init__(
            f"No calibration row for evaluator={evaluator!r}, judge_model={judge_model!r}. "
            f"Run benchmarks/run_threshold_calibration.py to produce one."
        )
        self.evaluator = evaluator
        self.judge_model = judge_model


class EvaluatorPrereqMissing(MultivonError):
    """Raised when an evaluator is run on a case that lacks a required field.

    Examples:
      - Faithfulness without case.context
      - ToolCallAccuracy without case.expected_tool_calls
      - ConversationRelevance without case.conversation
    """

    def __init__(self, evaluator: str, missing: str) -> None:
        super().__init__(
            f"{evaluator!r} requires case.{missing}, but it was not provided."
        )
        self.evaluator = evaluator
        self.missing = missing


class CacheError(MultivonError):
    """Raised by the judge-result cache when the on-disk store is unusable
    (corrupt SQLite file, permission denied, etc.). Callers can catch and
    proceed without caching."""


class SecretsError(MultivonError):
    """Raised by the secrets resolver when no resolver can produce a value
    for a required key."""

    def __init__(self, key: str, resolver: str = "") -> None:
        super().__init__(
            f"Secret {key!r} not found"
            + (f" via {resolver}" if resolver else "")
        )
        self.key = key
        self.resolver = resolver


class ComplianceError(MultivonError):
    """Raised by ComplianceReporter for hash-chain or anchor failures
    (e.g. anchor_fn raised, chain version mismatch, framework unknown)."""


# EvalGateFailure already exists in result.py for back-compat (SystemExit).
# We mark it as a sibling of MultivonError so users can catch either.
# The actual class definition stays in result.py to avoid an import cycle;
# this module only re-exports it after import for the public surface.
__all__ = [
    "MultivonError",
    "JudgeUnavailable",
    "CalibrationMissing",
    "EvaluatorPrereqMissing",
    "CacheError",
    "SecretsError",
    "ComplianceError",
]
