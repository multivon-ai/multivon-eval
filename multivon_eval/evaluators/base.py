from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import Any

from ..case import EvalCase
from ..result import EvalResult


class Evaluator(ABC):
    """Base class for all evaluators.

    Subclasses must implement :meth:`evaluate` (sync). Optionally override
    :meth:`aevaluate` if a true-async judge path is available — the default
    runs :meth:`evaluate` in a worker thread, which is enough to unlock
    concurrency in :meth:`EvalSuite.run_async` because the GIL releases
    around blocking I/O (the network call to the judge).
    """

    name: str = "evaluator"
    threshold: float = 0.5

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        ...

    async def aevaluate(self, case: EvalCase, output: str, **kwargs: Any) -> EvalResult:
        """Async sibling of :meth:`evaluate`.

        Default: ``await asyncio.to_thread(self.evaluate, case, output, **kwargs)``.
        That's correct for any sync evaluator and lets the surrounding
        :meth:`EvalSuite.run_async` interleave many cases concurrently.

        Override when an evaluator has a genuinely async implementation
        (e.g. the LLM-judge evaluators that can use ``make_judge_call_async``
        to avoid the thread hop entirely).
        """
        if kwargs:
            return await asyncio.to_thread(self._call_evaluate, case, output, **kwargs)
        return await asyncio.to_thread(self.evaluate, case, output)

    def _call_evaluate(self, case: EvalCase, output: str, **kwargs: Any) -> EvalResult:
        """Internal — forwards kwargs to evaluate() for subclasses that accept them."""
        return self.evaluate(case, output, **kwargs)  # type: ignore[call-arg]

    def _result(self, score: float, reason: str = "", **metadata) -> EvalResult:
        return EvalResult(
            evaluator=self.name,
            score=score,
            passed=score >= self.threshold,
            reason=reason,
            metadata=metadata,
        )

    def _skipped(self, reason: str) -> EvalResult:
        """Return a passing EvalResult flagged as skipped.

        Use when the case shape doesn't fit this evaluator (no context for
        a RAG metric, no expected_output for an exact-match metric, no
        agent_trace for a tool metric). Scoring 0.0 in those situations
        punishes the user for the *absence* of ground truth rather than a
        real quality failure, and contaminates aggregate pass rates.

        The reason is prefixed with "[skipped]" so consumers can filter
        on the reason string without inspecting metadata. metadata.skipped
        is also set to True for structured filtering.
        """
        return EvalResult(
            evaluator=self.name,
            score=1.0,
            passed=True,
            reason=f"[skipped] {reason}",
            metadata={"skipped": True},
        )
