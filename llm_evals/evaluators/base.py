from __future__ import annotations
from abc import ABC, abstractmethod
from ..case import EvalCase
from ..result import EvalResult


class Evaluator(ABC):
    """Base class for all evaluators."""

    name: str = "evaluator"
    threshold: float = 0.5

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        ...

    def _result(self, score: float, reason: str = "", **metadata) -> EvalResult:
        return EvalResult(
            evaluator=self.name,
            score=score,
            passed=score >= self.threshold,
            reason=reason,
            metadata=metadata,
        )
