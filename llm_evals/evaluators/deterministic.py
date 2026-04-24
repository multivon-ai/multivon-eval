from __future__ import annotations
import json
import re
from typing import Any

import jsonschema

from .base import Evaluator
from ..case import EvalCase
from ..result import EvalResult


class ExactMatch(Evaluator):
    """Passes only if the output exactly matches expected_output (stripped)."""
    name = "exact_match"

    def __init__(self, case_sensitive: bool = False, threshold: float = 1.0):
        super().__init__(threshold)
        self.case_sensitive = case_sensitive

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if case.expected_output is None:
            return self._result(0.0, "No expected_output provided")
        a = output.strip()
        b = case.expected_output.strip()
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()
        match = a == b
        return self._result(
            1.0 if match else 0.0,
            "Exact match" if match else f"Expected: {b!r} | Got: {a!r}",
        )


class Contains(Evaluator):
    """Passes if the output contains all required substrings."""
    name = "contains"

    def __init__(self, substrings: list[str], case_sensitive: bool = False, threshold: float = 1.0):
        super().__init__(threshold)
        self.substrings = substrings
        self.case_sensitive = case_sensitive

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        text = output if self.case_sensitive else output.lower()
        missing = [s for s in self.substrings if (s if self.case_sensitive else s.lower()) not in text]
        score = 1.0 - (len(missing) / len(self.substrings)) if self.substrings else 1.0
        reason = "All substrings found" if not missing else f"Missing: {missing}"
        return self._result(score, reason)


class RegexMatch(Evaluator):
    """Passes if the output matches a regex pattern."""
    name = "regex_match"

    def __init__(self, pattern: str, flags: int = re.IGNORECASE, threshold: float = 1.0):
        super().__init__(threshold)
        self.pattern = re.compile(pattern, flags)

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        match = bool(self.pattern.search(output))
        return self._result(
            1.0 if match else 0.0,
            f"Pattern matched" if match else f"Pattern not found: {self.pattern.pattern!r}",
        )


class JSONSchemaEval(Evaluator):
    """Passes if the output is valid JSON matching a JSON Schema."""
    name = "json_schema"

    def __init__(self, schema: dict[str, Any], threshold: float = 1.0):
        super().__init__(threshold)
        self.schema = schema

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        try:
            data = json.loads(output.strip())
        except json.JSONDecodeError as e:
            return self._result(0.0, f"Invalid JSON: {e}")
        try:
            jsonschema.validate(data, self.schema)
            return self._result(1.0, "Valid JSON matching schema")
        except jsonschema.ValidationError as e:
            return self._result(0.0, f"Schema violation: {e.message}")


class NotEmpty(Evaluator):
    """Passes if the output is non-empty after stripping whitespace."""
    name = "not_empty"

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        passed = bool(output.strip())
        return self._result(1.0 if passed else 0.0, "Non-empty" if passed else "Output is empty")


class WordCount(Evaluator):
    """Passes if the word count is within [min_words, max_words]."""
    name = "word_count"

    def __init__(self, min_words: int = 0, max_words: int = 10_000, threshold: float = 1.0):
        super().__init__(threshold)
        self.min_words = min_words
        self.max_words = max_words

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        count = len(output.split())
        in_range = self.min_words <= count <= self.max_words
        return self._result(
            1.0 if in_range else 0.0,
            f"Word count {count} in [{self.min_words}, {self.max_words}]"
            if in_range
            else f"Word count {count} outside [{self.min_words}, {self.max_words}]",
        )


class Latency(Evaluator):
    """Passes if response latency is under max_ms milliseconds."""
    name = "latency"

    def __init__(self, max_ms: float, threshold: float = 1.0):
        super().__init__(threshold)
        self.max_ms = max_ms

    def evaluate(self, case: EvalCase, output: str, latency_ms: float = 0.0) -> EvalResult:
        passed = latency_ms <= self.max_ms
        return self._result(
            1.0 if passed else max(0.0, 1.0 - (latency_ms - self.max_ms) / self.max_ms),
            f"{latency_ms:.0f}ms {'<=' if passed else '>'} {self.max_ms:.0f}ms limit",
        )
