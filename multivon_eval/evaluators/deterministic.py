from __future__ import annotations
import json
import re
import math
from collections import Counter
from typing import Any

import jsonschema

from .base import Evaluator
from ..case import EvalCase
from ..result import EvalResult


class NotEmpty(Evaluator):
    """Passes if the output is non-empty after stripping whitespace."""
    name = "not_empty"

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        passed = bool(output.strip())
        return self._result(1.0 if passed else 0.0, "Non-empty" if passed else "Output is empty")


class ExactMatch(Evaluator):
    """Passes only if the output exactly matches expected_output (stripped)."""
    name = "exact_match"

    def __init__(self, case_sensitive: bool = False, threshold: float = 1.0):
        super().__init__(threshold)
        self.case_sensitive = case_sensitive

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if case.expected_output is None:
            return self._result(0.0, "No expected_output provided")
        a, b = output.strip(), case.expected_output.strip()
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()
        match = a == b
        return self._result(
            1.0 if match else 0.0,
            "Exact match" if match else f"Expected: {b!r} | Got: {a!r}",
        )


class Contains(Evaluator):
    """Passes if the output contains all required substrings (score = fraction found)."""
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
            "Pattern matched" if match else f"Pattern not found: {self.pattern.pattern!r}",
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
            if in_range else f"Word count {count} outside [{self.min_words}, {self.max_words}]",
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


class BLEU(Evaluator):
    """
    BLEU score between output and expected_output.

    Computes corpus BLEU up to n-grams (default 4). Score of 1.0 = perfect match.
    No external dependencies — pure Python implementation.
    """
    name = "bleu"

    def __init__(self, n: int = 4, threshold: float = 0.5):
        super().__init__(threshold)
        self.n = n

    def _ngrams(self, tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def _modified_precision(self, hypothesis: list[str], reference: list[str], n: int) -> tuple[int, int]:
        hyp_ngrams = self._ngrams(hypothesis, n)
        ref_ngrams = self._ngrams(reference, n)
        clipped = {ng: min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items()}
        return sum(clipped.values()), max(sum(hyp_ngrams.values()), 1)

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if case.expected_output is None:
            return self._result(0.0, "No expected_output provided")
        hyp = output.lower().split()
        ref = case.expected_output.lower().split()
        if not hyp:
            return self._result(0.0, "Empty output")

        # Brevity penalty
        bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))

        # Cap n to shortest sequence length
        max_n = min(self.n, len(hyp), len(ref))

        # Geometric mean of modified precisions
        log_avg = 0.0
        for i in range(1, max_n + 1):
            if len(hyp) < i or len(ref) < i:
                break
            num, denom = self._modified_precision(hyp, ref, i)
            if num == 0:
                log_avg = float("-inf")
                break
            log_avg += math.log(num / denom) / self.n

        score = bp * math.exp(log_avg) if log_avg != float("-inf") else 0.0
        score = max(0.0, min(1.0, score))
        return self._result(round(score, 4), f"BLEU-{max_n}: {score:.4f} (BP={bp:.3f})")


class ROUGE(Evaluator):
    """
    ROUGE-L score (longest common subsequence) between output and expected_output.

    Score of 1.0 = perfect recall and precision on LCS.
    No external dependencies — pure Python implementation.
    """
    name = "rouge_l"

    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)

    def _lcs_length(self, a: list[str], b: list[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(2)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
                else:
                    dp[i % 2][j] = max(dp[(i-1) % 2][j], dp[i % 2][j-1])
        return dp[m % 2][n]

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if case.expected_output is None:
            return self._result(0.0, "No expected_output provided")
        hyp = output.lower().split()
        ref = case.expected_output.lower().split()
        if not hyp or not ref:
            return self._result(0.0, "Empty output or reference")
        lcs = self._lcs_length(hyp, ref)
        precision = lcs / len(hyp)
        recall = lcs / len(ref)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return self._result(round(f1, 4), f"ROUGE-L F1={f1:.4f} (P={precision:.3f}, R={recall:.3f})")


class StartsWith(Evaluator):
    """Passes if output starts with the given prefix."""
    name = "starts_with"

    def __init__(self, prefix: str, case_sensitive: bool = False, threshold: float = 1.0):
        super().__init__(threshold)
        self.prefix = prefix
        self.case_sensitive = case_sensitive

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        text = output.strip() if self.case_sensitive else output.strip().lower()
        prefix = self.prefix if self.case_sensitive else self.prefix.lower()
        passed = text.startswith(prefix)
        return self._result(1.0 if passed else 0.0,
                            f"Starts with {self.prefix!r}" if passed else f"Does not start with {self.prefix!r}")


class MaxLatency(Evaluator):
    """Alias for Latency — passes if response time is under the limit."""
    name = "max_latency"

    def __init__(self, max_ms: float, threshold: float = 1.0):
        super().__init__(threshold)
        self.max_ms = max_ms

    def evaluate(self, case: EvalCase, output: str, latency_ms: float = 0.0) -> EvalResult:
        passed = latency_ms <= self.max_ms
        return self._result(
            1.0 if passed else max(0.0, 1.0 - (latency_ms - self.max_ms) / self.max_ms),
            f"{latency_ms:.0f}ms {'<=' if passed else '>'} {self.max_ms:.0f}ms",
        )
