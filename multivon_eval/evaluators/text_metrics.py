"""Classical text-similarity evaluators.

Pure-Python implementations of metrics shipped by competing libraries
(Promptfoo's `levenshtein`, RAGAS's `ChrfScore`, etc.) that we lacked
prior to 0.7.0. Each evaluator is dependency-free and exposed via the
top-level package import.

Use these when you want a deterministic similarity score against a
reference output — cheaper and faster than an LLM-as-judge call, and
calibrated to a different signal (lexical / character-level similarity).
For semantic similarity, prefer :class:`BERTScore` from
``multivon_eval.evaluators.deterministic`` or one of the QAG-based
LLM-judge evaluators.
"""
from __future__ import annotations

from .base import Evaluator
from ..case import EvalCase
from ..result import EvalResult


def _levenshtein_distance(a: str, b: str) -> int:
    """Classic Wagner-Fischer edit distance with the O(min(m,n)) two-row
    optimization. Operates on characters (not tokens); for token-level
    distance you'd tokenize before calling.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure the shorter string drives the inner loop for memory efficiency.
    if len(a) < len(b):
        a, b = b, a

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,        # deletion
                curr_row[j - 1] + 1,    # insertion
                prev_row[j - 1] + cost, # substitution
            )
        prev_row = curr_row
    return prev_row[-1]


class Levenshtein(Evaluator):
    """Character-level edit-distance similarity against ``expected_output``.

    Score is normalized so that 1.0 = identical strings and 0.0 = entirely
    different (edit distance equal to the longer string's length).

    Useful for: short structured outputs (identifiers, formatted dates,
    SKUs), fuzzy-match acceptance, or detecting tiny regressions in
    deterministic-looking responses.

    Args:
        threshold:       Minimum similarity to pass. Default 0.8.
        case_sensitive:  If False (default), both strings are lowercased
                         before computing distance.

    Example::

        Levenshtein(threshold=0.95)  # nearly-exact match required
    """
    name = "levenshtein"

    def __init__(self, threshold: float = 0.8, case_sensitive: bool = False):
        super().__init__(threshold)
        self.case_sensitive = case_sensitive

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if case.expected_output is None:
            return self._skipped("Requires case.expected_output — supply the expected value to enable this evaluator.")
        a = output if self.case_sensitive else output.lower()
        b = case.expected_output if self.case_sensitive else case.expected_output.lower()
        if not a and not b:
            return self._result(1.0, "Both strings empty")
        dist = _levenshtein_distance(a, b)
        denom = max(len(a), len(b))
        score = 1.0 - (dist / denom) if denom else 1.0
        return self._result(
            score,
            f"Edit distance {dist} over {denom} chars → similarity {score:.3f}",
        )


def _char_ngrams(s: str, n: int) -> "list[str]":
    """All character n-grams in s. Returns empty list when len(s) < n."""
    if n <= 0 or len(s) < n:
        return []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def _ngram_precision_recall(
    hyp: "list[str]",
    ref: "list[str]",
) -> "tuple[float, float] | None":
    """Multiset precision + recall for one n-gram order.

    Returns ``None`` when either side has zero n-grams. Important for the
    chrF aggregation: short strings should SKIP the order entirely rather
    than contributing a zero — otherwise ``"a"`` vs ``"a"`` with ``max_n=6``
    would score ~1/6 because orders 2..6 emit empty lists.
    """
    if not hyp or not ref:
        return None
    from collections import Counter
    hyp_counts = Counter(hyp)
    ref_counts = Counter(ref)
    overlap = sum((hyp_counts & ref_counts).values())
    return overlap / len(hyp), overlap / len(ref)


class ChrfScore(Evaluator):
    """Character n-gram F-beta similarity against ``expected_output``.

    Implements chrF (Popović, 2015) using the standard sacreBLEU
    aggregation: average precision and average recall over orders
    1..``max_n``, then apply F-beta to the averages. This differs from
    averaging F-beta per order — important when precision/recall vary
    across orders.

    Default parameters match chrF++ defaults: ``max_n=6``, ``beta=2``
    (recall-favored). Whitespace is stripped by default to match
    sacreBLEU's behavior on word-tokenization-free baselines; pass
    ``include_whitespace=True`` to count spaces as characters.

    Args:
        max_n:     Highest character n-gram order (default 6).
        beta:      F-beta weight; >1 favors recall. Default 2.0.
        threshold: Minimum score to pass (default 0.5).
        case_sensitive: If False (default), lowercase both inputs.
        include_whitespace: If False (default), strip whitespace before
                            counting n-grams (sacreBLEU's chrF default).

    Example::

        ChrfScore(threshold=0.6, max_n=6)  # MT-style baseline
    """
    name = "chrf"

    def __init__(
        self,
        max_n: int = 6,
        beta: float = 2.0,
        threshold: float = 0.5,
        case_sensitive: bool = False,
        include_whitespace: bool = False,
    ):
        super().__init__(threshold)
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        self.max_n = max_n
        self.beta = beta
        self.case_sensitive = case_sensitive
        self.include_whitespace = include_whitespace

    def _prepare(self, s: str) -> str:
        if not self.case_sensitive:
            s = s.lower()
        if not self.include_whitespace:
            s = "".join(s.split())
        return s

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if case.expected_output is None:
            return self._skipped("Requires case.expected_output — supply the expected value to enable this evaluator.")
        hyp = self._prepare(output)
        ref = self._prepare(case.expected_output)
        if not hyp and not ref:
            return self._result(1.0, "Both strings empty")
        # Strings identical after normalization → 1.0 regardless of max_n.
        # Without this, "a" vs "a" with max_n=6 would score below 1.0
        # because orders 2..6 emit no n-grams.
        if hyp == ref:
            return self._result(1.0, "Exact match after normalization")

        precisions: list[float] = []
        recalls: list[float] = []
        orders_used: list[int] = []
        for n in range(1, self.max_n + 1):
            pr = _ngram_precision_recall(_char_ngrams(hyp, n), _char_ngrams(ref, n))
            if pr is None:
                continue   # order can't contribute (one side too short)
            precisions.append(pr[0])
            recalls.append(pr[1])
            orders_used.append(n)

        if not precisions:
            return self._result(
                0.0,
                f"No usable n-gram order (max_n={self.max_n}; strings too short)",
            )

        avg_p = sum(precisions) / len(precisions)
        avg_r = sum(recalls) / len(recalls)
        if avg_p + avg_r == 0:
            score = 0.0
        else:
            beta_sq = self.beta * self.beta
            score = (1 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)
        return self._result(
            score,
            f"chrF (orders={orders_used}, β={self.beta}, "
            f"avg_p={avg_p:.3f}, avg_r={avg_r:.3f}) → {score:.3f}",
        )


__all__ = ["Levenshtein", "ChrfScore"]
