"""Tests for the classical text-similarity evaluators added in 0.7.0.

These are pure-Python implementations with no LLM calls, so the tests
run in microseconds and don't require API keys.
"""
from __future__ import annotations

import pytest

from multivon_eval import Levenshtein, ChrfScore, EvalCase
from multivon_eval.evaluators.text_metrics import _levenshtein_distance, _char_ngrams


# ─────────────────────────────────────────────────────────────────────────────
# _levenshtein_distance — the pure algorithm
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,expected", [
    ("", "", 0),
    ("abc", "abc", 0),
    ("abc", "", 3),
    ("", "abc", 3),
    ("abc", "abd", 1),                  # one substitution
    ("kitten", "sitting", 3),           # classic Levenshtein example
    ("sunday", "saturday", 3),
])
def test_levenshtein_distance_known_pairs(a, b, expected):
    assert _levenshtein_distance(a, b) == expected


def test_levenshtein_distance_is_symmetric():
    """edit distance is metric → d(a,b) == d(b,a)."""
    pairs = [("flaw", "lawn"), ("intention", "execution"), ("abc", "")]
    for a, b in pairs:
        assert _levenshtein_distance(a, b) == _levenshtein_distance(b, a), (a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Levenshtein evaluator
# ─────────────────────────────────────────────────────────────────────────────

def test_levenshtein_identical_strings_score_one():
    ev = Levenshtein()
    case = EvalCase(input="x", expected_output="Hello world")
    res = ev.evaluate(case, "Hello world")
    assert res.score == 1.0
    assert res.passed is True


def test_levenshtein_completely_different_strings_score_zero():
    """Different strings of equal length with no shared chars → score 0."""
    ev = Levenshtein()
    case = EvalCase(input="x", expected_output="aaa")
    res = ev.evaluate(case, "bbb")
    assert res.score == 0.0


def test_levenshtein_partial_match():
    """3 chars same out of 4 → similarity 3/4 = 0.75."""
    ev = Levenshtein(threshold=0.7)
    case = EvalCase(input="x", expected_output="abcd")
    res = ev.evaluate(case, "abce")
    assert abs(res.score - 0.75) < 1e-9
    assert res.passed is True   # 0.75 >= 0.7


def test_levenshtein_threshold_gates_pass_fail():
    case = EvalCase(input="x", expected_output="hello")
    near = "hxllo"   # one substitution → similarity 0.8
    assert Levenshtein(threshold=0.9).evaluate(case, near).passed is False
    assert Levenshtein(threshold=0.7).evaluate(case, near).passed is True


def test_levenshtein_case_insensitive_by_default():
    ev = Levenshtein()
    case = EvalCase(input="x", expected_output="Hello")
    assert ev.evaluate(case, "hello").score == 1.0


def test_levenshtein_case_sensitive_distinguishes_case():
    ev = Levenshtein(case_sensitive=True)
    case = EvalCase(input="x", expected_output="Hello")
    assert ev.evaluate(case, "hello").score < 1.0


def test_levenshtein_missing_expected_output_returns_zero():
    """Without an expected_output to compare to, we can't compute distance.
    Surfaces as a clear reason rather than crashing."""
    ev = Levenshtein()
    case = EvalCase(input="x")  # no expected_output
    res = ev.evaluate(case, "anything")
    assert res.passed and res.metadata.get('skipped') and res.reason.startswith('[skipped]')
    assert "expected_output" in res.reason.lower()


def test_levenshtein_both_empty_scores_one():
    """Edge case: empty model output AND empty expected. They're equal."""
    ev = Levenshtein()
    case = EvalCase(input="x", expected_output="")
    res = ev.evaluate(case, "")
    assert res.score == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _char_ngrams + ChrfScore
# ─────────────────────────────────────────────────────────────────────────────

def test_char_ngrams_basic():
    assert _char_ngrams("hello", 2) == ["he", "el", "ll", "lo"]
    assert _char_ngrams("hi", 3) == []   # shorter than n
    assert _char_ngrams("", 1) == []


def test_chrf_identical_strings_score_one():
    ev = ChrfScore()
    case = EvalCase(input="x", expected_output="The cat sat on the mat.")
    res = ev.evaluate(case, "The cat sat on the mat.")
    assert res.score == 1.0


def test_chrf_disjoint_chars_score_low():
    """Strings with no character overlap should score ~0."""
    ev = ChrfScore()
    case = EvalCase(input="x", expected_output="abcdef")
    res = ev.evaluate(case, "ghijkl")
    assert res.score == 0.0


def test_chrf_partial_overlap_scores_between_zero_and_one():
    ev = ChrfScore(threshold=0.3)
    case = EvalCase(input="x", expected_output="The quick brown fox jumps")
    res = ev.evaluate(case, "A quick fox")
    # Partial char overlap; well within (0, 1).
    assert 0.0 < res.score < 1.0


def test_chrf_invalid_max_n_raises():
    with pytest.raises(ValueError):
        ChrfScore(max_n=0)


def test_chrf_reason_includes_beta_and_orders():
    ev = ChrfScore(max_n=4, beta=1.5)
    # Distinct strings so we exercise the aggregation path
    # (identical strings short-circuit before the reason is built).
    res = ev.evaluate(EvalCase(input="x", expected_output="abcd"), "abce")
    assert "β=1.5" in res.reason
    assert "orders=" in res.reason


# Codex review regressions ────────────────────────────────────────────────────

def test_chrf_short_exact_strings_score_one():
    """Codex P1: 'a' vs 'a' with default max_n=6 used to score ~1/6
    because orders 2..6 emitted no n-grams. Exact match must always be
    1.0 regardless of max_n."""
    ev = ChrfScore(max_n=6)
    assert ev.evaluate(EvalCase(input="x", expected_output="a"), "a").score == 1.0
    assert ev.evaluate(EvalCase(input="x", expected_output="abc"), "abc").score == 1.0


def test_chrf_uses_avg_p_avg_r_then_fbeta_not_avg_fbeta():
    """Codex P1: chrF averages precision per order, recall per order,
    then applies F-beta to the AVERAGES — not F-beta per order then
    averaged. The two diverge when precision/recall differ by order.

    For a precision-heavy short hypothesis on a longer reference, the
    correct chrF aggregation produces a score close to ``recall``
    (because beta=2 weights recall). The reason-string surfaces the
    average precision and average recall so we can directly assert the
    aggregation is the standard one.
    """
    ev = ChrfScore(max_n=6, beta=2.0)
    res = ev.evaluate(
        EvalCase(input="x", expected_output="The cat sat on the mat"),
        "the cat",
    )
    # Reason confirms standard aggregation: avg_p first, then avg_r,
    # then F-beta on those averages.
    assert "avg_p=" in res.reason and "avg_r=" in res.reason
    # avg_p should be high (precision-heavy short hypothesis); avg_r low.
    # Score = F-beta of (avg_p, avg_r). Beta=2 favors recall.
    # For avg_p ≈ 1.0, avg_r ≈ 0.23, F2 = 5 * 1 * 0.23 / (4 + 0.23) ≈ 0.27.
    # Just verify it's bounded between recall-ish and precision-ish.
    assert 0.1 < res.score < 0.5, f"unexpected chrF score: {res.score}"


def test_chrf_strips_whitespace_by_default():
    """Codex P2: sacreBLEU's chrF default doesn't count spaces. Default
    must match sacreBLEU; with whitespace included the score moves about
    +0.04 on this fixture."""
    ev = ChrfScore()
    res = ev.evaluate(
        EvalCase(input="x", expected_output="The Eiffel Tower is located in Paris."),
        "The Eiffel Tower is located in India.",
    )
    # Allow some tolerance — but we ARE testing the no-whitespace value,
    # not the higher with-whitespace number (~0.84).
    assert 0.78 < res.score < 0.82, f"expected ~0.80 (sacreBLEU default), got {res.score}"


def test_chrf_with_whitespace_opt_in():
    """include_whitespace=True restores the whitespace-counting behavior
    for users who want it explicitly."""
    ev = ChrfScore(include_whitespace=True)
    res = ev.evaluate(
        EvalCase(input="x", expected_output="The Eiffel Tower is located in Paris."),
        "The Eiffel Tower is located in India.",
    )
    assert res.score > 0.82, (
        f"include_whitespace should produce the higher chrF, got {res.score}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API integration
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluators_importable_from_top_level():
    """0.7.0 surfaces these as top-level imports."""
    import multivon_eval
    assert hasattr(multivon_eval, "Levenshtein")
    assert hasattr(multivon_eval, "ChrfScore")
    # Construction should work without any keyword args.
    multivon_eval.Levenshtein()
    multivon_eval.ChrfScore()


def test_evaluators_work_in_a_suite():
    """End-to-end smoke: drop into an EvalSuite, run, get a sensible report."""
    from multivon_eval import EvalSuite, EvalCase
    suite = EvalSuite("text-metrics-smoke")
    suite.add_cases([
        EvalCase(input="q1", expected_output="Hello world"),
        EvalCase(input="q2", expected_output="Goodbye"),
    ])
    suite.add_evaluators(Levenshtein(threshold=0.9), ChrfScore(threshold=0.5))

    def model_fn(inp: str) -> str:
        return "Hello world" if inp == "q1" else "Farewell"  # partial match for q2

    report = suite.run(model_fn, verbose=False)
    # q1 is an exact match → both evaluators pass.
    # q2 has only partial overlap → at least one threshold fails.
    case_results = report.case_results
    assert case_results[0].passed is True
    assert case_results[1].passed is False
