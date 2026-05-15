"""Tests for the 0.7.0 HTML report enhancements.

Priya persona ask: "plain-language error messages — 'your model crashed
on case 3', not just 'score: 0.0'." Surfacing the EvalStatus enum as
distinctly-colored status badges + an Errors summary card so a reader
sees infrastructure failures separately from quality failures.
"""
from __future__ import annotations

import re

import pytest

from multivon_eval import (
    EvalReport, EvalResult, EvalStatus,
)
from multivon_eval.result import CaseResult
from multivon_eval.reporters.html import _status_pill, to_html


def _make_case(*, results=None, model_error=None, judge_error=None,
               evaluator_error=None, skipped=False, runs=1, pass_count=-1) -> CaseResult:
    return CaseResult(
        case_input="q",
        actual_output="ans",
        results=results or [],
        model_error=model_error,
        judge_error=judge_error,
        evaluator_error=evaluator_error,
        skipped=skipped,
        runs=runs,
        pass_count=pass_count,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _status_pill — per-case badge rendering
# ─────────────────────────────────────────────────────────────────────────────

def test_pill_passed_renders_pass_pill():
    cr = _make_case(results=[EvalResult("e", 1.0, True)])
    html = _status_pill(cr)
    assert "pill pass" in html
    assert ">PASS<" in html


def test_pill_failed_quality_renders_fail_pill():
    cr = _make_case(results=[EvalResult("e", 0.3, False)])
    html = _status_pill(cr)
    assert "pill fail" in html
    assert ">FAIL<" in html


def test_pill_model_error_distinguishes_from_quality_fail():
    """Codex's whole D1 thesis: judge_error ≠ failed_quality. The HTML
    must reflect that — a model crash gets a 'MODEL ERR' pill in the
    distinct error color (orange), with a tooltip explaining the kind."""
    cr = _make_case(
        results=[EvalResult("e", 0.0, False)],
        model_error="ConnectionRefused",
    )
    html = _status_pill(cr)
    assert "pill error" in html
    assert ">MODEL ERR<" in html
    assert "title=" in html and "not a quality issue" in html


def test_pill_judge_error_distinct_from_model_error():
    cr = _make_case(
        results=[EvalResult("e", 0.0, False)],
        judge_error="429 rate limit",
    )
    html = _status_pill(cr)
    assert "pill error" in html
    assert ">JUDGE ERR<" in html


def test_pill_evaluator_error_distinct_from_judge_error():
    cr = _make_case(
        results=[EvalResult("e", 0.0, False)],
        evaluator_error="ValueError: bad config",
    )
    html = _status_pill(cr)
    assert "pill error" in html
    assert ">EVAL ERR<" in html
    # The tooltip routes the reader differently from judge errors.
    assert "likely a bug" in html


def test_pill_skipped_uses_skipped_class():
    cr = _make_case(skipped=True, results=[EvalResult("e", 1.0, True)])
    html = _status_pill(cr)
    assert "pill skipped" in html
    assert ">SKIPPED<" in html
    # Not styled as an error — skipping is a deliberate choice.
    assert "pill error" not in html


def test_pill_flaky_takes_precedence_over_quality_outcome():
    """A flaky case is more actionable than its final majority-vote
    outcome — show the FLAKY badge regardless of whether the majority
    passed or failed."""
    cr = _make_case(
        results=[EvalResult("e", 1.0, True)],
        runs=5,
        pass_count=2,   # 2/5 → inconsistent → flaky
    )
    html = _status_pill(cr)
    assert ">FLAKY<" in html
    # Not classified as a passing case visually even though majority-vote
    # is "passed=True" on the aggregate per-evaluator row.
    assert "pill pass" not in html


# ─────────────────────────────────────────────────────────────────────────────
# Full HTML render: Errors / Skipped cards surface in the summary
# ─────────────────────────────────────────────────────────────────────────────

def test_html_includes_errors_card_when_errors_present():
    passing = _make_case(results=[EvalResult("e", 1.0, True)])
    judge_err = _make_case(judge_error="x", results=[EvalResult("e", 0.0, False)])
    model_err = _make_case(model_error="x", results=[EvalResult("e", 0.0, False)])
    report = EvalReport(suite_name="t", case_results=[passing, judge_err, model_err])
    html = to_html(report)
    # The Errors card is present with count 2 and a tooltip enumerating
    # the kinds.
    assert ">Errors<" in html
    # Count of errors is 2.
    assert re.search(r'class="val">2</span>\s*<span class="lbl">Errors', html)
    # Tooltip includes both kinds.
    assert "judge error" in html.lower()
    assert "model error" in html.lower()


def test_html_omits_errors_card_when_no_errors():
    """The Errors card should only appear when there are errors —
    keeping the summary uncluttered for clean runs."""
    report = EvalReport(
        suite_name="clean",
        case_results=[_make_case(results=[EvalResult("e", 1.0, True)])] * 3,
    )
    html = to_html(report)
    assert ">Errors<" not in html


def test_html_includes_skipped_card_when_skipped_present():
    s = _make_case(skipped=True)
    report = EvalReport(suite_name="t", case_results=[s, s])
    html = to_html(report)
    assert ">Skipped<" in html
    assert re.search(r'class="val">2</span>\s*<span class="lbl">Skipped', html)


def test_html_renders_distinct_pills_for_a_mixed_report():
    """End-to-end: a report with passing, failing, judge-error, and
    skipped cases must produce four DISTINCT pill classes in the
    output. Catches accidental regressions to a single-class pill."""
    report = EvalReport(
        suite_name="mix",
        case_results=[
            _make_case(results=[EvalResult("e", 1.0, True)]),                            # PASS
            _make_case(results=[EvalResult("e", 0.0, False)]),                           # FAIL
            _make_case(judge_error="x", results=[EvalResult("e", 0.0, False)]),          # JUDGE ERR
            _make_case(skipped=True),                                                    # SKIPPED
        ],
    )
    html = to_html(report)
    for css_class in ("pill pass", "pill fail", "pill error", "pill skipped"):
        assert css_class in html, f"expected {css_class!r} in HTML output"
