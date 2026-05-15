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
    """A flaky case is more actionable than the quality-outcome it
    would otherwise carry. Because :class:`CaseResult.passed` requires
    ``pass_count == runs``, any flaky case is ``FAILED_QUALITY`` at
    the status level — but the pill should still surface FLAKY, not
    FAIL, so the reader sees the inconsistency rather than a stale
    one-shot verdict."""
    cr = _make_case(
        results=[EvalResult("e", 1.0, True)],
        runs=5,
        pass_count=2,   # 2/5 → inconsistent → flaky
    )
    html = _status_pill(cr)
    assert ">FLAKY<" in html
    # Despite the per-evaluator EvalResult carrying passed=True, the
    # pill must not render as PASS — flaky dominates the quality outcome.
    assert "pill pass" not in html
    assert ">FAIL<" not in html


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


# ─────────────────────────────────────────────────────────────────────────────
# Codex round-2 regressions
# ─────────────────────────────────────────────────────────────────────────────

def test_pill_infra_error_dominates_flaky_in_multirun():
    """Codex caught this: in multi-run, a case with a judge outage
    AND flaky-vote semantics used to render FLAKY (which hid the infra
    error). The infra error is the underlying signal — the
    flaky-looking pass counts are an artifact of the outage."""
    cr = _make_case(
        results=[EvalResult("e", 1.0, True)],
        judge_error="429 rate limit",
        runs=5,
        pass_count=2,   # would be flaky if not for the judge error
    )
    html = _status_pill(cr)
    assert ">JUDGE ERR<" in html
    assert ">FLAKY<" not in html
    # The case really is in JUDGE_ERROR status, and the badge tracks that.
    assert cr.status == EvalStatus.JUDGE_ERROR


def test_pill_flaky_with_majority_pass_count_still_renders_flaky():
    """Even when the run majority passed (pass_count=3/5), the case is
    still flaky and the pill must say FLAKY in the absence of an error.
    Codex round-3 noted that a flaky case can never be ``PASSED`` at the
    CaseResult level (that requires ``pass_count == runs``), so this
    is really 'flaky dominates the majority-pass quality outcome', which
    is what is actually being verified here."""
    cr = _make_case(
        results=[EvalResult("e", 1.0, True)],
        runs=5,
        pass_count=3,   # majority pass, but not consistent — still flaky
    )
    html = _status_pill(cr)
    assert ">FLAKY<" in html
    assert "pill pass" not in html


def test_pill_has_aria_label_for_screen_readers():
    """The tooltip text is also exposed as ``aria-label`` so keyboard,
    touch, and screen-reader users see it even without hover. Codex
    accessibility finding."""
    cr = _make_case(
        results=[EvalResult("e", 0.0, False)],
        model_error="ConnectionRefused",
    )
    html = _status_pill(cr)
    assert "aria-label=" in html
    assert "MODEL ERR" in html
    # Both title (hover) and aria-label (assistive tech) get the
    # explanation text.
    assert html.count("not a quality issue") >= 2


def test_pass_pill_has_no_aria_label():
    """A bare PASS pill doesn't need extra explanation — no aria-label."""
    cr = _make_case(results=[EvalResult("e", 1.0, True)])
    html = _status_pill(cr)
    assert "aria-label" not in html


def test_errors_summary_card_has_aria_label():
    """Same accessibility fix for the Errors summary card."""
    judge_err = _make_case(judge_error="x", results=[EvalResult("e", 0, False)])
    report = EvalReport(suite_name="t", case_results=[judge_err])
    html = to_html(report)
    assert "aria-label=" in html
    # The aria-label includes the kind breakdown.
    assert "judge error" in html.lower()
