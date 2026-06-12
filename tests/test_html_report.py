"""Tests for EvalReport.to_html() / save_html()."""
from __future__ import annotations
import os
import tempfile
import pytest
from multivon_eval.result import EvalResult, CaseResult, EvalReport


def _make_report(multi_run: bool = False) -> EvalReport:
    cr1 = CaseResult(
        case_input="What is the return policy?",
        actual_output="30-day returns on all items.",
        results=[
            EvalResult("Faithfulness", 0.90, True, "Grounded in context."),
            EvalResult("Relevancy", 0.85, True, "On topic."),
        ],
        latency_ms=200.0,
        tags=["policy"],
    )
    cr2 = CaseResult(
        case_input="Can I return a used item?",
        actual_output="Used items are not accepted.",
        results=[
            EvalResult("Faithfulness", 0.40, False, "Not in context."),
            EvalResult("Relevancy", 0.70, True, "Answers the question."),
        ],
        latency_ms=180.0,
    )
    if multi_run:
        cr1.runs = 3
        cr1.all_scores = [0.90, 0.88, 0.92]
        cr1.pass_count = 3
        cr2.runs = 3
        cr2.all_scores = [0.55, 0.40, 0.45]
        cr2.pass_count = 1  # flaky

    return EvalReport(
        suite_name="Test Suite",
        model_id="gpt-4o",
        case_results=[cr1, cr2],
    )


class TestToHtml:
    def test_returns_string(self):
        report = _make_report()
        result = report.to_html()
        assert isinstance(result, str)

    def test_contains_suite_name(self):
        report = _make_report()
        assert "Test Suite" in report.to_html()

    def test_contains_model_id(self):
        report = _make_report()
        assert "gpt-4o" in report.to_html()

    def test_contains_input_text(self):
        report = _make_report()
        html = report.to_html()
        assert "return policy" in html

    def test_contains_evaluator_names(self):
        report = _make_report()
        html = report.to_html()
        assert "Faithfulness" in html
        assert "Relevancy" in html

    def test_no_external_dependencies(self):
        html = _make_report().to_html()
        for cdn in ("cdn.js", "unpkg", "jsdelivr", "cdnjs", "googleapis"):
            assert cdn not in html, f"Found external CDN reference: {cdn}"

    def test_valid_html_structure(self):
        html = _make_report().to_html()
        assert html.startswith("<!DOCTYPE html>")
        assert "<html" in html
        assert "</html>" in html
        assert "<style>" in html
        assert "<script>" in html

    def test_pass_fail_status(self):
        html = _make_report().to_html()
        assert "PASS" in html
        assert "FAIL" in html

    def test_pass_rate_ci_rendered(self):
        # Console + JSON carry a Wilson CI on the pass rate; the HTML
        # report must too (release-readiness campaign finding).
        report = _make_report()
        html = report.to_html()
        lo, hi = report.pass_rate_ci()
        assert "95% CI (Wilson)" in html
        assert f"[{lo:.1%}, {hi:.1%}]" in html

    def test_multi_run_shows_flaky(self):
        html = _make_report(multi_run=True).to_html()
        assert "FLAKY" in html
        assert "flaky" in html.lower()

    def test_multi_run_shows_stability(self):
        html = _make_report(multi_run=True).to_html()
        assert "Stability" in html

    def test_tag_rendered(self):
        html = _make_report().to_html()
        assert "policy" in html

    def test_score_values_present(self):
        html = _make_report().to_html()
        assert "0.90" in html or "0.9" in html


class TestSaveHtml:
    def test_save_creates_file(self):
        report = _make_report()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.save_html(path)
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "Test Suite" in content
        finally:
            os.unlink(path)

    def test_save_matches_to_html(self):
        report = _make_report()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.save_html(path)
            with open(path, encoding="utf-8") as f:
                saved = f.read()
            assert saved == report.to_html()
        finally:
            os.unlink(path)
