"""Tests for the v0.6 compliance additions: per-case records, anchor callback, HTML rollup."""
from __future__ import annotations

import json
import os

import pytest

from multivon_eval import (
    ComplianceReporter,
    ComplianceHtmlReporter,
    ComplianceError,
    EvalSuite,
    github_actions_anchor,
)
from multivon_eval.result import CaseResult, EvalReport, EvalResult


def _make_report(suite_name: str = "v2 Suite") -> EvalReport:
    return EvalReport(
        suite_name=suite_name,
        model_id="test-model",
        case_results=[
            CaseResult(
                case_input="q1",
                actual_output="a1",
                results=[
                    EvalResult(evaluator="faithfulness", score=0.9, passed=True, reason="ok"),
                    EvalResult(evaluator="hallucination", score=0.95, passed=True),
                ],
            ),
            CaseResult(
                case_input="q2",
                actual_output="a2",
                results=[
                    EvalResult(evaluator="faithfulness", score=0.3, passed=False, reason="claim X not in context"),
                ],
            ),
            CaseResult(
                case_input="q3",
                actual_output="a3",
                results=[
                    EvalResult(evaluator="pii_detection", score=1.0, passed=True),
                    EvalResult(evaluator="not_empty", score=1.0, passed=True),
                ],
            ),
        ],
    )


def _log_lines(tmp_path, suite_name: str = "v2 Suite") -> list[dict]:
    safe = suite_name.replace(" ", "_")
    log = tmp_path / f"{safe}.audit.ndjson"
    return [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]


class TestPerCaseMode:
    def test_per_case_writes_one_record_per_case(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        ids = rep.record(_make_report(), mode="case")
        assert isinstance(ids, list)
        assert len(ids) == 3
        lines = _log_lines(tmp_path)
        assert len(lines) == 3
        assert all(line["record_type"] == "case" for line in lines)

    def test_summary_mode_is_default_and_returns_one_id(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        rec_id = rep.record(_make_report())
        assert isinstance(rec_id, str)
        lines = _log_lines(tmp_path)
        assert len(lines) == 1
        assert lines[0]["record_type"] == "summary"

    def test_per_case_chain_links_records(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        rep.record(_make_report(), mode="case")
        lines = _log_lines(tmp_path)
        for prev, curr in zip(lines, lines[1:]):
            assert curr["prev_hash"] == prev["record_hash"]

    def test_per_case_chain_verifies(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        rep.record(_make_report(), mode="case")
        assert rep.verify("v2 Suite") is True

    def test_per_case_payload_contains_evaluator_breakdown(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        rep.record(_make_report(), mode="case")
        lines = _log_lines(tmp_path)
        case = lines[1]["case"]
        assert case["case_index"] == 1
        assert case["input"] == "q2"
        assert case["output"] == "a2"
        assert case["passed"] is False
        faith = next(e for e in case["evaluators"] if e["evaluator"] == "faithfulness")
        assert faith["passed"] is False
        assert faith["reason"] == "claim X not in context"

    def test_per_case_controls_attached_to_evaluators(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        rep.record(_make_report(), mode="case")
        first = _log_lines(tmp_path)[0]
        faith = next(e for e in first["case"]["evaluators"] if e["evaluator"] == "faithfulness")
        assert faith["controls"] == [{"id": "Art. 15(1)", "description": "Accuracy"}]

    def test_unknown_mode_raises_compliance_error(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        with pytest.raises(ComplianceError):
            rep.record(_make_report(), mode="bogus")  # type: ignore[arg-type]

    def test_per_case_followed_by_summary_links_correctly(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        rep.record(_make_report(), mode="case")
        rep.record(_make_report(), mode="summary")
        lines = _log_lines(tmp_path)
        assert len(lines) == 4
        assert lines[3]["record_type"] == "summary"
        assert lines[3]["prev_hash"] == lines[2]["record_hash"]
        assert rep.verify("v2 Suite") is True


class TestAnchor:
    def test_anchor_called_after_summary_record(self, tmp_path):
        tips: list[str] = []
        rep = ComplianceReporter(
            output_dir=str(tmp_path),
            framework="eu-ai-act",
            anchor_fn=tips.append,
            verbose=False,
        )
        rep.record(_make_report())
        assert len(tips) == 1
        assert len(tips[0]) == 64  # SHA-256 hex

    def test_anchor_called_once_per_case_batch(self, tmp_path):
        tips: list[str] = []
        rep = ComplianceReporter(
            output_dir=str(tmp_path),
            framework="eu-ai-act",
            anchor_fn=tips.append,
            verbose=False,
        )
        rep.record(_make_report(), mode="case")
        # One anchor call after the last case in the batch.
        assert len(tips) == 1
        # Anchored hash equals the tip of the chain.
        last_line = _log_lines(tmp_path)[-1]
        assert tips[0] == last_line["record_hash"]

    def test_anchor_failure_wraps_in_compliance_error(self, tmp_path):
        def bad_anchor(_: str) -> None:
            raise RuntimeError("ledger unreachable")

        rep = ComplianceReporter(
            output_dir=str(tmp_path),
            framework="eu-ai-act",
            anchor_fn=bad_anchor,
            verbose=False,
        )
        with pytest.raises(ComplianceError):
            rep.record(_make_report())

    def test_github_actions_anchor_writes_when_var_set(self, tmp_path, monkeypatch):
        output_file = tmp_path / "gh_output.txt"
        output_file.write_text("")
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

        github_actions_anchor("abc123")
        contents = output_file.read_text()
        assert "multivon_audit_tip=abc123" in contents

    def test_github_actions_anchor_noop_without_var(self, monkeypatch):
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        # Must not raise.
        github_actions_anchor("abc123")


class TestHtmlRollup:
    def test_html_render_has_all_sections(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        report = _make_report("HTML test")
        rep.record(report, mode="case")
        suite = EvalSuite.eu_ai_act_high_risk()
        html = ComplianceHtmlReporter(rep).render(report, suite=suite)
        assert "<!doctype html>" in html
        assert "Compliance report" in html
        assert "Run summary" in html
        assert "Regulatory coverage" in html
        assert "Audit log integrity" in html
        assert "Per-case detail" in html
        assert "Evaluators" in html

    def test_html_render_works_without_suite(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        report = _make_report("HTML no suite")
        rep.record(report)
        html = ComplianceHtmlReporter(rep).render(report)
        assert "Regulatory coverage" not in html
        assert "Run summary" in html

    def test_html_renders_chain_status(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        report = _make_report("HTML chain")
        rep.record(report)
        html = ComplianceHtmlReporter(rep).render(report)
        assert "PASS" in html

    def test_html_renders_tamper_status(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        report = _make_report("HTML tamper")
        rep.record(report)
        rep.record(report)
        log = tmp_path / "HTML_tamper.audit.ndjson"
        lines = log.read_text().splitlines()
        first = json.loads(lines[0])
        first["summary"]["pass_rate"] = 0.0
        lines[0] = json.dumps(first, separators=(",", ":"))
        log.write_text("\n".join(lines) + "\n")

        html = ComplianceHtmlReporter(rep).render(report)
        assert "TAMPERED" in html or "FAIL" in html

    def test_html_escapes_inputs(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act", verbose=False)
        report = EvalReport(
            suite_name="escape <script>",
            model_id="m",
            case_results=[
                CaseResult(
                    case_input="<script>alert(1)</script>",
                    actual_output="<b>x</b>",
                    results=[EvalResult(evaluator="not_empty", score=1.0, passed=True)],
                )
            ],
        )
        html = ComplianceHtmlReporter(rep).render(report)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
