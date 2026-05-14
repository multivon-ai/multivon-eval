"""Tests for the 0.7.0 JUnit XML output + `multivon-eval view` command."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from multivon_eval import EvalReport, EvalResult, EvalStatus
from multivon_eval.result import CaseResult


def _make_case(*, results=None, model_error=None, judge_error=None,
               evaluator_error=None, skipped=False, latency_ms=10.0) -> CaseResult:
    return CaseResult(
        case_input="q",
        actual_output="ans",
        results=results or [],
        model_error=model_error,
        judge_error=judge_error,
        evaluator_error=evaluator_error,
        skipped=skipped,
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JUnit XML serialization
# ─────────────────────────────────────────────────────────────────────────────

def test_junit_xml_well_formed():
    """The produced XML must parse — most CI consumers strict-validate."""
    report = EvalReport(
        suite_name="parse-check",
        case_results=[_make_case(results=[EvalResult("e", 1.0, True)])],
    )
    xml = report.to_junit_xml()
    # Round-trip parse must not raise.
    root = ET.fromstring(xml)
    assert root.tag == "testsuites"


def test_junit_xml_declaration_present():
    report = EvalReport(suite_name="x", case_results=[])
    assert report.to_junit_xml().startswith('<?xml version="1.0"')


def test_junit_xml_passing_evaluator_emits_bare_testcase():
    cr = _make_case(results=[EvalResult("eval_a", 1.0, True, "ok")])
    report = EvalReport(suite_name="pass-suite", case_results=[cr])
    root = ET.fromstring(report.to_junit_xml())
    suites = root.findall("testsuite")
    assert len(suites) == 1
    cases = suites[0].findall("testcase")
    assert len(cases) == 1
    # Passing case must NOT have <failure>, <error>, or <skipped> children.
    assert cases[0].find("failure") is None
    assert cases[0].find("error") is None
    assert cases[0].find("skipped") is None


def test_junit_xml_quality_failure_emits_failure():
    cr = _make_case(results=[EvalResult("ev", 0.4, False, "below threshold")])
    report = EvalReport(suite_name="fail-suite", case_results=[cr])
    root = ET.fromstring(report.to_junit_xml())
    case = root.find("testsuite/testcase")
    failure = case.find("failure")
    assert failure is not None
    assert failure.get("type") == "quality"
    assert "below threshold" in failure.get("message", "")


def test_junit_xml_model_error_emits_error_element():
    """Plumbing failures get <error>, not <failure> — CI dashboards
    distinguish these two states meaningfully."""
    cr = _make_case(
        results=[EvalResult("ev", 0.0, False)],
        model_error="ConnectionError: refused",
    )
    report = EvalReport(suite_name="err-suite", case_results=[cr])
    root = ET.fromstring(report.to_junit_xml())
    case = root.find("testsuite/testcase")
    err = case.find("error")
    assert err is not None
    assert err.get("type") == "model_error"
    assert "ConnectionError" in (err.get("message", "") + (err.text or ""))


def test_junit_xml_judge_error_emits_error_element():
    cr = _make_case(
        results=[EvalResult("ev", 0.0, False, "[judge unavailable: 429]")],
        judge_error="429 rate limit",
    )
    report = EvalReport(suite_name="je", case_results=[cr])
    root = ET.fromstring(report.to_junit_xml())
    err = root.find("testsuite/testcase/error")
    assert err is not None
    assert err.get("type") == "judge_error"


def test_junit_xml_skipped_emits_skipped_element():
    cr = _make_case(results=[EvalResult("ev", 1.0, True)], skipped=True)
    report = EvalReport(suite_name="sk", case_results=[cr])
    root = ET.fromstring(report.to_junit_xml())
    case = root.find("testsuite/testcase")
    assert case.find("skipped") is not None


def test_junit_xml_suite_aggregates_match_counts():
    """The top-level <testsuite> element's `tests`/`failures`/`errors` must
    match what the per-case rows imply."""
    passing = _make_case(results=[EvalResult("ev", 1.0, True)])
    failing = _make_case(results=[EvalResult("ev", 0.0, False)])
    errored = _make_case(judge_error="x", results=[EvalResult("ev", 0.0, False)])
    report = EvalReport(suite_name="mixed", case_results=[passing, passing, failing, errored])
    root = ET.fromstring(report.to_junit_xml())
    suite = root.find("testsuite")
    assert int(suite.get("tests")) == 4
    assert int(suite.get("failures")) == 1
    assert int(suite.get("errors")) == 1


def test_junit_xml_handles_xml_unsafe_characters_in_reasons():
    """Reasons can contain '<', '>', '&' from rendered tool calls or HTML.
    XML output must escape them so the document stays well-formed."""
    nasty = EvalResult("ev", 0.0, False, reason="<script>alert('x & y')</script>")
    cr = _make_case(results=[nasty])
    report = EvalReport(suite_name="esc", case_results=[cr])
    xml = report.to_junit_xml()
    # Must parse cleanly with raw special chars escaped.
    root = ET.fromstring(xml)
    failure = root.find("testsuite/testcase/failure")
    # Either body text or message attr will contain the original chars,
    # but parsed back as raw chars (not entities) — ET handles that.
    payload = (failure.get("message", "") + (failure.text or ""))
    assert "alert" in payload


def test_save_junit_xml_writes_file(tmp_path: Path):
    cr = _make_case(results=[EvalResult("ev", 1.0, True)])
    report = EvalReport(suite_name="save", case_results=[cr])
    out = tmp_path / "junit.xml"
    report.save_junit_xml(str(out))
    assert out.exists()
    # File must be valid XML.
    ET.fromstring(out.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# `multivon-eval report --junit` CLI flag
# ─────────────────────────────────────────────────────────────────────────────

def _save_report_json(path: Path) -> None:
    """Save a minimal report JSON to disk for CLI tests."""
    report = EvalReport(
        suite_name="cli-test",
        case_results=[_make_case(results=[EvalResult("ev", 1.0, True)])],
    )
    path.write_text(report.to_json())


def test_cli_report_with_junit_flag_writes_xml(tmp_path: Path):
    src = tmp_path / "in.json"
    out = tmp_path / "out.xml"
    _save_report_json(src)
    res = subprocess.run(
        [sys.executable, "-m", "multivon_eval", "report",
         str(src), "--junit", str(out)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists()
    ET.fromstring(out.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# `multivon-eval view` — local HTML server
# ─────────────────────────────────────────────────────────────────────────────

def _free_port() -> int:
    """Pick a port the OS thinks is free, then close it so the test can bind."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_cli_view_serves_html(tmp_path: Path):
    """End-to-end: spawn `multivon-eval view`, fetch the index page, verify
    it returns HTML (200) and the suite name is in the response."""
    src = tmp_path / "report.json"
    _save_report_json(src)
    port = _free_port()

    proc = subprocess.Popen(
        [sys.executable, "-m", "multivon_eval", "view",
         str(src), "--port", str(port), "--no-browser"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        # Wait up to ~3s for the server to bind.
        deadline = time.time() + 3.0
        last_err = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    assert resp.status == 200
                    assert "cli-test" in body or "html" in body.lower()
                    break
            except Exception as e:
                last_err = e
                time.sleep(0.1)
        else:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(f"server never came up; last error: {last_err}; stderr: {stderr}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
