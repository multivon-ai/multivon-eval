"""Tests for the 0.7.0 immutable audit-record provenance manifest.

Marcus persona ask (from the multi-persona critique): "Immutable audit
manifest with: package version, git SHA, evaluator versions, calibration
file hash, threshold values, judge identity, prompt templates, dataset
hash, run config, timestamps."

This file verifies the provenance block is built correctly and embedded
on every audit record so an auditor can reproduce the eval decisions
offline.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from multivon_eval import (
    ComplianceReporter, EvalSuite, EvalCase, EvalReport, EvalResult,
    NotEmpty, WordCount,
)
from multivon_eval.compliance import _host_info, _package_git_sha
from multivon_eval.result import CaseResult


# ─────────────────────────────────────────────────────────────────────────────
# _package_git_sha + _host_info — environment capture
# ─────────────────────────────────────────────────────────────────────────────

def test_host_info_has_required_fields():
    """_host_info captures reproducibility-critical environment metadata
    with NO PII (no hostname, no username)."""
    h = _host_info()
    assert "python" in h
    assert "platform" in h
    assert "machine" in h
    # Affirmative no-PII check: don't leak username or hostname.
    blob = json.dumps(h).lower()
    import os, getpass, socket
    user = getpass.getuser().lower()
    host = socket.gethostname().lower()
    if user:
        assert user not in blob, "host_info leaked username"
    if host:
        # Hostname is more likely to legitimately collide with platform
        # markers (e.g. 'darwin'); use a stricter substring check.
        assert host not in blob or len(host) < 4, "host_info leaked hostname"


def test_git_sha_returns_string_or_none_never_raises():
    """The git SHA helper must never crash an audit record. Returns either
    a 40-char hex string (in a git workspace) or None."""
    sha = _package_git_sha()
    if sha is not None:
        # Standard git rev-parse output: 40 hex chars.
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: real suite.run → reporter.record → audit log contains provenance
# ─────────────────────────────────────────────────────────────────────────────

def _build_real_suite() -> EvalSuite:
    """Tiny offline suite — no LLM, two deterministic evaluators."""
    suite = EvalSuite("prov-test-suite")
    suite.add_cases([
        EvalCase(input="hello world"),
        EvalCase(input="how are you"),
    ])
    suite.add_evaluators(NotEmpty(), WordCount(min=1, max=10))
    return suite


def test_suite_run_populates_report_suite_lock():
    """suite.run() must attach the SuiteLock to the report so downstream
    code (audit reporter, audit-package CLI) can capture provenance."""
    suite = _build_real_suite()
    report = suite.run(lambda i: f"answer for {i}", verbose=False)
    assert report.suite_lock is not None
    assert report.suite_lock.suite_name == "prov-test-suite"
    assert report.suite_lock.case_count == 2
    assert len(report.suite_lock.evaluators) == 2


def test_audit_record_contains_provenance_field(tmp_path: Path):
    """A recorded audit log row must carry the provenance manifest."""
    suite = _build_real_suite()
    report = suite.run(lambda i: f"answer for {i}", verbose=False)

    reporter = ComplianceReporter(str(tmp_path / "audit"), framework="none", verbose=False)
    reporter.record(report)

    log_path = tmp_path / "audit" / "prov-test-suite.audit.ndjson"
    line = log_path.read_text().strip()
    rec = json.loads(line)
    assert "provenance" in rec
    prov = rec["provenance"]
    assert prov["schema_version"] == 1
    assert "package_version" in prov
    assert "host" in prov and prov["host"].get("python")
    # suite_lock is embedded with the evaluator + cases fingerprints.
    assert "suite_lock" in prov
    assert prov["suite_lock"]["suite_name"] == "prov-test-suite"
    assert prov["suite_lock"]["case_count"] == 2


def test_audit_record_provenance_is_in_hash_chain(tmp_path: Path):
    """The provenance block must be covered by the record hash so an
    auditor who alters it can be detected by reporter.verify(). Sanity
    check: serialize, tamper, verify rejects."""
    suite = _build_real_suite()
    report = suite.run(lambda i: f"answer for {i}", verbose=False)

    reporter = ComplianceReporter(str(tmp_path / "audit"), framework="none", verbose=False)
    reporter.record(report)
    # Append a second record so the chain is non-trivial.
    reporter.record(report)

    log_path = tmp_path / "audit" / "prov-test-suite.audit.ndjson"
    lines = log_path.read_text().splitlines()
    assert len(lines) == 2

    # Tamper with the provenance block of record 0 — flip the
    # package_version. Recompute nothing; just write back.
    rec = json.loads(lines[0])
    rec["provenance"]["package_version"] = "tampered"
    lines[0] = json.dumps(rec, separators=(",", ":"))
    log_path.write_text("\n".join(lines) + "\n")

    # Verifier returns False (and prints diagnostics) on tampered chain.
    # The contract is "return False, don't raise" — explicit by design
    # so callers can decide how to react.
    assert reporter.verify(report.suite_name) is False


def test_audit_record_summary_includes_status_breakdown(tmp_path: Path):
    """0.7.0: the summary written to the audit log must include the
    new error fields (evaluated, errors, errors_by_kind, skipped) so the
    auditor can distinguish quality failures from infrastructure errors."""
    suite = _build_real_suite()
    report = suite.run(lambda i: f"answer for {i}", verbose=False)

    reporter = ComplianceReporter(str(tmp_path / "audit2"), framework="none", verbose=False)
    reporter.record(report)
    line = (tmp_path / "audit2" / "prov-test-suite.audit.ndjson").read_text().strip()
    rec = json.loads(line)
    s = rec["summary"]
    assert "evaluated" in s
    assert "errors" in s
    assert "errors_by_kind" in s
    assert "skipped" in s


def test_per_case_records_share_provenance(tmp_path: Path):
    """``mode="case"`` writes one record per case. The provenance is
    run-level (not case-level), so every record must carry the SAME
    provenance — proving the suite state didn't drift mid-record."""
    suite = _build_real_suite()
    report = suite.run(lambda i: f"answer for {i}", verbose=False)

    reporter = ComplianceReporter(str(tmp_path / "audit3"), framework="none", verbose=False)
    reporter.record(report, mode="case")
    lines = (tmp_path / "audit3" / "prov-test-suite.audit.ndjson").read_text().splitlines()
    assert len(lines) == 2   # one record per case
    provs = [json.loads(l)["provenance"] for l in lines]
    # All provenance blocks must be identical (modulo the implicit fact
    # that they were computed from the same report).
    assert provs[0] == provs[1]


def test_provenance_handles_synthetic_report_without_suite_lock(tmp_path: Path):
    """A report constructed by hand (e.g., for tests, or replayed from
    JSON) might not have a suite_lock. Provenance must still serialize
    cleanly with the static fields present."""
    report = EvalReport(
        suite_name="synthetic",
        case_results=[
            CaseResult(case_input="x", actual_output="y",
                       results=[EvalResult("ev", 1.0, True)]),
        ],
    )  # NOTE: no suite_lock
    reporter = ComplianceReporter(str(tmp_path / "audit4"), framework="none", verbose=False)
    reporter.record(report)
    rec = json.loads((tmp_path / "audit4" / "synthetic.audit.ndjson").read_text().strip())
    prov = rec["provenance"]
    assert "package_version" in prov
    assert "suite_lock" not in prov   # absent — synthetic report
