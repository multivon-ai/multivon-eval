"""Tests for the HIPAA framework + Compliance Evidence Package CLI."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from multivon_eval import ComplianceReporter, EvalSuite
from multivon_eval.audit_package import (
    _PACKAGE_FORMAT_VERSION,
    build_audit_package,
)
from multivon_eval.compliance import (
    _CATALOGS,
    _HIPAA_BY_EVALUATOR,
    _HIPAA_CONTROLS,
    _HIPAA_PROCESS_CONTROLS,
)
from multivon_eval.result import CaseResult, EvalReport, EvalResult


def _make_report(suite_name: str = "HIPAA Demo") -> EvalReport:
    return EvalReport(
        suite_name=suite_name,
        model_id="claude-haiku-4-5",
        case_results=[
            CaseResult(
                case_input="Summarize patient encounter",
                actual_output="Patient stable, follow up in 30 days.",
                results=[
                    EvalResult(evaluator="faithfulness", score=0.91, passed=True),
                    EvalResult(evaluator="pii_detection", score=1.0, passed=True),
                    EvalResult(evaluator="not_empty", score=1.0, passed=True),
                ],
            ),
            CaseResult(
                case_input="Summarize triage note",
                actual_output="Patient with mrn 12345 reports chest pain.",
                results=[
                    EvalResult(evaluator="faithfulness", score=0.8, passed=True),
                    EvalResult(evaluator="pii_detection", score=0.0, passed=False,
                               reason="MRN exposed"),
                    EvalResult(evaluator="not_empty", score=1.0, passed=True),
                ],
            ),
        ],
    )


# ─── HIPAA mapping ──────────────────────────────────────────────────────────


class TestHipaaCatalog:
    def test_hipaa_is_a_registered_framework(self):
        assert "hipaa" in _CATALOGS
        assert "measurable" in _CATALOGS["hipaa"]
        assert "process" in _CATALOGS["hipaa"]

    def test_security_rule_controls_present(self):
        ids = {c.id for c in _HIPAA_CONTROLS.values()}
        assert "45 CFR §164.312(a)" in ids
        assert "45 CFR §164.312(b)" in ids
        assert "45 CFR §164.312(c)" in ids
        assert "45 CFR §164.514(b)(2)" in ids

    def test_process_safeguards_surface_separately(self):
        ids = {c.id for c in _HIPAA_PROCESS_CONTROLS.values()}
        assert "45 CFR §164.308" in ids  # Administrative
        assert "45 CFR §164.310" in ids  # Physical
        assert "Business Associate Agreement" in ids

    def test_pii_detection_maps_to_safe_harbor(self):
        # PII detection is the load-bearing evidence for Safe Harbor + access ctrl.
        assert "hipaa_514_b2" in _HIPAA_BY_EVALUATOR["pii_detection"]
        assert "hipaa_312_a" in _HIPAA_BY_EVALUATOR["pii_detection"]

    def test_clinical_quality_evaluators_map_to_audit_controls(self):
        assert "hipaa_312_b" in _HIPAA_BY_EVALUATOR["faithfulness"]
        assert "hipaa_312_b" in _HIPAA_BY_EVALUATOR["hallucination"]
        assert "hipaa_312_b" in _HIPAA_BY_EVALUATOR["answer_accuracy"]

    def test_schema_evaluators_map_to_integrity(self):
        assert "hipaa_312_c" in _HIPAA_BY_EVALUATOR["schema_compliance"]
        assert "hipaa_312_c" in _HIPAA_BY_EVALUATOR["json_schema"]


class TestHipaaFactory:
    def test_factory_wires_expected_evaluators(self):
        from multivon_eval.evaluators.compliance import PIIEvaluator
        from multivon_eval.evaluators.deterministic import NotEmpty
        from multivon_eval.evaluators.llm_judge import AnswerAccuracy, Faithfulness, Hallucination
        suite = EvalSuite.hipaa_safe_harbor()
        types = {type(e) for e in suite._evaluators}
        assert types == {
            PIIEvaluator, NotEmpty, Faithfulness, Hallucination, AnswerAccuracy,
        }

    def test_factory_uses_hipaa_jurisdiction_for_pii(self):
        from multivon_eval.evaluators.compliance import PIIEvaluator
        suite = EvalSuite.hipaa_safe_harbor()
        pii = next(e for e in suite._evaluators if isinstance(e, PIIEvaluator))
        # HIPAA-specific patterns must be loaded
        assert "medical_record_number" in pii._compiled

    def test_factory_with_schema(self):
        from pydantic import BaseModel
        from multivon_eval.evaluators.compliance import SchemaEvaluator

        class ClinicalOut(BaseModel):
            diagnosis: str

        suite = EvalSuite.hipaa_safe_harbor(schema=ClinicalOut)
        assert any(isinstance(e, SchemaEvaluator) for e in suite._evaluators)


class TestHipaaCoverage:
    def test_coverage_marks_pii_as_satisfying_safe_harbor(self, tmp_path):
        suite = EvalSuite.hipaa_safe_harbor()
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="hipaa", verbose=False)
        cov = rep.coverage(suite)
        assert "hipaa_514_b2" in cov.covered  # Safe Harbor de-id
        assert "hipaa_312_b" in cov.covered  # Audit controls (faithfulness/halu/aa)
        assert "hipaa_312_c" in cov.covered  # Integrity (not_empty)

    def test_process_safeguards_listed_separately(self, tmp_path):
        suite = EvalSuite.hipaa_safe_harbor()
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="hipaa", verbose=False)
        cov = rep.coverage(suite)
        ids = {c.id for c in cov.process}
        assert "45 CFR §164.308" in ids
        assert "Business Associate Agreement" in ids

    def test_record_attaches_hipaa_controls_to_evaluators(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="hipaa", verbose=False)
        rep.record(_make_report("HIPAA Records"))
        lines = (tmp_path / "HIPAA_Records.audit.ndjson").read_text().splitlines()
        record = json.loads(lines[0])
        pii_entry = next(e for e in record["evaluator_results"] if e["evaluator"] == "pii_detection")
        assert any("§164.514" in c["id"] or "§164.312(a)" in c["id"] for c in pii_entry["controls"])


# ─── Audit package ──────────────────────────────────────────────────────────


class TestBuildAuditPackage:
    def _seed_audit_log(self, tmp_path: Path, suite_name: str) -> Path:
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="hipaa", verbose=False)
        rep.record(_make_report(suite_name))
        rep.record(_make_report(suite_name))  # 2 records → chain has 2 links
        return tmp_path

    def test_builds_a_zip(self, tmp_path):
        logs = self._seed_audit_log(tmp_path / "logs", "HIPAA Demo")
        out = tmp_path / "pkg.zip"
        result = build_audit_package(
            logs_dir=logs,
            suite_name="HIPAA Demo",
            framework="hipaa",
            out_path=out,
            period_label="2026-Q2",
        )
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_zip_contents_match_contract(self, tmp_path):
        logs = self._seed_audit_log(tmp_path / "logs", "HIPAA Demo")
        out = tmp_path / "pkg.zip"
        build_audit_package(
            logs_dir=logs, suite_name="HIPAA Demo", framework="hipaa",
            out_path=out, period_label="2026-Q2",
        )
        with zipfile.ZipFile(out) as zf:
            names = sorted(zf.namelist())
        # All paths must be under the same prefix.
        prefix = "compliance-evidence-2026-Q2/"
        assert all(n.startswith(prefix) for n in names)
        relnames = [n.removeprefix(prefix) for n in names]
        # Calibration file name varies with shipped version (v2 preferred over v1).
        cal_names = [n for n in relnames if n.startswith("calibration_v") and n.endswith(".json")]
        assert len(cal_names) == 1, f"expected exactly one calibration file, got {cal_names}"
        for expected in ("audit_log.ndjson", "coverage_report.md",
                         "verify.py", "README.md", "manifest.json"):
            assert expected in relnames

    def test_manifest_records_per_file_sha256(self, tmp_path):
        logs = self._seed_audit_log(tmp_path / "logs", "HIPAA Demo")
        out = tmp_path / "pkg.zip"
        build_audit_package(
            logs_dir=logs, suite_name="HIPAA Demo", framework="hipaa", out_path=out,
        )
        with zipfile.ZipFile(out) as zf:
            prefix = [n for n in zf.namelist() if n.endswith("manifest.json")][0].removesuffix(
                "manifest.json"
            )
            manifest = json.loads(zf.read(prefix + "manifest.json"))
            # Every listed file's hash should match the actual content.
            for entry in manifest["files"]:
                if entry["path"] == "manifest.json":
                    continue  # the manifest hashes everything else
                actual = hashlib.sha256(zf.read(prefix + entry["path"])).hexdigest()
                assert actual == entry["sha256"], entry["path"]
        assert manifest["package_format_version"] == _PACKAGE_FORMAT_VERSION
        assert manifest["framework"] == "hipaa"
        assert manifest["suite_name"] == "HIPAA Demo"

    def test_calibration_data_included(self, tmp_path):
        logs = self._seed_audit_log(tmp_path / "logs", "HIPAA Demo")
        out = tmp_path / "pkg.zip"
        build_audit_package(
            logs_dir=logs, suite_name="HIPAA Demo", framework="hipaa", out_path=out,
        )
        with zipfile.ZipFile(out) as zf:
            # The shipped calibration is now versioned: v2 preferred, falls
            # back to v1 if v2 isn't packaged. Find whichever one ended up
            # in the bundle.
            cal_names = [n for n in zf.namelist()
                         if n.endswith("calibration_v1.json") or n.endswith("calibration_v2.json")]
            assert cal_names, "no calibration_v*.json in package"
            cal_blob = zf.read(cal_names[0])
            cal_data = json.loads(cal_blob)
        assert cal_data["schema_version"] >= 1
        assert len(cal_data["entries"]) > 0

    def test_verifier_script_runs_clean(self, tmp_path):
        """The bundled verifier should report PASSED on its own bundle."""
        logs = self._seed_audit_log(tmp_path / "logs", "HIPAA Demo")
        out = tmp_path / "pkg.zip"
        build_audit_package(
            logs_dir=logs, suite_name="HIPAA Demo", framework="hipaa", out_path=out,
            period_label="2026-Q2",
        )
        # Extract and run verify.py with the venv python.
        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(out) as zf:
            zf.extractall(extract_dir)
        inner = extract_dir / "compliance-evidence-2026-Q2"
        result = subprocess.run(
            [sys.executable, str(inner / "verify.py")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"verify.py failed:\n{result.stdout}\n{result.stderr}"
        assert "VERIFICATION PASSED" in result.stdout

    def test_missing_audit_log_raises(self, tmp_path):
        # No log seeded.
        (tmp_path / "logs").mkdir()
        with pytest.raises(FileNotFoundError):
            build_audit_package(
                logs_dir=tmp_path / "logs", suite_name="Does Not Exist",
                framework="hipaa", out_path=tmp_path / "p.zip",
            )

    def test_extra_files_included(self, tmp_path):
        logs = self._seed_audit_log(tmp_path / "logs", "HIPAA Demo")
        out = tmp_path / "pkg.zip"
        build_audit_package(
            logs_dir=logs, suite_name="HIPAA Demo", framework="hipaa", out_path=out,
            extra_files={"cover_letter.txt": b"signed: Jane Compliance Officer"},
        )
        with zipfile.ZipFile(out) as zf:
            prefix = [n for n in zf.namelist() if n.endswith("cover_letter.txt")][0].removesuffix(
                "cover_letter.txt"
            )
            assert zf.read(prefix + "cover_letter.txt") == b"signed: Jane Compliance Officer"

    def test_extras_cannot_overwrite_canonical_files(self, tmp_path):
        logs = self._seed_audit_log(tmp_path / "logs", "HIPAA Demo")
        with pytest.raises(ValueError):
            build_audit_package(
                logs_dir=logs, suite_name="HIPAA Demo", framework="hipaa",
                out_path=tmp_path / "p.zip",
                extra_files={"manifest.json": b"forged"},
            )

    def test_verifier_detects_tampered_file(self, tmp_path):
        logs = self._seed_audit_log(tmp_path / "logs", "HIPAA Demo")
        out = tmp_path / "pkg.zip"
        build_audit_package(
            logs_dir=logs, suite_name="HIPAA Demo", framework="hipaa", out_path=out,
            period_label="2026-Q2",
        )
        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(out) as zf:
            zf.extractall(extract_dir)
        inner = extract_dir / "compliance-evidence-2026-Q2"
        # Tamper with the audit log (don't touch manifest).
        log = inner / "audit_log.ndjson"
        lines = log.read_text().splitlines()
        d = json.loads(lines[0])
        d["summary"]["pass_rate"] = 0.0  # change the content
        lines[0] = json.dumps(d, separators=(",", ":"))
        log.write_text("\n".join(lines) + "\n")
        # Verifier should catch the SHA mismatch.
        result = subprocess.run(
            [sys.executable, str(inner / "verify.py")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        # Either the manifest hash mismatched or the chain detected it.
        assert "FAIL" in result.stdout or "FAIL" in result.stderr
