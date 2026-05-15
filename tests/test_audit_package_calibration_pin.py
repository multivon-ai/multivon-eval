"""D12: calibration version pinned through fingerprint → audit log →
audit-package.

The bug being fixed: ``audit-package`` always bundled the shipped
default calibration (v2 today). If the audit was recorded against v1
(via ``MULTIVON_CALIBRATION_VERSION=v1`` or an explicit
``load_calibration(version="v1")`` pin), the bundle would include v2
even though the threshold decisions in the log were made against v1.
A regulator running ``verify.py`` would see "VERIFICATION PASSED" even
though the calibration evidence doesn't match the audit's decisions.

This test set exercises the entire chain.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from multivon_eval import (
    EvalCase, EvalSuite, JudgeConfig, calibration_versions, load_calibration,
)
from multivon_eval.audit_package import (
    _calibration_version_from_log, _read_calibration_data, build_audit_package,
)
from multivon_eval.calibration import effective_calibration_version
from multivon_eval.compliance import ComplianceReporter
from multivon_eval.evaluators.llm_judge import Faithfulness
from multivon_eval.lockfile import fingerprint_evaluator


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationTable.version_label is populated by load_calibration
# ─────────────────────────────────────────────────────────────────────────────

def test_load_calibration_records_version_label():
    """The loaded table tags itself with the label it was loaded under,
    so downstream code (audit-package, fingerprint) doesn't have to
    infer it from a dataset hash."""
    t1 = load_calibration(reload=True, version="v1")
    t2 = load_calibration(reload=True, version="v2")
    assert t1.version_label == "v1"
    assert t2.version_label == "v2"


def test_effective_calibration_version_returns_resolved_label(monkeypatch):
    """``effective_calibration_version()`` reports the label that *would*
    be loaded right now, honoring the env override."""
    monkeypatch.setenv("MULTIVON_CALIBRATION_VERSION", "v1")
    load_calibration(reload=True)
    assert effective_calibration_version() == "v1"

    monkeypatch.setenv("MULTIVON_CALIBRATION_VERSION", "v2")
    load_calibration(reload=True)
    assert effective_calibration_version() == "v2"


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint carries the calibration version
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluator_fingerprint_records_calibration_version(monkeypatch):
    """A judge whose calibration is in v2 (gpt-5.5) renders a fingerprint
    that carries ``calibration.version == "v2"`` when the default is v2."""
    monkeypatch.setenv("MULTIVON_CALIBRATION_VERSION", "v2")
    load_calibration(reload=True)  # honor the env

    judge = JudgeConfig(provider="openai", model="gpt-5.5").resolve()
    fp = fingerprint_evaluator(Faithfulness(judge=judge))
    assert fp.calibration is not None, "gpt-5.5 must be in v2 — calibration entry expected"
    assert fp.calibration["version"] == "v2"


def test_evaluator_fingerprint_records_v1_under_v1_pin(monkeypatch):
    """Same evaluator + judge under a v1 pin records ``version == "v1"``
    when the judge HAS a v1 entry. Use claude-haiku-4-5 which is in
    both v1 and v2."""
    monkeypatch.setenv("MULTIVON_CALIBRATION_VERSION", "v1")
    load_calibration(reload=True)

    judge = JudgeConfig(provider="anthropic", model="claude-haiku-4-5").resolve()
    fp = fingerprint_evaluator(Faithfulness(judge=judge))
    assert fp.calibration is not None
    assert fp.calibration["version"] == "v1"


# ─────────────────────────────────────────────────────────────────────────────
# Audit log → calibration version extraction
# ─────────────────────────────────────────────────────────────────────────────

def _record_with_calibration_version(version: str) -> dict[str, Any]:
    return {
        "record_hash": "x" * 64,
        "chain_version": 1,
        "prev_hash": "0" * 64,
        "provenance": {
            "suite_lock": {
                "evaluators": [
                    {"name": "faithfulness", "calibration": {"version": version}},
                ],
            },
        },
    }


def test_extract_calibration_version_picks_from_first_record():
    log = (
        json.dumps(_record_with_calibration_version("v1")) + "\n"
        + json.dumps(_record_with_calibration_version("v2")) + "\n"
    ).encode("utf-8")
    # First record wins — suite_lock is stable within a session.
    assert _calibration_version_from_log(log) == "v1"


def test_extract_calibration_version_returns_none_for_legacy_log():
    """A pre-0.7.0 record with no provenance block returns None so the
    builder falls back to the shipped default."""
    log = json.dumps({"record_hash": "x" * 64, "summary": {}}).encode("utf-8")
    assert _calibration_version_from_log(log) is None


def test_extract_calibration_version_handles_empty_log():
    assert _calibration_version_from_log(b"") is None
    assert _calibration_version_from_log(b"\n\n  \n") is None


def test_extract_calibration_version_skips_empty_calibration_field():
    """An evaluator with no calibration entry shouldn't break the lookup —
    the first record may have non-judged evaluators (Contains, WordCount)."""
    rec = {
        "provenance": {
            "suite_lock": {
                "evaluators": [
                    {"name": "wordcount", "calibration": None},
                    {"name": "faithfulness", "calibration": {"version": "v2"}},
                ],
            },
        },
    }
    log = (json.dumps(rec) + "\n").encode("utf-8")
    assert _calibration_version_from_log(log) == "v2"


# ─────────────────────────────────────────────────────────────────────────────
# _read_calibration_data honors the requested label
# ─────────────────────────────────────────────────────────────────────────────

def test_read_calibration_data_honors_requested_label():
    version, data = _read_calibration_data("v1")
    assert version == "v1"
    parsed = json.loads(data)
    assert parsed["schema_version"] == 1

    version, data = _read_calibration_data("v2")
    assert version == "v2"
    parsed = json.loads(data)
    assert parsed["schema_version"] == 2


def test_read_calibration_data_falls_back_when_label_none():
    """Without a label, falls back to v2 → v1 preference order, matching
    the default loader behavior."""
    version, _ = _read_calibration_data(None)
    assert version in ("v2", "v1")
    # Either of these is shipped — at minimum v1 must be available.


def test_read_calibration_data_unknown_label_raises():
    """If the audit log references a label the installed package
    doesn't ship, fail loudly so the operator knows to install the
    right version of multivon-eval."""
    with pytest.raises(FileNotFoundError) as exc:
        _read_calibration_data("v99-not-shipped")
    msg = str(exc.value)
    assert "v99-not-shipped" in msg
    assert "Install the version" in msg or "not shipped" in msg


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: audit-package bundles the version the log records
# ─────────────────────────────────────────────────────────────────────────────

def _build_log(tmp_path: Path, suite_name: str, calibration_version: str) -> Path:
    """Write a minimal audit log with the given calibration version."""
    logs_dir = tmp_path / "audit-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe = suite_name.replace(" ", "_")
    log_path = logs_dir / f"{safe}.audit.ndjson"
    record = _record_with_calibration_version(calibration_version)
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return logs_dir


def test_audit_package_bundles_v1_calibration_when_log_records_v1(tmp_path):
    """The bug-fix headline test: a log recorded against v1 produces a
    package containing calibration_v1.json, not v2.json."""
    logs_dir = _build_log(tmp_path, "trial-suite", "v1")
    out = tmp_path / "evidence.zip"

    build_audit_package(
        logs_dir=logs_dir,
        suite_name="trial-suite",
        framework="eu-ai-act",
        out_path=out,
    )

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        # The bundled calibration is named after the LOGGED version.
        cal_names = [n for n in names if "calibration_" in n]
        assert any("calibration_v1.json" in n for n in cal_names), (
            f"expected v1 in bundle, got {cal_names}"
        )
        assert not any("calibration_v2.json" in n for n in cal_names), (
            f"v2 must not be bundled when log records v1: {cal_names}"
        )

        manifest_path = next(n for n in names if n.endswith("/manifest.json"))
        manifest = json.loads(zf.read(manifest_path).decode("utf-8"))
        assert manifest["calibration_version"] == "v1"
        assert manifest["calibration_source"] == "logged"


def test_audit_package_falls_back_to_default_for_legacy_log(tmp_path):
    """Legacy log (no provenance) gets the shipped default. Backward-
    compatible behavior — pre-0.7 audits can still be packaged."""
    logs_dir = tmp_path / "audit-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "legacy-suite.audit.ndjson"
    legacy_record = {"record_hash": "y" * 64, "summary": {"pass_rate": 1.0}}
    log_path.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")

    out = tmp_path / "legacy-evidence.zip"
    build_audit_package(
        logs_dir=logs_dir,
        suite_name="legacy-suite",
        framework="none",
        out_path=out,
    )

    with zipfile.ZipFile(out) as zf:
        manifest_path = next(n for n in zf.namelist() if n.endswith("/manifest.json"))
        manifest = json.loads(zf.read(manifest_path).decode("utf-8"))
        assert manifest["calibration_source"] == "default"
        assert manifest["calibration_version"] in calibration_versions()


def test_audit_package_unknown_calibration_version_in_log_raises(tmp_path):
    """If a log was somehow produced against a calibration label that
    isn't shipped (e.g. someone bumped the lib, dropped v0.5), fail
    rather than silently swap in v2."""
    logs_dir = _build_log(tmp_path, "stale-suite", "v_unshipped")
    out = tmp_path / "stale.zip"
    with pytest.raises(FileNotFoundError):
        build_audit_package(
            logs_dir=logs_dir,
            suite_name="stale-suite",
            framework="eu-ai-act",
            out_path=out,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Codex round-2 issues: suite-level pinning + unshipped-label propagation
# ─────────────────────────────────────────────────────────────────────────────

def test_suite_lock_records_calibration_version_at_top_level(monkeypatch):
    """Codex round-2 ISSUE 1: the version must land in the suite lock
    even when NO evaluator has a calibration entry — e.g., a suite of
    only deterministic evaluators (WordCount, Contains). Without this
    the audit log records no version and the package falls back to the
    default."""
    from multivon_eval.evaluators import WordCount
    from multivon_eval.suite import EvalSuite
    monkeypatch.setenv("MULTIVON_CALIBRATION_VERSION", "v1")
    load_calibration(reload=True)

    suite = EvalSuite("no-judge-suite")
    suite.add_case(EvalCase("q", expected_output="ans"))
    suite.add_evaluator(WordCount(min_words=1))
    lock = suite.lock()
    assert lock.calibration_version == "v1", (
        "suite-level field must be populated regardless of evaluator coverage"
    )


def test_suite_lock_serializes_calibration_version(monkeypatch):
    """The suite-level field must survive ``to_dict`` / ``from_dict``
    so it lands in the audit log provenance block."""
    from multivon_eval.evaluators import WordCount
    from multivon_eval.suite import EvalSuite
    monkeypatch.setenv("MULTIVON_CALIBRATION_VERSION", "v1")
    load_calibration(reload=True)

    suite = EvalSuite("round-trip")
    suite.add_case(EvalCase("q", expected_output="a"))
    suite.add_evaluator(WordCount(min_words=1))
    lock = suite.lock()
    serialized = lock.to_dict()
    assert serialized["calibration_version"] == "v1"

    from multivon_eval.lockfile import SuiteLock
    restored = SuiteLock.from_dict(serialized)
    assert restored.calibration_version == "v1"


def test_build_suite_lock_propagates_unshipped_version_error(monkeypatch):
    """Codex round-2 ISSUE 2: an unknown env-var label must raise during
    lock construction, NOT silently leave the field None. Otherwise the
    user thinks they're pinned but the audit log records nothing."""
    from multivon_eval.evaluators import WordCount
    from multivon_eval.suite import EvalSuite

    monkeypatch.setenv("MULTIVON_CALIBRATION_VERSION", "v_does_not_exist")
    # Clear the cache so the bad label is actually probed.
    from multivon_eval.calibration import _TABLE_CACHE
    _TABLE_CACHE.pop("v_does_not_exist", None)

    suite = EvalSuite("broken")
    suite.add_case(EvalCase("q", expected_output="a"))
    suite.add_evaluator(WordCount(min_words=1))
    with pytest.raises(FileNotFoundError):
        suite.lock()


def test_audit_package_uses_suite_level_version_over_per_evaluator(tmp_path):
    """Codex round-2: when both fields are set and DIFFER (which can
    happen if the per-evaluator field is stale), the suite-level field
    wins because it reflects what the loader actually returned at lock
    build time."""
    logs_dir = tmp_path / "audit-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "mixed-suite.audit.ndjson"
    record = {
        "record_hash": "z" * 64,
        "chain_version": 1,
        "prev_hash": "0" * 64,
        "provenance": {
            "suite_lock": {
                "calibration_version": "v1",   # authoritative
                "evaluators": [
                    {"name": "faithfulness", "calibration": {"version": "v2"}},
                ],
            },
        },
    }
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    out = tmp_path / "mixed.zip"
    build_audit_package(
        logs_dir=logs_dir,
        suite_name="mixed-suite",
        framework="eu-ai-act",
        out_path=out,
    )
    with zipfile.ZipFile(out) as zf:
        manifest_path = next(n for n in zf.namelist() if n.endswith("/manifest.json"))
        manifest = json.loads(zf.read(manifest_path).decode("utf-8"))
        assert manifest["calibration_version"] == "v1"
        names = zf.namelist()
        assert any("calibration_v1.json" in n for n in names)
        assert not any("calibration_v2.json" in n for n in names)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: real suite.run → ComplianceReporter.record → audit-package
# Codex round-2 ISSUE 3: handwritten provenance blocks bypass the real
# fingerprint → suite_lock → record flow. Cover it for real.
# ─────────────────────────────────────────────────────────────────────────────

def test_suite_run_propagates_unshipped_calibration_pin(monkeypatch):
    """Codex round-2 ISSUE 1 (follow-up): the safe-lock wrapper in
    ``suite.run`` must still propagate ``FileNotFoundError`` from an
    unshipped calibration label. Otherwise a misconfigured pin gets
    swallowed in the hot path (suite_lock=None → audit log empty →
    audit-package falls back to default), defeating the pin silently."""
    from multivon_eval.evaluators import WordCount
    from multivon_eval.suite import EvalSuite
    from multivon_eval.calibration import _TABLE_CACHE

    monkeypatch.setenv("MULTIVON_CALIBRATION_VERSION", "v_does_not_exist")
    _TABLE_CACHE.pop("v_does_not_exist", None)

    suite = EvalSuite("propagate-test")
    suite.add_case(EvalCase("q", expected_output="a"))
    suite.add_evaluator(WordCount(min_words=1))

    with pytest.raises(FileNotFoundError):
        suite.run(lambda _: "a", verbose=False)


def test_e2e_v1_pin_round_trip_through_audit_package(tmp_path, monkeypatch):
    """Real end-to-end: pin to v1, run a suite, record via the reporter,
    package, verify the bundle holds v1."""
    from multivon_eval.evaluators import WordCount
    from multivon_eval.suite import EvalSuite

    monkeypatch.setenv("MULTIVON_CALIBRATION_VERSION", "v1")
    load_calibration(reload=True)

    logs_dir = tmp_path / "audit-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    suite = EvalSuite("e2e-v1-suite")
    suite.add_case(EvalCase("hello", expected_output="hello world"))
    suite.add_evaluator(WordCount(min_words=1, max_words=10))

    def fn(_input: str) -> str:  # deterministic, no judge needed
        return "hello world"

    report = suite.run(fn)
    assert report.suite_lock is not None
    assert report.suite_lock.calibration_version == "v1"

    reporter = ComplianceReporter(
        framework="eu-ai-act",
        output_dir=str(logs_dir),
    )
    reporter.record(report)

    # Inspect the on-disk log: the suite-level version made it into the
    # provenance block.
    log_path = logs_dir / "e2e-v1-suite.audit.ndjson"
    first_record = json.loads(log_path.read_text().splitlines()[0])
    prov = first_record["provenance"]
    assert prov["suite_lock"]["calibration_version"] == "v1"

    # Package and verify v1 calibration is what's bundled.
    out = tmp_path / "e2e.zip"
    build_audit_package(
        logs_dir=logs_dir,
        suite_name=suite.name,
        framework="eu-ai-act",
        out_path=out,
    )
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(
            zf.read(next(n for n in zf.namelist() if n.endswith("manifest.json")))
            .decode("utf-8")
        )
        assert manifest["calibration_version"] == "v1"
        assert manifest["calibration_source"] == "logged"
        assert any("calibration_v1.json" in n for n in zf.namelist())
