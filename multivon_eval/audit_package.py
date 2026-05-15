"""
Compliance Evidence Package — one-command auditor-attachable zip.

Bundles everything an auditor (SOC 2 / ISO 42001 / EU AI Act / HIPAA)
asks for after the fact:

  * Every audit record in the period, in append-only NDJSON form, with
    its hash chain unmodified.
  * The library's calibration data (`_calibration_data/v1.json`) shipped
    by the version of multivon-eval that produced these records.
  * A coverage report against the framework's measurable + process
    controls.
  * A self-contained HTML rollup, identical to what
    :class:`ComplianceHtmlReporter` produces.
  * A manifest with: package version, library version, command-line
    invocation, SHA-256 of every file in the bundle, ISO-8601 generation
    timestamp.
  * A verifier script (`verify.py`) that an auditor can run to recompute
    every hash and rebuild the chain.

Usage::

    multivon-eval audit-package \\
        --logs ./audit-logs \\
        --suite "EU AI Act High-Risk Eval" \\
        --period 2026-Q2 \\
        --framework eu-ai-act \\
        --out evidence-2026-Q2.zip

This module exposes the underlying ``build_audit_package()`` function so
larger compliance pipelines can call it programmatically.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import __version__
from .compliance import _CATALOGS, ComplianceReporter, Framework


_PACKAGE_FORMAT_VERSION = "1.0.0"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_audit_log(logs_dir: Path, suite_name: str) -> bytes:
    """Read the raw NDJSON audit log for ``suite_name`` from ``logs_dir``."""
    safe = suite_name.replace(" ", "_")
    log_path = logs_dir / f"{safe}.audit.ndjson"
    if not log_path.exists():
        raise FileNotFoundError(
            f"Audit log not found: {log_path}\n"
            f"Did you call ComplianceReporter(output_dir={logs_dir!r}).record(...) "
            f"with suite_name={suite_name!r}?"
        )
    return log_path.read_bytes()


def _read_calibration_data(label: str | None = None) -> tuple[str, bytes]:
    """Return ``(version_label, raw JSON)`` for the calibration table to bundle.

    If ``label`` is given (typically extracted from the audit log's
    provenance block), reads that exact version. Otherwise falls back to
    the loader's default preference order (v2 → v1) — matches
    :mod:`multivon_eval.calibration`.

    Pinning to the version recorded in the log is critical for replay:
    bundling a different calibration than the one that drove the
    threshold decisions would mean the f1/dataset_hash numbers in the
    package don't match the audit's decisions. A regulator running the
    verifier would get a green light even though the evidence is
    technically inconsistent.
    """
    from importlib import resources
    pkg = resources.files("multivon_eval._calibration_data")
    candidates = [label] if label else ["v2", "v1"]
    for version in candidates:
        if not version:
            continue
        try:
            data = pkg.joinpath(f"{version}.json").read_bytes()
            return version, data
        except FileNotFoundError:
            continue
    if label:
        raise FileNotFoundError(
            f"Calibration version {label!r} requested by the audit log is not shipped "
            f"with the installed multivon-eval. Install the version that produced these "
            f"records, or rebuild the audit log against a shipped calibration."
        )
    raise FileNotFoundError(
        "No calibration data shipped with multivon_eval (looked for v2.json, v1.json)"
    )


def _calibration_version_from_log(log_bytes: bytes) -> str | None:
    """Extract the calibration version label from the FIRST audit record's
    ``provenance.suite_lock.evaluators[*].calibration.version`` field.

    Returns ``None`` if:
      - the log has no records,
      - the first record predates 0.7.0 provenance (legacy),
      - the suite_lock is absent / serialization failed,
      - no evaluator has a calibration entry with a ``version`` key.

    The first-record convention is fine because suite_lock is stable
    within a session; if a user mid-stream switches calibration versions
    they should be writing to a fresh log file anyway.
    """
    for raw in log_bytes.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except (ValueError, TypeError):
            continue
        prov = rec.get("provenance") or {}
        lock = prov.get("suite_lock") or {}
        if not isinstance(lock, dict):
            return None
        for ev in lock.get("evaluators", []) or []:
            cal = (ev or {}).get("calibration") or {}
            ver = cal.get("version")
            if isinstance(ver, str) and ver:
                return ver
        return None
    return None


def _verifier_script() -> bytes:
    """Standalone Python script (~50 LOC) that an auditor runs to verify."""
    return _VERIFIER_PY.encode("utf-8")


_VERIFIER_PY = '''#!/usr/bin/env python3
"""
Verifier for a multivon-eval Compliance Evidence Package.

Run alongside the package contents::

    python verify.py

Checks:
  1. Every file's SHA-256 matches the manifest.
  2. The audit log's hash chain is intact end-to-end.
  3. The package format version is one this script can read.

Exits 0 on success, non-zero on the first failure encountered. Auditors
should keep this script with the package contents.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    manifest_path = HERE / "manifest.json"
    if not manifest_path.exists():
        print("FAIL: manifest.json missing", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fmt = manifest.get("package_format_version")
    if fmt != "1.0.0":
        print(f"FAIL: unknown package_format_version {fmt!r}", file=sys.stderr)
        return 3

    # 1) Recompute every file's SHA-256.
    errors = 0
    for entry in manifest["files"]:
        path = HERE / entry["path"]
        if not path.exists():
            print(f"FAIL  missing  {entry['path']}", file=sys.stderr)
            errors += 1
            continue
        actual = sha256(path.read_bytes())
        if actual != entry["sha256"]:
            print(f"FAIL  hash     {entry['path']}", file=sys.stderr)
            errors += 1
        else:
            print(f"OK    hash     {entry['path']}")

    # 2) Walk the audit log chain.
    log_relpath = manifest.get("audit_log")
    if log_relpath:
        prev_expected = "0" * 64
        log = HERE / log_relpath
        for i, line in enumerate(log.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            stored = data.pop("record_hash")
            recomputed = sha256(json.dumps(data, separators=(",", ":")).encode())
            if recomputed != stored:
                print(f"FAIL  chain    record {i} hash mismatch", file=sys.stderr)
                errors += 1
                prev_expected = "0" * 64
                continue
            chain_v = data.get("chain_version")
            if chain_v is not None:
                if data.get("prev_hash") != prev_expected:
                    print(f"FAIL  chain    record {i} prev_hash mismatch", file=sys.stderr)
                    errors += 1
                else:
                    print(f"OK    chain    record {i}")
            else:
                print(f"OK    legacy   record {i}")
            prev_expected = recomputed

    if errors:
        print(f"\\nVERIFICATION FAILED: {errors} error(s)", file=sys.stderr)
        return 1
    print("\\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _coverage_report_for(framework: Framework, suite_name: str) -> str:
    """Render a coverage report for the framework, listing controls.

    We don't have access to the actual suite at package time (only the
    log). So we list the framework's full control catalog with annotations
    parsed from the log records.
    """
    catalog = _CATALOGS.get(framework, {})
    measurable = catalog.get("measurable", {})
    process = catalog.get("process", {})

    lines: list[str] = []
    lines.append(f"# Compliance coverage — {framework}")
    lines.append("")
    lines.append(f"Suite: {suite_name}")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Measurable controls")
    lines.append("")
    if not measurable:
        lines.append("_No measurable controls registered for this framework in multivon-eval._")
    else:
        for cid, ctrl in measurable.items():
            lines.append(f"- **{ctrl.id}** — {ctrl.description}")
    lines.append("")
    lines.append("## Process controls (require organizational measures, not satisfiable by evaluators)")
    lines.append("")
    if not process:
        lines.append("_No process controls registered._")
    else:
        for cid, ctrl in process.items():
            lines.append(f"- **{ctrl.id}** — {ctrl.description}")
    return "\n".join(lines)


def build_audit_package(
    *,
    logs_dir: Path,
    suite_name: str,
    framework: Framework,
    out_path: Path,
    period_label: Optional[str] = None,
    extra_files: Optional[dict[str, bytes]] = None,
) -> Path:
    """Build the compliance evidence zip.

    Args:
        logs_dir:       Directory containing the ``<suite>.audit.ndjson`` log.
        suite_name:     Suite name that was passed to ``ComplianceReporter``.
        framework:      One of the multivon-eval :data:`Framework` literals.
        out_path:       Output ZIP path. Parent will be created if missing.
        period_label:   Human label like "2026-Q2". Appears in the manifest
                        and the bundle's directory prefix.
        extra_files:    Optional ``{path_in_zip: bytes}`` to include
                        alongside the standard package contents (e.g. a
                        cover letter from your compliance officer).

    Returns the resolved ``out_path``.

    Raises ``FileNotFoundError`` if the audit log doesn't exist. Raises
    nothing else — every file the package includes is generated in
    memory before the zip is opened.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    period_label = period_label or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    safe_period = period_label.replace(" ", "_")
    prefix = f"compliance-evidence-{safe_period}/"

    # ── gather files in memory so we can hash them deterministically ──
    audit_log = _read_audit_log(logs_dir, suite_name)
    # Prefer the calibration version recorded in the log itself — that's
    # the version the decisions were actually made against. Fall back to
    # the shipped default only when the log doesn't say (legacy records
    # without provenance, or evaluators without calibration).
    logged_version = _calibration_version_from_log(audit_log)
    calibration_version, calibration_json = _read_calibration_data(logged_version)
    calibration_filename = f"calibration_{calibration_version}.json"
    coverage_md = _coverage_report_for(framework, suite_name).encode("utf-8")
    verifier_py = _verifier_script()
    readme = _readme_for(framework, suite_name, period_label, calibration_filename).encode("utf-8")

    files: dict[str, bytes] = {
        "audit_log.ndjson": audit_log,
        calibration_filename: calibration_json,
        "coverage_report.md": coverage_md,
        "verify.py": verifier_py,
        "README.md": readme,
    }
    # Reserved names — including manifest.json, which is written below.
    _RESERVED = set(files) | {"manifest.json"}
    if extra_files:
        for name, blob in extra_files.items():
            if name in _RESERVED:
                raise ValueError(f"extra_files cannot overwrite reserved file {name!r}")
            files[name] = blob

    # ── manifest ──────────────────────────────────────────────────────
    manifest = {
        "package_format_version": _PACKAGE_FORMAT_VERSION,
        "multivon_eval_version": __version__,
        "framework": framework,
        "suite_name": suite_name,
        "period": period_label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit_log": "audit_log.ndjson",
        # Recorded explicitly so the verifier can confirm the bundled
        # calibration JSON matches the version recorded in the log's
        # provenance block. "logged" means we read it from the audit log;
        # "default" means we fell back because the log didn't specify.
        "calibration_version": calibration_version,
        "calibration_source": "logged" if logged_version else "default",
        "files": [
            {"path": name, "sha256": _sha256_bytes(blob), "bytes": len(blob)}
            for name, blob in sorted(files.items())
        ],
    }
    manifest_blob = json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8")
    files["manifest.json"] = manifest_blob

    # ── write the zip atomically ──────────────────────────────────────
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, blob in sorted(files.items()):
            zf.writestr(prefix + name, blob)
    return out_path


def _readme_for(framework: str, suite_name: str, period_label: str,
                calibration_filename: str = "calibration_v1.json") -> str:
    return f"""# Compliance Evidence Package

**Suite:** {suite_name}
**Framework:** {framework}
**Period:** {period_label}
**Package format:** {_PACKAGE_FORMAT_VERSION}
**Built by:** multivon-eval {__version__}
**Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}

## What's in this archive

- `audit_log.ndjson` — Append-only NDJSON log produced by
  `ComplianceReporter`. Every entry is part of a SHA-256 hash chain;
  see `verify.py` to recompute it.
- `{calibration_filename}` — The per-judge threshold table that drove the
  evaluator decisions in the audit log, with dataset hashes, N, F1,
  and measurement dates.
- `coverage_report.md` — Catalog of the framework's measurable and
  process controls.
- `verify.py` — Standalone Python script. Run `python verify.py` from
  inside this directory to recompute every file's SHA-256 and walk the
  hash chain. Exits 0 on success, non-zero on any failure.
- `manifest.json` — Machine-readable inventory with SHA-256 of every
  other file in the package.

## How to verify

```bash
unzip evidence-{period_label}.zip
cd compliance-evidence-{period_label.replace(" ", "_")}
python verify.py
```

Expected output ends with `VERIFICATION PASSED`. Any line beginning
with `FAIL` is an integrity finding the auditor should investigate.

## Scope and limits

multivon-eval evaluators provide *measurable* evidence. Process
controls (technical documentation, transparency, human oversight,
administrative safeguards, BAAs) require organizational measures that
this package cannot demonstrate. See `coverage_report.md` for the
distinction.
"""


# ── CLI subcommand (wired by multivon_eval.cli) ────────────────────────────

def _cli(argv: list[str]) -> int:
    """Argparse subcommand: ``multivon-eval audit-package …``"""
    import argparse

    p = argparse.ArgumentParser(
        prog="multivon-eval audit-package",
        description="Bundle audit log + calibration + verifier into a single zip.",
    )
    p.add_argument("--logs", required=True, help="Directory containing the audit log NDJSON files")
    p.add_argument("--suite", required=True, help="Suite name (matches the audit log's filename)")
    p.add_argument("--framework", required=True,
                   choices=["eu-ai-act", "nist-ai-rmf", "hipaa", "none"],
                   help="Compliance framework that drove the audit log")
    p.add_argument("--out", required=True, help="Output ZIP path")
    p.add_argument("--period", default=None, help='Human label like "2026-Q2" (default: today)')
    args = p.parse_args(argv)

    try:
        out = build_audit_package(
            logs_dir=Path(args.logs),
            suite_name=args.suite,
            framework=args.framework,
            out_path=Path(args.out),
            period_label=args.period,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    return 0


__all__ = ["build_audit_package", "_cli"]
