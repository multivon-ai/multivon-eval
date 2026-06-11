"""Prompt-drift staleness detection (drift modes 1-3).

Compares a blessed scan snapshot (``prompt_baseline.json`` — ARTIFACT A)
against a live scan of the repo, joins in per-case provenance
(``metadata["_provenance"]`` — ARTIFACT B, see :mod:`multivon_eval.provenance`),
and reports which prompts changed since cases were authored, which call
sites no cases cover, and which cases point at prompts that no longer exist.

Scope honesty (enforced in output code, not docs):
  - covers prompt drift, coverage gaps, and dead cases ONLY. Shape drift
    and threshold staleness are suite.lock territory
    (``verify_suite_against_lock``) and this module never claims them.
  - every report opens with the determinacy headline; dynamic prompts are
    UNKNOWN forever (never fresh, never stale); REMOVED always carries the
    three-way caveat; a standing blind-spots footer closes every report.

Matching is content-first, structure-second. The line number and the git
SHA are NEVER matching inputs — lines shift on every edit and SHAs break
on rebase/squash. The baseline is named "baseline", not ".lock": it is a
snapshot you consciously refresh, not a regenerated fingerprint that must
verify, and it never touches suite.lock (``_cases_hash`` excludes metadata
by design — the two drift detectors stay orthogonal).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .attribution import PromptRecord, SCANNER_VERSION, scan
from .provenance import (
    PROVENANCE_KEY,
    atomic_write_text,
    git_info,
    read_provenance,
)
from .recorder import (
    DEFAULT_RECORDINGS_NAME,
    ObservedVerdict,
    RecorderError,
    TRUST_TIERS,
    compare_observed,
    load_recordings,
    unmerged_runtime_sites,
)

SCHEMA_VERSION = 1
DEFAULT_BASELINE_NAME = "prompt_baseline.json"

# Staleness scans add these to the scanner's build-dir ignores: tests and
# examples are full of SDK-shaped fixture calls that would flood the report
# (recorded in the baseline so a re-scan with different ignores warns
# instead of producing false orphans). Override with --include-tests/--ignore.
DEFAULT_STALENESS_IGNORES = ("examples", "tests", "third_party", "vendor")

BLIND_SPOTS = [
    "static scan sees kwarg-only anthropic/openai/litellm Python call sites only",
    "does not see the OpenAI Responses API or positional message args",
    "does not see prompts in YAML/Jinja/templates/files or prompt hubs",
    "does not see non-Python services",
]

_REMOVED_CAVEAT = (
    "feature removed, OR renamed+edited in one commit, OR prompt moved "
    "beyond static reach (kwarg-only anthropic/openai/litellm Python call sites)."
)

_RECORD_FIELDS = (
    "file_path", "line", "sdk", "call_site", "role", "role_position",
    "qualname", "fingerprint", "loose_fingerprint", "is_dynamic",
)


class BaselineError(Exception):
    """The baseline file exists but cannot be used (malformed / future schema)."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── baseline (ARTIFACT A) ─────────────────────────────────────────────


@dataclass
class Baseline:
    schema_version: int
    scanner_version: int
    created_at: str
    git: dict[str, Any]
    scan_root: str
    ignore_dirs: list[str]
    records: list[PromptRecord]
    # Runtime-recorded sites (source:"runtime", fingerprint SETS) — a
    # separate trust tier from static records, stored under a separate
    # key and merged only via `staleness baseline --merge-recordings`.
    runtime_records: list[dict[str, Any]] = field(default_factory=list)


def _record_to_dict(r: PromptRecord) -> dict[str, Any]:
    # Prompt TEXT is deliberately not stored: `git show <sha>:<file>`
    # recovers it; a stored copy would be a second prompt that itself drifts.
    return {f: getattr(r, f) for f in _RECORD_FIELDS}


def _record_from_dict(d: dict[str, Any]) -> PromptRecord:
    return PromptRecord(
        file_path=str(d.get("file_path", "")),
        line=int(d.get("line", 0)),
        sdk=str(d.get("sdk", "")),
        call_site=str(d.get("call_site", "")),
        role=str(d.get("role", "")),
        role_position=int(d.get("role_position", -1)),
        qualname=str(d.get("qualname", "<module>")),
        text="",  # not stored in the baseline
        is_dynamic=bool(d.get("is_dynamic", False)),
        fingerprint=str(d.get("fingerprint", "")),
        loose_fingerprint=str(d.get("loose_fingerprint", "")),
    )


def staleness_ignores(
    ignore_dirs: Any = None, include_tests: bool = False
) -> frozenset[str]:
    extra = set(ignore_dirs or ())
    if not include_tests:
        extra |= set(DEFAULT_STALENESS_IGNORES)
    return frozenset(extra)


def scan_repo(
    repo_root: str | Path,
    ignore_dirs: Any = None,
    include_tests: bool = False,
) -> tuple[list[PromptRecord], list[str]]:
    """Live scan with the staleness-default ignores. Returns (records, ignores)."""
    extra = staleness_ignores(ignore_dirs, include_tests)
    return scan(str(repo_root), ignore_dirs=extra), sorted(extra)


def load_baseline(path: str | Path) -> Baseline:
    """Read a prompt_baseline.json. Raises BaselineError on malformed input
    or a schema_version from the future (warn-and-skip, never crash)."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"baseline {path} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineError(f"baseline {path} is not a JSON object")
    sv = payload.get("schema_version")
    if not isinstance(sv, int):
        raise BaselineError(f"baseline {path} has no integer schema_version")
    if sv > SCHEMA_VERSION:
        raise BaselineError(
            f"baseline {path} has schema_version {sv} (written by a newer "
            f"multivon-eval; this reader understands ≤{SCHEMA_VERSION}) — skipped"
        )
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise BaselineError(f"baseline {path} has no records list")
    raw_runtime = payload.get("runtime_records")
    if not isinstance(raw_runtime, list):
        raw_runtime = []
    return Baseline(
        schema_version=sv,
        scanner_version=int(payload.get("scanner_version", 1)),
        created_at=str(payload.get("created_at", "")),
        git=payload.get("git") or {"sha": None, "dirty": False},
        scan_root=str(payload.get("scan_root", ".")),
        ignore_dirs=list(payload.get("ignore_dirs") or []),
        records=[_record_from_dict(d) for d in raw_records if isinstance(d, dict)],
        runtime_records=[d for d in raw_runtime if isinstance(d, dict)],
    )


def write_baseline(
    repo_root: str | Path,
    out: str | Path | None = None,
    *,
    ignore_dirs: Any = None,
    include_tests: bool = False,
    dry_run: bool = False,
) -> tuple[Baseline, list[str]]:
    """Fresh scan → baseline. Returns (baseline, diff-lines vs any existing
    baseline). Writes via temp file + os.replace; --dry-run writes nothing."""
    root = Path(repo_root)
    out_path = Path(out) if out else root / DEFAULT_BASELINE_NAME
    records, ignores = scan_repo(root, ignore_dirs, include_tests)
    baseline = Baseline(
        schema_version=SCHEMA_VERSION,
        scanner_version=SCANNER_VERSION,
        created_at=_utc_now(),
        git=git_info(root),
        scan_root=".",
        ignore_dirs=ignores,
        records=records,
    )

    diff_lines: list[str] = []
    if out_path.exists():
        try:
            old = load_baseline(out_path)
        except BaselineError as exc:
            diff_lines.append(f"existing baseline not comparable: {exc}")
        else:
            # A static rescan never discards runtime observations — the
            # two trust tiers live in separate keys and refresh separately.
            baseline.runtime_records = list(old.runtime_records)
            verdicts, added = match_records(old.records, records)
            for v in verdicts:
                if v.status == "unchanged" and not v.labels:
                    continue
                if v.status == "unknown" and "became-dynamic" not in v.labels:
                    continue  # a persisting dynamic site is not a change
                anchor = v.baseline.call_site_id if v.baseline else "?"
                label = f" [{', '.join(v.labels)}]" if v.labels else ""
                diff_lines.append(f"{v.status}{label}: {anchor}")
            for rec in added:
                diff_lines.append(f"added: {rec.call_site_id}")
            if not diff_lines:
                diff_lines.append("no site changes vs existing baseline")

    if not dry_run:
        payload = {
            "schema_version": baseline.schema_version,
            "scanner_version": baseline.scanner_version,
            "created_at": baseline.created_at,
            "git": baseline.git,
            "scan_root": baseline.scan_root,
            "ignore_dirs": baseline.ignore_dirs,
            "records": [_record_to_dict(r) for r in records],
        }
        if baseline.runtime_records:
            payload["runtime_records"] = baseline.runtime_records
        atomic_write_text(out_path, json.dumps(payload, indent=2) + "\n")
    return baseline, diff_lines


# ── matcher ───────────────────────────────────────────────────────────


@dataclass
class SiteVerdict:
    """One baseline record's verdict against the live scan.

    status: "unchanged" | "changed" | "removed" | "unknown"
    labels: extra context — "moved", "formatting-only", "file-renamed",
            "dynamic", "became-dynamic"
    confidence: "exact" | "structural" | "moved" | "ambiguous"
    """
    status: str
    labels: list[str]
    confidence: str
    baseline: Optional[PromptRecord]
    live: Optional[PromptRecord]
    bound_cases: list[dict[str, Any]] = field(default_factory=list)


def _anchor(r: PromptRecord) -> tuple:
    return (r.file_path, r.qualname, r.sdk, r.call_site, r.role, r.role_position)


def _anchor_nofile(r: PromptRecord) -> tuple:
    return (r.qualname, r.sdk, r.call_site, r.role, r.role_position)


def match_records(
    baseline_records: list[PromptRecord],
    live_records: list[PromptRecord],
) -> tuple[list[SiteVerdict], list[PromptRecord]]:
    """Content-first, structure-second matching. Returns (verdicts, added).

    STEP 0 — dynamic gate FIRST: a dynamic record on either side never
    enters the freshness corpus. Placeholder fingerprints are node-type-
    stable, so equality is vacuous — comparing them would report a totally
    rewritten constant as "fresh".
    STEP 1 — exact fingerprint (anchor-consistent claims first so duplicate
    prompt text resolves one-to-one).
    STEP 2 — structural anchor rescue, tiered, line-free. Tier B
    (file-renamed) additionally requires loose-fingerprint equality:
    rename+edit in one commit is statically unbridgeable and must degrade
    to REMOVED, never a fuzzy CHANGED.
    Line numbers are tie-breakers only; git SHAs never participate.
    """
    live_static = [l for l in live_records if not l.is_dynamic]
    fp_index: dict[str, list[PromptRecord]] = {}
    for l in live_static:
        fp_index.setdefault(l.fingerprint, []).append(l)
    anchor_index: dict[tuple, list[PromptRecord]] = {}
    for l in live_records:
        anchor_index.setdefault(_anchor(l), []).append(l)

    consumed: set[int] = set()
    results: dict[int, SiteVerdict] = {}
    pending: list[tuple[int, PromptRecord]] = []

    # STEP 0 — dynamic gate.
    for i, b in enumerate(baseline_records):
        if not b.is_dynamic:
            pending.append((i, b))
            continue
        cands = [l for l in anchor_index.get(_anchor(b), []) if id(l) not in consumed]
        if cands:
            live = cands[0]
            consumed.add(id(live))
            results[i] = SiteVerdict("unknown", ["dynamic"], "structural", b, live)
        else:
            # dynamic baseline whose structural anchor disappeared degrades
            # to REMOVED normally.
            results[i] = SiteVerdict("removed", ["dynamic"], "structural", b, None)

    # STEP 1a — anchor-consistent exact-fingerprint claims.
    still: list[tuple[int, PromptRecord]] = []
    for i, b in pending:
        hits = [l for l in fp_index.get(b.fingerprint, [])
                if id(l) not in consumed and _anchor(l) == _anchor(b)]
        if hits:
            live = hits[0]
            consumed.add(id(live))
            results[i] = SiteVerdict("unchanged", [], "exact", b, live)
        else:
            still.append((i, b))

    # STEP 1b — remaining exact-fingerprint hits (moved / ambiguous).
    still2: list[tuple[int, PromptRecord]] = []
    for i, b in still:
        hits = [l for l in fp_index.get(b.fingerprint, []) if id(l) not in consumed]
        if not hits:
            still2.append((i, b))
            continue
        if len(hits) == 1:
            live = hits[0]
            consumed.add(id(live))
            results[i] = SiteVerdict("unchanged", ["moved"], "moved", b, live)
        else:
            # duplicate prompt text with no anchor-consistent site left —
            # surfaced, not silently resolved.
            live = min(hits, key=lambda l: (l.file_path != b.file_path,
                                            abs(l.line - b.line)))
            consumed.add(id(live))
            results[i] = SiteVerdict("unchanged", ["moved"], "ambiguous", b, live)

    # STEP 2 — structural anchor rescue (line-free).
    for i, b in still2:
        tier_a = [l for l in anchor_index.get(_anchor(b), []) if id(l) not in consumed]
        tier_a_static = [l for l in tier_a if not l.is_dynamic]
        tier_a_dynamic = [l for l in tier_a if l.is_dynamic]
        if tier_a_static:
            if len(tier_a_static) == 1:
                live, conf = tier_a_static[0], "structural"
            else:
                live = min(tier_a_static, key=lambda l: abs(l.line - b.line))
                conf = "ambiguous"
            consumed.add(id(live))
            labels = []
            if b.loose_fingerprint and b.loose_fingerprint == live.loose_fingerprint:
                labels.append("formatting-only")
            results[i] = SiteVerdict("changed", labels, conf, b, live)
            continue
        if tier_a_dynamic:
            # prompt moved out of static reach — cannot compare. Never
            # CHANGED, never REMOVED.
            live = tier_a_dynamic[0]
            consumed.add(id(live))
            results[i] = SiteVerdict("unknown", ["became-dynamic"], "structural", b, live)
            continue
        # Tier B — same anchor minus file_path in exactly one other file,
        # AND loose fingerprints match (content survived modulo whitespace).
        cands_b = [
            l for l in live_static
            if id(l) not in consumed
            and l.file_path != b.file_path
            and _anchor_nofile(l) == _anchor_nofile(b)
            and b.loose_fingerprint
            and l.loose_fingerprint == b.loose_fingerprint
        ]
        if len(cands_b) == 1:
            live = cands_b[0]
            consumed.add(id(live))
            results[i] = SiteVerdict(
                "changed", ["file-renamed", "formatting-only"], "structural", b, live,
            )
            continue
        # Tier C — gone. Mandatory three-way caveat lives in the renderers.
        results[i] = SiteVerdict("removed", [], "structural", b, None)

    baseline_fps = {b.fingerprint for b in baseline_records if not b.is_dynamic}
    baseline_anchors = {_anchor(b) for b in baseline_records}
    added = [
        l for l in live_static
        if id(l) not in consumed
        and l.fingerprint not in baseline_fps
        and _anchor(l) not in baseline_anchors
    ]
    verdicts = [results[i] for i in sorted(results)]
    return verdicts, added


# ── case-level (ARTIFACT B join) ──────────────────────────────────────


@dataclass
class CaseVerdict:
    file: str
    index: int
    case_uid: Optional[str]
    status: str            # "unstamped" | "unreadable" | "stamped" | "bound"
    rollup: Optional[str]  # bound only: unchanged|unverifiable|unknown|changed|removed
    targets: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    git_sha: Optional[str] = None
    authored_by: Optional[str] = None


_ROLLUP_ORDER = {"unchanged": 0, "unverifiable": 1, "unknown": 2, "changed": 3,
                 "removed": 4}


def _record_from_target(t: dict[str, Any]) -> PromptRecord:
    a = t.get("anchor") or {}
    return PromptRecord(
        file_path=str(a.get("file_path", "")),
        line=int(a.get("line") or 0),
        sdk=str(a.get("sdk", "")),
        call_site=str(a.get("call_site", "")),
        role=str(a.get("role", "")),
        role_position=int(a.get("role_position", -1)),
        qualname=str(a.get("qualname", "<module>")),
        text="",
        is_dynamic=bool(t.get("is_dynamic", False)),
        fingerprint=str(t.get("fingerprint", "")),
        loose_fingerprint=str(t.get("loose_fingerprint", "")),
    )


def match_target(
    target: dict[str, Any], live_records: list[PromptRecord]
) -> tuple[str, list[str]]:
    """Per-target verdict — algorithm identical to site-level matching."""
    if target.get("source") == "external":
        # prompt the scanner cannot see — UNVERIFIABLE, never orphaned.
        return ("unverifiable", [])
    if target.get("source") == "runtime":
        # observed rendering — comparable against recordings, NEVER against
        # the static scan (a rendered fingerprint can't match a placeholder).
        return ("unverifiable", ["runtime"])
    if target.get("is_dynamic"):
        # a dynamic target is reported unknown forever, by rule.
        return ("unknown", ["dynamic"])
    verdicts, _added = match_records([_record_from_target(target)], live_records)
    v = verdicts[0]
    return (v.status, v.labels)


def _rollup(statuses: list[str]) -> str:
    if statuses and all(s == "removed" for s in statuses):
        return "removed"
    # partially-orphaned multi-target = CHANGED-class, still actionable.
    mapped = ["changed" if s == "removed" else s for s in statuses]
    return max(mapped, key=lambda s: _ROLLUP_ORDER.get(s, 0)) if mapped else "unchanged"


# Sentinel for a JSONL line that isn't valid JSON at all — reported as
# unreadable, never a crash.
_UNPARSEABLE = object()


def _iter_jsonl_cases(path: Path):
    """Yield (index, metadata) per case line; _UNPARSEABLE = broken line."""
    idx = -1
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        idx += 1
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            yield idx, _UNPARSEABLE
            continue
        yield idx, (data.get("metadata") if isinstance(data, dict) else None)


def _cases_from_suite(spec: str) -> list[tuple[int, Any]]:
    """Load `module:attr` and read runtime case metadata from an EvalSuite
    (or a callable returning one, or a list of EvalCase)."""
    import importlib

    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"--suite expects module:attr, got {spec!r}")
    obj = getattr(importlib.import_module(module_name), attr)
    if callable(obj) and not hasattr(obj, "_cases"):
        obj = obj()
    cases = getattr(obj, "_cases", obj)
    return [(i, getattr(c, "metadata", None)) for i, c in enumerate(cases)]


def evaluate_cases(
    case_sources: list[tuple[str, list[tuple[int, Any]]]],
    live_records: list[PromptRecord],
) -> list[CaseVerdict]:
    out: list[CaseVerdict] = []
    for label, entries in case_sources:
        for idx, metadata in entries:
            if metadata is _UNPARSEABLE:
                out.append(CaseVerdict(label, idx, None, "unreadable", None,
                                       notes=["line is not valid JSON"]))
                continue
            status, prov = read_provenance(metadata)
            if status == "unstamped":
                out.append(CaseVerdict(label, idx, None, "unstamped", None))
                continue
            if status in ("unreadable", "unreadable_newer"):
                note = (
                    "unreadable provenance (stamped by a newer multivon-eval)"
                    if status == "unreadable_newer" else "unreadable provenance"
                )
                uid = prov.get("case_uid") if isinstance(prov, dict) else None
                out.append(CaseVerdict(label, idx, uid, "unreadable", None,
                                       notes=[note]))
                continue
            assert prov is not None
            targets = [t for t in (prov.get("targets") or []) if isinstance(t, dict)]
            git = prov.get("git") if isinstance(prov.get("git"), dict) else {}
            notes: list[str] = []
            restamped = prov.get("stamped_at") and prov.get("authored_at") \
                and prov["stamped_at"] != prov["authored_at"]
            if restamped and not prov.get("evidence"):
                # self-attestation is visible, not silent.
                notes.append("restamped with no recorded run evidence")
            if not targets:
                out.append(CaseVerdict(
                    label, idx, prov.get("case_uid"), "stamped", None,
                    notes=notes, git_sha=git.get("sha"),
                    authored_by=prov.get("authored_by"),
                ))
                continue
            statuses = []
            for t in targets:
                t_status, _labels = match_target(t, live_records)
                statuses.append(t_status)
            out.append(CaseVerdict(
                label, idx, prov.get("case_uid"), "bound", _rollup(statuses),
                targets=targets, notes=notes, git_sha=git.get("sha"),
                authored_by=prov.get("authored_by"),
            ))
    return out


# ── report assembly ───────────────────────────────────────────────────


@dataclass
class StalenessReport:
    repo_root: str
    git: dict[str, Any]
    baseline_path: str
    baseline: Optional[Baseline]
    no_baseline: bool
    baseline_warning: Optional[str]
    scanner_mismatch: bool
    ignore_warning: Optional[str]
    live_records: list[PromptRecord]
    verdicts: list[SiteVerdict]
    added: list[PromptRecord]
    case_verdicts: list[CaseVerdict]
    fail_on: tuple[str, ...] = ()
    # — runtime recorder tier (source:"runtime"; recordings-vs-recordings) —
    observed: list[ObservedVerdict] = field(default_factory=list)
    recordings_path: Optional[str] = None
    has_current_recordings: bool = False
    recordings_warning: Optional[str] = None
    unmerged_runtime: int = 0

    # — determinacy: live-scan counts —
    @property
    def determinacy(self) -> dict[str, int]:
        total = len(self.live_records)
        dynamic = sum(1 for r in self.live_records if r.is_dynamic)
        return {
            "call_sites": total,
            "static": total - dynamic,
            "dynamic": dynamic,
            "observed_runtime": len(self.observed),
        }

    def counts(self) -> dict[str, int]:
        c = {"unchanged": 0, "moved": 0, "changed": 0, "formatting_only": 0,
             "removed": 0, "unknown": 0, "added": len(self.added)}
        for v in self.verdicts:
            if v.status == "unchanged":
                c["moved" if "moved" in v.labels else "unchanged"] += 1
            elif v.status == "changed":
                c["changed"] += 1
                if "formatting-only" in v.labels:
                    c["formatting_only"] += 1
            elif v.status == "removed":
                c["removed"] += 1
            elif v.status == "unknown":
                c["unknown"] += 1
        return c

    def case_stats(self) -> dict[str, int]:
        s = {"total": 0, "stamped": 0, "bound": 0, "unstamped": 0,
             "unreadable": 0, "restamped_no_evidence": 0}
        for cv in self.case_verdicts:
            s["total"] += 1
            if cv.status == "unstamped":
                s["unstamped"] += 1
            elif cv.status == "unreadable":
                s["unreadable"] += 1
            else:
                s["stamped"] += 1
                if cv.status == "bound":
                    s["bound"] += 1
            if "restamped with no recorded run evidence" in cv.notes:
                s["restamped_no_evidence"] += 1
        return s

    def coverage(self) -> tuple[int, int]:
        """(covered, static-site count) — an explicit lower bound. A bound
        target's fingerprint covers ALL duplicate sites with that text, or
        the numbers lie. Dynamic sites never enter the denominator."""
        live_static = [r for r in self.live_records if not r.is_dynamic]
        fps: set[str] = set()
        anchors: set[tuple] = set()
        for cv in self.case_verdicts:
            for t in cv.targets:
                if t.get("source") in ("external", "runtime") or t.get("is_dynamic"):
                    # runtime targets never enter the STATIC coverage join —
                    # the denominator is static sites only, by label.
                    continue
                if t.get("fingerprint"):
                    fps.add(t["fingerprint"])
                anchors.add(_anchor(_record_from_target(t)))
        covered = sum(
            1 for r in live_static
            if r.fingerprint in fps or _anchor(r) in anchors
        )
        return covered, len(live_static)

    @property
    def exit_code(self) -> int:
        """0 = clean or report-only; 1 = a --fail-on category fired;
        2 = warn-only (no baseline / unreadable baseline / scanner mismatch).
        Case-level unreadable provenance is counted but never moves the exit
        code — a newer teammate's stamp must not break an older CI."""
        c = self.counts()
        fired = (
            ("changed" in self.fail_on and c["changed"] > 0)
            or ("removed" in self.fail_on and c["removed"] > 0)
            or ("added" in self.fail_on and c["added"] > 0)
        )
        if fired:
            return 1
        if self.no_baseline or self.baseline_warning or self.scanner_mismatch:
            return 2
        return 0


def _attach_bound_cases(report: StalenessReport) -> None:
    """Join bound case targets onto site verdicts for display."""
    for cv in report.case_verdicts:
        for t in cv.targets:
            t_fp = t.get("fingerprint")
            t_anchor = _anchor(_record_from_target(t))
            for v in report.verdicts:
                if v.baseline is None:
                    continue
                if (t_fp and v.baseline.fingerprint == t_fp) or \
                        _anchor(v.baseline) == t_anchor:
                    ref = {"file": cv.file, "index": cv.index,
                           "case_uid": cv.case_uid}
                    if ref not in v.bound_cases:
                        v.bound_cases.append(ref)


def discover_case_files(repo_root: str | Path) -> list[Path]:
    """Find bootstrap-named case files (seed_cases.jsonl) under the repo,
    pruning the scanner's build-dir ignores."""
    from .attribution.ast_extractor import DEFAULT_IGNORE_DIRS

    found: list[Path] = []
    root = Path(repo_root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in DEFAULT_IGNORE_DIRS and not d.startswith(".")]
        if "seed_cases.jsonl" in filenames:
            found.append(Path(dirpath) / "seed_cases.jsonl")
    return sorted(found)


def build_staleness_report(
    repo_root: str | Path = ".",
    *,
    baseline_path: str | Path | None = None,
    case_files: list[str | Path] | None = None,
    suite: str | None = None,
    ignore_dirs: Any = None,
    include_tests: bool = False,
    fail_on: tuple[str, ...] = (),
    recordings_path: str | Path | None = None,
) -> StalenessReport:
    """Read-only staleness report: live scan vs baseline vs case provenance.

    ``recordings_path`` points at the current prompt_recordings.jsonl
    (default: REPO/prompt_recordings.jsonl when present). Runtime-sourced
    baseline sites are compared recordings-vs-recordings into the OBSERVED
    tier — never against the static scan, never collapsed into it.
    """
    root = Path(repo_root)
    bpath = Path(baseline_path) if baseline_path else root / DEFAULT_BASELINE_NAME
    rpath = Path(recordings_path) if recordings_path \
        else root / DEFAULT_RECORDINGS_NAME

    live, ignores = scan_repo(root, ignore_dirs, include_tests)

    baseline: Optional[Baseline] = None
    no_baseline = False
    baseline_warning: Optional[str] = None
    if not bpath.exists():
        no_baseline = True
    else:
        try:
            baseline = load_baseline(bpath)
        except BaselineError as exc:
            baseline_warning = str(exc)

    scanner_mismatch = (
        baseline is not None and baseline.scanner_version != SCANNER_VERSION
    )
    ignore_warning = None
    if baseline is not None and sorted(baseline.ignore_dirs) != sorted(ignores):
        ignore_warning = (
            f"baseline was scanned with ignore_dirs={baseline.ignore_dirs} but "
            f"this scan uses {sorted(ignores)} — added/removed findings may be "
            f"scan-scope artifacts, not drift"
        )

    if baseline is not None:
        verdicts, added = match_records(baseline.records, live)
    else:
        verdicts, added = [], []

    sources: list[tuple[str, list[tuple[int, Any]]]] = []
    files = [Path(f) for f in case_files] if case_files is not None \
        else discover_case_files(root)
    for f in files:
        sources.append((str(f), list(_iter_jsonl_cases(Path(f)))))
    if suite:
        sources.append((f"<suite {suite}>", _cases_from_suite(suite)))
    case_verdicts = evaluate_cases(sources, live)

    # — runtime tier: recordings vs recordings, never vs the static scan —
    runtime_records = baseline.runtime_records if baseline else []
    current_recordings: Optional[list[dict[str, Any]]] = None
    recordings_warning: Optional[str] = None
    if rpath.exists():
        try:
            current_recordings = load_recordings(rpath)
        except RecorderError as exc:
            recordings_warning = str(exc)
    observed = compare_observed(runtime_records, current_recordings)
    unmerged = unmerged_runtime_sites(runtime_records, current_recordings)

    report = StalenessReport(
        repo_root=str(root),
        git=git_info(root),
        baseline_path=str(bpath),
        baseline=baseline,
        no_baseline=no_baseline,
        baseline_warning=baseline_warning,
        scanner_mismatch=scanner_mismatch,
        ignore_warning=ignore_warning,
        live_records=live,
        verdicts=verdicts,
        added=added,
        case_verdicts=case_verdicts,
        fail_on=tuple(fail_on),
        observed=observed,
        recordings_path=str(rpath),
        has_current_recordings=current_recordings is not None,
        recordings_warning=recordings_warning,
        unmerged_runtime=unmerged,
    )
    _attach_bound_cases(report)
    return report


# ── renderers ─────────────────────────────────────────────────────────


def _sha_display(git: dict[str, Any]) -> str:
    sha = git.get("sha")
    if not sha:
        return "no git"
    if git.get("dirty"):
        return f"{sha} (dirty tree — SHA approximate)"
    return sha


def _age_display(created_at: str) -> str:
    try:
        created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        created = created.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return "unknown age"
    days = (datetime.now(timezone.utc) - created).days
    if days <= 0:
        return "today"
    return f"{days} day{'s' if days != 1 else ''} ago"


def _site_line(rec: PromptRecord) -> str:
    suffix = rec.role if rec.role_position < 0 else f"{rec.role}#{rec.role_position}"
    return f"{rec.file_path}:{rec.line}  {rec.sdk}.{suffix}  in {rec.qualname}"


def _determinacy_line(report: StalenessReport) -> str:
    d = report.determinacy
    line = (
        f"determinacy: {d['static']} of {d['call_sites']} call sites statically "
        f"resolvable; verdicts below cover only those."
    )
    if d["dynamic"]:
        line += f" {d['dynamic']} dynamic site{'s' if d['dynamic'] != 1 else ''} " \
                f"{'are' if d['dynamic'] != 1 else 'is'} unknown-by-construction."
    if d["observed_runtime"]:
        line += (
            f" {d['observed_runtime']} site"
            f"{'s' if d['observed_runtime'] != 1 else ''} observed at runtime "
            f"(recorder — see OBSERVED tier)."
        )
    return line


def _bound_cases_line(v: SiteVerdict, report: StalenessReport) -> Optional[str]:
    if not v.bound_cases:
        unbound = sum(1 for cv in report.case_verdicts if cv.status == "stamped")
        if unbound:
            return f"bound cases: none   ({unbound} cases carry repo-state provenance only)"
        return None
    refs = " ".join(f"{Path(r['file']).name} #{r['index']}" for r in v.bound_cases)
    return f"bound cases: {refs}"


def _footer_lines(report: StalenessReport) -> list[str]:
    s = report.case_stats()
    cov, denom = report.coverage()
    out = []
    if s["total"]:
        by = next((cv.authored_by for cv in report.case_verdicts
                   if cv.authored_by), None)
        sha = next((cv.git_sha for cv in report.case_verdicts if cv.git_sha), None)
        stamp_note = f" ({by}, {sha})" if by and sha else (f" ({by})" if by else "")
        out.append(
            f"cases: {s['total']} total · {s['stamped']} stamped{stamp_note} · "
            f"{s['bound']} bound to sites · {s['unreadable']} unreadable"
        )
        if s["unstamped"]:
            out.append(
                f"  {s['unstamped']} unstamped — stamp with "
                f"`multivon-eval staleness stamp --cases F.jsonl --site ...`"
            )
        if s["restamped_no_evidence"]:
            out.append(
                f"  {s['restamped_no_evidence']} case(s) restamped with no "
                f"recorded run evidence"
            )
    if s["bound"]:
        out.append(
            f"coverage (lower bound, static sites only): {cov}/{denom} sites "
            f"referenced by a bound case"
        )
    else:
        out.append(
            "coverage (lower bound, static sites only): no cases are bound to "
            "call sites yet — binding via `staleness stamp --site ...` is "
            "required before coverage means anything"
        )
    d = report.determinacy
    if d["dynamic"]:
        out.append(f"not statically coverable: {d['dynamic']} dynamic site(s)")
    out.append("trust tiers (never collapsed): " + "; ".join(TRUST_TIERS) + ".")
    out.append("blind spots: " + "; ".join(BLIND_SPOTS) + ".")
    return out


def _observed_lines(report: StalenessReport) -> list[str]:
    """The OBSERVED tier — runtime-sourced sites, recordings vs recordings.

    Honesty rules enforced here: always k-of-N (a site is NEVER called
    fresh because one rendering matched), and the tier states explicitly
    that runtime-only sites cannot be compared against a static scan.
    """
    if not report.observed:
        return []
    lines = [
        f"OBSERVED at runtime ({len(report.observed)}) — recorder-sourced sites; "
        f"compared recordings-vs-recordings (these sites cannot be compared "
        f"against a static scan). A recording proves the renderings observed, "
        f"not all renderings.",
    ]
    for ov in report.observed:
        a = ov.anchor
        lines.append(
            f"  {a.get('file_path', '?')}  {a.get('sdk', '?')}.{a.get('role', '?')} "
            f"in {a.get('qualname', '?')}"
        )
        if not ov.has_current:
            lines.append(
                f"    {ov.baseline_renderings} previously observed rendering(s); "
                f"no current recordings to compare — re-run with "
                f"--record-prompts (or record_prompts()) to refresh"
            )
        else:
            extra = (
                f"; {ov.new_renderings} new rendering(s) not in the baseline"
                if ov.new_renderings else ""
            )
            lines.append(
                f"    current recordings matched {ov.matched} of "
                f"{ov.baseline_renderings} previously observed renderings{extra}"
            )
        if ov.case_uids:
            lines.append(
                f"    cases observed at this site: {len(ov.case_uids)} "
                f"({ov.observations} observation(s))"
            )
    if report.unmerged_runtime:
        lines.append(
            f"  +{report.unmerged_runtime} runtime site(s) recorded but not yet "
            f"in the baseline — `multivon-eval staleness baseline "
            f"--merge-recordings`"
        )
    return lines


def render_text(report: StalenessReport) -> str:
    lines: list[str] = []

    if report.baseline is not None:
        b = report.baseline
        lines.append(
            f"baseline: {report.baseline_path} ({_sha_display(b.git)}, "
            f"{_age_display(b.created_at)}, scanner v{b.scanner_version})"
        )
    lines.append(_determinacy_line(report))

    for warn in (report.baseline_warning, report.ignore_warning,
                 report.recordings_warning):
        if warn:
            lines.append(f"warning: {warn}")
    if report.scanner_mismatch and report.baseline is not None:
        lines.append(
            f"warning: baseline written by scanner "
            f"v{report.baseline.scanner_version}, this scan is "
            f"v{SCANNER_VERSION} — rescan recommended "
            f"(`multivon-eval staleness baseline .`)"
        )

    if report.no_baseline:
        lines.append("")
        lines.append(
            "no baseline found — run `multivon-eval staleness baseline .` "
            "(bootstrap writes one automatically)"
        )
        lines.append("")
        lines.extend(_footer_lines(report))
        lines.append(f"exit {report.exit_code}")
        return "\n".join(lines) + "\n"

    lines.append("")

    changed = [v for v in report.verdicts if v.status == "changed"]
    removed = [v for v in report.verdicts if v.status == "removed"]
    unknown = [v for v in report.verdicts if v.status == "unknown"]
    moved = [v for v in report.verdicts
             if v.status == "unchanged" and "moved" in v.labels]

    if changed:
        lines.append(f"CHANGED ({len(changed)}) — prompt text differs from baseline")
        base_sha = report.baseline.git.get("sha") if report.baseline else None
        for v in changed:
            assert v.baseline is not None
            tags = []
            if "formatting-only" in v.labels:
                tags.append("formatting-only — loose fingerprint unchanged")
            if "file-renamed" in v.labels:
                tags.append("file-renamed")
            if v.confidence == "ambiguous":
                tags.append("confidence=ambiguous")
            tag = f"   [{'; '.join(tags)}]" if tags else ""
            lines.append(f"  {_site_line(v.live or v.baseline)}{tag}")
            new_fp = v.live.fingerprint[:8] if v.live else "?"
            lines.append(f"    fp {v.baseline.fingerprint[:8]}… → {new_fp}…")
            bc = _bound_cases_line(v, report)
            if bc:
                lines.append(f"    {bc}")
            if base_sha:
                # `git diff <sha> -- file` (no ..HEAD) also shows uncommitted
                # edits — the common case when running staleness pre-commit,
                # where <sha>..HEAD would print nothing.
                lines.append(
                    f"    what changed: git diff {base_sha} -- "
                    f"{(v.live or v.baseline).file_path}"
                )
        lines.append("")

    if removed:
        lines.append(f"REMOVED ({len(removed)}) — call site not found by static scan")
        for v in removed:
            assert v.baseline is not None
            lines.append(f"  {v.baseline.file_path}  "
                         f"{v.baseline.sdk}.{v.baseline.role} in {v.baseline.qualname}")
            lines.append(f"    note: {_REMOVED_CAVEAT}")
        lines.append("")

    if report.added:
        lines.append(f"ADDED since baseline ({len(report.added)})")
        for rec in report.added:
            lines.append(f"  {_site_line(rec)}   → no cases reference this prompt")
        lines.append("")

    if moved:
        lines.append(
            f"MOVED ({len(moved)}) — content unchanged, anchor moved; "
            f"baseline refresh suggested"
        )
        for v in moved:
            assert v.baseline is not None and v.live is not None
            amb = "   [confidence=ambiguous]" if v.confidence == "ambiguous" else ""
            lines.append(f"  {v.baseline.file_path} → {_site_line(v.live)}{amb}")
        lines.append("")

    if unknown:
        lines.append(
            f"UNKNOWN ({len(unknown)}) — dynamic prompts; static scan cannot "
            f"verify their text"
        )
        for v in unknown:
            rec = v.live or v.baseline
            assert rec is not None
            extra = "  [became-dynamic — prompt moved out of static reach]" \
                if "became-dynamic" in v.labels else ""
            lines.append(f"  {_site_line(rec)}  {rec.text or '<dynamic>'}{extra}".rstrip())
        lines.append("")

    observed_lines = _observed_lines(report)
    if observed_lines:
        lines.extend(observed_lines)
        lines.append("")

    if not (changed or removed or report.added or unknown or moved):
        lines.append("all call sites unchanged vs baseline.")
        lines.append("")

    lines.extend(_footer_lines(report))
    code = report.exit_code
    if code == 0 and not report.fail_on:
        lines.append("exit 0 (report-only — add --fail-on changed,removed in CI)")
    else:
        lines.append(f"exit {code}")
    if changed or removed:
        lines.append(
            "next: review CHANGED, re-run bound cases, then "
            "`multivon-eval staleness baseline .`"
        )
    return "\n".join(lines) + "\n"


def render_json(report: StalenessReport) -> str:
    c = report.counts()
    s = report.case_stats()
    cov, denom = report.coverage()

    def _verdict_dict(v: SiteVerdict) -> dict[str, Any]:
        return {
            "status": v.status,
            "labels": list(v.labels),
            "confidence": v.confidence,
            "anchor": _record_to_dict(v.baseline) if v.baseline else None,
            "old_fingerprint": v.baseline.fingerprint if v.baseline else None,
            "new_fingerprint": v.live.fingerprint if v.live else None,
            "bound_cases": list(v.bound_cases),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "repo": {
            "root": report.repo_root,
            "git_sha": report.git.get("sha"),
            "dirty": bool(report.git.get("dirty")),
        },
        "baseline": None if report.baseline is None else {
            "path": report.baseline_path,
            "git_sha": report.baseline.git.get("sha"),
            "created_at": report.baseline.created_at,
            "scanner_version": report.baseline.scanner_version,
        },
        "determinacy": report.determinacy,
        "summary": {
            "changed": c["changed"],
            "formatting_only": c["formatting_only"],
            "removed": c["removed"],
            "added": c["added"],
            "moved": c["moved"],
            "unknown_dynamic": c["unknown"],
            "cases_total": s["total"],
            "cases_bound": s["bound"],
            "cases_unstamped": s["unstamped"],
            "unreadable": s["unreadable"],
            "coverage_covered": cov,
            "coverage_static_sites": denom,
        },
        "sites": [_verdict_dict(v) for v in report.verdicts],
        "added": [_record_to_dict(r) for r in report.added],
        # OBSERVED tier — runtime-sourced, recordings-vs-recordings only.
        # Always k-of-N: a site is never "fresh" because one rendering matched.
        "observed": [
            {
                "source": "runtime",
                "anchor": ov.anchor,
                "baseline_renderings": ov.baseline_renderings,
                "matched": ov.matched,
                "current_renderings": ov.current_renderings,
                "new_renderings": ov.new_renderings,
                "case_uids": ov.case_uids,
                "observations": ov.observations,
                "has_current_recordings": ov.has_current,
                "caveat": (
                    "runtime recordings prove only the renderings observed, "
                    "not all renderings; not comparable to the static scan"
                ),
            }
            for ov in report.observed
        ],
        "unmerged_runtime_sites": report.unmerged_runtime,
        "trust_tiers": list(TRUST_TIERS),
        "cases": [
            {
                "file": cv.file, "index": cv.index, "case_uid": cv.case_uid,
                "status": cv.status, "rollup": cv.rollup, "notes": cv.notes,
            }
            for cv in report.case_verdicts
        ],
        "warnings": [w for w in (
            report.baseline_warning, report.ignore_warning,
            report.recordings_warning,
            "no baseline found" if report.no_baseline else None,
            "scanner version mismatch — rescan recommended"
            if report.scanner_mismatch else None,
        ) if w],
        "blind_spots": list(BLIND_SPOTS),
        "exit_code": report.exit_code,
    }
    return json.dumps(payload, indent=2)


def render_markdown(report: StalenessReport) -> str:
    """PR-summary-ready markdown (GITHUB_STEP_SUMMARY-friendly)."""
    c = report.counts()
    lines = ["## Prompt staleness", ""]
    lines.append(f"_{_determinacy_line(report)}_")
    lines.append("")
    if report.no_baseline:
        lines.append(
            "No baseline found — run `multivon-eval staleness baseline .` "
            "(bootstrap writes one automatically)."
        )
        lines.append("")
    else:
        if report.baseline is not None:
            b = report.baseline
            lines.append(
                f"Baseline `{report.baseline_path}` ({_sha_display(b.git)}, "
                f"{_age_display(b.created_at)}, scanner v{b.scanner_version})."
            )
            lines.append("")
        summary = (
            f"**{c['changed']} changed** ({c['formatting_only']} formatting-only) · "
            f"{c['removed']} removed · {c['added']} added · "
            f"{c['unknown']} unknown (dynamic) · {c['moved']} moved"
        )
        lines.append(summary)
        lines.append("")
        for v in report.verdicts:
            if v.status == "unchanged" and "moved" not in v.labels:
                continue
            rec = v.live or v.baseline
            assert rec is not None
            label = f" _[{', '.join(v.labels)}]_" if v.labels else ""
            lines.append(f"- **{v.status}** `{_site_line(rec)}`{label}")
            if v.status == "removed":
                lines.append(f"  - note: {_REMOVED_CAVEAT}")
        for rec in report.added:
            lines.append(
                f"- **added** `{_site_line(rec)}` — no cases reference this prompt"
            )
        for ov in report.observed:
            a = ov.anchor
            site = (f"{a.get('file_path', '?')} {a.get('sdk', '?')}."
                    f"{a.get('role', '?')} in {a.get('qualname', '?')}")
            if ov.has_current:
                detail = (f"current recordings matched {ov.matched} of "
                          f"{ov.baseline_renderings} previously observed renderings")
            else:
                detail = (f"{ov.baseline_renderings} previously observed "
                          f"rendering(s); no current recordings to compare")
            lines.append(
                f"- **observed (runtime)** `{site}` — {detail} "
                f"_(recordings-vs-recordings; proves observed renderings only)_"
            )
        lines.append("")
    for line in _footer_lines(report):
        lines.append(f"_{line}_" if line else line)
    # No "_exit N_" footer here: markdown's home is CI step summaries,
    # where a raw exit code reads as leaked debug output. The exit code
    # stays in the text renderer and the JSON payload.
    return "\n".join(lines) + "\n"


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_BASELINE_NAME",
    "DEFAULT_RECORDINGS_NAME",
    "DEFAULT_STALENESS_IGNORES",
    "BLIND_SPOTS",
    "TRUST_TIERS",
    "ObservedVerdict",
    "Baseline",
    "BaselineError",
    "SiteVerdict",
    "CaseVerdict",
    "StalenessReport",
    "write_baseline",
    "load_baseline",
    "scan_repo",
    "match_records",
    "match_target",
    "evaluate_cases",
    "discover_case_files",
    "build_staleness_report",
    "render_text",
    "render_json",
    "render_markdown",
]
