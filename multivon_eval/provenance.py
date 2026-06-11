"""Case provenance — the ``metadata["_provenance"]`` layer (write side).

Records *when and against what repo state a case was authored*, and
optionally which prompt call sites it is bound to. Lives inside the
existing free-form ``EvalCase.metadata`` dict under a library-reserved
underscore key — no new EvalCase field, and stamping never perturbs
``suite.lock`` because ``lockfile._cases_hash`` excludes metadata by
design (the load-bearing storage assumption, pinned by a regression test).

Three stamping paths:
  - bootstrap: automatic, ``authored_by="bootstrap"``, ``targets=[]``
    (honest "authored against this repo state" — bindings are NEVER
    fabricated).
  - hand-written JSONL: ``multivon-eval staleness stamp`` → ``stamp_jsonl``
    here. Raw-line-preserving: each line goes ``json.loads`` → inject →
    ``json.dumps`` of the SAME dict. It never round-trips through
    ``load_jsonl``/``EvalCase`` (that would drop ``expected_tool_calls``).
  - Python-inline cases: the ``stamp(sites=[...])`` helper builds the
    metadata dict at authoring time.

Versioning contract: v1 readers ignore unknown keys; a ``schema_version``
above the reader's max makes the case ``unreadable`` (counted, never
fatal) — a newer teammate's stamp must never break an older teammate's CI.

CSV-loaded cases cannot carry provenance (``load_csv`` reads no metadata)
— a documented, permanent limitation.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .attribution import PromptRecord, scan

PROVENANCE_KEY = "_provenance"
PROVENANCE_SCHEMA_VERSION = 1

_ROLES = ("system", "user", "assistant", "developer")


class ProvenanceError(Exception):
    """Base error for provenance stamping problems."""


class AmbiguousSiteError(ProvenanceError):
    """A --site spec matched zero or multiple call sites; never guess."""

    def __init__(self, message: str, candidates: list[PromptRecord] | None = None):
        super().__init__(message)
        self.candidates = candidates or []


class NonConformingProvenanceError(ProvenanceError):
    """Existing _provenance value is malformed or from a newer schema;
    refusing to overwrite without force."""


# ── small shared helpers ──────────────────────────────────────────────


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_info(repo: str | Path = ".") -> dict[str, Any]:
    """Display-only git facts: {"sha": short-sha-or-None, "dirty": bool}.

    NEVER a matching input — SHAs don't survive rebase/squash/shallow
    clones. ``dirty=True`` is rendered as a "(dirty tree — SHA approximate)"
    caveat downstream.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, timeout=5,
        )
        if proc.returncode != 0:
            return {"sha": None, "dirty": False}
        sha = proc.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo), capture_output=True, text=True, timeout=10,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else False
        return {"sha": sha, "dirty": dirty}
    except (OSError, subprocess.TimeoutExpired):
        return {"sha": None, "dirty": False}


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write via temp file + os.replace — never a half-written artifact."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── targets + provenance dicts ────────────────────────────────────────


def anchor_for_record(record: PromptRecord) -> dict[str, Any]:
    """Structural fallback anchor. ``line`` is a hint + tie-breaker only —
    matching never keys on it."""
    return {
        "file_path": record.file_path,
        "qualname": record.qualname,
        "sdk": record.sdk,
        "call_site": record.call_site,
        "role": record.role,
        "role_position": record.role_position,
        "line": record.line,
    }


def target_from_record(record: PromptRecord, source: str = "scan") -> dict[str, Any]:
    """Build one provenance target from a live PromptRecord.

    ``bound`` is always "manual" in v1 — auto-binding is rejected
    (confidently-wrong links poison every downstream verdict).
    ``source="external"`` marks prompts the scanner cannot see (YAML / hub /
    Responses API); those are reported UNVERIFIABLE, never orphaned.
    """
    return {
        "fingerprint": record.fingerprint,
        "loose_fingerprint": record.loose_fingerprint,
        "is_dynamic": record.is_dynamic,
        "anchor": anchor_for_record(record),
        "bound": "manual",
        "source": source,
    }


def new_provenance(
    *,
    authored_by: str,
    git: dict[str, Any],
    targets: list[dict[str, Any]],
    evidence: dict[str, Any] | None = None,
    case_uid: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    ts = now or _utc_now()
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "case_uid": case_uid or uuid.uuid4().hex,
        "authored_at": ts,
        "stamped_at": ts,
        "authored_by": authored_by,
        "git": dict(git),
        "evidence": evidence,
        "targets": [dict(t) for t in targets],
    }


def read_provenance(metadata: Any) -> tuple[str, Optional[dict[str, Any]]]:
    """Classify a case's metadata. Returns (status, provenance-or-None).

    status: "unstamped" | "ok" | "unreadable" | "unreadable_newer".
    Malformed values are *unreadable*, never a crash; a future
    schema_version is "stamped by a newer multivon-eval" — counted,
    exit-code unaffected.
    """
    if not isinstance(metadata, dict) or PROVENANCE_KEY not in metadata:
        return ("unstamped", None)
    prov = metadata[PROVENANCE_KEY]
    if not isinstance(prov, dict):
        return ("unreadable", None)
    sv = prov.get("schema_version")
    # bool is a subclass of int — `schema_version: true` is malformed,
    # not "version 1".
    if not isinstance(sv, int) or isinstance(sv, bool):
        return ("unreadable", None)
    if sv > PROVENANCE_SCHEMA_VERSION:
        return ("unreadable_newer", prov)
    targets = prov.get("targets")
    if targets is not None and not isinstance(targets, list):
        return ("unreadable", None)
    return ("ok", prov)


def stamp_metadata_inplace(
    metadata: dict[str, Any],
    *,
    authored_by: str,
    git: dict[str, Any],
    targets: list[dict[str, Any]] | tuple = (),
    evidence: dict[str, Any] | None = None,
    now: str | None = None,
) -> None:
    """Inject _provenance into a metadata dict (used by bootstrap on
    in-memory cases before the existing ``_case_to_jsonl`` write path)."""
    metadata[PROVENANCE_KEY] = new_provenance(
        authored_by=authored_by, git=git, targets=list(targets),
        evidence=evidence, now=now,
    )


# ── site-spec parsing + resolution ────────────────────────────────────


def _parse_role_position(pos: str, spec: str) -> int:
    """Parse the '#POS' suffix of a --site spec — a clean error, never a
    bare int() traceback (the CLI maps ProvenanceError to exit 2)."""
    try:
        return int(pos)
    except ValueError:
        raise AmbiguousSiteError(
            f"--site {spec!r}: position {pos!r} after '#' must be an integer "
            f"(e.g. 'app.py.user#0')"
        ) from None


def parse_site_spec(spec: str) -> dict[str, Any]:
    """Parse 'FILE[::QUALNAME][.ROLE[#POS]]' into its parts.

    ROLE must be one of system/user/assistant/developer for the trailing
    segment to be treated as a role (so 'extractors/invoice.py' parses as
    a bare file path).
    """
    qualname: str | None = None
    role: str | None = None
    role_position: int | None = None

    if "::" in spec:
        file_part, rest = spec.split("::", 1)
        # rest = QUALNAME[.ROLE[#POS]]
        head, sep, tail = rest.rpartition(".")
        candidate = tail
        if "#" in candidate:
            candidate, _, pos = candidate.partition("#")
        else:
            pos = None
        if sep and candidate in _ROLES:
            qualname = head
            role = candidate
            if pos is not None:
                role_position = _parse_role_position(pos, spec)
        elif rest.partition("#")[0] in _ROLES:
            # bare ROLE[#POS] after '::' (module-level call)
            candidate, _, pos = rest.partition("#")
            role = candidate
            role_position = _parse_role_position(pos, spec) if pos else None
        else:
            qualname = rest or None
    else:
        file_part = spec
        head, sep, tail = spec.rpartition(".")
        candidate, _, pos = tail.partition("#")
        if sep and candidate in _ROLES:
            file_part = head
            role = candidate
            if pos:
                role_position = _parse_role_position(pos, spec)

    return {
        "file_path": file_part,
        "qualname": qualname,
        "role": role,
        "role_position": role_position,
    }


def resolve_site_spec(records: list[PromptRecord], spec: str) -> PromptRecord:
    """Resolve a --site spec against a live scan to exactly one record.

    Zero or multiple matches → AmbiguousSiteError listing candidates —
    the resolver never guesses. A duplicated prompt fingerprint additionally
    requires an explicit qualname anchor in the spec.
    """
    parsed = parse_site_spec(spec)
    fp = parsed["file_path"]
    cands = [
        r for r in records
        if r.file_path == fp or r.file_path.endswith("/" + fp)
    ]
    if parsed["qualname"] is not None:
        cands = [r for r in cands if r.qualname == parsed["qualname"]]
    if parsed["role"] is not None:
        cands = [r for r in cands if r.role == parsed["role"]]
    if parsed["role_position"] is not None:
        cands = [r for r in cands if r.role_position == parsed["role_position"]]

    if not cands:
        raise AmbiguousSiteError(
            f"--site {spec!r} matched no call site in the live scan. "
            f"Run `multivon-eval attribution scan .` to list sites."
        )
    if len(cands) > 1:
        listing = "\n  ".join(
            f"{r.call_site_id}  qualname={r.qualname}" for r in cands
        )
        raise AmbiguousSiteError(
            f"--site {spec!r} is ambiguous ({len(cands)} candidates) — "
            f"add ::QUALNAME and/or .ROLE#POS:\n  {listing}",
            candidates=cands,
        )
    resolved = cands[0]
    if not resolved.is_dynamic and parsed["qualname"] is None:
        dups = [
            r for r in records
            if not r.is_dynamic and r.fingerprint == resolved.fingerprint
        ]
        if len(dups) > 1:
            listing = "\n  ".join(
                f"{r.call_site_id}  qualname={r.qualname}" for r in dups
            )
            raise AmbiguousSiteError(
                f"--site {spec!r}: this prompt text is duplicated at "
                f"{len(dups)} call sites — refusing to stamp without an "
                f"explicit FILE::QUALNAME anchor:\n  {listing}",
                candidates=dups,
            )
    return resolved


# ── JSONL stamping (raw-line-preserving) ──────────────────────────────


def _target_key(t: dict[str, Any]) -> tuple:
    a = t.get("anchor") or {}
    return (
        t.get("fingerprint"),
        a.get("file_path"), a.get("qualname"), a.get("sdk"),
        a.get("call_site"), a.get("role"), a.get("role_position"),
    )


def merge_targets(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge new targets into existing — same (fingerprint, anchor) replaces,
    otherwise appends. One case may bind several prompts."""
    out = [dict(t) for t in existing]
    keys = {_target_key(t): i for i, t in enumerate(out)}
    for t in new:
        k = _target_key(t)
        if k in keys:
            out[keys[k]] = dict(t)
        else:
            out.append(dict(t))
            keys[k] = len(out) - 1
    return out


@dataclass
class StampResult:
    path: Path
    selected: int
    updated: int
    unchanged: int
    dry_run: bool


def stamp_jsonl(
    path: str | Path,
    targets: list[dict[str, Any]],
    *,
    authored_by: str = "human",
    evidence: dict[str, Any] | None = None,
    indices: list[int] | None = None,
    tag: str | None = None,
    select_all: bool = False,
    repo: str | Path = ".",
    force: bool = False,
    dry_run: bool = False,
    _git: dict[str, Any] | None = None,
    _now: str | None = None,
) -> StampResult:
    """Stamp selected cases in a JSONL file with provenance targets.

    Raw-line-preserving: ``json.loads`` each selected line → inject
    ``metadata._provenance`` → ``json.dumps`` the SAME dict. Unselected and
    already-identical lines are copied byte-for-byte, so idempotent restamps
    produce a byte-identical file (``authored_at`` and ``case_uid`` are
    preserved across restamps; ``stamped_at`` only moves when targets or
    evidence actually change).
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    git = _git if _git is not None else git_info(repo)

    out_lines: list[str] = []
    case_idx = -1
    selected = updated = unchanged = 0

    def _loads(stripped: str, line_no: int) -> Any:
        # A broken line raises ProvenanceError (CLI → clean exit 2) with
        # file:line of the bad input — never a bare JSONDecodeError traceback.
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ProvenanceError(
                f"{path}:{line_no}: line is not valid JSON ({exc})"
            ) from exc
        if not isinstance(data, dict):
            raise ProvenanceError(
                f"{path}:{line_no}: expected a JSON object, "
                f"got {type(data).__name__}"
            )
        return data

    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        case_idx += 1

        data: dict[str, Any] | None = None
        sel = select_all or (indices is not None and case_idx in indices)
        if not sel and tag is not None:
            data = _loads(stripped, line_no)
            sel = tag in (data.get("tags") or [])
        if not sel:
            out_lines.append(line)
            continue
        selected += 1
        if data is None:
            data = _loads(stripped, line_no)

        metadata = data.get("metadata")
        if metadata is None:
            metadata = {}
            data["metadata"] = metadata
        if not isinstance(metadata, dict):
            raise ProvenanceError(
                f"{path} case #{case_idx}: metadata is not a JSON object"
            )

        status, existing = read_provenance(metadata)
        if status == "ok":
            assert existing is not None
            old_targets = existing.get("targets") or []
            merged = merge_targets(old_targets, targets)
            if merged == old_targets and (
                evidence is None or existing.get("evidence") == evidence
            ):
                unchanged += 1
                out_lines.append(line)  # byte-identical — untouched
                continue
            prov = dict(existing)
            prov["targets"] = merged
            prov["stamped_at"] = _now or _utc_now()
            prov["git"] = dict(git)
            if evidence is not None:
                prov["evidence"] = evidence
        elif status in ("unreadable", "unreadable_newer"):
            if not force:
                kind = (
                    "stamped by a newer multivon-eval"
                    if status == "unreadable_newer" else "malformed"
                )
                raise NonConformingProvenanceError(
                    f"{path} case #{case_idx}: existing _provenance is {kind} "
                    f"— refusing to overwrite without --force"
                )
            prov = new_provenance(
                authored_by=authored_by, git=git,
                targets=merge_targets([], targets),
                evidence=evidence, now=_now,
            )
        else:  # unstamped
            prov = new_provenance(
                authored_by=authored_by, git=git,
                targets=merge_targets([], targets),
                evidence=evidence, now=_now,
            )

        metadata[PROVENANCE_KEY] = prov
        updated += 1
        ending = "\n" if line.endswith("\n") else ""
        out_lines.append(json.dumps(data, ensure_ascii=False) + ending)

    if updated and not dry_run:
        atomic_write_text(path, "".join(out_lines))
    return StampResult(
        path=path, selected=selected, updated=updated,
        unchanged=unchanged, dry_run=dry_run,
    )


# ── Python-inline helper ──────────────────────────────────────────────


def stamp(
    sites: list[str] | None = None,
    repo: str | Path = ".",
    *,
    authored_by: str = "human",
    evidence: dict[str, Any] | None = None,
    case_uid: str | None = None,
) -> dict[str, Any]:
    """Build a metadata dict carrying _provenance for Python-inline cases.

    Usage in an eval_suite.py::

        from multivon_eval.provenance import stamp
        case = EvalCase(
            input="...",
            metadata=stamp(sites=["extractors/invoice.py::Extractor.extract.system"]),
        )

    ``sites`` are resolved against a live scan of ``repo`` — ambiguity is
    an error, never a guess. ``sites=[]`` (or None) stamps repo-state
    provenance only (targets=[]), which is honest and unbound.
    """
    targets: list[dict[str, Any]] = []
    if sites:
        records = scan(str(repo))
        for spec in sites:
            targets.append(target_from_record(resolve_site_spec(records, spec)))
    return {
        PROVENANCE_KEY: new_provenance(
            authored_by=authored_by,
            git=git_info(repo),
            targets=targets,
            evidence=evidence,
            case_uid=case_uid,
        )
    }


__all__ = [
    "PROVENANCE_KEY",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceError",
    "AmbiguousSiteError",
    "NonConformingProvenanceError",
    "StampResult",
    "git_info",
    "atomic_write_text",
    "anchor_for_record",
    "target_from_record",
    "new_provenance",
    "read_provenance",
    "stamp_metadata_inplace",
    "parse_site_spec",
    "resolve_site_spec",
    "merge_targets",
    "stamp_jsonl",
    "stamp",
]
