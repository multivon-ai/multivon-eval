"""Runtime prompt recorder (v1) — opt-in capture of rendered prompts.

The determinacy gate (issue #9 / epic #4) measured scanner v3 against five
real repos: 20.9% of call sites are statically resolvable. The other ~79%
build prompts dynamically — statically unbridgeable by construction. This
module is the designed answer: during an eval run it intercepts outgoing
SDK calls and records the *rendered* prompt text per call site,
fingerprinted with the SAME ``fingerprint_text`` the static scanner uses.

Design constraints (carried over from the issue-#9 review, non-negotiable):
  - Opt-in only. Importing multivon_eval performs NO patching; zero
    overhead when off. Enabled via the ``--record-prompts`` pytest flag or
    the :func:`record_prompts` context manager.
  - Recordings stay local (``prompt_recordings.jsonl``); no telemetry.
    Fingerprints only by default — rendered TEXT is stored only behind an
    explicit ``record_text=True`` / ``--record-text``.
  - Runtime fingerprints are labeled ``source: "runtime"`` in the baseline,
    never silently mixed with static ones. A recording proves the
    renderings OBSERVED, not all renderings — variable renderings per site
    are a fingerprint SET, and reports speak in "matched k of N previously
    observed renderings", never "fresh".
  - Case→site bindings come from OBSERVATION (a contextvar carrying the
    active case_uid), and are still written only on explicit ``--apply`` —
    observation removes the fabrication objection; human confirmation stays.

Patch mechanics: method-level wrapping of exactly the three SDK surfaces
the static scanner knows — anthropic ``Messages.create``, openai
``chat.completions.create``, ``litellm.completion``/``acompletion``. Save
original, wrap, restore on exit. No HTTP cassettes, no import-time wrapper
clients. Missing SDKs are skipped silently (litellm is optional).

Capture scope (v1, honest): string ``system=`` kwargs and string
``content`` entries in ``messages=`` lists. Content-block lists (vision,
tool results) are not captured. The caller anchor is the first stack frame
whose file lives under the repo root (build dirs excluded); ``line`` is an
advisory hint, never a matching input.
"""
from __future__ import annotations

import contextvars
import functools
import importlib
import json
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .attribution.ast_extractor import DEFAULT_IGNORE_DIRS
from .attribution.fingerprint import fingerprint_text, loose_fingerprint_text
from .provenance import PROVENANCE_KEY, atomic_write_text, stamp_jsonl

RECORDER_VERSION = 1
RECORDINGS_SCHEMA_VERSION = 1
DEFAULT_RECORDINGS_NAME = "prompt_recordings.jsonl"
RUNTIME_SOURCE = "runtime"

# The three labeled trust tiers. Rendered verbatim in staleness reports —
# never collapsed into one another.
TRUST_TIERS = (
    "static scan proves prompt text",
    "runtime recordings prove only the renderings observed, not all renderings",
    "template/external prompts deferred (unverifiable)",
)

_THIS_FILE = str(Path(__file__).resolve())


class RecorderError(Exception):
    """Recorder storage/merge problem (unreadable baseline or recordings)."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── case binding (contextvar) ─────────────────────────────────────────

_active_case: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "multivon_eval_active_case", default=None,
)

# Module-level active recorder. One at a time, by design — nested
# recorders would double-wrap and double-count.
_ACTIVE: Optional["PromptRecorder"] = None


def recording_active() -> bool:
    """True iff a PromptRecorder is currently patched in."""
    return _ACTIVE is not None


def set_active_case(case_uid: str) -> contextvars.Token:
    """Mark ``case_uid`` as the case currently exercising the model.

    Recordings made while a case is active carry its uid — binding by
    observation, not guessing. Returns a token for
    :func:`reset_active_case`. Cheap whether or not recording is on.
    """
    return _active_case.set(case_uid)


def reset_active_case(token: contextvars.Token) -> None:
    _active_case.reset(token)


def get_active_case() -> Optional[str]:
    return _active_case.get()


def bind_case(metadata: Any) -> Optional[contextvars.Token]:
    """Set the active case from ``metadata["_provenance"]["case_uid"]``.

    Returns a reset token, or None when recording is off / the case
    carries no uid. The ``_ACTIVE is None`` early-out keeps the per-case
    cost of the suite-runner hook at one attribute check when off.
    """
    if _ACTIVE is None:
        return None
    if not isinstance(metadata, dict):
        return None
    prov = metadata.get(PROVENANCE_KEY)
    uid = prov.get("case_uid") if isinstance(prov, dict) else None
    if not isinstance(uid, str) or not uid:
        return None
    return _active_case.set(uid)


def unbind_case(token: Optional[contextvars.Token]) -> None:
    if token is not None:
        _active_case.reset(token)


# ── recordings storage (prompt_recordings.jsonl) ──────────────────────


def _anchor_key(anchor: dict[str, Any]) -> tuple:
    """Merge identity for an anchor — ``line`` is advisory, never a key."""
    return (
        anchor.get("file_path"), anchor.get("qualname"), anchor.get("sdk"),
        anchor.get("call_site"), anchor.get("role"),
    )


def _rec_key(rec: dict[str, Any]) -> tuple:
    return (*_anchor_key(rec.get("anchor") or {}), rec.get("fingerprint"))


def load_recordings(path: str | Path) -> list[dict[str, Any]]:
    """Read a recordings JSONL. Malformed lines and future-schema lines
    are skipped, never a crash (same warn-and-skip posture as baselines)."""
    out: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RecorderError(f"recordings {path} unreadable: {exc}") from exc
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("anchor"), dict):
            continue
        sv = data.get("schema_version")
        if isinstance(sv, int) and sv > RECORDINGS_SCHEMA_VERSION:
            continue  # written by a newer multivon-eval — skipped, not fatal
        if not data.get("fingerprint"):
            continue
        out.append(data)
    return out


def merge_recording_dicts(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge recordings by (anchor, role, fingerprint) key.

    Counts add, case_uids union, first_seen takes the min, last_seen the
    max, the advisory line follows the newest observation. Merging an
    empty list is the identity (append-safe rewrites stay idempotent).
    """
    merged: dict[tuple, dict[str, Any]] = {}
    for rec in existing:
        merged[_rec_key(rec)] = dict(rec)
    for rec in new:
        k = _rec_key(rec)
        old = merged.get(k)
        if old is None:
            merged[k] = dict(rec)
            continue
        out = dict(old)
        out["count"] = int(old.get("count") or 0) + int(rec.get("count") or 0)
        out["case_uids"] = sorted(
            set(old.get("case_uids") or []) | set(rec.get("case_uids") or [])
        )
        out["first_seen"] = min(
            filter(None, (old.get("first_seen"), rec.get("first_seen"))),
            default=None,
        )
        out["last_seen"] = max(
            filter(None, (old.get("last_seen"), rec.get("last_seen"))),
            default=None,
        )
        anchor = dict(out.get("anchor") or {})
        new_line = (rec.get("anchor") or {}).get("line")
        if new_line:
            anchor["line"] = new_line
        out["anchor"] = anchor
        if rec.get("text") is not None:
            out["text"] = rec["text"]
        merged[k] = out
    return sorted(merged.values(), key=_rec_key)


def write_recordings(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Atomic rewrite (temp file + os.replace) — never a half-written file."""
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


# ── the recorder ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _PatchSpec:
    sdk: str
    module: str
    cls: Optional[str]
    attr: str
    call_site: str
    is_async: bool = False


# Exactly the three SDK surfaces the static scanner knows. Method-level
# wrapping on the class/module attribute — by call time, **kwargs unpacks
# are real kwargs, which is the whole point.
_PATCH_SPECS = (
    _PatchSpec("anthropic", "anthropic.resources.messages", "Messages",
               "create", "messages.create"),
    _PatchSpec("openai", "openai.resources.chat.completions", "Completions",
               "create", "chat.completions.create"),
    _PatchSpec("litellm", "litellm", None, "completion", "completion"),
    _PatchSpec("litellm", "litellm", None, "acompletion", "acompletion",
               is_async=True),
)


class PromptRecorder:
    """Opt-in runtime prompt recorder. Use as a context manager::

        from multivon_eval.recorder import record_prompts
        with record_prompts(repo_root="."):
            run_my_evals()
        # prompt_recordings.jsonl now holds fingerprints per call site

    ``start()`` patches the SDK surfaces; ``stop()`` restores them
    byte-identically and flushes recordings to disk. A recording failure
    never breaks the user's SDK call (counted on ``record_errors``).
    """

    def __init__(
        self,
        repo_root: str | Path = ".",
        out: str | Path | None = None,
        *,
        record_text: bool = False,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.out_path = Path(out) if out else self.repo_root / DEFAULT_RECORDINGS_NAME
        self.record_text = record_text
        self.patched_sdks: list[str] = []
        self.record_errors = 0
        self._records: dict[tuple, dict[str, Any]] = {}
        self._lock = threading.Lock()
        # (target, attr, original, attr_was_own) — restore plan.
        self._patched: list[tuple[Any, str, Any, bool]] = []
        self._frame_cache: dict[str, Optional[str]] = {}

    # — lifecycle —

    def start(self) -> "PromptRecorder":
        global _ACTIVE
        if _ACTIVE is not None:
            raise RuntimeError(
                "a PromptRecorder is already active — recorders do not nest"
            )
        try:
            for spec in _PATCH_SPECS:
                try:
                    module = importlib.import_module(spec.module)
                except ImportError:
                    continue  # optional SDK not installed — skipped
                target = getattr(module, spec.cls) if spec.cls else module
                original = getattr(target, spec.attr, None)
                if original is None or getattr(original, "__multivon_recorder__", False):
                    continue
                was_own = spec.attr in vars(target)
                setattr(target, spec.attr, self._make_wrapper(spec, original))
                self._patched.append((target, spec.attr, original, was_own))
                self.patched_sdks.append(f"{spec.sdk}.{spec.call_site}")
        except BaseException:
            self._restore()
            raise
        _ACTIVE = self
        return self

    def stop(self) -> None:
        global _ACTIVE
        self._restore()
        if _ACTIVE is self:
            _ACTIVE = None
        self.flush()

    def _restore(self) -> None:
        """Put every patched attribute back exactly as found. An attribute
        that was INHERITED (not in the target's own __dict__) is restored
        by delattr so the target's __dict__ ends byte-identical."""
        for target, attr, original, was_own in reversed(self._patched):
            if was_own:
                setattr(target, attr, original)
            else:
                try:
                    delattr(target, attr)
                except AttributeError:
                    pass
        self._patched.clear()

    def __enter__(self) -> "PromptRecorder":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    def flush(self) -> Path:
        """Merge in-memory recordings into the JSONL on disk (append-safe:
        load existing, merge duplicate keys, atomic rewrite) and clear the
        in-memory buffer. A flush with nothing new touches nothing."""
        with self._lock:
            new = list(self._records.values())
            self._records.clear()
        if not new:
            return self.out_path
        existing: list[dict[str, Any]] = []
        if self.out_path.exists():
            existing = load_recordings(self.out_path)
        write_recordings(self.out_path, merge_recording_dicts(existing, new))
        return self.out_path

    # — interception —

    def _make_wrapper(self, spec: _PatchSpec, original: Any) -> Any:
        if spec.is_async:
            @functools.wraps(original)
            async def awrapper(*args, **kwargs):
                try:
                    self._record_call(spec.sdk, spec.call_site, kwargs)
                except Exception:
                    self.record_errors += 1
                return await original(*args, **kwargs)
            awrapper.__multivon_recorder__ = True
            return awrapper

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            try:
                self._record_call(spec.sdk, spec.call_site, kwargs)
            except Exception:
                self.record_errors += 1
            return original(*args, **kwargs)
        wrapper.__multivon_recorder__ = True
        return wrapper

    def _caller_anchor(self) -> Optional[dict[str, Any]]:
        """First stack frame whose file lives under the repo root (build
        dirs excluded). None when the call originated outside the repo —
        v1 records anchor only to your own code, honestly skipping the rest."""
        frame = sys._getframe(1)
        while frame is not None:
            filename = frame.f_code.co_filename
            if filename != _THIS_FILE:
                rel = self._rel_path(filename)
                if rel is not None:
                    code = frame.f_code
                    return {
                        "file_path": rel,
                        "qualname": getattr(code, "co_qualname", code.co_name),
                        "line": frame.f_lineno,  # advisory hint only
                    }
            frame = frame.f_back
        return None

    def _rel_path(self, filename: str) -> Optional[str]:
        cached = self._frame_cache.get(filename)
        if cached is not None or filename in self._frame_cache:
            return cached
        rel: Optional[str] = None
        if not filename.startswith("<"):
            try:
                rel_p = Path(filename).resolve().relative_to(self.repo_root)
            except (ValueError, OSError):
                rel_p = None
            if rel_p is not None and not any(
                part in DEFAULT_IGNORE_DIRS for part in rel_p.parts
            ):
                rel = str(rel_p)
        self._frame_cache[filename] = rel
        return rel

    def _record_call(self, sdk: str, call_site: str, kwargs: dict) -> None:
        anchor = self._caller_anchor()
        if anchor is None:
            return
        entries: list[tuple[str, str]] = []
        system = kwargs.get("system")
        if isinstance(system, str):
            entries.append(("system", system))
        messages = kwargs.get("messages")
        if isinstance(messages, (list, tuple)):
            for m in messages:
                if isinstance(m, dict):
                    role, content = m.get("role"), m.get("content")
                    if isinstance(role, str) and isinstance(content, str):
                        entries.append((role, content))
        if not entries:
            return
        now = _utc_now()
        case_uid = _active_case.get()
        with self._lock:
            for role, text in entries:
                fp = fingerprint_text(text)
                key = (anchor["file_path"], anchor["qualname"], sdk,
                       call_site, role, fp)
                rec = self._records.get(key)
                if rec is None:
                    rec = {
                        "schema_version": RECORDINGS_SCHEMA_VERSION,
                        "anchor": {
                            "file_path": anchor["file_path"],
                            "qualname": anchor["qualname"],
                            "sdk": sdk,
                            "call_site": call_site,
                            "role": role,
                            "line": anchor["line"],
                        },
                        "fingerprint": fp,
                        "loose_fingerprint": loose_fingerprint_text(text),
                        "case_uids": [],
                        "count": 0,
                        "first_seen": now,
                        "last_seen": now,
                        "recorder_version": RECORDER_VERSION,
                    }
                    if self.record_text:
                        rec["text"] = text
                    self._records[key] = rec
                rec["count"] += 1
                rec["last_seen"] = now
                rec["anchor"]["line"] = anchor["line"]
                if case_uid and case_uid not in rec["case_uids"]:
                    rec["case_uids"] = sorted({*rec["case_uids"], case_uid})

    def snapshot(self) -> list[dict[str, Any]]:
        """Copy of the unflushed in-memory recordings (for summaries)."""
        with self._lock:
            return [dict(r) for r in self._records.values()]


def record_prompts(
    repo_root: str | Path = ".",
    out: str | Path | None = None,
    *,
    record_text: bool = False,
) -> PromptRecorder:
    """Context-manager entry point for non-pytest use. See PromptRecorder."""
    return PromptRecorder(repo_root, out, record_text=record_text)


# ── baseline merge (source: "runtime", fingerprint SETS) ──────────────


def runtime_records_from_recordings(
    recordings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Group recordings by anchor into baseline-ready runtime records.

    One record per call-site anchor; ``fingerprints`` is a SET (sorted
    list) — the variable renderings observed at that site. Labeled
    ``source: "runtime"``: a different trust tier from static records,
    never mixed with them.
    """
    grouped: dict[tuple, dict[str, Any]] = {}
    for rec in recordings:
        anchor = dict(rec.get("anchor") or {})
        key = _anchor_key(anchor)
        g = grouped.get(key)
        if g is None:
            g = {
                "source": RUNTIME_SOURCE,
                "anchor": anchor,
                "fingerprints": set(),
                "loose_fingerprints": set(),
                "case_uids": set(),
                "observations": 0,
                "first_seen": rec.get("first_seen"),
                "last_seen": rec.get("last_seen"),
                "recorder_version": rec.get("recorder_version", RECORDER_VERSION),
            }
            grouped[key] = g
        g["fingerprints"].add(rec["fingerprint"])
        if rec.get("loose_fingerprint"):
            g["loose_fingerprints"].add(rec["loose_fingerprint"])
        g["case_uids"] |= set(rec.get("case_uids") or [])
        g["observations"] += int(rec.get("count") or 0)
        g["first_seen"] = min(
            filter(None, (g["first_seen"], rec.get("first_seen"))), default=None)
        g["last_seen"] = max(
            filter(None, (g["last_seen"], rec.get("last_seen"))), default=None)
    out = []
    for g in grouped.values():
        g["fingerprints"] = sorted(g["fingerprints"])
        g["loose_fingerprints"] = sorted(g["loose_fingerprints"])
        g["case_uids"] = sorted(g["case_uids"])
        out.append(g)
    out.sort(key=lambda g: _anchor_key(g["anchor"]))
    return out


def merge_runtime_records(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Union per anchor. The recordings JSONL is the accumulating source
    of truth for counts, so ``observations`` takes the max — re-merging
    the same recordings file is idempotent, never double-counted."""
    merged: dict[tuple, dict[str, Any]] = {}
    for r in existing:
        merged[_anchor_key(r.get("anchor") or {})] = dict(r)
    for r in new:
        k = _anchor_key(r.get("anchor") or {})
        old = merged.get(k)
        if old is None:
            merged[k] = dict(r)
            continue
        out = dict(old)
        out["fingerprints"] = sorted(
            set(old.get("fingerprints") or []) | set(r.get("fingerprints") or []))
        out["loose_fingerprints"] = sorted(
            set(old.get("loose_fingerprints") or [])
            | set(r.get("loose_fingerprints") or []))
        out["case_uids"] = sorted(
            set(old.get("case_uids") or []) | set(r.get("case_uids") or []))
        out["observations"] = max(
            int(old.get("observations") or 0), int(r.get("observations") or 0))
        out["first_seen"] = min(
            filter(None, (old.get("first_seen"), r.get("first_seen"))), default=None)
        out["last_seen"] = max(
            filter(None, (old.get("last_seen"), r.get("last_seen"))), default=None)
        out["anchor"] = dict(r.get("anchor") or old.get("anchor") or {})
        merged[k] = out
    return sorted(merged.values(), key=lambda g: _anchor_key(g.get("anchor") or {}))


def merge_recordings_into_baseline(
    baseline_path: str | Path, recordings_path: str | Path
) -> tuple[int, int]:
    """Add runtime records to prompt_baseline.json under ``runtime_records``.

    The static ``records`` list is NEVER touched — different trust
    profiles stay in different keys, and v1 baseline readers ignore the
    unknown key. Returns (runtime sites merged, renderings merged).
    """
    bpath = Path(baseline_path)
    try:
        payload = json.loads(bpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecorderError(f"baseline {bpath} unreadable: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise RecorderError(f"baseline {bpath} is not a baseline payload")
    new = runtime_records_from_recordings(load_recordings(recordings_path))
    existing = payload.get("runtime_records") or []
    if not isinstance(existing, list):
        existing = []
    payload["runtime_records"] = merge_runtime_records(
        [r for r in existing if isinstance(r, dict)], new)
    atomic_write_text(bpath, json.dumps(payload, indent=2) + "\n")
    n_renderings = sum(len(r.get("fingerprints") or []) for r in new)
    return len(new), n_renderings


# ── observed-tier comparison (recordings vs recordings) ───────────────


@dataclass
class ObservedVerdict:
    """One runtime-sourced baseline site vs the CURRENT recordings.

    Always k-of-N: ``matched`` of ``baseline_renderings`` previously
    observed renderings were seen again. A site is never called fresh
    because one rendering matched. Runtime-only sites cannot be compared
    against a static scan — this is recordings-vs-recordings, and the
    renderers say so.
    """
    anchor: dict[str, Any]
    baseline_renderings: int
    matched: int
    current_renderings: int
    new_renderings: int
    case_uids: list[str] = field(default_factory=list)
    observations: int = 0
    has_current: bool = False


def compare_observed(
    runtime_records: list[dict[str, Any]],
    current_recordings: Optional[list[dict[str, Any]]],
) -> list[ObservedVerdict]:
    cur_by_anchor: dict[tuple, set[str]] = {}
    for rec in current_recordings or []:
        cur_by_anchor.setdefault(
            _anchor_key(rec.get("anchor") or {}), set()).add(rec["fingerprint"])
    out: list[ObservedVerdict] = []
    for rr in runtime_records:
        fps = set(rr.get("fingerprints") or [])
        key = _anchor_key(rr.get("anchor") or {})
        if current_recordings is None:
            out.append(ObservedVerdict(
                anchor=dict(rr.get("anchor") or {}),
                baseline_renderings=len(fps), matched=0,
                current_renderings=0, new_renderings=0,
                case_uids=list(rr.get("case_uids") or []),
                observations=int(rr.get("observations") or 0),
                has_current=False,
            ))
            continue
        cur = cur_by_anchor.get(key, set())
        out.append(ObservedVerdict(
            anchor=dict(rr.get("anchor") or {}),
            baseline_renderings=len(fps),
            matched=len(fps & cur),
            current_renderings=len(cur),
            new_renderings=len(cur - fps),
            case_uids=list(rr.get("case_uids") or []),
            observations=int(rr.get("observations") or 0),
            has_current=True,
        ))
    return out


def unmerged_runtime_sites(
    runtime_records: list[dict[str, Any]],
    current_recordings: Optional[list[dict[str, Any]]],
) -> int:
    """Anchors in the current recordings absent from the baseline's
    runtime tier — recorded but not yet blessed via --merge-recordings."""
    known = {_anchor_key(r.get("anchor") or {}) for r in runtime_records}
    seen = {_anchor_key(r.get("anchor") or {}) for r in current_recordings or []}
    return len(seen - known)


# ── binding proposals (observed case → site) ──────────────────────────


@dataclass(frozen=True)
class BindingProposal:
    case_uid: str
    anchor: dict[str, Any]
    fingerprint: str
    loose_fingerprint: str
    count: int


def propose_bindings(recordings: list[dict[str, Any]]) -> list[BindingProposal]:
    """Observed case→site bindings, one per (case_uid, anchor, fingerprint).

    Proposals only — writing them requires explicit --apply. Observation
    removes the fabrication objection; the human confirmation stays.
    """
    out: list[BindingProposal] = []
    for rec in recordings:
        for uid in rec.get("case_uids") or []:
            out.append(BindingProposal(
                case_uid=uid,
                anchor=dict(rec.get("anchor") or {}),
                fingerprint=rec["fingerprint"],
                loose_fingerprint=rec.get("loose_fingerprint", ""),
                count=int(rec.get("count") or 0),
            ))
    out.sort(key=lambda p: (p.case_uid, _anchor_key(p.anchor), p.fingerprint))
    return out


def runtime_target(p: BindingProposal) -> dict[str, Any]:
    """Provenance target for an observed binding. ``source: "runtime"``
    + ``bound: "observed"`` — staleness verifies it against recordings,
    never against the static scan."""
    return {
        "fingerprint": p.fingerprint,
        "loose_fingerprint": p.loose_fingerprint,
        "is_dynamic": False,
        "anchor": {**p.anchor, "role_position": -1},
        "bound": "observed",
        "source": RUNTIME_SOURCE,
    }


def apply_bindings(
    cases_path: str | Path,
    proposals: list[BindingProposal],
    *,
    repo: str | Path = ".",
    dry_run: bool = False,
) -> int:
    """Stamp observed bindings onto the JSONL cases whose
    ``_provenance.case_uid`` matches. Returns cases updated."""
    by_uid: dict[str, list[BindingProposal]] = {}
    for p in proposals:
        by_uid.setdefault(p.case_uid, []).append(p)

    # Map case_uid → line index (same skip-blank indexing as stamp_jsonl).
    path = Path(cases_path)
    idx_targets: list[tuple[int, list[dict[str, Any]]]] = []
    idx = -1
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        idx += 1
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        meta = data.get("metadata") if isinstance(data, dict) else None
        prov = meta.get(PROVENANCE_KEY) if isinstance(meta, dict) else None
        uid = prov.get("case_uid") if isinstance(prov, dict) else None
        if uid in by_uid:
            idx_targets.append((idx, [runtime_target(p) for p in by_uid[uid]]))

    updated = 0
    for case_idx, targets in idx_targets:
        result = stamp_jsonl(
            path, targets, indices=[case_idx],
            authored_by="recorder", repo=repo, dry_run=dry_run,
        )
        updated += result.updated
    return updated


__all__ = [
    "RECORDER_VERSION",
    "RECORDINGS_SCHEMA_VERSION",
    "DEFAULT_RECORDINGS_NAME",
    "RUNTIME_SOURCE",
    "TRUST_TIERS",
    "RecorderError",
    "PromptRecorder",
    "record_prompts",
    "recording_active",
    "set_active_case",
    "reset_active_case",
    "get_active_case",
    "bind_case",
    "unbind_case",
    "load_recordings",
    "merge_recording_dicts",
    "write_recordings",
    "runtime_records_from_recordings",
    "merge_runtime_records",
    "merge_recordings_into_baseline",
    "ObservedVerdict",
    "compare_observed",
    "unmerged_runtime_sites",
    "BindingProposal",
    "propose_bindings",
    "runtime_target",
    "apply_bindings",
]
