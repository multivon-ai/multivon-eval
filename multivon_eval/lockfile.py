"""
suite.lock — content-addressed fingerprint of an EvalSuite at run time.

The problem this solves: every persona in the 2026-05-14 deliberation
(11 of 12) flagged that a silent prompt change inside an evaluator
silently invalidates every historical comparison. A reviewer cannot
tell whether "Faithfulness dropped 3pp" is a regression or a prompt
update.

Solution: fingerprint the suite. Every component contributes a
deterministic hash; the suite's lock is the SHA-256 of the merged
manifest. A lock written today can be compared against a lock written
six months from now, and if any meaningful field has changed, the
comparison fails loudly with a structured diff.

What's fingerprinted (per the cross-model synthesis):

* ``library_version`` — multivon_eval.__version__
* ``evaluators[]`` — for each:
    - ``name``, ``version``, ``class_path``, ``threshold``
    - ``prompt_hash`` — SHA-256 of any QAG/judge prompt template the
      evaluator carries (LLM-judge evaluators)
    - ``judge`` — resolved judge config (provider, model, base_url,
      temperature, max_tokens) for evaluators that use one
    - ``calibration`` — calibration entry (dataset hash, N, F1, date)
      that drove the threshold, if any
* ``cases`` — count + SHA-256 of the canonical JSON serialization
* ``run_config`` — judge config defaults at lock time

Usage::

    suite = EvalSuite.eu_ai_act_high_risk()
    suite.add_cases(cases)
    lock = suite.lock()
    Path("suite.lock").write_text(lock.to_json())

    # six months later, in CI:
    saved = SuiteLock.from_json(Path("suite.lock").read_text())
    suite.verify_lock(saved)  # raises LockMismatch on any drift
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any

from . import __version__
from .exceptions import MultivonError

if TYPE_CHECKING:
    from .case import EvalCase
    from .evaluators.base import Evaluator
    from .suite import EvalSuite


class LockMismatch(MultivonError):
    """Raised when a saved SuiteLock doesn't match the current suite.

    The exception carries the structured diff via :attr:`differences`
    so CI tooling can surface every drift, not just the first one.
    """

    def __init__(self, differences: list[str]):
        self.differences = list(differences)
        joined = "\n  - ".join(differences) if differences else "(no detail)"
        super().__init__(
            f"Suite has drifted from the saved lockfile ({len(differences)} differences):\n  - {joined}"
        )


@dataclass
class EvaluatorFingerprint:
    """One row in the SuiteLock's ``evaluators`` array."""
    name: str
    class_path: str
    threshold: float
    prompt_hash: str | None = None
    judge: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    version: str = "1"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuiteLock:
    """Content-addressed fingerprint of an :class:`EvalSuite`."""

    library_version: str
    suite_name: str
    suite_hash: str  # SHA-256 of the canonical manifest below
    evaluators: list[EvaluatorFingerprint] = field(default_factory=list)
    case_count: int = 0
    cases_hash: str | None = None
    run_config: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "library_version": self.library_version,
            "suite_name": self.suite_name,
            "suite_hash": self.suite_hash,
            "evaluators": [asdict(e) for e in self.evaluators],
            "case_count": self.case_count,
            "cases_hash": self.cases_hash,
            "run_config": self.run_config,
            "extra": self.extra,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> "SuiteLock":
        return cls(
            library_version=data["library_version"],
            suite_name=data["suite_name"],
            suite_hash=data["suite_hash"],
            evaluators=[EvaluatorFingerprint(**e) for e in data.get("evaluators", [])],
            case_count=data.get("case_count", 0),
            cases_hash=data.get("cases_hash"),
            run_config=data.get("run_config"),
            extra=data.get("extra", {}),
        )

    @classmethod
    def from_json(cls, raw: str) -> "SuiteLock":
        return cls.from_dict(json.loads(raw))

    def diff(self, other: "SuiteLock") -> list[str]:
        """Return a list of human-readable differences vs another lock."""
        diffs: list[str] = []
        if self.suite_hash != other.suite_hash:
            diffs.append(f"suite_hash: {self.suite_hash[:8]} vs {other.suite_hash[:8]}")
        if self.library_version != other.library_version:
            diffs.append(
                f"library_version: {self.library_version} vs {other.library_version}"
            )
        if self.suite_name != other.suite_name:
            diffs.append(f"suite_name: {self.suite_name!r} vs {other.suite_name!r}")
        if self.case_count != other.case_count:
            diffs.append(f"case_count: {self.case_count} vs {other.case_count}")
        if self.cases_hash != other.cases_hash:
            diffs.append(f"cases_hash: {self.cases_hash} vs {other.cases_hash}")

        # Evaluator-level diff — match by class_path+name. `self` is
        # the current (live) lock; `other` is the saved baseline.
        current_by_key = {(e.class_path, e.name): e for e in self.evaluators}
        saved_by_key = {(e.class_path, e.name): e for e in other.evaluators}
        for k in sorted(set(current_by_key) | set(saved_by_key)):
            cur = current_by_key.get(k)
            saved = saved_by_key.get(k)
            if saved is None:  # exists in current but not in saved
                diffs.append(f"evaluator added: {k[0]}::{k[1]}")
                continue
            if cur is None:  # was in saved but no longer
                diffs.append(f"evaluator removed: {k[0]}::{k[1]}")
                continue
            for fld in ("threshold", "prompt_hash", "judge", "calibration", "version"):
                if getattr(cur, fld) != getattr(saved, fld):
                    diffs.append(f"evaluator {k[1]}.{fld}: {getattr(cur, fld)!r} vs {getattr(saved, fld)!r}")
            # Compare per-evaluator config dict — catches changes to
            # `Contains.substrings`, `WordCount.min_words`, etc. that
            # would otherwise produce a different suite_hash with no
            # visible explanation in the diff output.
            cur_cfg = (cur.extra or {}).get("config", {})
            saved_cfg = (saved.extra or {}).get("config", {})
            for cfg_key in sorted(set(cur_cfg) | set(saved_cfg)):
                if cur_cfg.get(cfg_key) != saved_cfg.get(cfg_key):
                    diffs.append(
                        f"evaluator {k[1]}.config.{cfg_key}: "
                        f"{cur_cfg.get(cfg_key)!r} vs {saved_cfg.get(cfg_key)!r}"
                    )
        return diffs


# ── builders ────────────────────────────────────────────────────────────────


def _canonical_json(obj: Any) -> bytes:
    """Deterministic JSON for hashing."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _evaluator_prompt_hash(evaluator: "Evaluator") -> str | None:
    """Return a SHA-256 over any prompt template the evaluator carries.

    Evaluators expose their prompt(s) via a few attributes — we pick up
    the common ones. None is returned for evaluators with no prompt
    (deterministic ones).
    """
    parts: list[str] = []
    for attr in (
        "_PROMPT", "PROMPT", "_TEMPLATE", "TEMPLATE",
        "_QUESTION_PROMPT", "_VERIFICATION_PROMPT",
        "_CLAIM_PROMPT", "criterion", "_check_criterion",
    ):
        v = getattr(evaluator, attr, None)
        if isinstance(v, str):
            parts.append(f"{attr}={v}")
    if not parts:
        return None
    return _sha256("\n".join(sorted(parts)).encode("utf-8"))


def _resolved_judge(evaluator: "Evaluator") -> Any | None:
    """Pull a resolved judge config from an LLM-judge evaluator.

    Different LLM evaluators store the config under different names
    (``_judge_cfg`` for the QAG evaluators, ``judge`` for older ones,
    ``_judge`` for some intermediates). We probe all three and resolve()
    the result so the fingerprint sees env-var fallbacks.
    """
    raw = (
        getattr(evaluator, "_judge_cfg", None)
        or getattr(evaluator, "judge", None)
        or getattr(evaluator, "_judge", None)
    )
    if raw is None:
        return None
    try:
        from .judge import resolve_judge
        return resolve_judge(raw)
    except Exception:
        return raw


def _evaluator_judge_fingerprint(evaluator: "Evaluator") -> dict[str, Any] | None:
    """If the evaluator uses an LLM judge, fingerprint the resolved config."""
    judge = _resolved_judge(evaluator)
    if judge is None:
        return None
    return {
        "provider": getattr(judge, "provider", ""),
        "model": getattr(judge, "model", ""),
        "base_url": getattr(judge, "base_url", ""),
        "temperature": getattr(judge, "temperature", 0.0),
        "max_tokens": getattr(judge, "max_tokens", 0),
    }


def _evaluator_calibration_fingerprint(evaluator: "Evaluator") -> dict[str, Any] | None:
    """Look up the calibration entry that drove this evaluator's threshold."""
    judge = _resolved_judge(evaluator)
    if judge is None or not getattr(judge, "model", ""):
        return None
    try:
        from .calibration import calibration_provenance
        entry = calibration_provenance(getattr(evaluator, "name", ""), judge)
        if entry is None:
            return None
        return {
            "dataset": entry.dataset,
            "dataset_hash": entry.dataset_hash,
            "n": entry.n,
            "f1": entry.f1,
            "measured_at": entry.measured_at,
            "threshold": entry.threshold,
        }
    except Exception:
        return None


def _evaluator_config_fingerprint(evaluator: "Evaluator") -> dict[str, Any]:
    """Capture JSON-safe public configuration knobs an evaluator was built with.

    Critical for reproducibility: ``Contains.substrings``, ``WordCount.min_words``,
    ``RegexMatch.pattern``, ``BLEU.n``, ``BERTScore.model`` — anything that
    changes a decision — must be in the fingerprint. Otherwise two suites
    with different config but the same name + threshold + judge would share
    a ``suite_hash`` and an audit replay would diverge silently.

    Conservative rules:
      - Public attributes only (no leading underscore).
      - Skip attributes already captured by other fingerprint fields
        (name, threshold, judge — those have dedicated slots).
      - Skip non-serializable values (callables, classes, complex objects)
        rather than failing — we err on the side of keeping the lock
        builder non-fatal.
      - Compiled regex patterns are rendered as their source pattern so
        the fingerprint stays stable across runs.
    """
    import re

    _SKIP_FIELDS = {"name", "threshold", "judge"}
    out: dict[str, Any] = {}
    for attr_name in sorted(vars(evaluator)):
        if attr_name.startswith("_") or attr_name in _SKIP_FIELDS:
            continue
        value = getattr(evaluator, attr_name)
        # Render regex objects as their source so the fingerprint is portable.
        if isinstance(value, re.Pattern):
            out[attr_name] = {"_kind": "regex", "pattern": value.pattern, "flags": value.flags}
            continue
        # Probe JSON-serializability — anything we can't round-trip is
        # silently skipped (better to miss a field than to crash the lock).
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        out[attr_name] = value
    return out


def fingerprint_evaluator(evaluator: "Evaluator") -> EvaluatorFingerprint:
    return EvaluatorFingerprint(
        name=getattr(evaluator, "name", type(evaluator).__name__),
        class_path=f"{type(evaluator).__module__}.{type(evaluator).__name__}",
        threshold=float(getattr(evaluator, "threshold", 0.5)),
        prompt_hash=_evaluator_prompt_hash(evaluator),
        judge=_evaluator_judge_fingerprint(evaluator),
        calibration=_evaluator_calibration_fingerprint(evaluator),
        version="2",  # bumped from 1 — config dict is now in `extra`
        extra={"config": _evaluator_config_fingerprint(evaluator)},
    )


def _cases_hash(cases: list["EvalCase"]) -> str | None:
    if not cases:
        return None
    serialized = [
        _canonical_json({
            "input": c.input,
            "expected_output": c.expected_output,
            "context": c.context,
            # tags/metadata may legitimately differ across runs; exclude.
        }).decode()
        for c in cases
    ]
    return _sha256("\n".join(serialized).encode("utf-8"))


def build_suite_lock(suite: "EvalSuite") -> SuiteLock:
    """Fingerprint a suite into a :class:`SuiteLock`."""
    eval_fps = [fingerprint_evaluator(e) for e in suite._evaluators]
    cases_hash = _cases_hash(suite._cases)
    manifest = {
        "library_version": __version__,
        "suite_name": suite.name,
        "evaluators": [asdict(e) for e in eval_fps],
        "case_count": len(suite._cases),
        "cases_hash": cases_hash,
    }
    return SuiteLock(
        library_version=__version__,
        suite_name=suite.name,
        suite_hash=_sha256(_canonical_json(manifest)),
        evaluators=eval_fps,
        case_count=len(suite._cases),
        cases_hash=cases_hash,
        run_config=None,
        extra={},
    )


def verify_suite_against_lock(suite: "EvalSuite", saved: SuiteLock) -> None:
    """Raise :class:`LockMismatch` if ``suite`` drifted from ``saved``."""
    current = build_suite_lock(suite)
    diffs = current.diff(saved)
    if diffs:
        raise LockMismatch(diffs)


__all__ = [
    "SuiteLock",
    "EvaluatorFingerprint",
    "LockMismatch",
    "build_suite_lock",
    "fingerprint_evaluator",
    "verify_suite_against_lock",
]
