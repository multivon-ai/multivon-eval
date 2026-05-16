"""
Per-judge threshold calibration for LLM-as-judge evaluators.

Calibrated thresholds are loaded from a versioned JSON file shipped with
the package (`_calibration_data/v1.json`). Each entry carries provenance:
the dataset it was measured against, dataset content hash, N, F1 / P / R
(where available), the exact judge model id, and the date measured.

Auditors can read the JSON directly to verify our threshold claims, and
calling :func:`calibration_provenance` returns the structured entry that
drove a given decision.

Usage:
    from multivon_eval.calibration import (
        calibrated_threshold, calibration_provenance, load_calibration,
    )

    threshold = calibrated_threshold("hallucination", judge_config)
    # → e.g. 0.55 for claude-haiku-4-5

    prov = calibration_provenance("hallucination", judge_config)
    # → CalibrationEntry(dataset="HaluEval QA", n=100, f1=0.812, …)

To regenerate the file after a fresh calibration run::

    python benchmarks/run_threshold_calibration.py \\
        --output multivon_eval/_calibration_data/v1.json
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Iterable, Literal

from .exceptions import CalibrationMissing

if TYPE_CHECKING:
    from .judge import JudgeConfig


_DEFAULT_THRESHOLD = 0.7

# When `calibrated_threshold` is asked for an (evaluator, judge_model) pair
# that isn't in the shipped calibration table, this policy decides what
# happens. "warn" is the default as of 2026-05-16 — earlier releases were
# silently using 0.7, which is uncalibrated and can produce 5-15pp F1 drift
# on real data. "silent" is opt-in for backward compatibility.
#
# Override globally via `set_calibration_fallback_policy("strict")` or the
# `MULTIVON_CALIBRATION_FALLBACK` env var.
FallbackPolicy = Literal["silent", "warn", "strict"]
_FALLBACK_POLICY: FallbackPolicy = "warn"
_FALLBACK_WARNED: set[tuple[str, str]] = set()


def _resolve_fallback_policy() -> FallbackPolicy:
    forced = os.environ.get("MULTIVON_CALIBRATION_FALLBACK", "").strip().lower()
    if forced in ("silent", "warn", "strict"):
        return forced  # type: ignore[return-value]
    return _FALLBACK_POLICY


def set_calibration_fallback_policy(policy: FallbackPolicy) -> None:
    """Configure what happens when ``calibrated_threshold`` cannot find
    a calibration row for an (evaluator, judge_model) pair.

    - ``"warn"`` (default, new in 0.7.3): emit a :class:`UserWarning` once
      per (evaluator, judge_model) pair, then return ``0.7``. Pre-0.7.3
      behaviour can be restored with ``"silent"``.
    - ``"strict"``: raise :class:`CalibrationMissing` — recommended for
      regulated-AI deployments and CI gates where a missing calibration
      row should itself be an audit finding, not a silent default.
    - ``"silent"``: fall back to ``0.7`` with no warning. Backward
      compatible. Not recommended for new code.

    Per-call ``strict=True`` always wins, regardless of the global policy.
    The ``MULTIVON_CALIBRATION_FALLBACK`` env var overrides this at process
    start without requiring a code change — useful for CI gates that want
    "strict" without changing application code.
    """
    global _FALLBACK_POLICY
    if policy not in ("silent", "warn", "strict"):
        raise ValueError(
            f"set_calibration_fallback_policy: unknown policy {policy!r}; "
            "expected 'silent', 'warn', or 'strict'."
        )
    _FALLBACK_POLICY = policy


@dataclass
class CalibrationEntry:
    """One row of the calibration table.

    Attributes are derived from the methodology used by
    `benchmarks/run_threshold_calibration.py`: every row corresponds to
    one (evaluator, judge_model) sweep that maximised F1 on the named
    dataset. ``precision``, ``recall``, and ``f1`` may be ``None`` if the
    original benchmark output didn't retain them — the threshold is still
    usable; only the audit trail is partial.
    """

    evaluator: str
    judge_model: str
    threshold: float
    dataset: str
    dataset_hash: str
    n: int
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    measured_at: str = ""
    notes: str = ""
    judge_aliases: list[str] = field(default_factory=list)

    def all_judge_keys(self) -> list[str]:
        """Every model id this entry can be matched by (canonical + aliases)."""
        return [self.judge_model, *self.judge_aliases]


@dataclass
class CalibrationTable:
    """Loaded calibration data plus the metadata block from the file."""

    schema_version: int
    generated_at: str
    methodology: str
    entries: list[CalibrationEntry]
    # The label this table was loaded under ("v1", "v2", …). Critical for
    # audit-package replay: the audit log records this label so the package
    # bundles the matching JSON file, not whatever happens to be the default
    # when the package is built. Empty string for tables built without a
    # label (e.g. constructed directly in tests).
    version_label: str = ""

    def lookup(self, evaluator: str, judge_model: str) -> CalibrationEntry | None:
        for entry in self.entries:
            if entry.evaluator != evaluator:
                continue
            if judge_model in entry.all_judge_keys():
                return entry
        return None

    def iter_thresholds(self) -> Iterable[tuple[tuple[str, str], float]]:
        for e in self.entries:
            for key in e.all_judge_keys():
                yield (e.evaluator, key), e.threshold


def _read_data_file(filename: str) -> dict:
    """Read a JSON file from the bundled `_calibration_data/` package."""
    pkg = resources.files("multivon_eval._calibration_data")
    with pkg.joinpath(filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_table(raw: dict) -> CalibrationTable:
    entries: list[CalibrationEntry] = []
    for row in raw.get("entries", []):
        entries.append(
            CalibrationEntry(
                evaluator=row["evaluator"],
                judge_model=row["judge_model"],
                threshold=float(row["threshold"]),
                dataset=row.get("dataset", ""),
                dataset_hash=row.get("dataset_hash", ""),
                n=int(row.get("n", 0)),
                precision=row.get("precision"),
                recall=row.get("recall"),
                f1=row.get("f1"),
                measured_at=row.get("measured_at", ""),
                notes=row.get("notes", ""),
                judge_aliases=list(row.get("judge_aliases", [])),
            )
        )
    return CalibrationTable(
        schema_version=int(raw.get("schema_version", 1)),
        generated_at=raw.get("generated_at", ""),
        methodology=raw.get("methodology", ""),
        entries=entries,
    )


# Cached table keyed by version label. Lets a single process load v1 and
# v2 simultaneously (e.g., for comparing thresholds across releases) without
# re-parsing the JSON on every lookup.
_TABLE_CACHE: dict[str, CalibrationTable] = {}

# Default version preference order when no version is requested explicitly.
# v2 wins; v1 is the long-term fallback. Override globally via the env var
# ``MULTIVON_CALIBRATION_VERSION`` for reproducibility-critical CI runs.
_DEFAULT_PREFERENCE = ("v2", "v1")


def _resolve_default_version() -> str:
    """Pick the calibration version label honoring the env-var override."""
    import os as _os
    forced = _os.environ.get("MULTIVON_CALIBRATION_VERSION", "").strip()
    if forced:
        return forced
    for cand in _DEFAULT_PREFERENCE:
        try:
            _read_data_file(f"{cand}.json")
            return cand
        except FileNotFoundError:
            continue
    # Final fallback — caller will get a FileNotFoundError when load_calibration
    # tries to read the file, which is the right behavior.
    return _DEFAULT_PREFERENCE[-1]


def load_calibration(reload: bool = False, *, version: str | None = None) -> CalibrationTable:
    """Return the loaded calibration table (lazy, cached).

    By default, picks the version label from
    ``MULTIVON_CALIBRATION_VERSION`` if set, then falls back to the
    shipped preference order (v2 → v1). Pass ``version="v1"`` (or any
    other shipped label) to pin explicitly — useful for CI runs that
    need to reproduce historical behavior even after a new calibration
    sweep ships.

    Args:
        reload: Force a re-read from disk; primarily for tests that
            mutate ``_calibration_data/*.json`` and want to see the
            change without restarting the process.
        version: Explicit version label (e.g. ``"v1"``, ``"v2"``). When
            omitted, the env-var or shipped preference order decides.

    Raises:
        FileNotFoundError: If the requested ``version`` isn't shipped
            with the package.
    """
    # Explicit kwarg always wins, even an empty string — pinning to a
    # known-invalid label should fail loudly, not silently fall back to
    # the env / default. Codex review caught this.
    if version is None:
        label = _resolve_default_version()
    elif not version:
        raise FileNotFoundError(
            "load_calibration: empty version label — pass a label like 'v1' or 'v2'"
        )
    else:
        label = version
    if reload:
        _TABLE_CACHE.pop(label, None)
    cached = _TABLE_CACHE.get(label)
    if cached is not None:
        return cached
    parsed = _parse_table(_read_data_file(f"{label}.json"))
    parsed.version_label = label
    _TABLE_CACHE[label] = parsed
    return _TABLE_CACHE[label]


def effective_calibration_version(*, version: str | None = None) -> str:
    """Return the label that would be loaded for ``version`` (or the
    default if ``None``). Cheap; loads the table to honor the cache.

    Used by the lockfile fingerprint and the audit-package builder so
    "which calibration drove these decisions" is recorded explicitly
    rather than inferred from a dataset hash.
    """
    return load_calibration(version=version).version_label


def calibrated_threshold(
    evaluator: str,
    judge: "JudgeConfig",
    *,
    strict: bool = False,
    version: str | None = None,
) -> float:
    """Return the calibrated threshold for ``(evaluator, judge.model)``.

    Behaviour when no calibration row exists is governed by the global
    fallback policy (see :func:`set_calibration_fallback_policy`):

    - ``"warn"`` (default since 0.7.3): emits a :class:`UserWarning` once
      per (evaluator, judge_model) pair, then returns ``0.7``.
    - ``"strict"``: raises :class:`CalibrationMissing`.
    - ``"silent"``: falls back to ``0.7`` with no warning (pre-0.7.3
      behaviour, opt-in).

    Per-call ``strict=True`` overrides the global policy. The
    ``MULTIVON_CALIBRATION_FALLBACK`` env var overrides the in-process
    default.

    Pass ``version="v1"`` (or another shipped label) to pin the lookup
    to a specific calibration release for reproducibility.
    """
    table = load_calibration(version=version)
    model = judge.model or ""
    entry = table.lookup(evaluator, model)
    if entry is not None:
        return entry.threshold
    # Per-call strict=True always wins.
    effective = "strict" if strict else _resolve_fallback_policy()
    if effective == "strict":
        raise CalibrationMissing(evaluator, model)
    if effective == "warn":
        cache_key = (evaluator, model)
        if cache_key not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(cache_key)
            warnings.warn(
                f"calibrated_threshold: no calibration row for evaluator="
                f"{evaluator!r} judge_model={model!r}; falling back to "
                f"{_DEFAULT_THRESHOLD}. This default is uncalibrated and "
                "may produce 5-15pp F1 drift on real data. Call "
                "multivon_eval.calibration.set_calibration_fallback_policy"
                "(\"strict\") to fail closed, or run "
                "benchmarks/run_threshold_calibration.py to add a row.",
                UserWarning,
                stacklevel=2,
            )
    return _DEFAULT_THRESHOLD


def calibration_provenance(
    evaluator: str,
    judge: "JudgeConfig",
    *,
    version: str | None = None,
) -> CalibrationEntry | None:
    """Return the full :class:`CalibrationEntry` (dataset / N / F1 / date)
    that produced this threshold, or ``None`` if uncalibrated.

    Use this when generating audit reports — the entry's ``dataset_hash``,
    ``n``, and ``measured_at`` are exactly what an auditor will ask for.
    Honors ``version=`` so an audit can be re-run against the calibration
    that was active when the original eval was recorded.
    """
    table = load_calibration(version=version)
    return table.lookup(evaluator, judge.model or "")


def threshold_table(*, version: str | None = None) -> dict[tuple[str, str], float]:
    """Back-compat helper: flat ``{(evaluator, judge_model): threshold}``.

    Includes every alias so callers using either the canonical id or a
    short alias both hit. New code should prefer :func:`load_calibration`
    so it can read provenance too.
    """
    return {key: thr for key, thr in load_calibration(version=version).iter_thresholds()}


def calibration_versions() -> list[str]:
    """List the calibration version labels shipped with the installed package.

    Useful for CI gates that want to assert ``"v2" in calibration_versions()``
    before relying on threshold pinning at that version.
    """
    from importlib import resources
    pkg = resources.files("multivon_eval._calibration_data")
    labels: list[str] = []
    for entry in pkg.iterdir():
        name = entry.name
        if name.endswith(".json"):
            labels.append(name[:-5])  # strip ".json"
    return sorted(labels)
