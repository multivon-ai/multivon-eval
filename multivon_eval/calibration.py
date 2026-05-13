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
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Iterable

from .exceptions import CalibrationMissing

if TYPE_CHECKING:
    from .judge import JudgeConfig


_DEFAULT_THRESHOLD = 0.7


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


_TABLE: CalibrationTable | None = None


def load_calibration(reload: bool = False) -> CalibrationTable:
    """Return the loaded calibration table (lazy, cached).

    Pass ``reload=True`` to force a re-read from disk; primarily for tests
    that mutate the file or want to test the loader directly.
    """
    global _TABLE
    if _TABLE is None or reload:
        _TABLE = _parse_table(_read_data_file("v1.json"))
    return _TABLE


def calibrated_threshold(
    evaluator: str,
    judge: "JudgeConfig",
    *,
    strict: bool = False,
) -> float:
    """Return the calibrated threshold for ``(evaluator, judge.model)``.

    Falls back to ``0.7`` if the combination has not been benchmarked,
    unless ``strict=True``, in which case :class:`CalibrationMissing` is
    raised. Strict mode is meant for procurement-style policy gates where
    "no calibrated threshold" is itself a finding.
    """
    table = load_calibration()
    model = judge.model or ""
    entry = table.lookup(evaluator, model)
    if entry is not None:
        return entry.threshold
    if strict:
        raise CalibrationMissing(evaluator, model)
    return _DEFAULT_THRESHOLD


def calibration_provenance(
    evaluator: str,
    judge: "JudgeConfig",
) -> CalibrationEntry | None:
    """Return the full :class:`CalibrationEntry` (dataset / N / F1 / date)
    that produced this threshold, or ``None`` if uncalibrated.

    Use this when generating audit reports — the entry's ``dataset_hash``,
    ``n``, and ``measured_at`` are exactly what an auditor will ask for.
    """
    table = load_calibration()
    return table.lookup(evaluator, judge.model or "")


def threshold_table() -> dict[tuple[str, str], float]:
    """Back-compat helper: flat ``{(evaluator, judge_model): threshold}``.

    Includes every alias so callers using either the canonical id or a
    short alias both hit. New code should prefer :func:`load_calibration`
    so it can read provenance too.
    """
    return {key: thr for key, thr in load_calibration().iter_thresholds()}
