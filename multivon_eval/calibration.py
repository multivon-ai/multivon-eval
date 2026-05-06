"""
Per-judge threshold calibration for LLM-as-judge evaluators.

Default thresholds are calibrated against human-labeled datasets
(HaluEval QA, HaluEval Summarization, curated relevance golden set)
for each supported judge model. Using a calibrated threshold instead of the
library-wide default of 0.7 closes the gap between "the score crossed the
line" and "a human would agree the output is bad."

Calibration benchmark results (F1 at optimal threshold):
  claude-haiku-4-5-20251001: hallucination=0.812, faithfulness=0.783, relevance=0.976
  claude-sonnet-4-6:         hallucination=0.787, faithfulness=0.656, relevance=1.000
  gpt-4o-mini:               hallucination=0.756, faithfulness=0.793, relevance=1.000

Usage:
    from multivon_eval.calibration import calibrated_threshold

    threshold = calibrated_threshold("hallucination", judge_config)
    # Returns the calibrated value, or 0.7 if the judge is not in the table.

To update the table after re-running the calibration benchmark:
    python benchmarks/run_threshold_calibration.py --output benchmarks/results/calibration.json
    # Then copy the optimal thresholds here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .judge import JudgeConfig


# Calibrated thresholds per (evaluator, judge_model).
# Derived by maximising F1 against human labels across a threshold sweep.
# Models not in this table fall back to the evaluator's own default (0.7).
_THRESHOLDS: dict[tuple[str, str], float] = {
    # Hallucination — HaluEval QA, 100 cases
    ("hallucination", "claude-haiku-4-5-20251001"): 0.55,
    ("hallucination", "claude-haiku-4-5"):           0.55,
    ("hallucination", "claude-sonnet-4-6"):           0.30,
    ("hallucination", "claude-opus-4-7"):             0.70,
    ("hallucination", "gpt-4o-mini"):                 0.30,
    ("hallucination", "gpt-4o"):                      0.65,

    # Faithfulness — HaluEval Summarization, 60 cases
    ("faithfulness", "claude-haiku-4-5-20251001"):   0.90,
    ("faithfulness", "claude-haiku-4-5"):             0.90,
    ("faithfulness", "claude-sonnet-4-6"):            0.90,
    ("faithfulness", "claude-opus-4-7"):              0.70,
    ("faithfulness", "gpt-4o-mini"):                  0.90,
    ("faithfulness", "gpt-4o"):                       0.65,

    # Relevance — curated golden set, 40 cases
    ("relevance", "claude-haiku-4-5-20251001"):      0.30,
    ("relevance", "claude-haiku-4-5"):                0.30,
    ("relevance", "claude-sonnet-4-6"):               0.30,
    ("relevance", "claude-opus-4-7"):                 0.75,
    ("relevance", "gpt-4o-mini"):                     0.30,
    ("relevance", "gpt-4o"):                          0.70,
}

_DEFAULT_THRESHOLD = 0.7


def calibrated_threshold(evaluator: str, judge: "JudgeConfig") -> float:
    """
    Return the calibrated threshold for (evaluator, judge_model), or the
    library default (0.7) if the combination has not been benchmarked.

    Args:
        evaluator: evaluator name, e.g. "hallucination", "faithfulness", "relevance"
        judge:     resolved JudgeConfig (provider + model must be set)
    """
    model = judge.model or ""
    return _THRESHOLDS.get((evaluator, model), _DEFAULT_THRESHOLD)


def threshold_table() -> dict[tuple[str, str], float]:
    """Return a copy of the full calibration table."""
    return dict(_THRESHOLDS)
