"""
Add CI fields to existing benchmark result JSONs.

For each result JSON in benchmarks/results/, computes:
  - precision_ci_lo, precision_ci_hi  via Wilson on TP / (TP+FP)
  - recall_ci_lo,    recall_ci_hi     via Wilson on TP / (TP+FN)
  - f1_ci_lo,        f1_ci_hi         via bootstrap (1000 resamples)

Idempotent — re-running on a JSON that already has CIs overwrites them
with freshly computed values. Skips entries that don't have the required
TP/FP/FN/TN counts.

Run once after a benchmark output lands. The benchmark runners themselves
should be updated to compute CIs on the fly in a follow-up commit.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Stable seed so CIs are reproducible run-to-run.
_RNG = random.Random(20260603)

# Import Wilson from the package.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from multivon_eval.experiments import wilson_interval  # noqa: E402


def _bootstrap_f1_ci(
    tp: int, fp: int, fn: int, tn: int, n_resamples: int = 1000, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Bootstrap CI on F1 by resampling reconstructed case indicators.

    Each case becomes a (predicted, actual) tuple:
      TP cases → (1, 1)
      FP cases → (1, 0)
      FN cases → (0, 1)
      TN cases → (0, 0)

    Resample N pairs with replacement, recompute F1, take percentiles.
    Uses a module-level seeded RNG for reproducibility.
    """
    n = tp + fp + fn + tn
    if n == 0:
        return (0.0, 0.0)
    cases: list[tuple[int, int]] = (
        [(1, 1)] * tp + [(1, 0)] * fp + [(0, 1)] * fn + [(0, 0)] * tn
    )
    f1s: list[float] = []
    for _ in range(n_resamples):
        sample = [cases[_RNG.randint(0, n - 1)] for _ in range(n)]
        s_tp = sum(1 for p, a in sample if p == 1 and a == 1)
        s_fp = sum(1 for p, a in sample if p == 1 and a == 0)
        s_fn = sum(1 for p, a in sample if p == 0 and a == 1)
        if s_tp + s_fp == 0 or s_tp + s_fn == 0:
            f1s.append(0.0)
            continue
        prec = s_tp / (s_tp + s_fp)
        rec = s_tp / (s_tp + s_fn)
        if prec + rec == 0:
            f1s.append(0.0)
            continue
        f1s.append(2 * prec * rec / (prec + rec))
    f1s.sort()
    lo_idx = int((1 - confidence) / 2 * n_resamples)
    hi_idx = int((1 + confidence) / 2 * n_resamples) - 1
    return (round(f1s[lo_idx], 4), round(f1s[hi_idx], 4))


def _add_cis_to_metrics(metrics: dict) -> dict:
    """Add CI fields to a single metrics dict if TP/FP/FN/TN present."""
    if not all(k in metrics for k in ("tp", "fp", "fn", "tn")):
        return metrics
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    # Wilson on precision and recall.
    if tp + fp > 0:
        p_lo, p_hi = wilson_interval(tp, tp + fp)
        metrics["precision_ci_lo"] = round(p_lo, 4)
        metrics["precision_ci_hi"] = round(p_hi, 4)
    if tp + fn > 0:
        r_lo, r_hi = wilson_interval(tp, tp + fn)
        metrics["recall_ci_lo"] = round(r_lo, 4)
        metrics["recall_ci_hi"] = round(r_hi, 4)
    # Bootstrap on F1.
    f1_lo, f1_hi = _bootstrap_f1_ci(tp, fp, fn, tn)
    metrics["f1_ci_lo"] = f1_lo
    metrics["f1_ci_hi"] = f1_hi
    return metrics


def process_file(path: Path) -> bool:
    """Process one JSON file in place. Returns True if mutated."""
    raw = json.loads(path.read_text())
    mutated = False

    # Pattern 1: top-level "results" dict of {evaluator_name: metrics_dict}.
    if isinstance(raw, dict) and isinstance(raw.get("results"), dict):
        for name, metrics in raw["results"].items():
            if isinstance(metrics, dict):
                before = dict(metrics)
                _add_cis_to_metrics(metrics)
                if metrics != before:
                    mutated = True

    # Pattern 2: per_judge_metrics in multi_judge_agreement.json.
    if isinstance(raw, dict) and isinstance(raw.get("per_judge_metrics"), dict):
        for judge, metrics in raw["per_judge_metrics"].items():
            if isinstance(metrics, dict):
                before = dict(metrics)
                _add_cis_to_metrics(metrics)
                if metrics != before:
                    mutated = True

    if mutated:
        path.write_text(json.dumps(raw, indent=2) + "\n")
    return mutated


def main():
    results_dir = HERE / "results"
    if not results_dir.exists():
        print(f"results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)
    touched = 0
    for path in sorted(results_dir.glob("*.json")):
        if process_file(path):
            print(f"  updated  {path.name}")
            touched += 1
        else:
            print(f"  no-op    {path.name}")
    print(f"\n{touched}/{len(list(results_dir.glob('*.json')))} files updated")


if __name__ == "__main__":
    main()
