"""
Threshold calibration benchmark.

For each evaluator × judge combination, sweeps threshold values from 0.30 to 0.90
and finds a development threshold that maximises F1 against dataset labels.
HaluEval task subsets pair references with generated hallucinations; these are
not independently human-adjudicated labels. This legacy sweep has no held-out
evaluation. Errors abort the sweep rather than supplying fabricated scores.

Datasets:
  hallucination — HaluEval QA (100 cases, 50/50 faithful/hallucinated)
  faithfulness  — HaluEval Summarization (60 cases, 50/50)
  relevance     — curated golden set (40 cases, 50/50 relevant/irrelevant)

Usage:
  python benchmarks/run_threshold_calibration.py
  python benchmarks/run_threshold_calibration.py --judges anthropic/claude-haiku-4-5-20251001 openai/gpt-4o-mini
  python benchmarks/run_threshold_calibration.py --output benchmarks/results/calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


DEFAULT_JUDGES = [
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-4o-mini",
]

HALUEVAL_REVISION = "b7253db3cdaa0ab2c382f92b26b390109174f77e"

THRESHOLD_RANGE = [round(t, 2) for t in [x / 100 for x in range(30, 95, 5)]]


# ── Datasets ──────────────────────────────────────────────────────────────────

def _halueval_rows(filename: str, n: int, required: tuple[str, ...]) -> list[dict]:
    if type(n) is not int or n < 1:
        raise ValueError("Source count must be a positive integer")
    url = f"https://raw.githubusercontent.com/RUCAIBox/HaluEval/{HALUEVAL_REVISION}/data/{filename}"
    # A failed download must not silently replace the named benchmark with fixtures.
    with urllib.request.urlopen(url, timeout=30) as response:
        data = [json.loads(line) for line in response.read().decode().splitlines() if line.strip()][:n]
    if len(data) != n or any(not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip() for key in required) for row in data):
        raise ValueError("HaluEval selection is incomplete or has missing scoring fields")
    return data


def _load_halueval_qa(n: int = 50) -> list[dict]:
    data = _halueval_rows("qa_data.json", n,
                         ("knowledge", "question", "right_answer", "hallucinated_answer"))
    cases = []
    for index, item in enumerate(data):
        for field, label in (("right_answer", 0), ("hallucinated_answer", 1)):
            cases.append({"question": item["question"], "context": item["knowledge"],
                          "output": item[field], "label": label,
                          "source_id": f"halueval-qa:{HALUEVAL_REVISION}:{index}"})
    return cases


def _load_halueval_summ(n: int = 30) -> list[dict]:
    data = _halueval_rows("summarization_data.json", n,
                         ("document", "right_summary", "hallucinated_summary"))
    cases = []
    for index, item in enumerate(data):
        for field, label in (("right_summary", 0), ("hallucinated_summary", 1)):
            cases.append({"document": item["document"], "output": item[field], "label": label,
                          "source_id": f"halueval-summ:{HALUEVAL_REVISION}:{index}"})
    return cases


def _load_relevance_golden() -> list[dict]:
    from benchmarks.run_relevance_benchmark import GOLDEN_SET
    return [{"question": g["question"], "output": g["answer"], "label": g["label"]}
            for g in GOLDEN_SET]


# ── Evaluation ────────────────────────────────────────────────────────────────

def _valid_score(score) -> float:
    if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Calibration requires a finite score between zero and one")
    return float(score)


def _measured_score(result) -> float:
    if result.metadata.get("skipped"):
        raise ValueError("Skipped measurements cannot calibrate a threshold")
    return _valid_score(result.score)


def _score_hallucination(item: dict, ev) -> float:
    from multivon_eval import EvalCase
    case = EvalCase(input=item["question"], context=item["context"])
    result = ev.evaluate(case, item["output"])
    return _measured_score(result)


def _score_faithfulness(item: dict, ev) -> float:
    from multivon_eval import EvalCase
    case = EvalCase(input="Summarize the following document.", context=item["document"])
    result = ev.evaluate(case, item["output"])
    return _measured_score(result)


def _score_relevance(item: dict, ev) -> float:
    from multivon_eval import EvalCase
    case = EvalCase(input=item["question"])
    result = ev.evaluate(case, item["output"])
    return _measured_score(result)


def _collect_scores(items: list[dict], score_fn, ev, workers: int = 4) -> list[tuple[float, int]]:
    if not items or any(type(item.get("label")) is not int or item["label"] not in (0, 1) for item in items):
        raise ValueError("Calibration requires nonempty binary-labeled data")
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_fn, item, ev): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = (_valid_score(fut.result()), items[i]["label"])
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"Calibration aborted: item {i} has no valid measurement") from exc
    return [r for r in results if r is not None]


# ── Statistics ────────────────────────────────────────────────────────────────

def _f1_at_threshold(scores_labels: list[tuple[float, int]], threshold: float,
                     invert: bool = False) -> tuple[float, float, float]:
    tp = fp = fn = tn = 0
    for score, label in scores_labels:
        # label=1 means "bad" (hallucinated / irrelevant) — evaluator should flag it (not pass)
        detected = (score < threshold) if not invert else (score >= threshold)
        actual_bad = label == 1
        if detected and actual_bad:     tp += 1
        elif detected and not actual_bad: fp += 1
        elif not detected and actual_bad: fn += 1
        else:                             tn += 1
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def _best_threshold(scores_labels: list[tuple[float, int]],
                    invert: bool = False) -> dict:
    if not scores_labels or any(type(label) is not int or label not in (0, 1) for _, label in scores_labels):
        raise ValueError("Calibration requires nonempty binary-labeled measurements")
    for score, _ in scores_labels:
        _valid_score(score)
    best = None
    sweep = []
    for t in THRESHOLD_RANGE:
        p, r, f1 = _f1_at_threshold(scores_labels, t, invert=invert)
        sweep.append({"threshold": t, "precision": p, "recall": r, "f1": f1})
        if best is None or f1 > best["f1"]:
            best = {"threshold": t, "f1": f1, "precision": p, "recall": r}
    return {"optimal": best, "sweep": sweep, "n": len(scores_labels),
            "scope": "development fit; no held-out estimate"}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_calibration(judges: list[str], workers: int, verbose: bool) -> dict:
    from multivon_eval import configure, JudgeConfig, Hallucination, Faithfulness, Relevance

    if verbose:
        print("\n  Loading datasets...", end=" ", flush=True)
    hal_items = _load_halueval_qa(50)
    faith_items = _load_halueval_summ(30)
    try:
        rel_items = _load_relevance_golden()
    except Exception:
        # run_relevance_benchmark may not be importable as a module; fall back
        sys.path.insert(0, str(Path(__file__).parent))
        from run_relevance_benchmark import GOLDEN_SET
        rel_items = [{"question": g["question"], "output": g["answer"], "label": g["label"]}
                     for g in GOLDEN_SET]
    if verbose:
        print(f"done  (hal={len(hal_items)}, faith={len(faith_items)}, rel={len(rel_items)})")

    calibration = {}

    for judge_str in judges:
        if "/" not in judge_str:
            print(f"[skip] invalid judge format: {judge_str}", file=sys.stderr)
            continue
        provider, model = judge_str.split("/", 1)
        configure(JudgeConfig(provider=provider, model=model))

        if verbose:
            print(f"\n  Judge: {judge_str}")

        judge_results = {}

        # Hallucination
        if verbose:
            print(f"    Hallucination ({len(hal_items)} cases)...", end=" ", flush=True)
        t0 = time.time()
        hal_ev = Hallucination(threshold=0.5)  # threshold irrelevant; we sweep post-hoc
        hal_scores = _collect_scores(hal_items, _score_hallucination, hal_ev, workers)
        elapsed = time.time() - t0
        hal_cal = _best_threshold(hal_scores)
        judge_results["hallucination"] = hal_cal
        if verbose:
            opt = hal_cal["optimal"]
            print(f"done ({elapsed:.0f}s)  optimal threshold={opt['threshold']}  F1={opt['f1']:.3f}")

        # Faithfulness
        if verbose:
            print(f"    Faithfulness  ({len(faith_items)} cases)...", end=" ", flush=True)
        t0 = time.time()
        faith_ev = Faithfulness(threshold=0.5)
        faith_scores = _collect_scores(faith_items, _score_faithfulness, faith_ev, workers)
        elapsed = time.time() - t0
        faith_cal = _best_threshold(faith_scores)
        judge_results["faithfulness"] = faith_cal
        if verbose:
            opt = faith_cal["optimal"]
            print(f"done ({elapsed:.0f}s)  optimal threshold={opt['threshold']}  F1={opt['f1']:.3f}")

        # Relevance
        if verbose:
            print(f"    Relevance     ({len(rel_items)} cases)...", end=" ", flush=True)
        t0 = time.time()
        rel_ev = Relevance(threshold=0.5)
        rel_scores = _collect_scores(rel_items, _score_relevance, rel_ev, workers)
        elapsed = time.time() - t0
        rel_cal = _best_threshold(rel_scores)
        judge_results["relevance"] = rel_cal
        if verbose:
            opt = rel_cal["optimal"]
            print(f"done ({elapsed:.0f}s)  optimal threshold={opt['threshold']}  F1={opt['f1']:.3f}")

        calibration[judge_str] = judge_results

    return calibration


# ── Output ────────────────────────────────────────────────────────────────────

def print_table(calibration: dict) -> None:
    print(f"\n{'─' * 80}")
    print("  Threshold Calibration Results")
    print(f"{'─' * 80}")
    print(f"  {'Judge':<40}  {'Evaluator':<15}  {'Threshold':>9}  {'F1':>6}")
    print(f"  {'─'*40}  {'─'*15}  {'─'*9}  {'─'*6}")
    for judge, evals in calibration.items():
        model = judge.split("/", 1)[1]
        for ev_name, result in evals.items():
            opt = result["optimal"]
            print(f"  {model:<40}  {ev_name:<15}  {opt['threshold']:>9.2f}  {opt['f1']:>6.3f}")
    print(f"{'─' * 80}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold calibration benchmark for multivon-eval")
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES,
                    help="List of provider/model strings to calibrate")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", default="benchmarks/results/calibration.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    calibration = run_calibration(args.judges, args.workers, verbose)
    print_table(calibration)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"calibration": calibration}, indent=2))
    if verbose:
        print(f"  Results saved → {args.output}\n")


if __name__ == "__main__":
    main()
