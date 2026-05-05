"""
Threshold calibration benchmark.

For each evaluator × judge combination, sweeps threshold values from 0.30 to 0.90
and finds the threshold that maximises F1 against human-labeled ground truth.

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

THRESHOLD_RANGE = [round(t, 2) for t in [x / 100 for x in range(30, 95, 5)]]


# ── Datasets ──────────────────────────────────────────────────────────────────

def _load_halueval_qa(n: int = 50) -> list[dict]:
    url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            lines = r.read().decode().strip().splitlines()
        data = [json.loads(l) for l in lines if l.strip()][:n]
    except Exception:
        data = _HALUEVAL_QA_FALLBACK[:n]

    cases = []
    for item in data:
        ctx = item.get("knowledge", item.get("context", ""))
        q = item.get("question", "")
        cases.append({"question": q, "context": ctx,
                      "output": item.get("right_answer", ""), "label": 0})
        cases.append({"question": q, "context": ctx,
                      "output": item.get("hallucinated_answer", ""), "label": 1})
    return cases


def _load_halueval_summ(n: int = 30) -> list[dict]:
    url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            lines = r.read().decode().strip().splitlines()
        data = [json.loads(l) for l in lines if l.strip()][:n]
    except Exception:
        data = _HALUEVAL_SUMM_FALLBACK[:n]

    cases = []
    for item in data:
        doc = item.get("document", item.get("context", ""))
        cases.append({"document": doc,
                      "output": item.get("right_summary", ""), "label": 0})
        cases.append({"document": doc,
                      "output": item.get("hallucinated_summary", ""), "label": 1})
    return cases


def _load_relevance_golden() -> list[dict]:
    from benchmarks.run_relevance_benchmark import GOLDEN_SET
    return [{"question": g["question"], "output": g["answer"], "label": g["label"]}
            for g in GOLDEN_SET]


# ── Evaluation ────────────────────────────────────────────────────────────────

def _score_hallucination(item: dict, ev) -> float:
    from multivon_eval import EvalCase
    case = EvalCase(input=item["question"], context=item["context"])
    result = ev.evaluate(case, item["output"])
    return result.score


def _score_faithfulness(item: dict, ev) -> float:
    from multivon_eval import EvalCase
    case = EvalCase(input="Summarize the following document.", context=item["document"])
    result = ev.evaluate(case, item["output"])
    return result.score


def _score_relevance(item: dict, ev) -> float:
    from multivon_eval import EvalCase
    case = EvalCase(input=item["question"])
    result = ev.evaluate(case, item["output"])
    return result.score


def _collect_scores(items: list[dict], score_fn, ev, workers: int = 4) -> list[tuple[float, int]]:
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_fn, item, ev): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = (fut.result(), items[i]["label"])
            except Exception as exc:
                print(f"  [warn] eval error on item {i}: {exc}", file=sys.stderr)
                results[i] = (0.5, items[i]["label"])
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
    best = {"threshold": 0.7, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    sweep = []
    for t in THRESHOLD_RANGE:
        p, r, f1 = _f1_at_threshold(scores_labels, t, invert=invert)
        sweep.append({"threshold": t, "precision": p, "recall": r, "f1": f1})
        if f1 > best["f1"]:
            best = {"threshold": t, "f1": f1, "precision": p, "recall": r}
    return {"optimal": best, "sweep": sweep}


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


# ── Fallback data (used when GitHub fetch fails) ──────────────────────────────

_HALUEVAL_QA_FALLBACK = [
    {"question": "What is the capital of France?",
     "knowledge": "France is a country in Western Europe. Its capital city is Paris.",
     "right_answer": "The capital of France is Paris.",
     "hallucinated_answer": "The capital of France is Lyon."},
    {"question": "Who wrote Romeo and Juliet?",
     "knowledge": "Romeo and Juliet is a tragedy written by William Shakespeare, believed to have been written between 1594 and 1596.",
     "right_answer": "Romeo and Juliet was written by William Shakespeare.",
     "hallucinated_answer": "Romeo and Juliet was written by Christopher Marlowe in 1590."},
    {"question": "What is the speed of light?",
     "knowledge": "The speed of light in a vacuum is exactly 299,792,458 metres per second.",
     "right_answer": "The speed of light is approximately 299,792,458 metres per second.",
     "hallucinated_answer": "The speed of light is approximately 150,000 kilometres per second."},
]

_HALUEVAL_SUMM_FALLBACK = [
    {"document": "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest.",
     "right_summary": "The Amazon rainforest covers most of the Amazon basin in South America, spanning about 5.5 million km2.",
     "hallucinated_summary": "The Amazon rainforest is located primarily in Africa and covers about 3 million km2."},
]


if __name__ == "__main__":
    main()
