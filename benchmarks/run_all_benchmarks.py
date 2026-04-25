"""
Run all multivon-eval benchmarks and print a combined summary report.

Usage:
    python benchmarks/run_all_benchmarks.py

Results are written to benchmarks/results/<name>.json and summarized in stdout.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def print_header(title: str):
    print(f"\n{'#'*60}")
    print(f"  {title}")
    print(f"{'#'*60}\n")


def load_result(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def run_all():
    print_header("multivon-eval Full Benchmark Suite")

    from benchmarks.run_hallucination_benchmark import run_benchmark as run_hal
    from benchmarks.run_faithfulness_benchmark import run_benchmark as run_faith
    from benchmarks.run_relevance_benchmark import run_benchmark as run_rel

    results = {}

    print("\n[1/3] Hallucination Detection (HaluEval QA)")
    results["hallucination"] = run_hal(n_samples=50)

    print("\n[2/3] Faithfulness (HaluEval Summarization)")
    results["faithfulness"] = run_faith(n_samples=30)

    print("\n[3/3] Answer Relevance (Golden Set)")
    results["relevance"] = run_rel()

    # Combined summary
    print("\n" + "="*70)
    print("  COMBINED SUMMARY — multivon-eval vs baselines")
    print("="*70)

    EVALUATOR_DISPLAY = {
        "multivon_eval (QAG)": "multivon-eval",
        "multivon_eval (Faithfulness)": "multivon-eval",
        "multivon_eval (Relevance)": "multivon-eval",
        "deepeval (GPT-4o-mini)": "DeepEval",
        "simple_judge (1-10)": "Simple LLM judge",
        "simple_judge": "Simple LLM judge",
        "simple_judge (yes/no)": "Simple LLM judge",
        "keyword_overlap": "Keyword overlap",
    }

    for bench_name, bench_results in results.items():
        print(f"\n  {bench_name.upper()}")
        print(f"  {'Evaluator':<30} {'F1':>8} {'Precision':>10} {'Recall':>10}")
        print(f"  {'-'*60}")
        for ev_name, metrics in bench_results.items():
            display = EVALUATOR_DISPLAY.get(ev_name, ev_name)
            f1 = metrics.get("f1", 0)
            p = metrics.get("precision", 0)
            r = metrics.get("recall", 0)
            print(f"  {display:<30} {f1:>8.3f} {p:>10.3f} {r:>10.3f}")

    print("\n" + "="*70)
    print("  Results saved to benchmarks/results/")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all()
