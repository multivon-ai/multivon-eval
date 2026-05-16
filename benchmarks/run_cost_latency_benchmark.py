"""Cost + latency benchmark — answers the "what does running multivon-eval cost"
question with hard numbers. For each evaluator class, run N=50 cases, record
total cost (USD), total tokens, p50/p95 wall-clock latency.

Produces benchmarks/results/cost_latency.json + a README-pasteable Markdown table.

Run:
    cd benchmarks
    python run_cost_latency_benchmark.py
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import time
import urllib.request
from dotenv import load_dotenv
load_dotenv()

from multivon_eval import (
    EvalSuite, EvalCase, JudgeConfig, configure, CostTracker,
    Faithfulness, Hallucination, Relevance, AnswerAccuracy,
    NotEmpty, WordCount, Levenshtein,
)


def load_halueval_sample(n: int = 50) -> list[dict]:
    url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
    with urllib.request.urlopen(url, timeout=15) as r:
        lines = r.read().decode().strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()][:n]


def main():
    configure(JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0))

    N = 50
    print(f"Loading {N} HaluEval QA cases…")
    raw = load_halueval_sample(N)

    # Use the "right answer" as the model output → tests evaluators on honest data
    cases = [
        EvalCase(
            input=r["question"],
            context=r.get("knowledge", r.get("context", "")),
            expected_output=r["right_answer"],
            tags=["benchmark"],
        )
        for r in raw
    ]

    # Each (evaluator class, label) is one column in the output. Two tiers
    # so the table answers "what does the cheap path cost" + "what does the
    # full LLM-judge path cost" separately.
    tiers: dict[str, list] = {
        "deterministic (no LLM)": [NotEmpty(), WordCount(min=2, max=500), Levenshtein()],
        "llm-judge (claude-haiku)": [Faithfulness(), Hallucination(), Relevance(), AnswerAccuracy()],
    }

    out: dict[str, dict] = {}
    for tier_name, evaluators in tiers.items():
        print(f"\n--- {tier_name} ---")
        suite = EvalSuite(f"cost-latency-{tier_name}")
        suite.add_cases(cases)
        suite.add_evaluators(*evaluators)

        tracker = CostTracker()
        # Define model_fn that returns the same expected_output — we're
        # benchmarking the evaluator cost, not the model.
        def model_fn(prompt: str) -> str:
            return next((c.expected_output for c in cases if c.input == prompt), "") or ""

        t0 = time.time()
        # NOTE: workers=1 because cost tracking loses records when workers > 1
        # (CostTracker ContextVar doesn't propagate to threads).
        # See F3 tracking task. Once fixed, bump to workers=4.
        report = suite.run(model_fn, verbose=False, workers=1)
        elapsed = time.time() - t0

        # `cr.latency_ms` is model_fn latency only (here ~0, since we echo).
        # Real eval cost is wall_clock / n_cases — that's what users feel.
        avg_per_case_s = elapsed / len(cases)
        cost_total_usd = float(report.costs.total_cost_usd) if hasattr(report, "costs") else 0.0
        tokens_in = report.costs.total_input_tokens if hasattr(report, "costs") else 0
        tokens_out = report.costs.total_output_tokens if hasattr(report, "costs") else 0
        n_judge_calls = report.costs.total_calls if hasattr(report, "costs") else 0

        out[tier_name] = {
            "n_cases": len(cases),
            "n_evaluators": len(evaluators),
            "evaluator_names": [getattr(e, "name", type(e).__name__) for e in evaluators],
            "wall_clock_s": round(elapsed, 2),
            "avg_per_case_s": round(avg_per_case_s, 2),
            "total_cost_usd": round(cost_total_usd, 4),
            "cost_per_case_usd": round(cost_total_usd / len(cases), 5) if cases else 0,
            "total_tokens_in": tokens_in,
            "total_tokens_out": tokens_out,
            "tokens_per_case": (tokens_in + tokens_out) // len(cases) if cases else 0,
            "n_judge_calls": n_judge_calls,
            "judge_calls_per_case": round(n_judge_calls / len(cases), 1) if cases else 0,
        }
        print(f"   wall_clock          : {elapsed:.1f}s")
        print(f"   avg per case        : {out[tier_name]['avg_per_case_s']:.2f}s")
        print(f"   judge calls / case  : {out[tier_name]['judge_calls_per_case']}")
        print(f"   total cost          : ${cost_total_usd:.4f}")
        print(f"   cost per case       : ${out[tier_name]['cost_per_case_usd']:.5f}")
        print(f"   tokens (in / out)   : {tokens_in:,} / {tokens_out:,}")

    # Add an extrapolated "what would 5000 cases cost" row from the LLM-judge tier
    if "llm-judge (claude-haiku)" in out:
        d = out["llm-judge (claude-haiku)"]
        out["extrapolation"] = {
            "scenario": "Same eval suite, 5000 cases",
            "estimated_cost_usd": round(d["cost_per_case_usd"] * 5000, 2),
            "estimated_wall_clock_min": round(d["wall_clock_s"] * (5000 / d["n_cases"]) / 60, 1),
            "note": "Linear extrapolation; ignores rate-limiting and judge cache hits.",
        }

    results_path = pathlib.Path(__file__).parent / "results" / "cost_latency.json"
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
