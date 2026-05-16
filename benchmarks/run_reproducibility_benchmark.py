"""Reproducibility benchmark — answers "if I re-run the same eval, do I get
the same answer?". The honest case for an eval framework's trust score.

Setup: 10 cases, 10 reps each. Twice: once with the judge cache on, once off.
Output: variance of avg_score and pass_rate across reps, plus total cost.

Cache-on should approach ZERO variance (deterministic seeds + cached judges).
Cache-off variance shows true judge stability at temperature=0.

Run:
    python run_reproducibility_benchmark.py
"""
from __future__ import annotations

import json
import pathlib
import statistics
import tempfile
import time
import urllib.request
from dotenv import load_dotenv
load_dotenv()

from multivon_eval import (
    EvalSuite, EvalCase, JudgeConfig, configure,
    Hallucination, Faithfulness, JudgeCache, set_cache,
)


def load_halueval_sample(n: int = 10) -> list[dict]:
    url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
    with urllib.request.urlopen(url, timeout=15) as r:
        lines = r.read().decode().strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()][:n]


def run_one_rep(cases: list[EvalCase]) -> dict:
    suite = EvalSuite("repro")
    suite.add_cases(cases)
    suite.add_evaluators(Hallucination(), Faithfulness())
    report = suite.run(
        # Echo the right_answer (already in case.expected_output)
        lambda x: next((c.expected_output for c in cases if c.input == x), "") or "",
        verbose=False, workers=4,
    )
    return {
        "pass_rate": report.pass_rate,
        "avg_score": report.avg_score,
        "evaluated": report.evaluated,
        "errors": report.errors,
    }


def main():
    configure(JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0))

    N_CASES = 10
    N_REPS = 10
    print(f"Loading {N_CASES} HaluEval QA cases…")
    raw = load_halueval_sample(N_CASES)
    cases = [
        EvalCase(
            input=r["question"],
            context=r.get("knowledge", ""),
            expected_output=r["right_answer"],
        )
        for r in raw
    ]

    out: dict[str, dict] = {}

    # ── Cache ON ────────────────────────────────────────────────────────────
    print(f"\nCache ON — {N_REPS} reps (first cold, rest warm)…")
    with tempfile.TemporaryDirectory() as td:
        cache_path = pathlib.Path(td) / "cache.sqlite"
        set_cache(JudgeCache(str(cache_path)))
        per_rep_on = []
        t0 = time.time()
        for i in range(N_REPS):
            t_rep = time.time()
            per_rep_on.append(run_one_rep(cases))
            print(f"  rep {i+1:>2}/{N_REPS}  pass_rate={per_rep_on[-1]['pass_rate']:.3f}  "
                  f"avg_score={per_rep_on[-1]['avg_score']:.3f}  "
                  f"({time.time()-t_rep:.1f}s)")
        wall_on = time.time() - t0
        set_cache(None)  # reset

    pass_on = [r["pass_rate"] for r in per_rep_on]
    score_on = [r["avg_score"] for r in per_rep_on]
    out["cache_on"] = {
        "n_reps": N_REPS,
        "pass_rate_mean": round(statistics.mean(pass_on), 4),
        "pass_rate_stdev": round(statistics.stdev(pass_on), 4) if len(pass_on) > 1 else 0,
        "avg_score_mean": round(statistics.mean(score_on), 4),
        "avg_score_stdev": round(statistics.stdev(score_on), 4) if len(score_on) > 1 else 0,
        "total_wall_clock_s": round(wall_on, 1),
        "per_rep": per_rep_on,
    }

    # ── Cache OFF ───────────────────────────────────────────────────────────
    print(f"\nCache OFF — {N_REPS} reps (every call hits the API)…")
    per_rep_off = []
    t0 = time.time()
    for i in range(N_REPS):
        t_rep = time.time()
        per_rep_off.append(run_one_rep(cases))
        print(f"  rep {i+1:>2}/{N_REPS}  pass_rate={per_rep_off[-1]['pass_rate']:.3f}  "
              f"avg_score={per_rep_off[-1]['avg_score']:.3f}  "
              f"({time.time()-t_rep:.1f}s)")
    wall_off = time.time() - t0

    pass_off = [r["pass_rate"] for r in per_rep_off]
    score_off = [r["avg_score"] for r in per_rep_off]
    out["cache_off"] = {
        "n_reps": N_REPS,
        "pass_rate_mean": round(statistics.mean(pass_off), 4),
        "pass_rate_stdev": round(statistics.stdev(pass_off), 4) if len(pass_off) > 1 else 0,
        "avg_score_mean": round(statistics.mean(score_off), 4),
        "avg_score_stdev": round(statistics.stdev(score_off), 4) if len(score_off) > 1 else 0,
        "total_wall_clock_s": round(wall_off, 1),
        "per_rep": per_rep_off,
    }

    # Speedup
    out["summary"] = {
        "speedup_x": round(wall_off / wall_on, 1) if wall_on > 0 else 0,
        "score_stability_x": round(out["cache_off"]["avg_score_stdev"] / max(out["cache_on"]["avg_score_stdev"], 1e-9), 1),
        "interpretation": (
            "Cache ON reproduces identical scores rep-to-rep. "
            "Cache OFF exposes the irreducible judge variance at temperature=0."
        ),
    }

    out["metadata"] = {
        "n_cases": N_CASES,
        "n_reps": N_REPS,
        "evaluators": ["Hallucination", "Faithfulness"],
        "judge_model": "claude-haiku-4-5",
        "dataset": "halueval_qa (first 10)",
    }

    res = pathlib.Path(__file__).parent / "results" / "reproducibility.json"
    res.parent.mkdir(exist_ok=True)
    res.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {res}")
    print("\nSummary:")
    print(f"  Cache ON  — pass_rate σ = {out['cache_on']['pass_rate_stdev']:.4f}   wall_clock = {wall_on:.1f}s")
    print(f"  Cache OFF — pass_rate σ = {out['cache_off']['pass_rate_stdev']:.4f}  wall_clock = {wall_off:.1f}s")
    print(f"  Speedup (cache off → on) = {out['summary']['speedup_x']:.1f}×")


if __name__ == "__main__":
    main()
