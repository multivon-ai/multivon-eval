"""
SummEval benchmark — Spearman correlation between multivon-eval scores and
expert human annotations on machine-generated summaries.

Dataset: SummEval via mteb/summeval on HuggingFace (Fabbri et al., 2021).
100 CNN/DailyMail articles × 16 machine-generated summaries each = 1,600
evaluation units. Human expert annotations (averaged across 3 annotators)
for coherence, consistency, fluency, relevance on a 1-5 scale.

Evaluator mapping:
  coherence    → Coherence    (structural quality)
  relevance    → Relevance    (summary vs. source article)
  consistency  → Faithfulness (factual alignment with source)

Fluency has no direct evaluator and is omitted.

Usage:
  pip install multivon-eval datasets python-dotenv
  export ANTHROPIC_API_KEY=sk-ant-...
  export OPENAI_API_KEY=sk-...
  python benchmarks/run_summeval_benchmark.py
  python benchmarks/run_summeval_benchmark.py --n 100 --judge openai/gpt-3.5-turbo
  python benchmarks/run_summeval_benchmark.py --output benchmarks/results/summeval_sonnet.json

Estimated cost (default 100 samples, 3 evaluators):
  Haiku:   ~300 calls × ~$0.001  ≈ $0.30
  Sonnet:  ~300 calls × ~$0.003  ≈ $0.90
  gpt-4o-mini: ~300 calls × ~$0.0002 ≈ $0.06
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


# ── dataset ──────────────────────────────────────────────────────────────────

def _load_hf(n: int, seed: int = 42) -> list[dict]:
    """
    Load SummEval from HuggingFace and flatten to (article, summary, scores) rows.
    Each article has 16 machine summaries; we flatten then sample n.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets package required: pip install datasets", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset("mteb/summeval", split="test")

    rows = []
    for article in ds:
        text = article["text"]
        for i, summary in enumerate(article["machine_summaries"]):
            rows.append({
                "text": text,
                "summary": summary,
                "coherence":    article["coherence"][i],
                "relevance":    article["relevance"][i],
                "consistency":  article["consistency"][i],
            })

    import random
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


# ── statistics ────────────────────────────────────────────────────────────────

def _rank(vals: list[float]) -> list[float]:
    n = len(vals)
    idx = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[idx[j + 1]] == vals[idx[j]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman rho + two-tailed p (t→normal approx, accurate for n≥30)."""
    n = len(x)
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denom = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    rho = max(-1.0, min(1.0, num / denom if denom else 0.0))
    if abs(rho) >= 1.0:
        return round(rho, 4), 0.0
    t = rho * math.sqrt(n - 2) / math.sqrt(1 - rho ** 2)
    p = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return round(rho, 4), round(max(0.0, p), 4)


# ── evaluation ────────────────────────────────────────────────────────────────

_INPUT_PROMPT = "Summarize the following document."


def _eval_row(row: dict, coh_ev, rel_ev, faith_ev) -> dict:
    from multivon_eval import EvalCase

    case = EvalCase(input=_INPUT_PROMPT, context=row["text"])
    result = {
        "coh_human": row["coherence"],
        "rel_human": row["relevance"],
        "faith_human": row["consistency"],
    }

    try:
        result["coh_model"] = coh_ev.evaluate(case, row["summary"]).score
    except Exception as e:
        result["coh_error"] = str(e)

    try:
        result["rel_model"] = rel_ev.evaluate(case, row["summary"]).score
    except Exception as e:
        result["rel_error"] = str(e)

    try:
        result["faith_model"] = faith_ev.evaluate(case, row["summary"]).score
    except Exception as e:
        result["faith_error"] = str(e)

    return result


def run_benchmark(n: int, provider: str, model: str, workers: int, verbose: bool) -> dict:
    try:
        from multivon_eval import configure, JudgeConfig, Coherence, Relevance, Faithfulness
    except ImportError:
        print("multivon-eval not installed. Run: pip install multivon-eval", file=sys.stderr)
        sys.exit(1)

    configure(JudgeConfig(provider=provider, model=model))
    coh_ev   = Coherence()
    rel_ev   = Relevance()
    faith_ev = Faithfulness()

    if verbose:
        print(f"  Loading SummEval from HuggingFace...", end=" ", flush=True)
    rows = _load_hf(n)
    if verbose:
        print(f"done ({len(rows)} samples)")
        print(f"  Running {len(rows)} samples × 3 evaluators (workers={workers})...\n")

    scores = {k: [] for k in ("coh", "rel", "faith")}
    human  = {k: [] for k in ("coh", "rel", "faith")}
    errors = 0
    done   = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_eval_row, row, coh_ev, rel_ev, faith_ev): i
            for i, row in enumerate(rows)
        }
        for fut in as_completed(futures):
            done += 1
            try:
                res = fut.result()
            except Exception as exc:
                errors += 1
                if verbose:
                    print(f"\r  [{done:4d}/{n}] error: {exc}", end="", flush=True)
                continue

            for dim, key in [("coh", "coh"), ("rel", "rel"), ("faith", "faith")]:
                if f"{key}_model" in res:
                    scores[dim].append(res[f"{key}_model"])
                    human[dim].append(res[f"{key}_human"])
                else:
                    errors += 1

            if verbose:
                ns = {k: len(v) for k, v in scores.items()}
                print(f"\r  [{done:4d}/{n}] coh={ns['coh']} rel={ns['rel']} faith={ns['faith']}", end="", flush=True)

    if verbose:
        print()

    dim_labels = {"coh": "coherence", "rel": "relevance", "faith": "faithfulness"}
    results = {}
    for key, label in dim_labels.items():
        xs, ys = scores[key], human[key]
        if len(xs) < 20:
            if verbose:
                print(f"  [SKIP] {label}: only {len(xs)} valid samples", file=sys.stderr)
            continue
        rho, p = spearman(xs, ys)
        results[label] = {"rho": rho, "p": p, "n": len(xs)}

    return {"results": results, "errors": errors, "n_requested": n}


# ── output ────────────────────────────────────────────────────────────────────

def _interp(rho: float) -> str:
    a = abs(rho)
    if a >= 0.7: return "strong"
    if a >= 0.5: return "moderate"
    if a >= 0.3: return "weak"
    return "negligible"


def print_table(data: dict, judge: str) -> None:
    try:
        import multivon_eval
        version = f"v{multivon_eval.__version__}"
    except Exception:
        version = ""

    results = data["results"]
    n_req   = data["n_requested"]
    errors  = data["errors"]

    print(f"\n{'─' * 66}")
    print(f"  SummEval Benchmark — multivon-eval {version}")
    print(f"  Judge: {judge}")
    print(f"  Dataset: SummEval (Fabbri et al., 2021) — expert human annotations, 1–5 scale")
    print(f"{'─' * 66}")
    print(f"  {'Dimension':<14}  {'Spearman ρ':>10}  {'p-value':>8}  {'n':>5}  {'':>12}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*8}  {'─'*5}  {'─'*12}")
    for dim, r in results.items():
        p_str  = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        interp = _interp(r["rho"])
        print(f"  {dim:<14}  {r['rho']:>10.4f}  {p_str:>8}  {r['n']:>5}  {interp:>12}")
    print(f"{'─' * 66}")
    if errors:
        print(f"  Errors: {errors} / {n_req * 3} eval calls failed")
    print()
    print("  Notes:")
    print("  coherence     — structural quality of the summary text")
    print("  relevance     — does the summary capture key info from the source article")
    print("  faithfulness  — factual consistency with the source article")
    print()
    print("  RAGAS comparison (different dataset — directional only):")
    print("  RAGAS WikiEval (gpt-3.5-turbo-16k, Sept 2023):")
    print("    Faithfulness≈95%, AnswerRelevance≈78%, ContextRelevance≈70%")
    print("  Note: those are agreement % on a proprietary dataset, not Spearman ρ.")
    print(f"{'─' * 66}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="SummEval Spearman benchmark for multivon-eval")
    ap.add_argument("--n",       type=int, default=100,
                    help="Samples to evaluate (default 100; max 1600)")
    ap.add_argument("--judge",   default="anthropic/claude-haiku-4-5-20251001",
                    help="provider/model (default: anthropic/claude-haiku-4-5-20251001)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel eval workers (default 4)")
    ap.add_argument("--output",  help="Save raw results as JSON")
    ap.add_argument("--quiet",   action="store_true")
    args = ap.parse_args()

    if "/" not in args.judge:
        print("--judge must be provider/model, e.g. anthropic/claude-sonnet-4-6", file=sys.stderr)
        sys.exit(1)
    provider, model = args.judge.split("/", 1)

    verbose = not args.quiet
    if verbose:
        costs = {"claude-haiku-4-5-20251001": 0.001, "claude-sonnet-4-6": 0.003,
                 "gpt-4o-mini": 0.0002, "gpt-3.5-turbo": 0.0005, "gpt-4o": 0.005}
        cpp = costs.get(model, 0.002)
        print(f"\n  SummEval: {args.n} samples × 3 evaluators = {args.n * 3} API calls")
        print(f"  Judge: {args.judge}")
        print(f"  Estimated cost: ~${args.n * 3 * cpp:.2f}")

    data = run_benchmark(args.n, provider, model, args.workers, verbose)
    print_table(data, args.judge)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"judge": args.judge, "n_requested": args.n, **data}, indent=2))
        if verbose:
            print(f"  Results saved → {args.output}")


if __name__ == "__main__":
    main()
