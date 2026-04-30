"""
SummEval benchmark — Spearman correlation between multivon-eval scores and
expert human annotations on machine-generated summaries.

Dataset: SummEval (Fabbri et al., 2021). 1,600 CNN/DailyMail summaries from 16
models, each rated by 3 expert annotators on coherence, consistency, fluency,
relevance (1-5 scale).

Evaluator mapping:
  coherence  → Coherence  (structural quality; no source article needed)
  relevance  → Relevance  (machine summary vs. human reference as context proxy)

Consistency/faithfulness requires the original CNN/DailyMail source articles,
which are not bundled in the SummEval annotation file. That dimension is skipped.

RAGAS comparison note:
  RAGAS published on WikiEval (their own dataset) using gpt-3.5-turbo-16k.
  That dataset is not public. SummEval is an independent third-party benchmark —
  the only public neutral ground for this kind of comparison.

Usage:
  pip install multivon-eval python-dotenv
  export ANTHROPIC_API_KEY=sk-ant-...          # or OPENAI_API_KEY for --judge openai/...
  python benchmarks/run_summeval_benchmark.py
  python benchmarks/run_summeval_benchmark.py --n 100 --judge openai/gpt-3.5-turbo
  python benchmarks/run_summeval_benchmark.py --output benchmarks/results/summeval.json

Estimated cost (default 200 samples, Haiku):
  ~400 API calls × ~$0.001 each ≈ $0.40
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


# ── dataset ──────────────────────────────────────────────────────────────────

_DATA_URLS = [
    "https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.paired.jsonl",
    "https://raw.githubusercontent.com/Yale-LILY/SummEval/master/data/model_annotations.aligned.paired.jsonl",
]
_CACHE = Path.home() / ".cache" / "multivon-bench" / "summeval.jsonl"


def _ensure_data(verbose: bool) -> None:
    if _CACHE.exists():
        if verbose:
            print(f"  Using cached dataset ({_CACHE.stat().st_size / 1e6:.1f} MB): {_CACHE}")
        return
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    for url in _DATA_URLS:
        try:
            if verbose:
                print(f"  Downloading SummEval from {url} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, _CACHE)
            if verbose:
                print(f"done ({_CACHE.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as exc:
            if verbose:
                print(f"failed ({exc})")
            _CACHE.unlink(missing_ok=True)
    print("ERROR: could not download SummEval. Check your internet connection.", file=sys.stderr)
    sys.exit(1)


def _load(n: int, seed: int = 42) -> list[dict]:
    rows = []
    with open(_CACHE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    import random
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def _avg_expert(row: dict, dim: str) -> float | None:
    annotations = row.get("expert_annotations", [])
    vals = [a[dim] for a in annotations if dim in a]
    return sum(vals) / len(vals) if vals else None


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
    """Returns (rho, p_value). p_value uses t→normal approx; accurate for n≥30."""
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


def _eval_row(
    row: dict,
    coh_ev,
    rel_ev,
    EvalCase,
) -> dict | None:
    summary = row.get("decoded", "").strip()
    refs = row.get("references", [])
    context = refs[0].strip() if refs else ""

    if not summary:
        return None

    h_coh = _avg_expert(row, "coherence")
    h_rel = _avg_expert(row, "relevance")

    from multivon_eval.case import EvalCase as _EC

    case = _EC(input=_INPUT_PROMPT, context=context or None)

    result: dict = {}

    if h_coh is not None:
        try:
            r = coh_ev.evaluate(case, summary)
            result["coh_model"] = r.score
            result["coh_human"] = h_coh
        except Exception as exc:
            result["coh_error"] = str(exc)

    if h_rel is not None and context:
        try:
            r = rel_ev.evaluate(case, summary)
            result["rel_model"] = r.score
            result["rel_human"] = h_rel
        except Exception as exc:
            result["rel_error"] = str(exc)

    return result if result else None


def run_benchmark(n: int, provider: str, model: str, workers: int, verbose: bool) -> dict:
    try:
        from multivon_eval import configure, JudgeConfig, Coherence, Relevance
    except ImportError:
        print("multivon-eval not installed. Run: pip install multivon-eval", file=sys.stderr)
        sys.exit(1)

    configure(JudgeConfig(provider=provider, model=model))
    coh_ev = Coherence()
    rel_ev = Relevance()

    _ensure_data(verbose)
    rows = _load(n)

    if verbose:
        print(f"\n  Running {len(rows)} samples × 2 evaluators (workers={workers})...\n")

    coh_model, coh_human = [], []
    rel_model, rel_human = [], []
    errors = 0
    done = 0

    # Rate-limit single-threaded mode; parallel mode relies on API-side throttling.
    from multivon_eval.case import EvalCase

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_eval_row, row, coh_ev, rel_ev, EvalCase): i
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

            if res is None:
                continue

            if "coh_model" in res:
                coh_model.append(res["coh_model"])
                coh_human.append(res["coh_human"])
            else:
                errors += 1

            if "rel_model" in res:
                rel_model.append(res["rel_model"])
                rel_human.append(res["rel_human"])

            if verbose:
                print(f"\r  [{done:4d}/{n}] coherence n={len(coh_model)}  relevance n={len(rel_model)}", end="", flush=True)

    if verbose:
        print()

    results = {}
    for name, xs, ys in [("coherence", coh_model, coh_human), ("relevance", rel_model, rel_human)]:
        if len(xs) < 20:
            if verbose:
                print(f"  [SKIP] {name}: only {len(xs)} valid samples", file=sys.stderr)
            continue
        rho, p = spearman(xs, ys)
        results[name] = {"rho": rho, "p": p, "n": len(xs)}

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
    n_req = data["n_requested"]
    errors = data["errors"]

    print(f"\n{'─' * 62}")
    print(f"  SummEval Benchmark — multivon-eval {version}")
    print(f"  Judge: {judge}")
    print(f"  Dataset: SummEval (Fabbri et al., 2021) — expert annotations")
    print(f"{'─' * 62}")
    print(f"  {'Dimension':<14}  {'Spearman ρ':>10}  {'p-value':>8}  {'n':>5}  {'':>10}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*8}  {'─'*5}  {'─'*10}")
    for dim, r in results.items():
        p_str = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        interp = _interp(r["rho"])
        print(f"  {dim:<14}  {r['rho']:>10.4f}  {p_str:>8}  {r['n']:>5}  {interp:>10}")
    print(f"{'─' * 62}")
    if errors:
        print(f"  Errors: {errors} / {n_req} samples failed")
    print()
    print("  Dimension notes:")
    print("  coherence  — structural quality of summary text (no source needed)")
    print("  relevance  — machine summary vs. first human reference (context proxy)")
    print("  consistency/faithfulness — skipped: requires CNN/DailyMail source articles")
    print()
    print("  RAGAS comparison (different dataset — directional only):")
    print("  RAGAS WikiEval (gpt-3.5-turbo-16k, Sept 2023):")
    print("    Faithfulness=95%, AnswerRelevance=78%, ContextRelevance=70%")
    print("  Note: those are agreement %, not Spearman ρ, on a proprietary dataset.")
    print(f"{'─' * 62}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="SummEval Spearman benchmark for multivon-eval")
    ap.add_argument("--n", type=int, default=200, help="Samples to evaluate (default 200; max ~1600)")
    ap.add_argument("--judge", default="anthropic/claude-haiku-4-5-20251001",
                    help="provider/model (default: anthropic/claude-haiku-4-5-20251001)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4)")
    ap.add_argument("--output", help="Save raw results as JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if "/" not in args.judge:
        print("--judge must be provider/model, e.g. anthropic/claude-haiku-4-5-20251001", file=sys.stderr)
        sys.exit(1)
    provider, model = args.judge.split("/", 1)

    verbose = not args.quiet
    if verbose:
        cost = args.n * 2 * 0.001
        print(f"\n  SummEval benchmark: {args.n} samples × 2 evaluators")
        print(f"  Judge: {args.judge}")
        print(f"  Estimated cost: ~${cost:.2f}")

    data = run_benchmark(args.n, provider, model, args.workers, verbose)
    print_table(data, args.judge)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "judge": args.judge,
            "n_requested": args.n,
            **data,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        if verbose:
            print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
