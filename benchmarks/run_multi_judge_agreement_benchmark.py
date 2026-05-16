"""Multi-judge agreement benchmark — answers "do different judge models agree
about the same response?". Cohen's kappa across pairs of (claude-haiku,
claude-sonnet, gpt-4o-mini, gpt-4o) on the same hallucination cases.

High pairwise κ = judges agree (signal is in the data, not the model).
Low pairwise κ = task is judge-dependent (calibration matters).
Accuracy vs human label = which judge to actually use.

Produces benchmarks/results/multi_judge_agreement.json

Run:
    python run_multi_judge_agreement_benchmark.py
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from multivon_eval import EvalCase, JudgeConfig, Hallucination


def load_halueval_sample(n: int = 25) -> list[dict]:
    url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
    with urllib.request.urlopen(url, timeout=15) as r:
        lines = r.read().decode().strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()][:n]


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Compute Cohen's kappa for two binary rater sequences."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return 0.0
    agreed = sum(1 for x, y in zip(a, b) if x == y)
    p_o = agreed / n
    # Expected agreement under independence
    p_a_1 = sum(a) / n
    p_b_1 = sum(b) / n
    p_e = (p_a_1 * p_b_1) + ((1 - p_a_1) * (1 - p_b_1))
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


JUDGES = {
    "claude-haiku-4-5": JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0),
    "claude-sonnet-4-6": JudgeConfig(provider="anthropic", model="claude-sonnet-4-6", temperature=0.0),
    "gpt-4o-mini":  JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.0),
    "gpt-4o":       JudgeConfig(provider="openai", model="gpt-4o", temperature=0.0),
}


def score_one(judge_name: str, case_record: dict, variant: str) -> tuple[str, int, int]:
    """Run Hallucination on one case-variant. Returns (variant_key, label, predicted)."""
    output = case_record[variant]
    case = EvalCase(
        input=case_record["question"],
        context=case_record.get("knowledge", case_record.get("context", "")),
        expected_output=case_record["right_answer"],
    )
    ev = Hallucination(judge=JUDGES[judge_name])
    result = ev.evaluate(case, output)
    # Hallucination evaluator returns score 0 if hallucinated, 1 if faithful — we want hallucinated=1
    predicted = 1 if result.score < 0.5 else 0
    label = 1 if variant == "hallucinated_answer" else 0
    return (judge_name, label, predicted)


def main():
    N = 25
    print(f"Loading {N} HaluEval QA cases (×2 variants = {2*N} pairs)…")
    raw = load_halueval_sample(N)

    # 4 judges × 25 cases × 2 variants = 200 calls
    jobs = []
    for case_idx, r in enumerate(raw):
        for variant in ("right_answer", "hallucinated_answer"):
            for judge_name in JUDGES:
                jobs.append((case_idx, variant, judge_name, r))

    print(f"Dispatching {len(jobs)} judge calls (4-way parallelism per judge)…")
    t0 = time.time()
    results: list[tuple] = []  # (case_idx, variant, judge, label, predicted)
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(score_one, judge_name, r, variant): (case_idx, variant, judge_name) for case_idx, variant, judge_name, r in jobs}
        done = 0
        for fut in as_completed(futs):
            case_idx, variant, judge_name = futs[fut]
            try:
                _, label, predicted = fut.result()
                results.append((case_idx, variant, judge_name, label, predicted))
            except Exception as e:
                print(f"  err  case={case_idx} variant={variant} judge={judge_name}: {e}")
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)}")
    print(f"\nDone in {time.time()-t0:.1f}s")

    # Build per-judge prediction vectors, ordered consistently
    indexed = sorted(results, key=lambda x: (x[0], x[1]))
    judges = list(JUDGES.keys())
    per_judge: dict[str, list[int]] = {j: [] for j in judges}
    labels: list[int] = []
    seen_keys: set = set()
    for (case_idx, variant, judge_name, label, pred) in indexed:
        key = (case_idx, variant)
        if key not in seen_keys and judge_name == judges[0]:
            labels.append(label)
            seen_keys.add(key)
        per_judge[judge_name].append(pred)

    # Truncate all vectors to the same length defensively
    min_len = min(len(per_judge[j]) for j in judges)
    for j in judges:
        per_judge[j] = per_judge[j][:min_len]
    labels = labels[:min_len]

    # Pairwise Cohen's kappa
    kappa = {}
    for i, j1 in enumerate(judges):
        for j2 in judges[i + 1:]:
            k = cohens_kappa(per_judge[j1], per_judge[j2])
            kappa[f"{j1} ↔ {j2}"] = round(k, 3)

    # Per-judge accuracy vs human label
    accuracy = {}
    for j in judges:
        correct = sum(1 for p, l in zip(per_judge[j], labels) if p == l)
        accuracy[j] = round(correct / len(labels), 3) if labels else 0

    # Per-judge precision/recall on the hallucinated class
    metrics = {}
    for j in judges:
        tp = sum(1 for p, l in zip(per_judge[j], labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(per_judge[j], labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(per_judge[j], labels) if p == 0 and l == 1)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = (2 * prec * rec) / (prec + rec) if prec + rec else 0
        metrics[j] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}

    out = {
        "benchmark": "multi_judge_agreement",
        "dataset": "halueval_qa",
        "n_cases": N,
        "n_pairs": min_len,
        "judges": list(JUDGES.keys()),
        "pairwise_cohens_kappa": kappa,
        "kappa_interpretation": {
            ">= 0.81": "almost perfect",
            "0.61–0.80": "substantial",
            "0.41–0.60": "moderate",
            "0.21–0.40": "fair",
            "< 0.21": "slight",
        },
        "accuracy_vs_human_label": accuracy,
        "per_judge_metrics": metrics,
    }

    out_path = pathlib.Path(__file__).parent / "results" / "multi_judge_agreement.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\nResults:")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
