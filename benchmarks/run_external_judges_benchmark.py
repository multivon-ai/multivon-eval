"""External-judges benchmark — Patronus Lynx, Prometheus-2, Vectara HHEM.

This is the Phase 0, Week 1 "decide or pivot" task from the strategy plan.
If any of these open-weights / specialist judges land within ~3 F1 points of
where a hypothetical multivon-judge-1b would credibly land, the in-house
distillation thesis is already addressed by an existing OSS competitor and
the strategy needs to re-plan.

This script DOES NOT require any GPU on the host. It calls each external
judge through whatever hosted endpoint that judge offers (Patronus's API,
Together / Replicate for Prometheus-2, Vectara's HHEM API). Each path is
gated on its own env var; the script gracefully skips judges that aren't
reachable.

Run:
    cd benchmarks
    pip install -e ..
    # then export whichever of these you have access to:
    export PATRONUS_API_KEY=...          # for Lynx (https://www.patronus.ai/)
    export TOGETHER_API_KEY=...          # for Prometheus-2 (Together hosts the weights)
    export VECTARA_API_KEY=...           # for HHEM (https://vectara.com/)
    export ANTHROPIC_API_KEY=...         # for the existing in-house judges
    export OPENAI_API_KEY=...
    export GOOGLE_API_KEY=...
    python run_external_judges_benchmark.py
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from multivon_eval import EvalCase, JudgeConfig, Hallucination


# ─── Dataset (same as the in-house multi-judge benchmark) ────────────────────

def load_halueval_sample(n: int = 25) -> list[dict]:
    url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
    with urllib.request.urlopen(url, timeout=15) as r:
        lines = r.read().decode().strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()][:n]


# ─── In-house judges (reuse the existing config) ─────────────────────────────

IN_HOUSE_JUDGES: dict[str, JudgeConfig] = {
    "claude-haiku-4-5":   JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0),
    "claude-sonnet-4-6":  JudgeConfig(provider="anthropic", model="claude-sonnet-4-6", temperature=0.0),
    "gpt-4o-mini":        JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.0),
    "gpt-4o":             JudgeConfig(provider="openai", model="gpt-4o", temperature=0.0),
    "gemini-2.5-flash":   JudgeConfig(provider="google", model="gemini-2.5-flash", temperature=0.0),
}


# ─── External-judge adapters ─────────────────────────────────────────────────
# Each adapter takes (question, context, output_to_evaluate) and returns a
# float score in [0.0, 1.0] where 1.0 = faithful (no hallucination). We
# deliberately normalize to multivon-eval's Hallucination convention so the
# downstream F1/accuracy code is shared across in-house + external judges.


def _patronus_lynx_score(question: str, context: str, output: str) -> float | None:
    """Patronus Lynx judge via Patronus API.

    Lynx is an 8B specialist judge for hallucination detection. Patronus
    exposes it via their REST API. Docs: https://docs.patronus.ai/
    Returns None if API key absent OR the call fails — caller treats None
    as "judge unavailable" and excludes from comparison.
    """
    api_key = os.environ.get("PATRONUS_API_KEY")
    if not api_key:
        return None
    try:
        import urllib.request
        body = json.dumps({
            "model": "lynx-large",
            "evaluators": [{"evaluator": "lynx", "criteria": "patronus:hallucination"}],
            "evaluated_model_input": question,
            "evaluated_model_output": output,
            "evaluated_model_retrieved_context": context,
        }).encode()
        req = urllib.request.Request(
            "https://api.patronus.ai/v1/evaluate",
            data=body,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        # Patronus returns boolean "PASS" / "FAIL" per evaluator. PASS = faithful.
        results = data.get("results") or data.get("evaluation_results") or []
        if not results:
            return None
        passed = results[0].get("pass", results[0].get("evaluation_result") == "PASS")
        return 1.0 if passed else 0.0
    except Exception as exc:
        print(f"  [patronus-lynx] call failed: {type(exc).__name__}: {exc}")
        return None


def _prometheus2_score(question: str, context: str, output: str) -> float | None:
    """Prometheus-2 via Together.ai (which hosts the weights as a chat model).

    Prometheus-2 is a 7B/8x7B judge from KAIST. Together.ai serves both at:
        prometheus-eval/prometheus-7b-v2.0
        prometheus-eval/prometheus-8x7b-v2.0

    We prompt it with a faithfulness rubric and parse the 1–5 score it emits.
    """
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        return None
    try:
        import urllib.request
        # Prometheus expects a structured instruction + reference + rubric.
        prompt = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{question}

###Context:
{context}

###Response to evaluate:
{output}

###Score Rubrics:
[Is the response faithful to the context, free of hallucination and unsupported claims?]
Score 1: The response contains substantial hallucinations or unsupported claims, directly contradicting the context.
Score 2: The response is partially supported by the context but contains noticeable unsupported claims.
Score 3: The response is mostly supported by the context with minor unsupported additions.
Score 4: The response is well-supported by the context with negligible deviations.
Score 5: The response is fully faithful to the context with no hallucinations.

###Feedback:"""
        body = json.dumps({
            "model": "prometheus-eval/prometheus-7b-v2.0",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
        }).encode()
        req = urllib.request.Request(
            "https://api.together.xyz/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        # Parse [RESULT] N
        import re
        m = re.search(r"\[RESULT\]\s*(\d)", text)
        if not m:
            return None
        likert = int(m.group(1))
        # Map Likert 1-5 → [0.0, 1.0]. 4 or 5 = faithful (pass).
        return 1.0 if likert >= 4 else 0.0
    except Exception as exc:
        print(f"  [prometheus-2] call failed: {type(exc).__name__}: {exc}")
        return None


def _vectara_hhem_score(question: str, context: str, output: str) -> float | None:
    """Vectara HHEM via Vectara API.

    HHEM (Hughes Hallucination Evaluation Model) is a small specialist model
    Vectara published. Their hosted API takes (premise=context, hypothesis=output)
    and returns a [0,1] consistency score. We treat ≥0.5 as faithful.
    """
    api_key = os.environ.get("VECTARA_API_KEY")
    if not api_key:
        return None
    try:
        import urllib.request
        body = json.dumps({
            "model": "hhem-v2.1",
            "premise": context,
            "hypothesis": output,
        }).encode()
        # Vectara's HHEM v2 endpoint — the exact path is documented at
        # https://docs.vectara.com/docs/api-reference/predict-apis/predict-hhem
        # Adjust if Vectara has moved it.
        req = urllib.request.Request(
            "https://api.vectara.io/v1/hhem",
            data=body,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        # Vectara returns a single consistency score in [0, 1].
        score = data.get("score") or data.get("consistency_score")
        if score is None:
            return None
        return 1.0 if float(score) >= 0.5 else 0.0
    except Exception as exc:
        print(f"  [vectara-hhem] call failed: {type(exc).__name__}: {exc}")
        return None


EXTERNAL_JUDGES = {
    "patronus-lynx-large": _patronus_lynx_score,
    "prometheus-2-7b":     _prometheus2_score,
    "vectara-hhem-v2.1":   _vectara_hhem_score,
}


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_in_house(judge_name: str, case_record: dict, variant: str) -> tuple[int, int]:
    """Run multivon-eval's Hallucination evaluator with the given JudgeConfig.

    Returns (label, predicted) where label=1 means the variant is hallucinated.
    """
    output = case_record[variant]
    case = EvalCase(
        input=case_record["question"],
        context=case_record.get("knowledge", case_record.get("context", "")),
        expected_output=case_record["right_answer"],
    )
    ev = Hallucination(judge=IN_HOUSE_JUDGES[judge_name])
    result = ev.evaluate(case, output)
    predicted = 1 if result.score < 0.5 else 0
    label = 1 if variant == "hallucinated_answer" else 0
    return (label, predicted)


def score_external(judge_name: str, case_record: dict, variant: str) -> tuple[int, int] | None:
    """Run an external judge. Returns None if the judge is unreachable."""
    output = case_record[variant]
    score = EXTERNAL_JUDGES[judge_name](
        case_record["question"],
        case_record.get("knowledge", case_record.get("context", "")),
        output,
    )
    if score is None:
        return None
    # In external-judge convention: 1.0 = faithful. multivon-eval convention:
    # 1 = hallucinated. Invert.
    predicted = 0 if score >= 0.5 else 1
    label = 1 if variant == "hallucinated_answer" else 0
    return (label, predicted)


# ─── Metrics (same shape as the in-house multi-judge benchmark) ──────────────

def metrics_for(labels: list[int], predictions: list[int]) -> dict:
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    accuracy = (tp + tn) / max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    N = 25
    print(f"Loading {N} HaluEval QA cases (×2 variants = {2*N} pairs)…\n")
    raw = load_halueval_sample(N)
    case_variants = [(case_idx, variant) for case_idx in range(len(raw)) for variant in ("right_answer", "hallucinated_answer")]

    # Probe which external judges are actually reachable. Saves the user from
    # waiting through obvious "no key" failures.
    reachable_external: list[str] = []
    for name, fn in EXTERNAL_JUDGES.items():
        # Smoke test against the first case.
        smoke = fn(raw[0]["question"], raw[0].get("knowledge", ""), raw[0]["right_answer"])
        if smoke is None:
            print(f"  ✗ {name:<25} unreachable — skipping (set API key to include)")
        else:
            print(f"  ✓ {name:<25} reachable")
            reachable_external.append(name)
    print()

    if not reachable_external and not any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")):
        print("No judges configured. Set at least one API key to run this benchmark.")
        return

    # All judges to run (in-house first because they're already wired).
    judges = list(IN_HOUSE_JUDGES.keys()) + reachable_external

    # Build the work matrix: (case_idx, variant, judge_name).
    jobs = [(case_idx, variant, judge) for case_idx, variant in case_variants for judge in judges]
    print(f"Dispatching {len(jobs)} judge calls across {len(judges)} judges…\n")

    t0 = time.time()
    results: list[tuple[int, str, str, int, int]] = []  # (case_idx, variant, judge, label, predicted)

    def run_one(case_idx: int, variant: str, judge: str):
        r = raw[case_idx]
        if judge in IN_HOUSE_JUDGES:
            return (case_idx, variant, judge, *score_in_house(judge, r, variant))
        out = score_external(judge, r, variant)
        if out is None:
            return None
        return (case_idx, variant, judge, *out)

    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(run_one, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as exc:
                ci, var, j = futs[fut]
                print(f"  err  case={ci} variant={var} judge={j}: {exc}")
                done += 1
                continue
            done += 1
            if r is not None:
                results.append(r)
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}")

    print(f"\nCompleted in {time.time()-t0:.1f}s\n")

    # Per-judge metrics — ordered consistently across judges so labels align.
    indexed = sorted(results, key=lambda x: (x[0], x[1], x[2]))

    per_judge_metrics: dict[str, dict] = {}
    judge_labels: dict[str, list[int]] = {j: [] for j in judges}
    judge_preds: dict[str, list[int]] = {j: [] for j in judges}
    for (_, _, judge, label, predicted) in indexed:
        judge_labels[judge].append(label)
        judge_preds[judge].append(predicted)

    for judge in judges:
        if judge_labels[judge]:
            per_judge_metrics[judge] = metrics_for(judge_labels[judge], judge_preds[judge])

    # Output.
    out = {
        "benchmark": "external_judges_vs_in_house",
        "dataset": "halueval_qa",
        "n_cases": N,
        "n_pairs": min((len(judge_labels[j]) for j in judges if judge_labels[j]), default=0),
        "judges_in_house": list(IN_HOUSE_JUDGES.keys()),
        "judges_external": reachable_external,
        "per_judge_metrics": per_judge_metrics,
        "notes": [
            "All judges run the same Hallucination task on identical HaluEval QA case-variants.",
            "External judges normalize their scoring convention to multivon-eval's "
            "Hallucination convention (1 = hallucinated). See score_external().",
            "External judges are gracefully skipped if their API key is not set; "
            "see the 'reachable' list above.",
            "Decision rule from the strategy plan: if any external judge is within "
            "3 F1 points of the projected multivon-judge-1b landing zone (≥0.75), "
            "the distillation thesis needs to be re-evaluated.",
        ],
    }

    out_path = pathlib.Path(__file__).parent / "results" / "external_judges.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}\n")

    # Console summary, sorted by F1.
    print("Per-judge metrics (sorted by F1 descending):\n")
    print(f"  {'judge':<28} {'acc':>6} {'prec':>6} {'recall':>7} {'F1':>6}")
    for j, m in sorted(per_judge_metrics.items(), key=lambda kv: kv[1]["f1"], reverse=True):
        mark = " ←" if j in reachable_external else ""
        print(f"  {j:<28} {m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>7.3f} {m['f1']:>6.3f}{mark}")
    print("\n  ← = external judge")


if __name__ == "__main__":
    main()
