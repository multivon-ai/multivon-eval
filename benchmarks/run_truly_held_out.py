"""
Truly held-out test: Hallucination evaluator (calibrated on HaluEval-QA-100,
threshold=0.55, F1=0.812 in-distribution) tested on HaluEval-Sum n=60.

This is genuinely cross-distribution within the existing calibration data:
the evaluator's threshold was tuned on QA documents (Wikipedia-sourced
short-context QA), then applied without modification to summarization
(CNN/DailyMail-adjacent long-context summary faithfulness).
"""
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from multivon_eval import EvalCase, Hallucination

# Pull HaluEval-Sum (same loader the faithfulness benchmark uses)
URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json"
local = "/tmp/halueval_sum.json"
if not os.path.exists(local):
    urllib.request.urlretrieve(URL, local)

with open(local) as f:
    raw = [json.loads(line) for line in f if line.strip()]

# Take first 30 articles (60 cases total: 30 faithful + 30 hallucinated)
articles = raw[:30]
cases = []
for art in articles:
    # Faithful (label=0)
    cases.append((EvalCase(
        input=f"Summarize: {art['document'][:500]}...",
        expected_output=art['right_summary'],
        context=art['document'],
        metadata={'label': 0, 'summary': art['right_summary']},
    ), 0))
    # Hallucinated (label=1)
    cases.append((EvalCase(
        input=f"Summarize: {art['document'][:500]}...",
        expected_output=art['right_summary'],
        context=art['document'],
        metadata={'label': 1, 'summary': art['hallucinated_summary']},
    ), 1))

print(f"Running Hallucination evaluator on HaluEval-Sum, n={len(cases)} cases")
print(f"Calibration of this evaluator was on HaluEval-QA — this is genuine held-out.")

evaluator = Hallucination()
print(f"Evaluator threshold (frozen, calibrated on HaluEval-QA): {evaluator.threshold}")

def score(case_label):
    case, label = case_label
    try:
        # Hallucination evaluator scores the "output" against "context"
        # We pass the summary as the output
        result = evaluator.evaluate(case, case.metadata['summary'])
        pred = 1 if not result.passed else 0  # passed = faithful (0), not passed = hallucinated (1)
        return label, pred, result.score
    except Exception as e:
        return label, None, str(e)

tp = fp = fn = tn = 0
errors = 0
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(score, cl) for cl in cases]
    for i, fut in enumerate(as_completed(futures), 1):
        label, pred, _ = fut.result()
        if pred is None:
            errors += 1
            continue
        if pred == 1 and label == 1: tp += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 1: fn += 1
        else: tn += 1
        if i % 10 == 0:
            print(f"  {i}/{len(cases)} done")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print()
print(f"=== Hallucination evaluator on HaluEval-Sum n={len(cases)} (TRULY held-out) ===")
print(f"  TP={tp} FP={fp} FN={fn} TN={tn} errors={errors}")
print(f"  precision={precision:.4f}  recall={recall:.4f}  F1={f1:.4f}")

# Save result to JSON
out = {
    "benchmark": "hallucination_held_out_on_halueval_sum",
    "dataset": "halueval_summarization",
    "n_cases": len(cases) - errors,
    "evaluator": "Hallucination",
    "evaluator_calibration_dataset": "HaluEval QA (different task)",
    "evaluator_calibration_threshold": evaluator.threshold,
    "note": "Genuinely held-out: Hallucination threshold calibrated on QA, tested on Summarization. Cross-task within HaluEval corpora.",
    "results": {
        "multivon_eval (Hallucination, held-out)": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    }
}
with open('/Users/siddharthsrivastava/Projects/personal/multivon-eval/benchmarks/results/hallucination_held_out.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to benchmarks/results/hallucination_held_out.json")
