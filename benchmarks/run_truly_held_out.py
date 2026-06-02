"""Truly held-out test: Hallucination on HaluEval-Sum, explicit Haiku
JudgeConfig so the calibrated threshold (0.55 per _calibration_data/v2.json)
is applied. Reports the post-resolve threshold (not the init default)."""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from multivon_eval import EvalCase, Hallucination, JudgeConfig
from multivon_eval.judge import resolve_judge

URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json"
local = "/tmp/halueval_sum.json"
if not os.path.exists(local):
    urllib.request.urlretrieve(URL, local)
with open(local) as f:
    raw = [json.loads(line) for line in f if line.strip()]
articles = raw[:30]
cases = []
for art in articles:
    cases.append((EvalCase(input=f"Summarize: {art['document'][:500]}...",
        expected_output=art['right_summary'], context=art['document'],
        metadata={'label':0,'summary':art['right_summary']}), 0))
    cases.append((EvalCase(input=f"Summarize: {art['document'][:500]}...",
        expected_output=art['right_summary'], context=art['document'],
        metadata={'label':1,'summary':art['hallucinated_summary']}), 1))

judge = JudgeConfig(provider='anthropic', model='claude-haiku-4-5-20251001')
evaluator = Hallucination(judge=judge)
# Get the post-resolve threshold (what the benchmark will actually use)
actual_threshold = evaluator._resolve_threshold(resolve_judge(judge))
print(f"Calibrated threshold (post-resolve, used at runtime): {actual_threshold}")

def score(case_label):
    case, label = case_label
    try:
        result = evaluator.evaluate(case, case.metadata['summary'])
        pred = 1 if not result.passed else 0
        return label, pred, result.score
    except Exception as e:
        return label, None, str(e)

tp = fp = fn = tn = 0
errors = 0
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(score, cl) for cl in cases]
    for fut in as_completed(futures):
        label, pred, _ = fut.result()
        if pred is None: errors += 1; continue
        if pred==1 and label==1: tp+=1
        elif pred==1 and label==0: fp+=1
        elif pred==0 and label==1: fn+=1
        else: tn+=1

precision = tp/(tp+fp) if tp+fp>0 else 0
recall = tp/(tp+fn) if tp+fn>0 else 0
f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0
print(f"\nTP={tp} FP={fp} FN={fn} TN={tn} errors={errors}")
print(f"precision={precision:.4f}  recall={recall:.4f}  F1={f1:.4f}")

out = {
    "benchmark": "hallucination_held_out_on_halueval_sum",
    "dataset": "halueval_summarization",
    "n_samples": 30, "n_cases": len(cases) - errors,
    "evaluator": "Hallucination",
    "evaluator_calibration_dataset": "HaluEval QA (different task)",
    "evaluator_calibration_threshold": actual_threshold,
    "judge_model": "claude-haiku-4-5-20251001",
    "note": f"Truly held-out: Hallucination/Haiku threshold ({actual_threshold}) calibrated on HaluEval-QA-100, tested on HaluEval-Sum n={len(cases)}. Cross-task within HaluEval corpora.",
    "results": {
        "multivon_eval (Hallucination, held-out)": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    }
}
with open('benchmarks/results/hallucination_held_out.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved.")
