"""
Answer accuracy benchmark using a hardcoded set of 40 factual QA pairs.

Requires ANTHROPIC_API_KEY or OPENAI_API_KEY. Estimated cost: ~$0.10-0.20 per run using claude-haiku-4-5.

Task: given a question and a model response, determine whether the response
is factually correct. Tests capital cities, historical dates, scientific facts,
and basic math.

Compares:
- multivon-eval AnswerAccuracy evaluator (QAG approach)
- ExactMatch (deterministic string comparison)
- Simple LLM judge (1-10 numeric score)

Labels: 0 = correct answer (passes), 1 = wrong answer (should be flagged)
"""
from __future__ import annotations
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from multivon_eval import EvalCase, AnswerAccuracy, ExactMatch
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.result import EvalResult
import anthropic


# ── Baseline ───────────────────────────────────────────────────────────────

class SimpleJudgeBaseline(Evaluator):
    """Simple numeric 1-10 LLM judge — the approach we compare QAG against."""
    name = "simple_judge (1-10)"

    def __init__(self):
        super().__init__(threshold=0.6)
        self.client = anthropic.Anthropic()

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        prompt = f"""Rate how factually accurate this answer is on a scale of 1-10.
10 = completely correct, matches the known fact perfectly.
1 = completely wrong.
Reply with only a number.

Question: {case.input}
Correct answer: {case.expected_output}
Model answer: {output}"""
        resp = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            score = int(resp.content[0].text.strip()) / 10.0
        except Exception:
            score = 0.5
        return self._result(round(score, 2), f"Numeric judge: {score:.1f}")


class ExactMatchBaseline(Evaluator):
    """Wraps the built-in ExactMatch evaluator to fit the benchmark interface."""
    name = "exact_match"

    def __init__(self):
        super().__init__(threshold=1.0)
        self._inner = ExactMatch()

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        return self._inner.evaluate(case, output)


# ── Golden Dataset ─────────────────────────────────────────────────────────
# 40 QA pairs: 20 correct answers + 20 wrong answers
# Covers: capital cities (10), historical dates (10), scientific facts (10), math (10)
# Each wrong answer is a plausible but incorrect alternative.

GOLDEN_SET = [
    # ── Capital cities — correct answers (label=0) ───────────────────────
    {"question": "What is the capital of Australia?", "expected": "Canberra", "answer": "Canberra", "label": 0},
    {"question": "What is the capital of Canada?", "expected": "Ottawa", "answer": "Ottawa", "label": 0},
    {"question": "What is the capital of Brazil?", "expected": "Brasília", "answer": "Brasília", "label": 0},
    {"question": "What is the capital of Japan?", "expected": "Tokyo", "answer": "Tokyo", "label": 0},
    {"question": "What is the capital of Germany?", "expected": "Berlin", "answer": "Berlin", "label": 0},

    # ── Historical dates — correct answers (label=0) ─────────────────────
    {"question": "In what year did World War I begin?", "expected": "1914", "answer": "World War I began in 1914.", "label": 0},
    {"question": "In what year did the United States declare independence?", "expected": "1776", "answer": "The United States declared independence in 1776.", "label": 0},
    {"question": "In what year did Neil Armstrong first walk on the Moon?", "expected": "1969", "answer": "Neil Armstrong walked on the Moon in 1969.", "label": 0},
    {"question": "In what year did the Berlin Wall fall?", "expected": "1989", "answer": "The Berlin Wall fell in 1989.", "label": 0},
    {"question": "In what year was the Eiffel Tower completed?", "expected": "1889", "answer": "The Eiffel Tower was completed in 1889.", "label": 0},

    # ── Scientific facts — correct answers (label=0) ─────────────────────
    {"question": "What is the chemical symbol for gold?", "expected": "Au", "answer": "Au", "label": 0},
    {"question": "How many planets are in our solar system?", "expected": "8", "answer": "There are 8 planets in our solar system.", "label": 0},
    {"question": "What is the speed of light in a vacuum in metres per second?", "expected": "299792458", "answer": "The speed of light is approximately 299,792,458 metres per second.", "label": 0},
    {"question": "What gas do plants absorb during photosynthesis?", "expected": "carbon dioxide", "answer": "Plants absorb carbon dioxide during photosynthesis.", "label": 0},
    {"question": "What is the atomic number of hydrogen?", "expected": "1", "answer": "The atomic number of hydrogen is 1.", "label": 0},

    # ── Math — correct answers (label=0) ─────────────────────────────────
    {"question": "What is 12 multiplied by 13?", "expected": "156", "answer": "156", "label": 0},
    {"question": "What is the square root of 144?", "expected": "12", "answer": "12", "label": 0},
    {"question": "What is 15% of 200?", "expected": "30", "answer": "30", "label": 0},
    {"question": "What is 2 to the power of 10?", "expected": "1024", "answer": "1024", "label": 0},
    {"question": "What is 7 factorial (7!)?", "expected": "5040", "answer": "5040", "label": 0},

    # ── Capital cities — wrong answers (label=1) ─────────────────────────
    {"question": "What is the capital of Australia?", "expected": "Canberra", "answer": "Sydney is the capital of Australia.", "label": 1},
    {"question": "What is the capital of Canada?", "expected": "Ottawa", "answer": "Toronto is the capital of Canada.", "label": 1},
    {"question": "What is the capital of Brazil?", "expected": "Brasília", "answer": "Rio de Janeiro is the capital of Brazil.", "label": 1},
    {"question": "What is the capital of Japan?", "expected": "Tokyo", "answer": "Osaka is the capital of Japan.", "label": 1},
    {"question": "What is the capital of Germany?", "expected": "Berlin", "answer": "Munich is the capital of Germany.", "label": 1},

    # ── Historical dates — wrong answers (label=1) ───────────────────────
    {"question": "In what year did World War I begin?", "expected": "1914", "answer": "World War I began in 1912.", "label": 1},
    {"question": "In what year did the United States declare independence?", "expected": "1776", "answer": "The United States declared independence in 1783.", "label": 1},
    {"question": "In what year did Neil Armstrong first walk on the Moon?", "expected": "1969", "answer": "Neil Armstrong walked on the Moon in 1972.", "label": 1},
    {"question": "In what year did the Berlin Wall fall?", "expected": "1989", "answer": "The Berlin Wall fell in 1991.", "label": 1},
    {"question": "In what year was the Eiffel Tower completed?", "expected": "1889", "answer": "The Eiffel Tower was completed in 1900.", "label": 1},

    # ── Scientific facts — wrong answers (label=1) ───────────────────────
    {"question": "What is the chemical symbol for gold?", "expected": "Au", "answer": "Ag", "label": 1},
    {"question": "How many planets are in our solar system?", "expected": "8", "answer": "There are 9 planets in our solar system.", "label": 1},
    {"question": "What is the speed of light in a vacuum in metres per second?", "expected": "299792458", "answer": "The speed of light is approximately 150,000 kilometres per second.", "label": 1},
    {"question": "What gas do plants absorb during photosynthesis?", "expected": "carbon dioxide", "answer": "Plants absorb oxygen during photosynthesis.", "label": 1},
    {"question": "What is the atomic number of hydrogen?", "expected": "1", "answer": "The atomic number of hydrogen is 2.", "label": 1},

    # ── Math — wrong answers (label=1) ───────────────────────────────────
    {"question": "What is 12 multiplied by 13?", "expected": "156", "answer": "144", "label": 1},
    {"question": "What is the square root of 144?", "expected": "12", "answer": "14", "label": 1},
    {"question": "What is 15% of 200?", "expected": "30", "answer": "25", "label": 1},
    {"question": "What is 2 to the power of 10?", "expected": "1024", "answer": "2048", "label": 1},
    {"question": "What is 7 factorial (7!)?", "expected": "5040", "answer": "720", "label": 1},
]


# ── Runner ─────────────────────────────────────────────────────────────────

def accuracy_and_fp(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0
    return round(acc, 3), fp


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return round(p, 3), round(r, 3), round(f1, 3)


def evaluate_one(name, ev, case, output, label):
    t0 = time.time()
    r = ev.evaluate(case, output)
    ms = (time.time() - t0) * 1000
    # passed=True means correct, passed=False means wrong (detected error)
    detected = not r.passed
    actual_wrong = label == 1
    return name, ms, detected, actual_wrong


def run_benchmark():
    print(f"\n{'='*60}")
    print("  multivon-eval Answer Accuracy Benchmark")
    print(f"  Dataset: hardcoded golden set ({len(GOLDEN_SET)} QA pairs, 50/50 correct/wrong)")
    print(f"  Categories: capital cities, historical dates, scientific facts, math")
    print(f"{'='*60}\n")

    evaluators = {
        "multivon_eval (QAG)": AnswerAccuracy(),
        "exact_match": ExactMatchBaseline(),
        "simple_judge (1-10)": SimpleJudgeBaseline(),
    }

    results = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "latency_ms": []} for name in evaluators}
    total = len(GOLDEN_SET) * len(evaluators)
    done = 0

    print(f"  Running {total} evaluations...\n")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                evaluate_one, name, ev,
                EvalCase(input=item["question"], expected_output=item["expected"]),
                item["answer"],
                item["label"]
            ): name
            for item in GOLDEN_SET
            for name, ev in evaluators.items()
        }
        for fut in as_completed(futures):
            name, ms, detected, actual_wrong = fut.result()
            results[name]["latency_ms"].append(ms)
            if detected and actual_wrong:
                results[name]["tp"] += 1
            elif detected and not actual_wrong:
                results[name]["fp"] += 1
            elif not detected and actual_wrong:
                results[name]["fn"] += 1
            else:
                results[name]["tn"] += 1
            done += 1
            print(f"  [{done}/{total}] done", end="\r")

    print("\n")
    print(f"{'Evaluator':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'FP':>6} {'Avg ms':>10}")
    print("-" * 82)

    summary = {}
    for name, r in results.items():
        acc, fp_count = accuracy_and_fp(r["tp"], r["fp"], r["fn"], r["tn"])
        p, rec, f1 = precision_recall_f1(r["tp"], r["fp"], r["fn"])
        avg_ms = sum(r["latency_ms"]) / len(r["latency_ms"])
        print(f"{name:<30} {acc:>10.3f} {p:>10.3f} {rec:>10.3f} {fp_count:>6} {avg_ms:>9.0f}ms")
        summary[name] = {
            "accuracy": acc,
            "precision": p, "recall": rec, "f1": f1,
            "avg_latency_ms": round(avg_ms),
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
        }

    print()
    os.makedirs("benchmarks/results", exist_ok=True)
    out = {
        "benchmark": "answer_accuracy",
        "dataset": "hardcoded_golden_set",
        "n_cases": len(GOLDEN_SET),
        "categories": ["capital cities", "historical dates", "scientific facts", "math"],
        "note": (
            "20 correct answers + 20 plausible-but-wrong answers across four factual domains. "
            "Wrong answers are not gibberish — they are realistic confusions (e.g., Sydney vs Canberra)."
        ),
        "results": summary,
    }
    with open("benchmarks/results/answer_accuracy.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  Results saved to benchmarks/results/answer_accuracy.json\n")
    return summary


if __name__ == "__main__":
    run_benchmark()
