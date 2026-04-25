"""
Hallucination detection benchmark using HaluEval QA subset.

Compares multivon-eval's Hallucination evaluator against:
- Baseline A: keyword overlap (no LLM)
- Baseline B: simple LLM judge (1-10 numeric score)
- Baseline C: DeepEval HallucinationMetric (GPT-4o-mini)

Human labels from HaluEval: 1 = hallucinated, 0 = faithful
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from multivon_eval import EvalCase, Hallucination
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.result import EvalResult
import anthropic

# ── Baselines ──────────────────────────────────────────────────────────────

class KeywordOverlapBaseline(Evaluator):
    """Flags hallucination if answer contains words not in context."""
    name = "keyword_overlap"

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.context:
            return self._result(0.0, "No context")
        ctx_words = set(case.context.lower().split())
        ans_words = set(output.lower().split())
        stops = {"the","a","an","is","are","was","were","in","of","to","and","or","it","its","be"}
        novel = (ans_words - ctx_words) - stops
        ratio = len(novel) / max(len(ans_words), 1)
        faithful = ratio < 0.4
        return self._result(1.0 if faithful else 0.0, f"Novel word ratio: {ratio:.2f}")


class SimpleJudgeBaseline(Evaluator):
    """Simple numeric 1-10 LLM judge (the approach we replace with QAG)."""
    name = "simple_judge"

    def __init__(self):
        super().__init__(threshold=1.0)
        self.client = anthropic.Anthropic()

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        prompt = f"""Rate how faithful this answer is to the context on a scale of 1-10.
10 = completely faithful, no hallucinations.
1 = completely hallucinated.
Reply with only a number.

Context: {case.context}
Answer: {output}"""
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


class DeepEvalBaseline(Evaluator):
    """DeepEval HallucinationMetric using GPT-4o-mini."""
    name = "deepeval"

    def __init__(self):
        super().__init__(threshold=0.5)
        try:
            from deepeval.metrics import HallucinationMetric
            self.metric_cls = HallucinationMetric
            self.available = True
        except ImportError:
            self.available = False

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not self.available:
            return self._result(0.5, "DeepEval not available")
        try:
            from deepeval.test_case import LLMTestCase
            metric = self.metric_cls(
                threshold=0.5,
                model="gpt-4o-mini",
                include_reason=False,
                async_mode=False,
            )
            test_case = LLMTestCase(
                input=case.input or "",
                actual_output=output,
                context=[case.context] if case.context else [],
            )
            metric.measure(test_case)
            # score = hallucination rate (0=faithful, 1=hallucinated)
            # passed=True means score <= threshold (faithful)
            score = 1.0 - float(metric.score)  # invert so 1.0 = faithful
            return self._result(round(score, 3), f"DeepEval score: {metric.score:.3f}")
        except Exception as e:
            return self._result(0.5, f"DeepEval error: {e}")


# ── Dataset ────────────────────────────────────────────────────────────────

def load_halueval_sample(n: int = 50) -> list[dict]:
    """Load n samples from HaluEval QA subset (JSONL format)."""
    url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            lines = r.read().decode().strip().splitlines()
        data = [json.loads(l) for l in lines if l.strip()]
        return data[:n]
    except Exception as e:
        print(f"Could not fetch HaluEval dataset: {e}")
        print("Using built-in fallback samples...")
        return FALLBACK_SAMPLES


FALLBACK_SAMPLES = [
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital city is Paris, which is also the largest city in France.",
        "right_answer": "The capital of France is Paris.",
        "hallucinated_answer": "The capital of France is Lyon, which is the largest city in the country.",
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "context": "Romeo and Juliet is a tragedy written by William Shakespeare, believed to have been written between 1594 and 1596.",
        "right_answer": "Romeo and Juliet was written by William Shakespeare.",
        "hallucinated_answer": "Romeo and Juliet was written by Christopher Marlowe in 1590.",
    },
    {
        "question": "What is the speed of light?",
        "context": "The speed of light in a vacuum is exactly 299,792,458 metres per second, approximately 3×10^8 m/s.",
        "right_answer": "The speed of light is approximately 299,792,458 metres per second.",
        "hallucinated_answer": "The speed of light is approximately 150,000 kilometres per second.",
    },
    {
        "question": "When did World War II end?",
        "context": "World War II ended in 1945. The war in Europe ended on 8 May 1945 (V-E Day), and the war in the Pacific ended on 15 August 1945 (V-J Day).",
        "right_answer": "World War II ended in 1945, with V-E Day on May 8 and V-J Day on August 15.",
        "hallucinated_answer": "World War II ended in 1943 after the Allied forces defeated Germany.",
    },
    {
        "question": "What element has the symbol Au?",
        "context": "Gold is a chemical element with the symbol Au (from Latin: aurum) and atomic number 79.",
        "right_answer": "The element with symbol Au is Gold.",
        "hallucinated_answer": "The element with symbol Au is Silver, derived from the Latin word argentum.",
    },
]


# ── Runner ─────────────────────────────────────────────────────────────────

def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return round(p, 3), round(r, 3), round(f1, 3)


def evaluate_one(name, ev, item):
    """Evaluate a single case with a single evaluator. Thread-safe."""
    t0 = time.time()
    r = ev.evaluate(item["case"], item["output"])
    ms = (time.time() - t0) * 1000
    detected = not r.passed
    actual_hal = item["label"] == 1
    return name, ms, detected, actual_hal


def run_benchmark(n_samples: int = 50):
    print(f"\n{'='*60}")
    print("  multivon-eval Hallucination Benchmark")
    print(f"  Samples: {n_samples} (faithful + hallucinated pairs)")
    print(f"{'='*60}\n")

    raw = load_halueval_sample(n_samples)

    cases = []
    for item in raw:
        ctx = item.get("knowledge", item.get("context", ""))
        q = item.get("question", "")
        cases.append({
            "case": EvalCase(input=q, context=ctx),
            "output": item.get("right_answer", ""),
            "label": 0,
        })
        cases.append({
            "case": EvalCase(input=q, context=ctx),
            "output": item.get("hallucinated_answer", ""),
            "label": 1,
        })

    evaluators = {
        "multivon_eval (QAG)": Hallucination(),
        "keyword_overlap": KeywordOverlapBaseline(),
        "simple_judge (1-10)": SimpleJudgeBaseline(),
        "deepeval (GPT-4o-mini)": DeepEvalBaseline(),
    }

    results = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "latency_ms": []} for name in evaluators}
    total = len(cases) * len(evaluators)
    done = 0

    print(f"  Running {total} evaluations with {len(evaluators)} evaluators x {len(cases)} cases...")
    print(f"  (Using threading for LLM evaluators)\n")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(evaluate_one, name, ev, item): (name, i)
            for i, item in enumerate(cases)
            for name, ev in evaluators.items()
        }
        for fut in as_completed(futures):
            name, ms, detected, actual_hal = fut.result()
            results[name]["latency_ms"].append(ms)
            if detected and actual_hal:
                results[name]["tp"] += 1
            elif detected and not actual_hal:
                results[name]["fp"] += 1
            elif not detected and actual_hal:
                results[name]["fn"] += 1
            else:
                results[name]["tn"] += 1
            done += 1
            print(f"  [{done}/{total}] done", end="\r")

    print("\n")
    print(f"{'Evaluator':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg ms':>10}")
    print("-" * 70)

    summary = {}
    for name, r in results.items():
        p, rec, f1 = precision_recall_f1(r["tp"], r["fp"], r["fn"])
        avg_ms = sum(r["latency_ms"]) / len(r["latency_ms"])
        print(f"{name:<28} {p:>10.3f} {rec:>10.3f} {f1:>10.3f} {avg_ms:>9.0f}ms")
        summary[name] = {
            "precision": p, "recall": rec, "f1": f1,
            "avg_latency_ms": round(avg_ms),
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
        }

    print()
    os.makedirs("benchmarks/results", exist_ok=True)
    out = {
        "benchmark": "hallucination_detection",
        "dataset": "halueval_qa",
        "n_samples": n_samples,
        "n_cases": len(cases),
        "results": summary,
    }
    with open("benchmarks/results/hallucination.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  Results saved to benchmarks/results/hallucination.json\n")
    return summary


if __name__ == "__main__":
    run_benchmark(n_samples=50)
