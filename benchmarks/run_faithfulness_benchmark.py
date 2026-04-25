"""
Faithfulness benchmark using HaluEval summarization subset.

Task: given a document, detect whether its summary is faithful (grounded)
or hallucinated (contains claims not in the document).

Compares:
- multivon-eval Faithfulness evaluator (QAG-based)
- Baseline: simple LLM judge (1-10)
- Baseline: keyword overlap (no LLM)
- Baseline: DeepEval HallucinationMetric (GPT-4o-mini)

Human labels: 1 = hallucinated summary, 0 = faithful summary
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from multivon_eval import EvalCase, Faithfulness
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.result import EvalResult
import anthropic


# ── Baselines ──────────────────────────────────────────────────────────────

class KeywordOverlapBaseline(Evaluator):
    name = "keyword_overlap"

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.context:
            return self._result(0.0, "No context")
        ctx_words = set(case.context.lower().split())
        ans_words = set(output.lower().split())
        stops = {"the","a","an","is","are","was","were","in","of","to","and","or","it","its","be"}
        novel = (ans_words - ctx_words) - stops
        ratio = len(novel) / max(len(ans_words), 1)
        faithful = ratio < 0.35  # lower threshold for summarization (paraphrase is expected)
        return self._result(1.0 if faithful else 0.0, f"Novel ratio: {ratio:.2f}")


class SimpleJudgeBaseline(Evaluator):
    name = "simple_judge (1-10)"

    def __init__(self):
        super().__init__(threshold=1.0)
        self.client = anthropic.Anthropic()

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        prompt = f"""Rate how faithful this summary is to the source document on a scale of 1-10.
10 = completely faithful, all claims are supported by the document.
1 = completely unfaithful, contains fabricated or contradicted claims.
Reply with only a number.

Document: {case.context[:2000]}
Summary: {output}"""
        resp = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            score = int(resp.content[0].text.strip()) / 10.0
        except Exception:
            score = 0.5
        return self._result(round(score, 2), f"Judge: {score:.1f}")


class DeepEvalBaseline(Evaluator):
    name = "deepeval (GPT-4o-mini)"

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
            context_chunks = []
            if case.context:
                # Split document into ~500 char chunks for DeepEval
                text = case.context
                chunk_size = 500
                context_chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), 2000), chunk_size)]
            test_case = LLMTestCase(
                input=case.input or "Summarize the document.",
                actual_output=output,
                context=context_chunks,
            )
            metric.measure(test_case)
            score = 1.0 - float(metric.score)
            return self._result(round(score, 3), f"DeepEval: {metric.score:.3f}")
        except Exception as e:
            return self._result(0.5, f"Error: {e}")


# ── Dataset ────────────────────────────────────────────────────────────────

def load_halueval_summarization(n: int = 30) -> list[dict]:
    """Load summarization samples from HaluEval (JSONL format)."""
    url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            lines = r.read().decode().strip().splitlines()
        data = [json.loads(l) for l in lines if l.strip()]
        return data[:n]
    except Exception as e:
        print(f"Could not fetch HaluEval summarization dataset: {e}")
        print("Using built-in fallback samples...")
        return FALLBACK_SAMPLES


FALLBACK_SAMPLES = [
    {
        "document": "The Apollo 11 mission was the first crewed lunar landing mission. It launched on July 16, 1969, and landed on the Moon on July 20, 1969. Astronauts Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above.",
        "right_summary": "Apollo 11 landed on the Moon on July 20, 1969 — the first crewed lunar landing. Armstrong and Aldrin walked on the surface while Collins orbited.",
        "hallucinated_summary": "Apollo 11 was the second crewed mission to the Moon, landing on August 12, 1969. All three astronauts — Armstrong, Aldrin, and Collins — walked on the lunar surface.",
    },
    {
        "document": "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability and supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "right_summary": "Python is a high-level interpreted language created by Guido van Rossum in 1991, known for readability and supporting multiple paradigms.",
        "hallucinated_summary": "Python was created by Dennis Ritchie in 1985 as a low-level systems language primarily used for operating system development.",
    },
    {
        "document": "The Great Wall of China is a series of fortifications built across historical northern China. The wall was built to protect against invasions and raids from northern nomads. Construction began as early as the 7th century BC, with major sections built during the Ming Dynasty (1368-1644).",
        "right_summary": "The Great Wall of China is a series of northern fortifications, built to repel nomadic invasions, with major construction during the Ming Dynasty (1368-1644).",
        "hallucinated_summary": "The Great Wall of China was constructed in a single continuous effort during the Qin Dynasty around 221 BC, spanning the entire length of China's southern border.",
    },
]


# ── Runner ─────────────────────────────────────────────────────────────────

def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return round(p, 3), round(r, 3), round(f1, 3)


def evaluate_one(name, ev, item):
    t0 = time.time()
    r = ev.evaluate(item["case"], item["output"])
    ms = (time.time() - t0) * 1000
    detected = not r.passed
    actual_hal = item["label"] == 1
    return name, ms, detected, actual_hal


def run_benchmark(n_samples: int = 30):
    print(f"\n{'='*60}")
    print("  multivon-eval Faithfulness Benchmark")
    print(f"  Samples: {n_samples} (faithful + hallucinated pairs)")
    print(f"{'='*60}\n")

    raw = load_halueval_summarization(n_samples)

    cases = []
    for item in raw:
        doc = item.get("document", "")
        cases.append({
            "case": EvalCase(input="Summarize the document.", context=doc),
            "output": item.get("right_summary", ""),
            "label": 0,
        })
        cases.append({
            "case": EvalCase(input="Summarize the document.", context=doc),
            "output": item.get("hallucinated_summary", ""),
            "label": 1,
        })

    evaluators = {
        "multivon_eval (Faithfulness)": Faithfulness(),
        "keyword_overlap": KeywordOverlapBaseline(),
        "simple_judge (1-10)": SimpleJudgeBaseline(),
        "deepeval (GPT-4o-mini)": DeepEvalBaseline(),
    }

    results = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "latency_ms": []} for name in evaluators}
    total = len(cases) * len(evaluators)
    done = 0

    print(f"  Running {total} evaluations with {len(evaluators)} evaluators x {len(cases)} cases...\n")

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
    print(f"{'Evaluator':<32} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg ms':>10}")
    print("-" * 74)

    summary = {}
    for name, r in results.items():
        p, rec, f1 = precision_recall_f1(r["tp"], r["fp"], r["fn"])
        avg_ms = sum(r["latency_ms"]) / len(r["latency_ms"])
        print(f"{name:<32} {p:>10.3f} {rec:>10.3f} {f1:>10.3f} {avg_ms:>9.0f}ms")
        summary[name] = {
            "precision": p, "recall": rec, "f1": f1,
            "avg_latency_ms": round(avg_ms),
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
        }

    print()
    os.makedirs("benchmarks/results", exist_ok=True)
    out = {
        "benchmark": "faithfulness",
        "dataset": "halueval_summarization",
        "n_samples": n_samples,
        "n_cases": len(cases),
        "results": summary,
    }
    with open("benchmarks/results/faithfulness.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  Results saved to benchmarks/results/faithfulness.json\n")
    return summary


if __name__ == "__main__":
    run_benchmark(n_samples=30)
