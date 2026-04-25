"""
Answer relevance benchmark using a curated golden set.

Task: given a question, detect whether the answer actually addresses it
(relevant) or drifts off-topic / addresses a different question (irrelevant).

Dataset: 40 QA pairs manually curated with 50/50 relevant and irrelevant answers,
spanning factual QA, instructions, opinions, and edge cases.

Compares:
- multivon-eval Relevance evaluator
- Simple LLM judge (yes/no)
- DeepEval AnswerRelevancyMetric (GPT-4o-mini)

Labels: 0 = relevant answer (passes), 1 = irrelevant answer (should be flagged)
"""
from __future__ import annotations
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from multivon_eval import EvalCase, Relevance
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.result import EvalResult
import anthropic


# ── Baselines ──────────────────────────────────────────────────────────────

class SimpleJudgeBaseline(Evaluator):
    name = "simple_judge (yes/no)"

    def __init__(self):
        super().__init__(threshold=0.5)
        self.client = anthropic.Anthropic()

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        prompt = f"""Does this answer directly address the question asked?
Reply with only "yes" or "no".

Question: {case.input}
Answer: {output}"""
        resp = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip().lower()
        score = 1.0 if "yes" in text else 0.0
        return self._result(score, f"Judge: {text}")


class DeepEvalBaseline(Evaluator):
    name = "deepeval (GPT-4o-mini)"

    def __init__(self):
        super().__init__(threshold=0.5)
        try:
            from deepeval.metrics import AnswerRelevancyMetric
            self.metric_cls = AnswerRelevancyMetric
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
            )
            metric.measure(test_case)
            return self._result(round(float(metric.score), 3), f"DeepEval: {metric.score:.3f}")
        except Exception as e:
            return self._result(0.5, f"Error: {e}")


# ── Golden Dataset ─────────────────────────────────────────────────────────
# 40 QA pairs: 20 relevant + 20 irrelevant answers
# Spans factual, instructional, conversational, and numeric reasoning

GOLDEN_SET = [
    # Factual QA — relevant answers
    {"question": "What is the boiling point of water at sea level?", "answer": "Water boils at 100 degrees Celsius (212°F) at sea level.", "label": 0},
    {"question": "Who invented the telephone?", "answer": "Alexander Graham Bell is credited with inventing the telephone in 1876.", "label": 0},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter is the largest planet in our solar system.", "label": 0},
    {"question": "How many bones are in the adult human body?", "answer": "An adult human body has 206 bones.", "label": 0},
    {"question": "What language is most spoken worldwide?", "answer": "Mandarin Chinese has the most native speakers, but English is the most widely spoken second language.", "label": 0},
    {"question": "What year did the Berlin Wall fall?", "answer": "The Berlin Wall fell in 1989.", "label": 0},
    {"question": "What is the chemical formula for table salt?", "answer": "Table salt is sodium chloride, with the formula NaCl.", "label": 0},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci painted the Mona Lisa, likely between 1503 and 1519.", "label": 0},
    {"question": "What is the speed of sound in air?", "answer": "Sound travels at approximately 343 meters per second in air at 20°C.", "label": 0},
    {"question": "Which country is the Amazon River in?", "answer": "The Amazon River flows primarily through Brazil, though it originates in Peru.", "label": 0},
    # Instructional — relevant answers
    {"question": "How do I reverse a string in Python?", "answer": "You can reverse a string in Python using slicing: s[::-1], or the reversed() function with join.", "label": 0},
    {"question": "What is the difference between HTTP and HTTPS?", "answer": "HTTPS is the encrypted version of HTTP, using TLS/SSL to secure data in transit.", "label": 0},
    {"question": "How do you make a roux for a sauce?", "answer": "To make a roux, melt equal parts butter and flour together over medium heat, stirring constantly until it reaches the desired color.", "label": 0},
    {"question": "What does git rebase do?", "answer": "Git rebase moves or replays commits from one branch onto another, creating a cleaner linear history.", "label": 0},
    {"question": "How do you calculate compound interest?", "answer": "Compound interest is calculated as A = P(1 + r/n)^(nt), where P is principal, r is rate, n is compounds per year, and t is time.", "label": 0},
    # Opinion/advice — relevant answers
    {"question": "What are the benefits of regular exercise?", "answer": "Regular exercise improves cardiovascular health, strengthens muscles, boosts mood through endorphins, and helps maintain a healthy weight.", "label": 0},
    {"question": "Why is sleep important?", "answer": "Sleep allows the body to repair cells, consolidate memories, regulate hormones, and maintain immune function.", "label": 0},
    {"question": "What should I consider when choosing a database?", "answer": "Key factors include your data structure (relational vs. document), expected query patterns, scalability needs, and consistency requirements.", "label": 0},
    {"question": "How can I improve my writing skills?", "answer": "Reading widely, writing daily, seeking feedback, and studying grammar and style guides all help improve writing.", "label": 0},
    {"question": "What is the best way to learn a new programming language?", "answer": "Build small projects, read existing code, follow official tutorials, and practice solving problems in that language.", "label": 0},
    # Factual QA — irrelevant answers (wrong topic, evasive, or off-topic)
    {"question": "What is the boiling point of water at sea level?", "answer": "Weather patterns are influenced by many factors including humidity, pressure, and temperature gradients in the atmosphere.", "label": 1},
    {"question": "Who invented the telephone?", "answer": "The telegraph was an important communication device in the 19th century that preceded modern telecommunications.", "label": 1},
    {"question": "What is the largest planet in our solar system?", "answer": "There are eight planets in our solar system, each with unique characteristics and orbital periods.", "label": 1},
    {"question": "How many bones are in the adult human body?", "answer": "The human muscular system contains over 600 muscles that enable movement and support bodily functions.", "label": 1},
    {"question": "What language is most spoken worldwide?", "answer": "Language learning has many cognitive benefits including improved memory and multitasking ability.", "label": 1},
    {"question": "What year did the Berlin Wall fall?", "answer": "Germany is a country in Central Europe with a rich cultural history and strong economy.", "label": 1},
    {"question": "What is the chemical formula for table salt?", "answer": "Cooking with salt can enhance flavor in many dishes when used appropriately in recipes.", "label": 1},
    {"question": "Who painted the Mona Lisa?", "answer": "The Louvre museum in Paris houses many famous artworks and attracts millions of visitors each year.", "label": 1},
    {"question": "What is the speed of sound in air?", "answer": "Sound is a form of energy that travels in waves and can be heard by humans within a certain frequency range.", "label": 1},
    {"question": "Which country is the Amazon River in?", "answer": "Rivers are important freshwater sources and play a crucial role in ecosystems worldwide.", "label": 1},
    # Instructional — irrelevant answers
    {"question": "How do I reverse a string in Python?", "answer": "Python is a popular programming language with many libraries for data science and machine learning.", "label": 1},
    {"question": "What is the difference between HTTP and HTTPS?", "answer": "Web servers process incoming requests and return responses to clients over the internet.", "label": 1},
    {"question": "How do you make a roux for a sauce?", "answer": "French cuisine has a long history and is considered one of the world's great culinary traditions.", "label": 1},
    {"question": "What does git rebase do?", "answer": "Version control systems help teams collaborate on software projects by tracking changes over time.", "label": 1},
    {"question": "How do you calculate compound interest?", "answer": "Investing early is important because your money has more time to grow through market returns.", "label": 1},
    # Opinion/advice — irrelevant answers
    {"question": "What are the benefits of regular exercise?", "answer": "Gym memberships can be expensive, and many people find it hard to stay motivated to exercise regularly.", "label": 1},
    {"question": "Why is sleep important?", "answer": "Many adults report feeling tired at work due to long hours and high stress levels in modern workplaces.", "label": 1},
    {"question": "What should I consider when choosing a database?", "answer": "Software engineering is a complex field with many specializations, from frontend to infrastructure.", "label": 1},
    {"question": "How can I improve my writing skills?", "answer": "There are many famous authors throughout history who have written influential and celebrated novels.", "label": 1},
    {"question": "What is the best way to learn a new programming language?", "answer": "Programming languages vary in syntax, paradigm, and use case, from systems languages to scripting languages.", "label": 1},
]


# ── Runner ─────────────────────────────────────────────────────────────────

def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return round(p, 3), round(r, 3), round(f1, 3)


def evaluate_one(name, ev, case, output, label):
    t0 = time.time()
    r = ev.evaluate(case, output)
    ms = (time.time() - t0) * 1000
    # passed=True means relevant, passed=False means irrelevant (detected)
    detected = not r.passed
    actual_irrel = label == 1
    return name, ms, detected, actual_irrel


def run_benchmark():
    print(f"\n{'='*60}")
    print("  multivon-eval Answer Relevance Benchmark")
    print(f"  Dataset: golden set ({len(GOLDEN_SET)} QA pairs, 50/50 relevant/irrelevant)")
    print(f"{'='*60}\n")

    evaluators = {
        "multivon_eval (Relevance)": Relevance(),
        "simple_judge (yes/no)": SimpleJudgeBaseline(),
        "deepeval (GPT-4o-mini)": DeepEvalBaseline(),
    }

    results = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "latency_ms": []} for name in evaluators}
    total = len(GOLDEN_SET) * len(evaluators)
    done = 0

    print(f"  Running {total} evaluations...\n")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                evaluate_one, name, ev,
                EvalCase(input=item["question"]),
                item["answer"],
                item["label"]
            ): name
            for item in GOLDEN_SET
            for name, ev in evaluators.items()
        }
        for fut in as_completed(futures):
            name, ms, detected, actual_irrel = fut.result()
            results[name]["latency_ms"].append(ms)
            if detected and actual_irrel:
                results[name]["tp"] += 1
            elif detected and not actual_irrel:
                results[name]["fp"] += 1
            elif not detected and actual_irrel:
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
        "benchmark": "answer_relevance",
        "dataset": "multivon_golden_set",
        "n_cases": len(GOLDEN_SET),
        "note": "Golden set curated manually: 20 relevant + 20 irrelevant answers across factual, instructional, and opinion QA",
        "results": summary,
    }
    with open("benchmarks/results/relevance.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  Results saved to benchmarks/results/relevance.json\n")
    return summary


if __name__ == "__main__":
    run_benchmark()
