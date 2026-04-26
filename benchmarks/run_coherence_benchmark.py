"""
Coherence benchmark using a hardcoded golden set of 40 summaries.

Requires ANTHROPIC_API_KEY or OPENAI_API_KEY. Estimated cost: ~$0.10-0.20 per run using claude-haiku-4-5.

Task: detect whether a summary is internally coherent (logically ordered,
well-structured sentences) or incoherent (same sentences shuffled randomly).

Compares:
- multivon-eval Coherence evaluator (QAG approach)
- Simple LLM judge (1-10 numeric score)

Labels: 0 = coherent (passes), 1 = incoherent (should be flagged)
"""
from __future__ import annotations
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from multivon_eval import EvalCase, Coherence
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
        prompt = f"""Rate how coherent and well-structured this text is on a scale of 1-10.
10 = perfectly coherent, logical flow, clear structure.
1 = completely incoherent, random sentence order, no logical flow.
Reply with only a number.

Text: {output}"""
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


# ── Golden Dataset ─────────────────────────────────────────────────────────
# 40 summaries: 20 coherent + 20 incoherent (same sentences shuffled)
# Each coherent summary has a well-structured narrative; the incoherent
# version contains the exact same sentences reordered randomly.

GOLDEN_SET = [
    # ── Coherent summaries (label=0) ────────────────────────────────────────
    {
        "summary": (
            "The Amazon rainforest is the world's largest tropical rainforest, covering over 5.5 million square kilometers. "
            "It is home to an estimated 10% of all species on Earth, including thousands of plants, animals, and insects. "
            "Deforestation poses a major threat to this ecosystem, with millions of hectares cleared each year for agriculture. "
            "Conservation efforts are underway, but the pace of destruction remains a serious global concern."
        ),
        "label": 0,
    },
    {
        "summary": (
            "Python was created by Guido van Rossum and first released in 1991. "
            "It was designed to emphasize code readability and simplicity, making it accessible to beginners. "
            "Over the decades Python grew into one of the most popular languages, widely adopted in web development, data science, and AI. "
            "Today it is supported by a vast ecosystem of libraries and an active open-source community."
        ),
        "label": 0,
    },
    {
        "summary": (
            "The water cycle describes how water moves through Earth's systems continuously. "
            "Water evaporates from oceans and lakes, rises into the atmosphere, and condenses into clouds. "
            "When enough water collects in clouds, it falls back to Earth as precipitation—rain, snow, or sleet. "
            "It then flows into rivers and groundwater, eventually returning to the ocean to begin the cycle again."
        ),
        "label": 0,
    },
    {
        "summary": (
            "World War II began in 1939 when Germany invaded Poland, prompting Britain and France to declare war. "
            "The conflict rapidly expanded to involve dozens of nations across Europe, Africa, and the Pacific. "
            "Allied forces turned the tide after major victories in North Africa and the D-Day landings in 1944. "
            "Germany surrendered in May 1945, and Japan followed in September 1945, ending the deadliest war in history."
        ),
        "label": 0,
    },
    {
        "summary": (
            "Photosynthesis is the process by which plants convert sunlight into chemical energy. "
            "Chlorophyll in plant cells absorbs light, which drives a series of chemical reactions. "
            "Carbon dioxide from the air and water from the soil are combined to produce glucose and oxygen. "
            "The glucose fuels plant growth, while the oxygen is released into the atmosphere as a byproduct."
        ),
        "label": 0,
    },
    {
        "summary": (
            "The Roman Empire at its height controlled territories stretching from Britain to Mesopotamia. "
            "Its strength rested on a professional army, efficient administration, and an extensive road network. "
            "Economic troubles, military pressure on multiple frontiers, and political instability weakened the empire over centuries. "
            "The western half officially fell in 476 CE when the last Roman emperor was deposed."
        ),
        "label": 0,
    },
    {
        "summary": (
            "Machine learning is a branch of artificial intelligence focused on building systems that learn from data. "
            "Rather than being explicitly programmed with rules, these systems identify patterns through training. "
            "Common approaches include supervised learning, where labeled examples guide the model, and unsupervised learning, which finds structure without labels. "
            "Applications range from image recognition and natural language processing to medical diagnosis and financial forecasting."
        ),
        "label": 0,
    },
    {
        "summary": (
            "The human immune system defends the body against pathogens such as bacteria, viruses, and fungi. "
            "It consists of physical barriers like skin, as well as specialized cells and proteins that identify and destroy invaders. "
            "When the immune system encounters a new pathogen, it creates antibodies tailored to that threat. "
            "These antibodies remain in the body, providing faster protection if the same pathogen is encountered again."
        ),
        "label": 0,
    },
    {
        "summary": (
            "Climate change refers to long-term shifts in global temperatures and weather patterns. "
            "While natural factors contribute, scientific consensus holds that human activities—especially burning fossil fuels—are the primary driver since the mid-20th century. "
            "Rising temperatures are causing more frequent extreme weather events, melting polar ice, and rising sea levels. "
            "International agreements like the Paris Accord aim to limit warming by reducing greenhouse gas emissions."
        ),
        "label": 0,
    },
    {
        "summary": (
            "The internet began as ARPANET, a US military research network established in the late 1960s. "
            "Academic institutions joined through the 1970s and 1980s, expanding the network's reach and purpose. "
            "The invention of the World Wide Web by Tim Berners-Lee in 1991 made the internet accessible to the general public. "
            "Today it connects billions of people and underpins nearly every sector of the global economy."
        ),
        "label": 0,
    },
    {
        "summary": (
            "Antibiotics are medicines that kill or inhibit the growth of bacteria. "
            "Alexander Fleming discovered the first antibiotic, penicillin, in 1928 after noticing mold killing bacteria in his lab. "
            "Antibiotics transformed medicine, making previously deadly infections treatable and enabling complex surgeries. "
            "However, overuse has led to antibiotic-resistant bacteria, which pose a growing threat to public health worldwide."
        ),
        "label": 0,
    },
    {
        "summary": (
            "The stock market allows companies to raise capital by selling shares to investors. "
            "Investors buy shares hoping the company's value will increase, allowing them to sell at a profit. "
            "Market prices fluctuate based on company performance, economic data, and investor sentiment. "
            "Stock indices like the S&P 500 track the average performance of a large group of companies to reflect overall market health."
        ),
        "label": 0,
    },
    {
        "summary": (
            "DNA, or deoxyribonucleic acid, carries the genetic instructions for the development and functioning of living organisms. "
            "It is structured as a double helix—two strands wound together—held in place by pairs of chemical bases. "
            "Genes are segments of DNA that encode instructions for making proteins, which carry out most cellular functions. "
            "When cells divide, DNA is copied so each new cell receives a complete set of genetic information."
        ),
        "label": 0,
    },
    {
        "summary": (
            "The Renaissance was a cultural movement that began in 14th-century Italy and spread across Europe. "
            "It was characterized by renewed interest in classical Greek and Roman art, philosophy, and literature. "
            "Artists like Leonardo da Vinci and Michelangelo produced works that remain iconic today. "
            "The Renaissance also fostered scientific inquiry and laid groundwork for the Scientific Revolution that followed."
        ),
        "label": 0,
    },
    {
        "summary": (
            "Black holes form when massive stars collapse at the end of their life cycle. "
            "The collapse concentrates so much mass in a small area that gravity becomes so strong not even light can escape. "
            "The boundary beyond which nothing can escape is called the event horizon. "
            "Scientists study black holes through gravitational waves and the behavior of nearby gas and stars, since black holes themselves emit no light."
        ),
        "label": 0,
    },
    {
        "summary": (
            "The Industrial Revolution began in Britain in the late 18th century and transformed economies worldwide. "
            "Steam power and mechanized production replaced hand tools and agrarian labor. "
            "Factories drew workers from rural areas into rapidly growing cities, changing social structures profoundly. "
            "The period produced lasting advances in transportation, manufacturing, and living standards, while also introducing new labor and environmental challenges."
        ),
        "label": 0,
    },
    {
        "summary": (
            "Vaccines work by training the immune system to recognize and fight specific pathogens. "
            "A vaccine introduces a harmless form of a pathogen—such as a weakened virus or a protein fragment—into the body. "
            "The immune system responds by producing antibodies and memory cells tailored to that pathogen. "
            "If the vaccinated person later encounters the real pathogen, the immune system can respond rapidly before illness takes hold."
        ),
        "label": 0,
    },
    {
        "summary": (
            "Gravity is one of the four fundamental forces of nature and governs the motion of large objects. "
            "Isaac Newton formulated the law of universal gravitation in 1687, describing how any two masses attract each other. "
            "Albert Einstein later refined this understanding with general relativity, showing that gravity is the curvature of spacetime caused by mass. "
            "Gravitational effects explain the orbits of planets, the structure of galaxies, and the expansion of the universe."
        ),
        "label": 0,
    },
    {
        "summary": (
            "Supply and demand is the foundational model of how prices are determined in a market economy. "
            "When the supply of a good exceeds demand, prices tend to fall until buyers are found. "
            "When demand exceeds supply, prices rise until supply increases or buyers are priced out. "
            "This dynamic equilibrium guides resource allocation across most modern economies without central coordination."
        ),
        "label": 0,
    },
    {
        "summary": (
            "The human brain is divided into several regions, each responsible for distinct functions. "
            "The cerebral cortex handles higher-order tasks such as reasoning, language, and voluntary movement. "
            "The limbic system is associated with emotions and memory, while the cerebellum coordinates balance and fine motor control. "
            "All these regions work in concert through billions of neurons connected by synaptic pathways."
        ),
        "label": 0,
    },

    # ── Incoherent summaries — same sentences shuffled (label=1) ────────────
    {
        "summary": (
            "Conservation efforts are underway, but the pace of destruction remains a serious global concern. "
            "It is home to an estimated 10% of all species on Earth, including thousands of plants, animals, and insects. "
            "The Amazon rainforest is the world's largest tropical rainforest, covering over 5.5 million square kilometers. "
            "Deforestation poses a major threat to this ecosystem, with millions of hectares cleared each year for agriculture."
        ),
        "label": 1,
    },
    {
        "summary": (
            "Today it is supported by a vast ecosystem of libraries and an active open-source community. "
            "Python was created by Guido van Rossum and first released in 1991. "
            "Over the decades Python grew into one of the most popular languages, widely adopted in web development, data science, and AI. "
            "It was designed to emphasize code readability and simplicity, making it accessible to beginners."
        ),
        "label": 1,
    },
    {
        "summary": (
            "It then flows into rivers and groundwater, eventually returning to the ocean to begin the cycle again. "
            "When enough water collects in clouds, it falls back to Earth as precipitation—rain, snow, or sleet. "
            "The water cycle describes how water moves through Earth's systems continuously. "
            "Water evaporates from oceans and lakes, rises into the atmosphere, and condenses into clouds."
        ),
        "label": 1,
    },
    {
        "summary": (
            "Germany surrendered in May 1945, and Japan followed in September 1945, ending the deadliest war in history. "
            "World War II began in 1939 when Germany invaded Poland, prompting Britain and France to declare war. "
            "Allied forces turned the tide after major victories in North Africa and the D-Day landings in 1944. "
            "The conflict rapidly expanded to involve dozens of nations across Europe, Africa, and the Pacific."
        ),
        "label": 1,
    },
    {
        "summary": (
            "Carbon dioxide from the air and water from the soil are combined to produce glucose and oxygen. "
            "Chlorophyll in plant cells absorbs light, which drives a series of chemical reactions. "
            "The glucose fuels plant growth, while the oxygen is released into the atmosphere as a byproduct. "
            "Photosynthesis is the process by which plants convert sunlight into chemical energy."
        ),
        "label": 1,
    },
    {
        "summary": (
            "The western half officially fell in 476 CE when the last Roman emperor was deposed. "
            "Economic troubles, military pressure on multiple frontiers, and political instability weakened the empire over centuries. "
            "The Roman Empire at its height controlled territories stretching from Britain to Mesopotamia. "
            "Its strength rested on a professional army, efficient administration, and an extensive road network."
        ),
        "label": 1,
    },
    {
        "summary": (
            "Applications range from image recognition and natural language processing to medical diagnosis and financial forecasting. "
            "Rather than being explicitly programmed with rules, these systems identify patterns through training. "
            "Machine learning is a branch of artificial intelligence focused on building systems that learn from data. "
            "Common approaches include supervised learning, where labeled examples guide the model, and unsupervised learning, which finds structure without labels."
        ),
        "label": 1,
    },
    {
        "summary": (
            "These antibodies remain in the body, providing faster protection if the same pathogen is encountered again. "
            "It consists of physical barriers like skin, as well as specialized cells and proteins that identify and destroy invaders. "
            "When the immune system encounters a new pathogen, it creates antibodies tailored to that threat. "
            "The human immune system defends the body against pathogens such as bacteria, viruses, and fungi."
        ),
        "label": 1,
    },
    {
        "summary": (
            "Rising temperatures are causing more frequent extreme weather events, melting polar ice, and rising sea levels. "
            "While natural factors contribute, scientific consensus holds that human activities—especially burning fossil fuels—are the primary driver since the mid-20th century. "
            "International agreements like the Paris Accord aim to limit warming by reducing greenhouse gas emissions. "
            "Climate change refers to long-term shifts in global temperatures and weather patterns."
        ),
        "label": 1,
    },
    {
        "summary": (
            "Today it connects billions of people and underpins nearly every sector of the global economy. "
            "The internet began as ARPANET, a US military research network established in the late 1960s. "
            "The invention of the World Wide Web by Tim Berners-Lee in 1991 made the internet accessible to the general public. "
            "Academic institutions joined through the 1970s and 1980s, expanding the network's reach and purpose."
        ),
        "label": 1,
    },
    {
        "summary": (
            "However, overuse has led to antibiotic-resistant bacteria, which pose a growing threat to public health worldwide. "
            "Antibiotics are medicines that kill or inhibit the growth of bacteria. "
            "Antibiotics transformed medicine, making previously deadly infections treatable and enabling complex surgeries. "
            "Alexander Fleming discovered the first antibiotic, penicillin, in 1928 after noticing mold killing bacteria in his lab."
        ),
        "label": 1,
    },
    {
        "summary": (
            "Stock indices like the S&P 500 track the average performance of a large group of companies to reflect overall market health. "
            "The stock market allows companies to raise capital by selling shares to investors. "
            "Market prices fluctuate based on company performance, economic data, and investor sentiment. "
            "Investors buy shares hoping the company's value will increase, allowing them to sell at a profit."
        ),
        "label": 1,
    },
    {
        "summary": (
            "Genes are segments of DNA that encode instructions for making proteins, which carry out most cellular functions. "
            "DNA, or deoxyribonucleic acid, carries the genetic instructions for the development and functioning of living organisms. "
            "When cells divide, DNA is copied so each new cell receives a complete set of genetic information. "
            "It is structured as a double helix—two strands wound together—held in place by pairs of chemical bases."
        ),
        "label": 1,
    },
    {
        "summary": (
            "The Renaissance also fostered scientific inquiry and laid groundwork for the Scientific Revolution that followed. "
            "The Renaissance was a cultural movement that began in 14th-century Italy and spread across Europe. "
            "Artists like Leonardo da Vinci and Michelangelo produced works that remain iconic today. "
            "It was characterized by renewed interest in classical Greek and Roman art, philosophy, and literature."
        ),
        "label": 1,
    },
    {
        "summary": (
            "Scientists study black holes through gravitational waves and the behavior of nearby gas and stars, since black holes themselves emit no light. "
            "Black holes form when massive stars collapse at the end of their life cycle. "
            "The boundary beyond which nothing can escape is called the event horizon. "
            "The collapse concentrates so much mass in a small area that gravity becomes so strong not even light can escape."
        ),
        "label": 1,
    },
    {
        "summary": (
            "The period produced lasting advances in transportation, manufacturing, and living standards, while also introducing new labor and environmental challenges. "
            "The Industrial Revolution began in Britain in the late 18th century and transformed economies worldwide. "
            "Factories drew workers from rural areas into rapidly growing cities, changing social structures profoundly. "
            "Steam power and mechanized production replaced hand tools and agrarian labor."
        ),
        "label": 1,
    },
    {
        "summary": (
            "If the vaccinated person later encounters the real pathogen, the immune system can respond rapidly before illness takes hold. "
            "Vaccines work by training the immune system to recognize and fight specific pathogens. "
            "The immune system responds by producing antibodies and memory cells tailored to that pathogen. "
            "A vaccine introduces a harmless form of a pathogen—such as a weakened virus or a protein fragment—into the body."
        ),
        "label": 1,
    },
    {
        "summary": (
            "Gravitational effects explain the orbits of planets, the structure of galaxies, and the expansion of the universe. "
            "Gravity is one of the four fundamental forces of nature and governs the motion of large objects. "
            "Albert Einstein later refined this understanding with general relativity, showing that gravity is the curvature of spacetime caused by mass. "
            "Isaac Newton formulated the law of universal gravitation in 1687, describing how any two masses attract each other."
        ),
        "label": 1,
    },
    {
        "summary": (
            "This dynamic equilibrium guides resource allocation across most modern economies without central coordination. "
            "When the supply of a good exceeds demand, prices tend to fall until buyers are found. "
            "Supply and demand is the foundational model of how prices are determined in a market economy. "
            "When demand exceeds supply, prices rise until supply increases or buyers are priced out."
        ),
        "label": 1,
    },
    {
        "summary": (
            "All these regions work in concert through billions of neurons connected by synaptic pathways. "
            "The cerebral cortex handles higher-order tasks such as reasoning, language, and voluntary movement. "
            "The human brain is divided into several regions, each responsible for distinct functions. "
            "The limbic system is associated with emotions and memory, while the cerebellum coordinates balance and fine motor control."
        ),
        "label": 1,
    },
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
    # passed=True means coherent, passed=False means incoherent (detected)
    detected = not r.passed
    actual_incoherent = label == 1
    return name, ms, detected, actual_incoherent


def run_benchmark():
    print(f"\n{'='*60}")
    print("  multivon-eval Coherence Benchmark")
    print(f"  Dataset: hardcoded golden set ({len(GOLDEN_SET)} summaries, 50/50 coherent/incoherent)")
    print(f"{'='*60}\n")

    evaluators = {
        "multivon_eval (QAG)": Coherence(),
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
                EvalCase(input="Evaluate this summary for coherence."),
                item["summary"],
                item["label"]
            ): name
            for item in GOLDEN_SET
            for name, ev in evaluators.items()
        }
        for fut in as_completed(futures):
            name, ms, detected, actual_incoherent = fut.result()
            results[name]["latency_ms"].append(ms)
            if detected and actual_incoherent:
                results[name]["tp"] += 1
            elif detected and not actual_incoherent:
                results[name]["fp"] += 1
            elif not detected and actual_incoherent:
                results[name]["fn"] += 1
            else:
                results[name]["tn"] += 1
            done += 1
            print(f"  [{done}/{total}] done", end="\r")

    print("\n")
    print(f"{'Evaluator':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>6} {'Avg ms':>10}")
    print("-" * 78)

    summary = {}
    for name, r in results.items():
        p, rec, f1 = precision_recall_f1(r["tp"], r["fp"], r["fn"])
        avg_ms = sum(r["latency_ms"]) / len(r["latency_ms"])
        print(f"{name:<30} {p:>10.3f} {rec:>10.3f} {f1:>10.3f} {r['fp']:>6} {avg_ms:>9.0f}ms")
        summary[name] = {
            "precision": p, "recall": rec, "f1": f1,
            "avg_latency_ms": round(avg_ms),
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
        }

    print()
    os.makedirs("benchmarks/results", exist_ok=True)
    out = {
        "benchmark": "coherence_detection",
        "dataset": "hardcoded_golden_set",
        "n_cases": len(GOLDEN_SET),
        "note": (
            "20 coherent summaries with logical sentence ordering + 20 incoherent versions "
            "with same sentences shuffled randomly. Topics span science, history, and technology."
        ),
        "results": summary,
    }
    with open("benchmarks/results/coherence.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  Results saved to benchmarks/results/coherence.json\n")
    return summary


if __name__ == "__main__":
    run_benchmark()
