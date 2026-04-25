# multivon-eval Benchmarks

Independent evaluation of multivon-eval's evaluators against human-labeled datasets and competing tools.

All benchmarks are fully reproducible — code, datasets, and model versions are published here.

```bash
pip install multivon-eval deepeval python-dotenv
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...          # required for DeepEval baseline
python benchmarks/run_all_benchmarks.py
```

---

## Summary

| Task | multivon-eval | DeepEval | Simple LLM judge | Keyword overlap |
|------|:---:|:---:|:---:|:---:|
| Hallucination detection | **F1 0.804** | F1 0.586 | F1 0.763 | F1 0.523 |
| Answer relevance | F1 0.952 | **F1 0.974** | F1 0.976 | — |
| Faithfulness (summarization) | F1 0.480 | F1 0.448 | F1 0.667 | F1 0.627 |

Judge models: multivon-eval uses `claude-haiku-4-5-20251001`; DeepEval uses `gpt-4o-mini`. Same judge disclosed per run.

---

## Benchmark 1 — Hallucination Detection

**Dataset:** [HaluEval QA](https://github.com/RUCAIBox/HaluEval) — 50 QA pairs (100 cases) with human-annotated faithful and hallucinated answers.

**Task:** Given a context passage and an answer, detect whether the answer contains claims not supported by the context.

**Ground truth:** Human labels (1 = hallucinated, 0 = faithful). Balanced 50/50.

| Evaluator | Judge model | Precision | Recall | F1 | Avg latency |
|-----------|-------------|-----------|--------|----|-------------|
| **multivon-eval (QAG)** | claude-haiku-4-5 | **0.788** | 0.820 | **0.804** | 2955ms |
| Simple LLM judge (1-10) | claude-haiku-4-5 | 0.617 | **1.000** | 0.763 | 708ms |
| DeepEval (HallucinationMetric) | gpt-4o-mini | 0.456 | 0.820 | 0.586 | 1421ms |
| Keyword overlap (no LLM) | — | 0.605 | 0.460 | 0.523 | <1ms |

**Raw counts (n=100 cases):**

| Evaluator | TP | FP | FN | TN |
|-----------|----|----|----|----|
| multivon-eval (QAG) | 41 | 11 | 9 | 39 |
| Simple judge (1-10) | 50 | 31 | 0 | 19 |
| DeepEval | 41 | 49 | 9 | 1 |
| Keyword overlap | 23 | 15 | 27 | 35 |

**Key findings:**

- **QAG produces a more calibrated signal.** The simple judge achieves perfect recall but 31 false positives — it effectively treats every answer as suspicious. QAG's binary questions anchor the evaluation to specific claims, cutting false positives by 65% at the cost of 9 missed hallucinations.
- **DeepEval over-flags.** At F1=0.586 with only 1 true negative out of 50, the HallucinationMetric with gpt-4o-mini defaults to flagging almost everything as hallucinated. High recall, unusable precision in this configuration.
- **Keyword overlap misses 54% of hallucinations.** Useful as a free pre-filter, not as a standalone evaluator.

**What we got wrong:** QAG missed 9 hallucinations (recall 0.82). In most cases these were plausible-sounding wrong dates or proper nouns that shared vocabulary with the context — a known LLM-judge limitation on numerical reasoning.

---

## Benchmark 2 — Answer Relevance

**Dataset:** Hand-curated golden set of 40 QA pairs — 20 relevant answers, 20 off-topic answers that address different questions or provide evasive generic responses.

**Task:** Given a question and an answer, detect whether the answer is relevant to the question.

**Ground truth:** Manually labeled. Spans factual QA, instructional questions, and opinion/advice questions.

| Evaluator | Judge model | Precision | Recall | F1 | Avg latency |
|-----------|-------------|-----------|--------|----|-------------|
| multivon-eval (Relevance) | claude-haiku-4-5 | 0.909 | **1.000** | 0.952 | 3135ms |
| Simple LLM judge (yes/no) | claude-haiku-4-5 | 0.952 | **1.000** | **0.976** | 602ms |
| DeepEval (AnswerRelevancyMetric) | gpt-4o-mini | **1.000** | 0.950 | 0.974 | 7091ms |

**Raw counts (n=40 cases):**

| Evaluator | TP | FP | FN | TN |
|-----------|----|----|----|----|
| multivon-eval (Relevance) | 20 | 2 | 0 | 18 |
| Simple judge (yes/no) | 20 | 1 | 0 | 19 |
| DeepEval | 19 | 0 | 1 | 20 |

**Key findings:**

- **All three LLM-based evaluators perform well** on answer relevance — the task is relatively clear-cut when an answer is entirely off-topic.
- **multivon-eval has 2 false positives** — both were generic meta-answers (e.g., "Python has many useful libraries") that technically touch the topic but don't answer the question. The evaluator passed these as relevant; a stricter rubric would flag them.
- **DeepEval takes 2.3× longer** than multivon-eval for this task and DeepEval misses 1 irrelevant answer. Tradeoff is real.

**Dataset note:** This is a self-curated golden set, which we publish fully for inspection. It has not been independently reviewed. Treat these numbers as directionally correct, not as a definitive external benchmark.

---

## Benchmark 3 — Faithfulness (Summarization)

**Dataset:** [HaluEval Summarization](https://github.com/RUCAIBox/HaluEval) — 30 document-summary pairs (60 cases) with human-annotated faithful and hallucinated summaries.

**Task:** Given a source document and a summary, detect whether the summary introduces claims not present in the document.

**Ground truth:** Human labels (1 = hallucinated summary, 0 = faithful summary). Balanced 50/50.

| Evaluator | Judge model | Precision | Recall | F1 | Avg latency |
|-----------|-------------|-----------|--------|----|-------------|
| Simple LLM judge (1-10) | claude-haiku-4-5 | 0.500 | **1.000** | **0.667** | 782ms |
| Keyword overlap (no LLM) | — | **0.762** | 0.533 | 0.627 | <1ms |
| multivon-eval (Faithfulness) | claude-haiku-4-5 | 0.600 | 0.400 | 0.480 | 5071ms |
| DeepEval (HallucinationMetric) | gpt-4o-mini | 0.464 | 0.433 | 0.448 | 3028ms |

**Raw counts (n=60 cases):**

| Evaluator | TP | FP | FN | TN |
|-----------|----|----|----|----|
| Simple judge (1-10) | 30 | 30 | 0 | 0 |
| Keyword overlap | 16 | 5 | 14 | 25 |
| multivon-eval (Faithfulness) | 12 | 8 | 18 | 22 |
| DeepEval | 13 | 15 | 17 | 15 |

**Important finding — known limitation:**

multivon-eval's `Faithfulness` evaluator underperforms on this benchmark, and we want to be direct about why.

The evaluator is designed for **RAG faithfulness** — checking whether a short answer is grounded in retrieved context chunks. Summarization faithfulness is a different task: the "context" is the full source document (often 500-1500 words), and the hallucinations are plausible paraphrases, missing negations, or invented numerical details.

In this configuration the QAG approach struggles because:
1. Long documents make it harder to generate focused yes/no questions
2. Paraphrased-but-faithful summaries can look "novel" to a question-based evaluator
3. The evaluator's threshold is calibrated for shorter retrieval contexts

**Recommendation:** Use `Faithfulness()` for RAG pipelines (retrieving chunks + generating answers). For summarization-specific hallucination, the simple judge or a dedicated `Summarization()` evaluator will perform better until we ship calibration for long-document tasks.

---

## Methodology

### Why human labels matter

Comparing evaluators against each other tells you which one scores higher, not which one is correct. All benchmarks here use publicly available human-annotated datasets (HaluEval) or hand-curated golden sets where we publish all labels for inspection.

### Baselines chosen

| Baseline | Judge model | Why included |
|----------|-------------|-------------|
| Simple LLM judge (1-10 or yes/no) | claude-haiku-4-5 | Most common alternative approach — same judge model as multivon-eval for a fair comparison |
| DeepEval | gpt-4o-mini | Most popular open-source eval library; uses different judge model |
| Keyword overlap | None | Zero-cost reference point — shows what you get with no LLM |

### What we don't do

- We don't run benchmarks only on cases our evaluators handle well.
- We don't tune hyperparameters on the test set.
- We publish failures — including cases where we're outperformed (see faithfulness).
- We disclose judge models for every evaluator so you can reproduce with different LLMs.

### Limitations

- HaluEval QA represents a specific distribution (Wikipedia-sourced, single-hop). Performance may vary on domain-specific or multi-hop contexts.
- The answer relevance golden set is self-curated. It has not been externally reviewed.
- All LLM judges are non-deterministic — individual runs may vary slightly. Numbers reported are single-run results, not averages across seeds.
- DeepEval results depend on the gpt-4o-mini API response at time of evaluation. We cannot guarantee reproducibility on a different date.

---

## Planned benchmarks

- [ ] Faithfulness — calibrated for long-document summarization (SummEval dataset)
- [ ] Agent tool call accuracy — custom golden set from real agent traces
- [ ] RAGAS comparison — faithfulness and context precision on real RAG pipelines
- [ ] Cost per evaluation — tokens used and cost vs. accuracy tradeoff table

---

## Reproducing results

```bash
git clone https://github.com/multivon-ai/multivon-eval
cd multivon-eval
pip install -e ".[dev]"
pip install deepeval
cp .env.example .env  # add ANTHROPIC_API_KEY and OPENAI_API_KEY

# Run individual benchmarks
python benchmarks/run_hallucination_benchmark.py   # ~8 min, 400 API calls
python benchmarks/run_faithfulness_benchmark.py    # ~6 min, 240 API calls
python benchmarks/run_relevance_benchmark.py       # ~4 min, 120 API calls

# Or run all at once
python benchmarks/run_all_benchmarks.py
```

Results are saved to `benchmarks/results/*.json` and can be diffed across versions.
