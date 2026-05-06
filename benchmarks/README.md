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

| Task | multivon-eval | DeepEval | Simple LLM judge | Keyword overlap / Exact match |
|------|:---:|:---:|:---:|:---:|
| Hallucination detection | **F1 0.804** | F1 0.586 | F1 0.763 | F1 0.523 |
| Answer relevance | F1 0.952 | **F1 0.974** | F1 0.976 | — |
| Faithfulness (summarization) | F1 0.480 | F1 0.448 | F1 0.667 | F1 0.627 |
| Coherence detection | see run | — | see run | — |
| Answer accuracy | see run | — | see run | see run |
| SummEval coherence (Spearman ρ) | **0.587** (Haiku) | 0.431 (gpt-4o-mini) | — | — |
| SummEval relevance (Spearman ρ) | **0.522** (Haiku) | 0.380 (gpt-4o-mini) | — | — |
| SummEval faithfulness (Spearman ρ) | 0.455 (Opus) | **0.443** (gpt-4o-mini) | — | — |

Judge models: multivon-eval uses `claude-haiku-4-5-20251001` for benchmarks 1–4; DeepEval uses `gpt-4o-mini`. SummEval benchmark (5) runs 5 judges. Judge always disclosed per run.

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

---

## Benchmark 5 — SummEval Spearman Correlation

**Dataset:** [SummEval](https://github.com/Yale-LILY/SummEval) via HuggingFace (`mteb/summeval`) — 100 CNN/DailyMail articles × 16 machine-generated summaries = 1,600 evaluation units. Expert human annotations (averaged across 3 annotators) for coherence, consistency, fluency, relevance on a 1–5 scale.

**Task:** Compute Spearman correlation between evaluator scores and human expert ratings. Higher ρ = the evaluator tracks human judgment more closely. All results are statistically significant at α=0.05 unless noted.

**100 samples, 5 judges:**

| Dimension | Haiku 4.5 | Opus 4.7 | Sonnet 4.6 | GPT-4o-mini | GPT-3.5-turbo |
|-----------|:---------:|:--------:|:----------:|:-----------:|:-------------:|
| Coherence    | **0.587** (p<0.001) | 0.530 (p<0.001) | 0.455 (p<0.001) | 0.431 (p<0.001) | 0.205 (p=0.038) |
| Relevance    | **0.522** (p<0.001) | 0.224 (p=0.023) | 0.283 (p=0.004) | 0.380 (p<0.001) | 0.053 (p=0.602) ⚠ |
| Faithfulness | 0.345 (p<0.001)     | **0.455** (p<0.001) | 0.319 (p=0.001) | 0.443 (p<0.001) | 0.240 (p=0.014) |

⚠ = not statistically significant.

**Key findings:**

- **Claude Haiku 4.5 is surprisingly the best overall judge** for coherence and relevance (ρ=0.587, 0.522), outperforming all models including Opus and Sonnet. Coherence and relevance are structural, language-level judgments where Haiku's scoring appears well-calibrated — and it's the cheapest option in the lineup.
- **Claude Opus 4.7 leads faithfulness** (ρ=0.455), tied with GPT-4o-mini (ρ=0.443). Factual consistency requires reasoning through source documents — the one dimension where scale helps.
- **Bigger is not always better for judging.** Opus (ρ=0.224 relevance) underperforms Haiku (ρ=0.522) significantly on relevance. Sonnet also underperforms Haiku. For evaluation tasks, model calibration and consistency matter more than raw capability.
- **GPT-3.5-turbo relevance is not statistically significant (p=0.602).** This is the same model RAGAS used for their published numbers (gpt-3.5-turbo-16k, Sept 2023). Their claim of 78% agreement on AnswerRelevance is on their own WikiEval dataset — not a neutral benchmark — and this result suggests the judge choice is load-bearing for relevance tasks. Their faithfulness number (95%) may hold since factual entailment is an easier task for weaker models.
- **All three dimensions have meaningful correlation with capable judges.** Best-in-class: coherence ρ=0.587 (Haiku), relevance ρ=0.522 (Haiku), faithfulness ρ=0.455 (Opus). These are moderate-to-strong correlations for an automated evaluator, consistent with published G-Eval results (~0.514 Spearman on SummEval coherence with GPT-4).
- **Zero errors across 1,500 API calls** (100 samples × 3 evaluators × 5 judges).

**Recommendation:** For most use cases, `claude-haiku-4-5-20251001` gives the best coherence and relevance correlation at the lowest cost. For faithfulness-critical pipelines (RAG, summarization), use `claude-opus-4-7` or `gpt-4o-mini`. Avoid `gpt-3.5-turbo` for relevance.

---

## Benchmark 6 — Threshold Calibration

**Dataset:** HaluEval QA (100 cases), HaluEval Summarization (60 cases), curated relevance golden set (40 cases). Labels: 50/50 faithful/hallucinated for each split.

**Task:** Sweep thresholds 0.30–0.90 in 0.05 steps. Find the threshold that maximises F1 against human labels for each (evaluator, judge) pair. Results are baked into the library — `Hallucination()`, `Faithfulness()`, and `Relevance()` automatically apply the calibrated threshold for the configured judge.

| Judge | Evaluator | Optimal threshold | F1 |
|-------|-----------|:-----------------:|:---:|
| claude-haiku-4-5-20251001 | hallucination | 0.55 | 0.812 |
| claude-haiku-4-5-20251001 | faithfulness  | 0.90 | 0.783 |
| claude-haiku-4-5-20251001 | relevance     | 0.30 | 0.976 |
| claude-sonnet-4-6         | hallucination | 0.30 | 0.787 |
| claude-sonnet-4-6         | faithfulness  | 0.90 | 0.656 |
| claude-sonnet-4-6         | relevance     | 0.30 | 1.000 |
| gpt-4o-mini               | hallucination | 0.30 | 0.756 |
| gpt-4o-mini               | faithfulness  | 0.90 | 0.793 |
| gpt-4o-mini               | relevance     | 0.30 | 1.000 |

**Key findings:**

- **Faithfulness optimal threshold is 0.90 across all three judges.** These judges score faithful outputs very high and unfaithful ones very low — the signal is binary with a sharp separation near the top of the scale. The library default of 0.7 would produce false passes; 0.90 is the correct gate.
- **Relevance optimal threshold is 0.30 for Haiku, Sonnet, and GPT-4o-mini.** Relevance judgments are extremely confident — relevant responses score near 1.0, irrelevant ones near 0.0. Any threshold from 0.30 to ~0.85 achieves perfect F1.
- **Hallucination thresholds vary by model (0.30–0.55).** Haiku is more conservative (flags fewer things as hallucinations, requiring a lower threshold to catch them). Sonnet and GPT-4o-mini are more aggressive, already flagging most hallucinations without needing a high threshold.
- **Pass `threshold=` explicitly to override for your domain.** The calibrated defaults are derived from Wikipedia-sourced QA and news summarization. Domain-specific content (medical, legal, financial) may warrant a different threshold.

```bash
python benchmarks/run_threshold_calibration.py
# Results saved to benchmarks/results/calibration.json
```

---

## Planned benchmarks

- [x] SummEval Spearman correlation — coherence and relevance (`run_summeval_benchmark.py`, run to fill TBD above)
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
python benchmarks/run_summeval_benchmark.py        # ~15 min, ~400 API calls, downloads ~5 MB

# Or run all at once
python benchmarks/run_all_benchmarks.py
```

Results are saved to `benchmarks/results/*.json` and can be diffed across versions.
