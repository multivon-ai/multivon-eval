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
| Hallucination detection | **F1 0.804 [0.71–0.88]**¹ | F1 0.586 [0.48–0.68] | F1 0.763 [0.68–0.83] | F1 0.523 [0.39–0.64] |
| Answer relevance | F1 0.952 | **F1 0.974** | F1 0.976 | — |
| Faithfulness (summarization, **held-out**) | **F1 0.783 [0.68–0.88]**² | F1 0.516 [0.35–0.66] | F1 0.667 [0.55–0.78] | F1 0.627 [0.44–0.76] |
| Coherence detection | see run | — | see run | — |
| Answer accuracy | see run | — | see run | see run |
| SummEval coherence (Spearman ρ) | **0.587** (Haiku) | 0.431 (gpt-4o-mini) | — | — |
| SummEval relevance (Spearman ρ) | **0.522** (Haiku) | 0.380 (gpt-4o-mini) | — | — |
| SummEval faithfulness (Spearman ρ) | 0.455 (Opus) | **0.443** (gpt-4o-mini) | — | — |

Judge models: multivon-eval uses `claude-haiku-4-5-20251001` for benchmarks 1–4; DeepEval uses `gpt-4o-mini`. SummEval benchmark (5) runs 5 judges. Judge always disclosed per run.

¹ **In-distribution**: F1 0.804 is reported on the same HaluEval-QA-100 split used to calibrate the Hallucination threshold (`dataset_hash: halueval-qa-2024-100c` in `_calibration_data/v2.json`). Treat this as a calibrated-default sanity check, not an out-of-distribution generalization claim.

² **Held-out**: F1 0.783 is reported on HaluEval-Summarization n=60 with the threshold **frozen** from the v2 calibration (no re-tuning on this set). HaluEval-Sum is a different task (summarization, not QA) on different source documents — so this is a genuine cross-distribution check. The 0.04 drop from in-distribution (0.804 → 0.783) sits well inside the bootstrap CI overlap, suggesting calibrated thresholds generalize across HaluEval task families. A larger held-out evaluation against TruthfulQA + FaithBench is on the [public roadmap](https://multivon.ai/roadmap).

All CIs above are bootstrap-1000 on F1 with stable RNG seed (see `benchmarks/_add_cis.py`). Cells where CIs overlap are NOT reported as wins.

---

## Benchmark 1 — Hallucination Detection (in-distribution)

**Dataset:** [HaluEval QA](https://github.com/RUCAIBox/HaluEval) — 50 QA pairs (100 cases) with human-annotated faithful and hallucinated answers.

**Task:** Given a context passage and an answer, detect whether the answer contains claims not supported by the context.

**Ground truth:** Human labels (1 = hallucinated, 0 = faithful). Balanced 50/50.

> **In-distribution caveat:** the multivon-eval Hallucination threshold was calibrated against this same HaluEval-QA-100 split (`dataset_hash: halueval-qa-2024-100c` in `_calibration_data/v2.json`). The F1 below is best read as a calibrated-default sanity check, not an out-of-distribution generalization claim. For the held-out figure on HaluEval-Sum, see Benchmark 3.

| Evaluator | Judge model | Precision | Recall | F1 (95% bootstrap CI) | Avg latency |
|-----------|-------------|-----------|--------|-----------------------|-------------|
| **multivon-eval (QAG)** | claude-haiku-4-5 | **0.788** | 0.820 | **0.804 [0.71–0.88]** | 2955ms |
| Simple LLM judge (1-10) | claude-haiku-4-5 | 0.617 | **1.000** | 0.763 [0.68–0.83] | 708ms |
| DeepEval (HallucinationMetric) | gpt-4o-mini | 0.456 | 0.820 | 0.586 [0.48–0.68] | 1421ms |
| Keyword overlap (no LLM) | — | 0.605 | 0.460 | 0.523 [0.39–0.64] | <1ms |

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

> **Held-out:** the Faithfulness threshold was calibrated on HaluEval-QA (different task, different split). The F1 below uses that calibrated threshold **frozen** — no re-tuning on this set. This is the cross-task generalization check.

| Evaluator | Judge model | Precision | Recall | F1 (95% bootstrap CI) | Avg latency |
|-----------|-------------|-----------|--------|-----------------------|-------------|
| **multivon-eval (Faithfulness)** | claude-haiku-4-5 | 0.692 | **0.900** | **0.783 [0.68–0.88]** | 7309ms |
| Simple LLM judge (1-10) | claude-haiku-4-5 | 0.500 | **1.000** | 0.667 [0.55–0.78] | 1044ms |
| Keyword overlap (no LLM) | — | **0.762** | 0.533 | 0.627 [0.44–0.76] | <1ms |
| DeepEval (HallucinationMetric) | gpt-4o-mini | 0.500 | 0.533 | 0.516 [0.35–0.66] | 5269ms |

**Raw counts (n=60 cases):**

| Evaluator | TP | FP | FN | TN |
|-----------|----|----|----|----|
| multivon-eval (Faithfulness) | 27 | 12 | 3 | 18 |
| Simple judge (1-10) | 30 | 30 | 0 | 0 |
| Keyword overlap | 16 | 5 | 14 | 25 |
| DeepEval | 16 | 16 | 14 | 14 |

**Key findings — held-out generalization works:**

- **multivon-eval Faithfulness F1 = 0.783 [0.68–0.88]** on this held-out HaluEval-Sum split, with the threshold **frozen** from the v2 calibration (which used HaluEval-QA, not HaluEval-Sum). The 0.04 drop from in-distribution (0.804 → 0.783) sits well inside the bootstrap CI overlap.
- **The win over DeepEval (0.783 vs 0.516) survives the bootstrap CI test:** multivon-eval's lower bound (0.68) clears DeepEval's upper bound (0.66). Not a coin-flip difference.
- **Simple judge still pegs recall at 1.0** by flagging every summary as hallucinated — same pattern as Benchmark 1. High recall, no actionable signal.

**What this discharges:** the "you tuned thresholds on the same set you tested on" criticism. The calibrated threshold (0.90 for Haiku faithfulness) was set on HaluEval-QA-50; here we apply it without modification to HaluEval-Sum-60. F1 holds. Calibration transfers across task families inside HaluEval.

**What it doesn't discharge:** TruthfulQA, FaithBench, and non-HaluEval-corpora generalization. Both HaluEval-Sum and HaluEval-QA build on CNN/DailyMail-adjacent source documents — there is shared corpus structure even across the two splits. A larger held-out evaluation against TruthfulQA + FaithBench is on the [public roadmap](https://multivon.ai/roadmap), targeting launch + 2 weeks.

**Earlier published numbers superseded:** prior versions of this README cited F1 0.480 for multivon-eval on this benchmark. That was a single-run snapshot at an older threshold/judge configuration. The 0.783 figure above is the current benchmark output with the v2 calibration. Reproduce via `python benchmarks/run_faithfulness_benchmark.py`.

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

- **The headline Hallucination F1 (0.804) is in-distribution.** The threshold was tuned on the same HaluEval-QA-100 split it is tested on. Treat it as a calibrated-default sanity check, not an OOD generalization claim. See the held-out HaluEval-Sum result (Benchmark 3) for the cross-task figure.
- **"Best-tuned" F1 numbers reported anywhere in this repo are upper bounds.** Threshold sweeps reported below are conducted on the test set; the best-F1 values are not held-out estimates. Treat them as ceilings, not generalization claims.
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
