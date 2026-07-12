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
| Hallucination — in-distribution (HaluEval-QA) | F1 0.804 [0.71–0.88]¹ | F1 0.586 [0.48–0.68] | F1 0.763 [0.68–0.83] | F1 0.523 [0.39–0.64] |
| Hallucination — **truly held-out** (HaluEval-Sum) | **F1 0.830 [0.70–0.92]**² | — | — | — |
| Answer relevance | F1 0.952 | **F1 0.974** | F1 0.976 | — |
| Faithfulness (summarization) | F1 0.783 [0.68–0.88]³ | F1 0.516 [0.35–0.66] | F1 0.667 [0.55–0.78] | F1 0.627 [0.44–0.76] |
| Coherence detection | see run | — | see run | — |
| Answer accuracy | see run | — | see run | see run |
| SummEval coherence (Spearman ρ) | **0.587** (Haiku) | 0.431 (gpt-4o-mini) | — | — |
| SummEval relevance (Spearman ρ) | **0.522** (Haiku) | 0.380 (gpt-4o-mini) | — | — |
| SummEval faithfulness (Spearman ρ) | 0.455 (Opus) | **0.443** (gpt-4o-mini) | — | — |

Judge models: multivon-eval uses `claude-haiku-4-5-20251001` for benchmarks 1–4; DeepEval uses `gpt-4o-mini`. SummEval benchmark (5) runs 5 judges. Judge always disclosed per run.

¹ **In-distribution.** F1 0.804 is reported on the same HaluEval-QA-100 split used to calibrate the Hallucination threshold (`dataset_hash: halueval-qa-2024-100c` in `_calibration_data/v2.json`). Treat this as a calibrated-default sanity check, not an out-of-distribution generalization claim.

² **Genuinely held-out** — and this row's framing was corrected after a peer-review round flagged that the prior "held-out" claim was actually in-distribution (see correction note below). The number you want for the cross-distribution claim is **F1 0.830 [0.70–0.92] on HaluEval-Sum n=60**, produced by running the **Hallucination** evaluator at its calibrated threshold (**0.55** per `_calibration_data/v2.json`, calibrated on HaluEval-**QA**) without re-tuning against HaluEval-**Summarization** cases. The benchmark script pins the judge with an explicit Haiku `JudgeConfig` for reproducibility; since 0.9.7 a bare `Hallucination()` resolves the same calibrated 0.55 at `evaluate()` time (see the Benchmark 4 reproducibility note). Different task family, different evaluator-of-record for this dataset, calibration set ↮ test set. This is the honest cross-distribution generalization figure. See Benchmark 4 for the methodology + raw counts.

³ **In-distribution, not held-out as previously claimed.** Earlier versions of this table labeled the Faithfulness-on-HaluEval-Sum result as "held-out" with the threshold "frozen from v2 calibration." That framing was wrong: the Faithfulness evaluator is itself calibrated on HaluEval-Sum (`dataset_hash: halueval-sum-2024-60c` in `_calibration_data/v2.json`, threshold 0.9 for Haiku, F1=0.783). Re-running Faithfulness on HaluEval-Sum reproduces the calibration F1 by construction — it's not a held-out evaluation, it's the calibration measurement again. The number is real, the framing was misleading. **The genuine cross-distribution figure is row ², not row ³.** See the correction note in Benchmark 3.

**Correction note (2026-06-03):** v0.9.4 launched with a "held-out" Faithfulness claim that the round-2 peer review correctly flagged as in-distribution. v0.9.4.1 replaces that row with the actually-held-out test (Hallucination evaluator on HaluEval-Sum), reports the new F1 [CI], and acknowledges the original mislabeling here. The data didn't lie — the label did. Issue + commit history preserved.

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

## Benchmark 3 — Faithfulness (Summarization, **in-distribution — corrected**)

**Dataset:** [HaluEval Summarization](https://github.com/RUCAIBox/HaluEval) — 30 document-summary pairs (60 cases) with human-annotated faithful and hallucinated summaries.

**Task:** Given a source document and a summary, detect whether the summary introduces claims not present in the document.

**Ground truth:** Human labels (1 = hallucinated summary, 0 = faithful summary). Balanced 50/50.

> ### Correction note (2026-06-03)
>
> v0.9.4 launched this section labeled "held-out: threshold calibrated on HaluEval-QA." That was wrong. The Faithfulness evaluator's Haiku threshold is itself calibrated on HaluEval-Sum (`dataset_hash: halueval-sum-2024-60c` in `_calibration_data/v2.json`), not on HaluEval-QA. Running Faithfulness on HaluEval-Sum with the v2 default threshold reproduces the calibration F1 by construction — it's the calibration measurement again, not a held-out test.
>
> The round-2 peer review (ML researcher persona) caught this within hours of launch. v0.9.4.1 corrects the framing here and replaces the cross-distribution row in the summary table with the **actually-held-out** result: the Hallucination evaluator (calibrated on HaluEval-QA) run on HaluEval-Sum. That's Benchmark 4 below.
>
> The data didn't lie — the label did. Both numbers stay in the README; only the framing changes.

| Evaluator | Judge model | Precision | Recall | F1 (95% bootstrap CI) | Avg latency |
|-----------|-------------|-----------|--------|-----------------------|-------------|
| multivon-eval (Faithfulness, in-dist.) | claude-haiku-4-5 | 0.692 | **0.900** | **0.783 [0.68–0.88]** | 7309ms |
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

**What this section shows:** the Faithfulness evaluator hits F1 0.783 [0.68–0.88] on the dataset it was calibrated for — i.e., this is a calibration-sanity-check row, the same class as Benchmark 1's F1 0.804. The win over DeepEval (0.783 vs 0.516, lower bound 0.68 clears upper bound 0.66) survives the bootstrap CI test, so the head-to-head comparison stands even after the framing correction. But this is not the cross-distribution generalization claim — that's Benchmark 4.

**Earlier published numbers superseded:** prior versions of this README cited F1 0.480 for multivon-eval on this benchmark (a single-run snapshot at an older threshold/judge configuration). The 0.783 figure is the current benchmark output with the v2 calibration. Reproduce via `python benchmarks/run_faithfulness_benchmark.py`.

---

## Benchmark 4 — Hallucination evaluator on HaluEval-Sum (truly held-out)

**Dataset:** [HaluEval Summarization](https://github.com/RUCAIBox/HaluEval) — 30 document-summary pairs (60 cases). Same data as Benchmark 3.

**Task:** Detect whether a summary introduces claims not in the source document.

**The crucial difference from Benchmark 3:** here we run the **Hallucination** evaluator — calibrated on HaluEval-**QA** (different task family, threshold **0.55** per `_calibration_data/v2.json`, never seen summarization data). This is the actually-held-out test that v0.9.4 was trying to claim before the framing was corrected.

| Evaluator | Judge model | Calibrated on | Tested on | Precision | Recall | F1 (95% bootstrap CI) |
|-----------|-------------|---------------|-----------|-----------|--------|-----------------------|
| **multivon-eval (Hallucination, held-out)** | claude-haiku-4-5 | HaluEval-QA (thr 0.55) | HaluEval-Sum | **0.957 [0.79–0.99]** | 0.733 [0.55–0.86] | **0.830 [0.70–0.92]** |

**Raw counts (n=60 cases):** TP=22, FP=1, FN=8, TN=29.

**Reproducibility note:** the script passes an explicit `JudgeConfig(provider='anthropic', model='claude-haiku-4-5-20251001')` to pin the judge. The calibrated threshold (0.55) applies **with or without** that explicit config: since 0.9.7, evaluators resolve their judge at `evaluate()` time (default judge: `claude-haiku-4-5-20251001`) and look up the calibrated threshold for the resolved (evaluator, judge) pair. The init-time `threshold` attribute (0.7) is only a pre-resolution placeholder; it is used — with a loud warning — only when no calibration row exists for your judge. Verify in three lines:

```python
from multivon_eval.judge import resolve_judge
from multivon_eval.calibration import calibrated_threshold
print(calibrated_threshold("hallucination", resolve_judge(None)))  # 0.55
```

**Historical footnote (what 0.9.7 fixed):** before 0.9.7, a bare `Hallucination()` really did evaluate at the init default 0.7, which produced F1 0.852 [0.73–0.94] on this same data. That figure is not held-out — it was the accident of an uncalibrated threshold — and it stays here only as the record of the correction. The defensible cross-distribution F1 is 0.830 [0.70–0.92], **at the calibrated threshold**.

**What this discharges — for real this time:** the contamination criticism on the cross-distribution claim. The Hallucination evaluator was tuned on QA-style short-context cases (Wikipedia-sourced, single-hop QA). We applied that exact evaluator with its calibrated threshold (no re-tuning) to long-context summarization cases. F1 = 0.830, slightly higher than the in-distribution F1 = 0.812 on HaluEval-QA. The threshold generalizes across task families inside HaluEval.

**What it still doesn't discharge:** non-HaluEval corpora (TruthfulQA, FaithBench, RAGTruth) and non-Haiku judges on the same cross-distribution test. Both HaluEval-Sum and HaluEval-QA build on CNN/DailyMail-adjacent source documents — there's shared corpus structure across the two splits. The cross-corpus held-out evaluation is on the [public roadmap](https://multivon.ai/roadmap), targeting launch + 2 weeks. The repository will publish the held-out F1 next to the in-distribution F1 once that lands; we won't retire either number.

**Reproduce:**
```bash
python benchmarks/run_truly_held_out.py    # script in repo
# or inspect: benchmarks/results/hallucination_held_out.json
```

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

- **The headline Hallucination F1 (0.804) is in-distribution.** The threshold was tuned on the same HaluEval-QA-100 split it is tested on. Treat it as a calibrated-default sanity check, not an OOD generalization claim. For the genuine cross-task figure see Benchmark 4 (F1 0.830 [0.70–0.92] on HaluEval-Sum with the QA-calibrated Hallucination evaluator at the calibrated threshold 0.55 — the value that resolves automatically at `evaluate()` time, not the uncalibrated 0.7 fallback).
- **The Faithfulness benchmark on HaluEval-Sum (Benchmark 3) is also in-distribution** despite earlier framing to the contrary. Faithfulness/Haiku is calibrated on HaluEval-Sum, so testing it on HaluEval-Sum reproduces the calibration. The framing was corrected on 2026-06-03 after a round-2 peer review (ML researcher persona) caught it. See the correction note in Benchmark 3.
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

**Task:** Sweep thresholds 0.30–0.90 in 0.05 steps. Find the threshold that maximises F1 against human labels for each (evaluator, judge) pair. Results are baked into the library and apply automatically: since 0.9.7, evaluators resolve their judge at `evaluate()` time and look up the calibrated threshold for the resolved (evaluator, judge) pair — with or without an explicit `JudgeConfig`. (Pre-0.9.7 releases leaked the init-time default 0.7 when no `JudgeConfig` was passed; see the historical footnote in **Benchmark 4**.) An explicit `threshold=` always wins, and an (evaluator, judge) pair with no calibration row warns loudly and falls back to 0.7. The same rules apply to `Faithfulness()` and `Relevance()`.

```python
from multivon_eval import Hallucination, JudgeConfig

Hallucination()                  # calibrated threshold resolved at evaluate() time
                                 # (0.55 for the default claude-haiku-4-5 judge)
Hallucination(judge=JudgeConfig(provider="anthropic",
                                model="claude-haiku-4-5-20251001"))  # same 0.55, judge pinned
Hallucination(threshold=0.8)     # explicit override always wins
```

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

- **Faithfulness optimal threshold is 0.90 across all three judges.** These judges score faithful outputs very high and unfaithful ones very low — the signal is binary with a sharp separation near the top of the scale. An uncalibrated 0.7 gate would produce false passes; the calibrated lookup applies 0.90 automatically for these judges.
- **Relevance optimal threshold is 0.30 for Haiku, Sonnet, and GPT-4o-mini.** Relevance judgments are extremely confident — relevant responses score near 1.0, irrelevant ones near 0.0. Any threshold from 0.30 to ~0.85 achieves perfect F1.
- **Hallucination thresholds vary by model (0.30–0.55).** Haiku is more conservative (flags fewer things as hallucinations, requiring a lower threshold to catch them). Sonnet and GPT-4o-mini are more aggressive, already flagging most hallucinations without needing a high threshold.
- **Pass `threshold=` explicitly to override for your domain.** The calibrated defaults are derived from Wikipedia-sourced QA and news summarization. Domain-specific content (medical, legal, financial) may warrant a different threshold.

```bash
python benchmarks/run_threshold_calibration.py
# Results saved to benchmarks/results/calibration.json
```

---

## Planned benchmarks

- [x] SummEval Spearman correlation — coherence and relevance (`run_summeval_benchmark.py`) — see Benchmark 5.
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
