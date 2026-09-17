# Public benchmark program for Multivon

September 17, 2026. Researched in response to the owner's Supermemory comparison.
**No Multivon SoTA result exists on the benchmarks below at this checkpoint.**
Unit-test counts, synthetic recovery tests and task scores are not substitutes
for independent measurements of evaluator accuracy.

## The appropriate claim

Supermemory publishes a [LongMemEval-S result](https://supermemory.ai/research/longmembench/)
for its memory system. Its page labels the headline as 95% Recall@15 with
aggregation and also presents an LLM-judge results table. Those are vendor-reported
results, not independently reproduced here. A comparable Multivon claim should
name the verification system, frozen benchmark/metric, model, cost and date.

The most relevant target is **accurate, efficient verification of answers grounded
in enterprise documents**. A release artifact can then use those judgments and
independent state checks. “Best evaluation library” has no single meaningful
accuracy metric: an execution harness and a learned/prompted verifier are different
objects of evaluation.

## Established benchmarks to target

| Priority | Benchmark | What it measures | Multivon's role and limits |
|---|---|---|---|
| 1 | [LLM-AggreFact](https://llm-aggrefact.github.io/) | Grounded factuality across 11 constituent datasets; use the official balanced-accuracy aggregation | Evaluate a claim-verification pipeline. Strong match for faithfulness/hallucination checks; does not measure database state or regulatory compliance |
| 1 | [RAGChecker meta-evaluation](https://github.com/amazon-science/RAGChecker/tree/main/data/meta_evaluation) | Agreement with human evaluation of RAG outputs | Direct evaluation of an evaluator. Official files contain 280 instances with two annotators, yielding 560 annotation records; do not count these as 560 independent tasks |
| 2 | [RewardBench 2](https://github.com/allenai/reward-bench) | Reward/judge discrimination under the official multi-response and tie handling | Useful broader judge benchmark; general preferences/correctness differ from evidence-grounded enterprise acceptance |
| 2 | [JudgeBench](https://github.com/ScalerLab/JudgeBench) | Choosing the correct response in difficult pairs | Existing harness, response-order handling and public submission path; report both orders and use official scoring |
| Later | [Multimodal RewardBench 2](https://github.com/facebookresearch/MMRB2) | Judging multimodal outputs | Relevant if building a general multimodal judge. Its generation/editing profile does not directly validate document extraction or financial units |

For LLM-AggreFact, the [official evaluation notebook](https://github.com/Liyan06/MiniCheck/blob/main/benchmark_evaluation_demo.ipynb)
computes balanced accuracy for each constituent dataset, then averages those
values. Preserve that macro aggregation instead of pooling all examples.

[MiniCheck](https://github.com/Liyan06/MiniCheck) is an example of a project built
around a focused verification capability and public evidence, rather than a
generic framework claim. [RAGChecker](https://github.com/amazon-science/RAGChecker)
is especially relevant because its publication includes meta-evaluation of its
metrics against human judgments.

## Access and reuse constraints

[LLM-AggreFact's dataset card](https://huggingface.co/datasets/lytang/LLM-AggreFact)
lists CC BY-ND 4.0, gated access with contact-information sharing, and evaluation
use only: it explicitly prohibits pretraining or fine-tuning on the benchmark.
No access request or contact sharing was performed. Treat it as held-out
evaluation, preserve its contamination identifiers, and resolve access through
the owner's authorized account. Do not bypass the gate through a mirror.

[RewardBench 2's card](https://huggingface.co/datasets/allenai/reward-bench-2)
lists ODC-BY; retain required attribution and source notices. Check the pinned
data, scorer and any baseline weights separately. MiniCheck's Apache-licensed
code is not blanket commercial permission for every associated model: its README
separately discusses commercial licensing for Bespoke-MiniCheck-7B.

Start with the accessible RAGChecker meta-evaluation and released baseline
outputs. Reproduce their official scoring before adding Multivon. Historical
baseline files are valuable reproducibility checks, but a current SoTA comparison
also needs current strong baselines under a matching protocol.

The [first native reproduction](../benchmarks/industrial/results/ragchecker-meta-2026-09-17/README.md)
now matches the published RAGChecker Table 5 correlations on all 280 cases,
using unchanged upstream scoring and released predictions. The bundle records
data alignment, missing-score imputation, dependencies and artifact hashes.
This establishes a historical reference, not a Multivon result or fresh baseline
model execution.

## Technical hypothesis to test

Use a modular verification pipeline: identify checkable claims, bind them to the
provided evidence, apply deterministic numeric/unit checks when applicable, and
use a semantic judge for remaining claims. Return uncertainty when evidence or
judgment is missing. Calibrate decision thresholds only on allowed development
data. Reuse existing models and scoring code rather than training a foundation
model or writing another benchmark harness.

This is a hypothesis. Decomposition can omit difficult claims; numeric tools can
be supplied incorrect operands; judge ensembles can repeat correlated mistakes.
Expose these failure modes through ablations. A stronger underlying model may
explain any gain, so compare the same model without Multivon's added machinery.

## Measurement and claim discipline

1. Pin upstream data/scorer revisions, license notes, input projection, development
   selection and test protocol. Keep test labels out of prompts and tuning.
2. Reproduce official baselines and score saved judgments with official scripts.
   Preserve malformed outputs, provider failures, timeouts and abstentions.
3. Compare at least a direct same-model judge, an established specialized verifier
   where licensed/available, and the applicable upstream RAGChecker baseline.
   Keep input evidence, model revisions and token/call budgets matched where the
   comparison permits. Report mismatches explicitly.
4. Publish the official primary metric on the complete required population.
   Add cost, latency and source-grouped uncertainty. Report per-dataset results
   and uncertainty for differences; no selective subset should become an overall
   headline. Snapshot the leaderboard on the date of any ranking claim.
5. Also measure false acceptance of unsupported answers, false rejection of
   supported answers, and abstention versus coverage. Denote denominators: false
   accept among unsupported examples differs from error among accepted outputs.
   Do not gain apparent accuracy by dropping abstentions or operational failures.
6. Run a separate financial-document workflow study using
   [the researched public target](regulated-enterprise-study.md). Show whether
   verifier gains improve saved-record correctness and reviewer workload.

The first checkpoint should produce a credible baseline, not a marketing number.
Only after full held-out evaluation could we claim a leading result on a named
benchmark and date, or better accuracy at a specified cost. A narrow cost/accuracy
tradeoff can be valuable without being the overall leaderboard leader.

## Contribution and moat are different

A publishable contribution might demonstrate reduced false acceptance under
controlled evidence changes, numeric ambiguity or interrupted agent execution,
while preserving useful coverage. That requires matched baselines and independent
labels; our existing synthetic experiments are motivation, not proof of broad
novelty. Label any new stress tests as a Multivon extension, alongside unchanged
upstream benchmark scores.

A commercial moat would require trusted integration into repeated enterprise
release reviews and accumulated permissioned domain validation. Benchmark
leadership may establish credibility, but it can be copied or overtaken. Do not
replace the upstream dataset ecosystem with a proprietary generic dataset engine.
