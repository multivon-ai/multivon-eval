# RAGChecker published-prediction reproduction

September 17, 2026. **This is an upstream baseline reproduction, not a Multivon
accuracy result or a current leaderboard comparison.** No model inference was
performed. The unchanged native scorer processed released predictions from five
baseline frameworks and regenerated its three plots.

## Result

The 90 correlation values in the native output match Table 5 of the
[RAGChecker paper](https://arxiv.org/html/2408.08067) at its reported two-decimal
precision. Each metric uses 280 response-pair cases across ten domains, with two
human annotations per case (560 annotation records). The native output reports
correlations multiplied by 100; this table uses the ordinary [-1, 1] scale.

| Human dimension | RAGChecker Pearson | RAGChecker Spearman | RAGAS answer-similarity Pearson | RAGAS answer-similarity Spearman |
|---|---:|---:|---:|---:|
| Correctness | 0.4966 | 0.4695 | 0.4107 | 0.4321 |
| Completeness | 0.6067 | 0.5811 | 0.5316 | 0.6135 |
| Overall | 0.6193 | 0.6090 | 0.4831 | 0.5723 |

These are correlations, not accuracy percentages. RAGChecker does not lead every
column: RAGAS answer similarity has higher completeness Spearman correlation in
these released predictions. No confidence interval or significance claim is
made here. Inter-annotator correlation is a comparison, not a mathematical upper
bound on evaluator performance.

## Data and protocol audit

- All five baseline files match both human records on instance, domain, question
  ID, query, reference answer, model names and responses: zero conflicts.
- There are 280 unique instance IDs and 280 unique `(dataset, query_id)` keys,
  with 28 cases per domain. This does not establish source-document independence.
- The native scorer imputes missing score differences with the metric median:
  TruLens groundedness 14/280; RAGAS faithfulness 27/280; RAGAS answer correctness
  17/280. All other scored metrics have zero missing differences. These are
  metric-specific counts, not 58 distinct failed cases.
- Score differences are normalized and duplicated for the paired annotations.
  Treating the 560 labels as independent cases would overstate sample size.
- Exact human agreement is 140/280 for correctness, 168/280 for completeness and
  155/280 for overall. Within-one agreement is 254/280, 257/280 and 253/280,
  respectively. Disagreement must remain visible in subsequent comparisons.

## Reproduction

The upstream source is pinned to
[`6091f08c00e676e87a970f2aeb4a23a484746348`](https://github.com/amazon-science/RAGChecker/tree/6091f08c00e676e87a970f2aeb4a23a484746348/data/meta_evaluation).
The wrapper verifies consumed working files against that Git object and stages
them unchanged in a temporary directory. It rejects misaligned data and writes
the input audit and protocol manifest before invoking the native scorer.

```sh
git clone https://github.com/amazon-science/RAGChecker.git /tmp/ragchecker
git -C /tmp/ragchecker checkout 6091f08c00e676e87a970f2aeb4a23a484746348
python3.12 -m venv /tmp/ragchecker-repro
/tmp/ragchecker-repro/bin/pip install -r benchmarks/industrial/results/ragchecker-meta-2026-09-17/requirements-freeze.txt
/tmp/ragchecker-repro/bin/python benchmarks/industrial/reproduce_ragchecker_meta.py \
  --upstream /tmp/ragchecker --out /tmp/ragchecker-reproduction
```

Kaleido also needs a compatible Chrome installation. Renderer/platform changes
may change PNG bytes; compare numeric results separately. The recorded run used
Python 3.12.13 on macOS. The runner parses on Python 3.10; native scoring was only
executed in the recorded 3.12 environment. Browser rendering was not network
sandboxed. No credentials or model clients were used.

`manifest.json` records scorer/data/runner hashes and key dependency versions;
`requirements-freeze.txt` retains the environment. `audit.json` records alignment,
missing values and human agreement. `meta_eval_results.json`, `native-scorer.log`,
`execution.json` and the PNGs are native execution evidence. `artifact-hashes.json`
covers the generated artifacts, excluding itself and this explanatory README.

Validation accepted the original dataset and rejected four corrupted controls:
a mismatched second annotation, a missing baseline case, an out-of-range human
label and an infinite baseline score. Ruff and Python 3.10 compilation passed.

## Reuse and next comparison

Credit: Dongyu Ru and colleagues, RAGChecker, Amazon Science. The upstream code
uses Apache-2.0. Its paper describes different licenses for underlying datasets,
including CC BY-SA and unspecified LoTTE corpus terms. This repository retains
aggregate evidence and source hashes, not the raw corpus. A commercial data
bundle needs a separate source-specific review.

Next, freeze a Multivon-versus-direct-judge protocol using identical query,
reference and response inputs. Keep human labels, critiques and released scores
out of judge prompts. Existing `AnswerAccuracy` is a coarse reference-based
baseline; `Faithfulness` against retrieved context measures a different construct
and cannot simply be relabeled human correctness. Compare the same model with
and without added verification machinery, preserve failures and coverage, and
resample paired annotations together for uncertainty. Historical released
baselines alone cannot substantiate a September 2026 SoTA claim.
