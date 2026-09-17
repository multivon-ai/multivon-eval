# First full-population Multivon judge comparison

September 17, 2026. **This study does not establish QAG superiority or SoTA.**
The existing `AnswerAccuracy` evaluator produced complete scores, but used more
calls than a direct judge. The quality difference is inconclusive and its point
estimate changes direction under a post-hoc formatting diagnostic.

## Frozen primary result

Both configurations use `claude-haiku-4-5-20251001`, temperature 0, the same query,
reference and response inputs, and a 100-token cap per call. All 280 comparison
cases and 560 responses were processed. Each case has two human annotations;
these are not 560 independent cases. No human labels, critiques, baseline scores
or model identities enter the prompts.

| Configuration | Overall Pearson | 95% case-bootstrap interval | Overall Spearman | Scored responses | Imputed pair differences |
|---|---:|---|---:|---:|---:|
| Multivon AnswerAccuracy | 0.4986 | 0.4129–0.5717 | 0.5087 | 560/560 | 0/280 |
| Direct rating, frozen integer-only parser | 0.4521 | 0.3661–0.5288 | 0.4613 | 513/560 | 37/280 |

The paired Pearson difference (Multivon minus direct) has a 95% interval of
**−0.0310 to +0.1287**. It includes zero. Because the direct baseline has incomplete
format coverage, the frozen protocol also disallows a headline quality win.
All 47 unparseable direct replies remain in the results; the upstream median
score-difference imputation supplies 37 missing pairs (median zero).
Multivon had no unparseable or partial-coverage QAG scores in this run.

The bootstrap uses 2,000 case resamples stratified by the ten domains, preserving
both annotations together. It conditions on this one set of model outputs and
does not address unverified shared-source dependence or model-sampling variance.
`results.json` contains exploratory correctness/completeness and per-domain
results, with n=28 per domain. No per-domain significance claim is made.

## Formatting diagnostic, explicitly post-hoc

QAG accepts a leading Yes/No verdict with explanations, while the frozen direct
parser required a complete integer-only reply. This asymmetry can affect the
comparison independently of semantic judgment quality.

Accepting a leading standalone integer followed by whitespace or end-of-text
recovers **44 of the 47** rejected direct replies. Without new inference, the
direct judge's overall Pearson becomes **0.5334** and Spearman **0.5318**; three
responses/pairs remain missing and imputed. The paired Pearson difference interval
becomes **−0.1166 to +0.0470**, again including zero. This reverses the point-estimate
ordering and reinforces the lack of evidence for a QAG advantage. It is not a
replacement primary result or a newly validated parser. See `parser-sensitivity.json`.

## Execution and cost

| Configuration | Observed API calls | Input tokens | Output tokens | Catalog estimate | Median response scoring latency | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Multivon AnswerAccuracy | 2,240 | 870,904 | 10,138 | $0.921594 | 3.081 s | 4.015 s |
| Direct rating | 560 | 260,146 | 7,101 | $0.295651 | 0.780 s | 1.717 s |

All **2,800/2,800** observed requests returned HTTP 200, with native usage and
the pinned response model. Accounting found no evidence gaps. The estimate totals
**$1.217245**, using the frozen local LiteLLM 1.101.0 catalog and native usage;
it is not a provider invoice. `accounting.json` records the catalog digest,
entry, supported standard-tier assumptions and provenance. QAG used four times
as many calls and about 3.12 times the estimated API cost.

The shuffled twelve-worker run completed in 201.7 seconds. Response latency is
the duration of one method's scoring job, including its calls, not queue waiting,
production latency or a guarantee. No cache was used and no extra HTTP retries
were observed. Raw wire data and the durable journal remain local because they
contain upstream text; their hashes are published, not their raw contents.

## Reproduce and inspect

Verify the primary correlations, coverage and paired interval using only the
published numeric predictions and pinned upstream labels, without API calls:

```sh
python benchmarks/industrial/verify_ragchecker_published.py --upstream /tmp/ragchecker \
  --bundle benchmarks/industrial/results/ragchecker-judges-2026-09-17
```

This arithmetic check cannot independently verify the unpublished wire journal,
the format of original direct replies, or provider billing.

The inference protocol and runner were committed before inference at
`57ae75c`. Upstream data/scorer revision:
`6091f08c00e676e87a970f2aeb4a23a484746348`.
See [frozen protocol](../../../../plans/ragchecker-first-comparison.md) and the
[upstream baseline reproduction](../ragchecker-meta-2026-09-17/README.md).

```sh
# With the pinned upstream checkout and ANTHROPIC_API_KEY set:
python benchmarks/industrial/run_ragchecker_judges.py \
  --upstream /tmp/ragchecker --out /tmp/ragchecker-live
python benchmarks/industrial/score_ragchecker_judges.py \
  --upstream /tmp/ragchecker --run /tmp/ragchecker-live --out /tmp/ragchecker-scores
python benchmarks/industrial/ragchecker_parser_sensitivity.py \
  --upstream /tmp/ragchecker --run /tmp/ragchecker-live --out /tmp/parser-sensitivity.json
LITELLM_LOCAL_MODEL_COST_MAP=True python benchmarks/industrial/account_ragchecker_judges.py \
  --run /tmp/ragchecker-live --out /tmp/accounting.json --price
```

Use a clean committed checkout for inference. Install the project, NumPy and
SciPy for scoring; LiteLLM is only needed for optional pricing. Exact execution
and analysis environment versions are retained in `environment.json`.
`predictions.json` publishes all 1,120 numeric judgments with case IDs and
method/side/status/latency, without upstream text. `results.json` retains the
protocol, source hashes, coverage and statistical results. `artifact-hashes.json`
covers this bundle except itself. No result was selected from repeated live runs.

The scorer executes the pinned upstream correlation function unchanged. The
wrapper has controls for perfect/reversed/constant predictions and paired
bootstrap identity. The inference projection and parser/error controls were
checked before running. Analysis replay is deterministic on saved predictions.

## What this changes

The historical released RAGChecker overall Pearson is 0.6193, above this
Multivon configuration, but it uses a different model and cannot isolate an
algorithmic effect. These historical baselines do not establish current rank.
Public-data contamination is unmeasured; regulated customer usefulness remains
unvalidated. Credit the RAGChecker authors and original data sources; the upstream
reproduction describes source-specific reuse limitations.

Generic QAG is not an evidenced moat. Prioritize transparent coverage, stable
grading contracts and domain-specific verification. A candidate contribution is
evidence-linked numerical and claim checking that reduces false acceptance while
preserving useful coverage. Develop it on a separate permitted development set,
then freeze a new held-out protocol with strong same-model and specialized
baselines. Do not optimize against this study's test labels or describe this
result as validation of that unimplemented hypothesis.
