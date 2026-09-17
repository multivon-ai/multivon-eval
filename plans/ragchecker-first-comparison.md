# First full-population Multivon meta-evaluation protocol

Frozen before model inference; September 17, 2026. This is an initial measurement,
not a SoTA submission. Use all 280 cases from the pinned RAGChecker reproduction.

Compare the existing, unchanged `AnswerAccuracy` evaluator with one direct
0–100 overall-quality rating per response. Both use the pinned
`claude-haiku-4-5-20251001`, temperature 0, no cache, and a 100-token ceiling per
call. QAG uses four yes/no calls per response; direct rating uses one. This is
a same-model comparison with unequal computation, so report usage and latency.
It is not an ablation that isolates every prompt difference.

Inputs are only query, canonical answer and one response. The input projection
excludes human labels, critiques, baseline metric scores and model identities.
Do not tune prompts, thresholds, models or selection against this test. Historical
published values have already been inspected for reproduction; that fact and
possible foundation-model contamination preclude claiming pristine blind testing.

Primary endpoint: Pearson correlation against overall human preference, using
model2 score minus model1 score and the original two-annotation pairing.
Secondary: overall Spearman. Correctness and completeness are exploratory.
Use the upstream correlation function and report its native scale (×100) along
with ordinary correlations when useful. No binary threshold calibration occurs.

All 560 responses receive both methods (1,120 jobs; 2,800 planned logical model
calls). A fixed shuffled schedule with twelve workers interleaves the methods.
Retain failed and unparseable results. For descriptive full-population scoring,
use the upstream median-difference imputation, report its count, and make no
headline quality claim if coverage is incomplete. Partial QAG verdict coverage
must be disclosed separately even when the existing evaluator returns a score.

Use 2,000 paired case-bootstrap resamples stratified by the ten domains with seed
17092026. Resample each case's two annotations together and compare methods within
the same draw. These intervals cover case sampling conditional on this one
realization of model outputs; they do not cover repeated model sampling or unknown
shared-source dependence. Source-document identifiers have not been verified.
Show per-domain correlations with n=28, avoiding per-domain significance claims.

Retain raw requests, responses and usage in the existing durable ProviderJournal.
Publish aggregate results and numeric per-case predictions with identifiers;
keep raw upstream text/provider payloads local pending source-specific reuse
review. Provider failures, unknown billing and missing capture cannot be silently
dropped. List-price estimates, if reported, are not invoices.

Interpretation is bounded: historical RAGChecker Llama3-based outputs differ in
model and protocol details. Beating those alone would not establish current SoTA
or show that Multivon's mechanism caused the gain. A weak result should guide
verifier redesign and be published, not trigger test-driven prompt selection.
Any redesigned method needs a separate development set and new validation plan.
