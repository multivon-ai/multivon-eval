# Document-to-ledger study, 2026-09-17

Both models fail the frozen zero-failure acceptance policy. The experiment found
wrong persisted amounts, a missing post, and weaknesses in the mapping from a
public transcription label to a business outcome. It does **not** establish
SoTA performance, production readiness, customer value or a defensible moat.

[Protocol and deviations](DOCUMENT_PROTOCOL.md) ·
[Machine-readable results](results/document-ledger-2026-09-17/heldout-analysis.json) ·
[Failure review](results/document-ledger-2026-09-17/failure-review.json) ·
[Attribution](THIRD_PARTY.md) ·
[Raw evidence and offline reproduction](REPRODUCE_DOCUMENT_RESULTS.md)

## Frozen held-out results

39 source documents, two treatments each, two models: 156 completed scored
cases. CORD test rows 0–19 were selected; row 9 had no scalar total and was
excluded without replacement under the preregistered rule. Each synthetic
family contributes 10 new seeds. No exact source-image/PDF hash overlaps with
development. Merchant independence and unseen-layout generalization are unproven.

| Task / input | Haiku 4.5 | Sonnet 5 |
|---|---:|---:|
| CORD receipt image | 16/19 (84.2%; Wilson 95% 62.4–94.5%) | 18/19 (94.7%; 75.4–99.1%) |
| CORD annotated-word text | 18/19 (94.7%; 75.4–99.1%) | 17/19 (89.5%; 68.6–97.1%) |
| Currency conversion, native PDF | 9/10 (90%; 59.6–98.2%) | 10/10 (100%; 72.2–100%) |
| Currency conversion, 150 DPI pixels | 9/10 (90%; 59.6–98.2%) | 10/10 (100%; 72.2–100%) |
| Hidden OCR conflict, native PDF | 10/10 (100%; 72.2–100%) | 10/10 (100%; 72.2–100%) |
| Hidden OCR conflict, 150 DPI pixels | 10/10 (100%; 72.2–100%) | 10/10 (100%; 72.2–100%) |

Models: `claude-haiku-4-5-20251001`, `claude-sonnet-5`. One generation, maximum
512 output tokens, no repair loop. CORD images use the documented 1536-pixel
maximum edge; text contains upstream annotated words, not complete OCR or layout.
Intervals assume independent source draws; synthetic intervals describe seed
sampling within fixed templates. Variants are not additional independent sources.
Exploratory exact paired tests give p≥0.5 in every slice, without multiplicity
adjustment. This small sample supports neither a population ranking nor equivalence.

## What actually failed

- Haiku made two receipt numeric mismatches and two USD conversion mismatches.
  For example, EUR 12,345.67 × 1.09 requires USD 13,456.78, but the persisted
  row held 13,456.57. The independent SQLite query exposed the wrong end state.
- Sonnet's annotated-text case 14 exhausted the 512-token output budget, omitted
  a required tool field and wrote no row. This is a failure under the specified
  budget, not evidence about unconstrained model capability.
- Remaining strict mismatches concern separators or omission of the `Rp.`
  prefix. They violate the frozen verbatim-label contract, but cannot all be
  interpreted as wrong monetary values. An assistant reviewed the images and
  arithmetic; no independent human adjudication has occurred. Scores were not
  changed after looking at results.

The third point is a substantive limitation in **our task projection**. CORD's
transcription target is useful for testing perception; industrial money handling
also requires explicit locale, currency and normalization rules. Borrowing a
credible dataset does not make its labels automatically appropriate for a new
workflow. A follow-on study must preregister that domain mapping and use a fresh
held-out selection, rather than silently relabeling this run.

With the correct writer, posting arguments and persisted-state checks agree on
these outcomes. This study does not demonstrate extra detection power from SQL
verification alone. The separate [crash/retry experiment](README.md) demonstrates
its value under a specific duplicate-write fault. A response-only nonempty-text
check is merely a weak negative control, not a competitive evaluation baseline.

## Infrastructure findings and accounting

The experiment exposed two integration defects: local media needed explicit
materialization, and Inspect tool errors are dataclasses. Both were corrected.
Sonnet's scoring crash stopped the first held-out run; offline regrading retained
the malformed response as a failure. Native Inspect retry preserved 16 completed
generations and executed 62 remaining samples. Original grading errors and two
interrupted sample records remain in the evidence. The final policy also flags
this history; it does not present the run as uninterrupted.

Across development and held-out runs: **190 unique model events, 188 with
recorded usage, $0.699844 estimated at published API list prices**. Two cancelled
requests have unknown usage/billing. The final native log already includes usage
of preserved samples, so adding the original run's usage again would double-count
it. [Accounting](results/document-ledger-2026-09-17/provider-accounting.json)
deduplicates native event IDs and preserves the unknowns. These are usage-based
estimates, not an account invoice.

## Implications for the product

Build around interoperable task evidence: reuse datasets and execution tools,
make domain assumptions inspectable, retain interrupted trials and grading
history, and test actual side effects. Do not compete by collecting another
undifferentiated leaderboard or by claiming IDs and release gates are novel.
A potential advantage is a maintained set of independently reviewed workflow
assertions and failure cases that customers find useful. That remains a hypothesis;
this public-data sandbox does not establish customer demand or proprietary data.

## Reproduce

Use the library development revision recorded in `execution-lock.json` and the
pdfhell generator revision from the protocol. Install editable projects with
`multivon-eval[inspect,datasets]`, `pdfhell[pixels]`, and `scipy>=1.14,<2`.
Provide `ANTHROPIC_API_KEY` through your environment. Every output directory
must be new; no command overwrites an earlier run.

```bash
python benchmarks/industrial/prepare_documents.py --split held-out --output /tmp/document-inputs
python benchmarks/industrial/run_documents.py --root /tmp/document-inputs --output /tmp/haiku --model anthropic/claude-haiku-4-5-20251001
python benchmarks/industrial/run_documents.py --root /tmp/document-inputs --output /tmp/sonnet --model anthropic/claude-sonnet-5
python benchmarks/industrial/analyze_documents.py /tmp/haiku /tmp/sonnet --output /tmp/analysis.json
```

The repaired bridge should not reproduce the original grading crash. Its original
and resumed logs remain necessary to reproduce the historical accounting and
recovery analysis. Model serving may change; a fresh run need not match these
outputs. The checked-in manifests, source hashes, protocol, execution revisions,
results, failure review and raw-log hashes identify this specific run.
