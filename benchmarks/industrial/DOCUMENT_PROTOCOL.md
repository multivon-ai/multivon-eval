# Document-to-ledger protocol v1

Frozen before paid inference, 2026-09-17. This is a small workflow experiment,
not a new document benchmark, a CORD leaderboard submission, or a SoTA claim.

## Reuse and selection

- **CORD v2**, NAVER/clovaai, CC BY 4.0. Official repository:
  https://github.com/clovaai/cord ; license:
  https://github.com/clovaai/cord/blob/master/LICENSE-CC-BY . Load its official
  `naver-clova-ix/cord-v2` Hugging Face release at commit
  `7f0115a4b758a71d6473b8d085751692da2fef98` using Datasets streaming.
  Development: validation rows 0–3. Held-out: test rows 0–19. Retain official
  annotation `gt_parse.total.total_price` unchanged; do not infer currency.
  Keep image IDs, image hashes, original split and annotation provenance.
  Multiple/missing total labels are recorded as exclusions, without replacement.
  Publish selection/derived evidence with attribution; upstream owns the images.
- **pdfhell**, Apache-2.0, generators at commit `16d184b`: hidden OCR mismatch
  and currency mismatch conversion. Development seeds 42–43; held-out seeds
  91700–91709 in each family. These share generator templates with development;
  they test new parameter draws, **not unseen layouts**. Decimal arithmetic
  independently verifies generator amounts. Fail preparation on disagreement.
- Receipt images are ordinary public-document cases, not guaranteed easy or
  production representative. Synthetic cases isolate two known mechanisms.
  Do not combine them into a headline population accuracy.
- DocILE offers a more relevant invoice/line-item benchmark and established
  metrics; its official access process is required before adding its data:
  https://docile.rossum.ai/ . No substitute mirror or copied metric is used.

## Task and treatments

Use Inspect 0.3.263 for execution, model API logs, tools and sample limits.
Each sample has an isolated ledger partition and a `post_entry` tool. The tool
validates syntax, writes an idempotent row to SQLite and reads it back; it never
sees the answer label. Scoring independently queries persisted rows after the
solver. Saying an entry was posted does not satisfy the outcome check.

CORD task: transcribe the final total (not cash tendered or change) **verbatim**
into `amount`, set `currency="UNSPECIFIED"`. This narrowly evaluates transcription;
it does not establish payment authorization or numeric locale interpretation.
PDF task: post the visible total for hidden OCR, or USD settlement total using
the printed rate for currency conversion. Use two decimal places, no grouping,
`currency="USD"`. The visible-total priority is explicit in the task instruction.
This is a sandbox task policy, not a recommendation to pay disputed invoices.

Models: `anthropic/claude-haiku-4-5-20251001` and
`anthropic/claude-sonnet-5`. Same prompts and max output budget; no temperature
setting, no application retries, cache disabled. Record provider request and
returned model ID; provider internals and versioned-ID immutability are not
assumed. One generation with tool execution per sample, no repair loop.

Treatments: standardized image for CORD (RGB PNG, longest edge at most 1536
pixels, Lanczos downsampling, no upscaling; retain original bytes and both
hashes); native PDF and locally rendered 150 DPI
pixels for each pdfhell source. Images/pixels are different information channels;
report paired results by modality, not as a universal ranking. Also run a
**privileged text baseline** for CORD using the upstream human transcription
in `valid_line` order with categories/answer labels removed. It changes the
perception burden but also removes layout and any text not annotated upstream;
it is not a complete OCR transcript, OCR-system baseline or deployable competitor.
Each source remains one statistical unit across all modalities/models.

## Limits and evidence

Development at most 32 single-generation calls (8 sources × 2 treatments ×
2 models). Held-out at most 160 calls (40 sources × 2 treatments × 2 models).
Maximum 512 output tokens per call; per-sample timeout 90 s; max concurrency 2;
SDK retry limit zero; 20,000-token sample limit; $0.20 per-sample Inspect cost
limit as an additional stop signal, **not a hard prepaid spending cap**.
Before calls, restrict payloads to one page/image, at most 4 MB, text at most
32,000 characters. Estimated pre-tax ceiling is conservatively $40 for the
whole experiment; halt after any unexpected provider error for investigation.
No automatic model fallback. Persist all errors and do not silently replace them.
Pricing snapshot (2026-09-17): Haiku input/output $1/$5 per million tokens;
Sonnet 5 $2/$10. Source: https://platform.claude.com/docs/en/about-claude/pricing .
Report actual usage-based estimates separately from billable account charges.

Preserve .eval logs (API request/response bodies, usage, tool events), source
manifest, package versions, PDFs/images and their content hashes, SQLite state,
Multivon trial reports and decisions. Log no credentials. Native Inspect logs
are the authoritative provider record. Scorer evidence includes queried rows;
regrading an external-state assertion requires the original ledger snapshot.

## Decisions, statistics and critique

`ledger_outcome` is critical: exactly one row, exact task amount and currency.
Require complete check coverage, no infrastructure errors and every selected
case. Known wrong state rejects; missing evidence is indeterminate. This
zero-failure demonstration policy is not a production SLO. Separately report
whether a valid tool call occurred, failures by family/modality, and API errors.
Compare real outcomes to a weak response-only baseline (nonempty assistant
answer); count its false accepts without claiming it is a credible competitor.

Report per-track n and Wilson 95% intervals for marginal success, with the
assumption that source documents are independent made explicit. CORD merchant
independence is unknown. Synthetic seeds share layouts; their intervals only
characterize seed sampling under these generators. Compare paired outcomes
at source level; do not treat model/modality variants as independent examples.
With n≤20 per slice, wide intervals preclude a general reliability claim.
Publish all exclusions, disagreements, failures and limitations. Development
may expose bugs; any protocol change must be versioned before inspecting test
outputs. Do not tune prompts on held-out results. Customer usefulness remains
unproven without a real application target and independent review.


### Pre-inference preparation correction

Before any paid calls, development row 3 exceeded the 4 MB request bound even
with original PNG bytes (9.8 MB). Standardized CORD resizing above was added for
all image samples, not only large ones. This is part of the image treatment;
results cannot be described as original-resolution CORD performance. The text
track uses only upstream annotated words and loses layout/unannotated text.
The bound remains 4 MB per **submitted** asset; original archival files can be
larger. No held-out output has been viewed or prompt tuned at this correction.


### Development integration interruption

An offline native-media run exposed Inspect's requirement for explicit trusted
media materialization. A development Haiku run was inadvertently launched after
the offline assertion failed because the shell sequence did not stop on failure.
It halted before any completed provider response: one concurrent text sample
has an interrupted model event and unknown billable usage. Preserve that .eval
log; do not treat it as a zero-cost success or a model quality failure. Fix media
submission to use inline bytes whose hashes were verified, and require a full
successful offline run before resuming. The development cap is now **33 model
request attempts**, including that interrupted attempt, with the same $40 study
estimate ceiling. This engineering correction precedes held-out inference and
does not change labels, prompts, selection or acceptance rules. Report resumed
development outcomes separately from the failed integration run.

### Held-out scoring repair (no task or prompt change)

Sonnet's first held-out log stopped after 16 completed responses and two
interrupted samples. A max-output-token response omitted a required tool field;
Inspect represented that as a ToolCallError dataclass. Multivon's bridge tried
`.model_dump()` and crashed while grading it. The fix uses dataclass serialization.
The malformed response remains a quality failure. Native `score()` regrades
saved outputs, preserving original errors in a separate original report and
repair provenance; native `eval_retry()` preserves those 16 completed generations
and runs the remaining 62 samples. It may replay two interrupted requests whose
billable usage is unknown. Total held-out request attempts remain at most 158,
inside the original 160-attempt cap after the predeclared CORD exclusion.
No model prompt, label, budget or acceptance threshold changes. Retain both
original and derived grading artifacts; the final report explicitly flags this
repair and retained interruptions rather than presenting an uninterrupted run.
