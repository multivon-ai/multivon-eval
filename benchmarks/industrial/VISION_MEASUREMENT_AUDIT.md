# Vision measurement audit — 2026-09-17

This is a correctness and SDK-compatibility audit. It does not measure vision
judge accuracy, calibration, prompt-injection resistance or customer value.
All new judge responses are controlled test fixtures; no model inference calls
or paid requests were made for this checkpoint.

## Findings and corrections

| Trigger | Published 0.18.0 behavior | Development behavior |
|---|---|---|
| Empty claim array, or prose containing no array | Could receive a perfect VQA faithfulness score | An actual empty array is skipped; invalid extraction is a judge error |
| Ambiguous reply such as `Maybe yes` or `Yes or no` | First Yes/No word supplied the verdict | Entire verdict must match the response contract |
| Truncated/partial document verdict | Missing answers became negative quality judgments; a low threshold could pass | All three distinct valid answers are required before scoring |
| Duplicate/contradictory document answers | First matching answer could win | Judge error |
| Provider failure during one claim check | Could be counted as an unsupported claim | Judge error; no partial quality score |
| Missing image prerequisite | Numeric quality failure | Skipped/unmeasured |
| New/private vision model name | Static allowlist could reject before calling the endpoint | Only known text-only families are rejected early |
| Anthropic SDK 1.6.0 direct `temperature=` argument | `TypeError` before any request | Official `extra_body` migration preserves the requested setting |

The protocol is explicitly `vision-qag/v2`. Effective prompt templates are
fingerprinted by suite locks, and the protocol is part of grader configuration.
Complete successful responses and the resolved threshold are retained with each
result. Threshold resolution no longer mutates a shared evaluator during
concurrent calls. A genuine valid No still produces a measured quality failure.

The new prompt wording tells the judge to treat answer/image text as data.
This is not an adversarial robustness result. Strict parsing can increase
indeterminate outcomes for otherwise understandable prose; that is an explicit
contract tradeoff, not a claim that strict formatting improves model reasoning.

## Validation

The focused suite covers malformed claim shapes, refusals, truncated JSON,
brackets inside claim text, complete Markdown JSON fences, invalid image lists,
ambiguous verdicts, missing/duplicate document questions, provider exceptions
and concurrent threshold resolution. Sync and async suite runs preserve skipped
and judge-error states, with zero evaluated cases in those one-case fixtures.
Their acceptance policies remain indeterminate after saved JSON round trips.

A real Anthropic SDK 1.6.0 client sent three HTTP requests to a loopback fixture:
claim extraction, a Yes verdict and an authentication failure. The test checked
exact image bytes, MIME type, requested temperature and per-stage token limits.
It initially exposed the removed SDK keyword; the corrected requests and error
classification passed. The server did not run a model. No OpenAI/Google SDK or
live-provider compatibility claim is made by this test.

Focused Python 3.10 and 3.12 checks each passed 103 tests, with two expected
uncalibrated-threshold warnings. The full Python 3.12 suite passed 1,718 tests,
with five skips and seven warnings. All 67 MDX pages compiled.

Reproduce without API credentials:

```bash
python -m pytest tests/test_multimodal.py tests/test_multimodal_measurement.py \
  tests/test_acceptance_policy.py -q -p no:cacheprovider
```

## Remaining work and reuse

Legacy path/URL metadata does not freeze image bytes. Native PDFs/audio/video,
time/region evidence, grounded citations and artifact rendering are still R09
work. Full provider configuration, raw request/usage capture and interrupted
requests remain R02 work. Fixed document questions lack a task-specific
applicability oracle; three generated claims need not cover all answer errors.

The [reuse assessment](../../plans/reuse-decisions.md#multimodal-evidence-and-grader-audit)
selects Inspect for native media execution/logging, Hugging Face for loading and
W3C annotations for region/time semantics. TIFA and VQAScore are established
visual-scoring work to reuse or compare against where their tasks fit. These
parser fixes are ordinary correctness engineering and do not establish a moat.

Sources: [Anthropic SDK migration](https://github.com/anthropics/anthropic-sdk-python/blob/main/MIGRATION.md),
[Inspect multimodal support](https://inspect.aisi.org.uk/multimodal.html),
[TIFA](https://github.com/Yushi-Hu/tifa),
[VQAScore](https://github.com/linzhiqiu/t2v_metrics).
