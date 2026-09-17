# Provider evidence validation — 2026-09-17

The development capture path retained all **4 of 4** intended live Anthropic
requests and their native usage. Regrading the two saved targets issued **0** new
requests. This is a transport smoke study, not a model-quality benchmark.

## Frozen protocol and environment

Run source: `a11e344eccf9dc42886d90a94ace6c55efb57946`, with a clean checkout.
[Protocol and runner](provider_capture_experiment.py) were committed before calls.
The protocol wrote its source revision and environment before execution: Python
3.12.13, Anthropic SDK 1.6.0, OpenAI SDK 3.14.1, runtime module 0.18.0.
The editable installation's distribution metadata was stale at 0.17.0; the
artifact records that mismatch rather than rewriting it. The source revision
identifies the code exercised; this was not a published-wheel test.

Two parallel target calls, one sync judge call and one async judge call used
`claude-haiku-4-5`, temperature 0.2, max output 16 tokens and timeout 30 seconds.
Native retries stayed enabled. Prompts requested fixed Yes/No words; no external
dataset was needed to test transport evidence. Task benchmarks elsewhere reuse
credible datasets and upstream execution tools.

## Observations

| Observation | Result |
| --- | --- |
| Observed physical HTTP attempts | 4 |
| HTTP 200 responses with native usage | 4 / 4 |
| Requests without responses | 0 / 4 |
| Serialized temperature, max tokens and timeout matched protocol | 4 / 4 |
| Reported input / output tokens | 48 / 20 across 4 responses |
| Regrade requests | 0 across 2 saved outputs |
| ExactMatch target quality passes | 0 / 2 |

The target returned `Yes.` and `No.`, while the strict references were `Yes` and
`No`. Both original and regraded outcomes therefore remain `failed_quality`.
This is retained in the evidence; successful HTTP transport is separate from
task success. The sync and async judge responses were both `Yes.`.

Native usage also retained cache-creation/read fields, cache-window breakdown,
service tier and inference geography. No dollar estimate is made: raw token
usage is not an invoice, and legacy cost accounting remains incomplete.

## Evidence and checks

[Validation index](results/provider-capture-2026-09-17/validation.json) lists file
sizes and SHA-256 digests. [Archive](results/provider-capture-2026-09-17/evidence.tar.gz)
contains the pre-run protocol, original report, regraded report, native events
as JSON and SQLite, and outcomes. It is 24,673 bytes; SHA-256:

```text
67adbd7274e91a6e41476acd141a174f29caced93308670771335b6e332eaefd
```

Verification reopened the SQLite journal, checked event digests and equality to
exported JSON, matched every request to its response, restored both reports,
validated retained trial evidence against journal events, verified regrade
lineage and zero new requests, checked request settings and authentication-header
exclusion, scanned artifacts for the actual supplied credential and Anthropic
credential patterns, and reread archive members against the manifest hashes.
Only fixed synthetic prompt/response content is included.

Local checks before the run: **1,927 passed, 5 skipped** on Python 3.12; **73
focused checks passed** on Python 3.10; **70 MDX pages** parsed. The 18 new provider
checks use real installed SDKs with local transports. They cover sync/async
retries, missing usage, vision serialization, streaming preservation, cancellation,
Google transport gaps, snapshots and one actual child-process SIGKILL. Killing
that child retained its dispatched request, without inventing a response.

## Limits and next work

No live failure/retry occurred in these four calls. The local retry/crash tests
do not validate provider billing, power-loss durability or broad model accuracy.
Only Anthropic was tested against a live endpoint; OpenAI and Google tests used
their actual SDKs with local transports. The model alias may change; resolved
response IDs and model fields remain in the raw evidence.

Google async, LiteLLM and arbitrary supplied clients do not have automatic wire
coverage. Streaming usage, retention quotas, complete target configuration and
coverage-aware cost/budget reconciliation remain open in
[R02](../../plans/provider-evidence.md). No new PyPI release follows from this
smoke study alone.
