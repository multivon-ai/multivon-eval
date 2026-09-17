# Provider evidence implementation protocol

R02 remains open until requests, complete native usage, retries and interruptions
are retained across target/judge and sync/async/parallel/regrade paths. The prior
hooks count only selected successful judge usage fields; missing usage is silently
zero-like. Native SDK retries are not separately represented. Judge temperature
is omitted from Anthropic/OpenAI text requests despite configuration fingerprints.
The installed Anthropic 1.6 SDK also removed the temperature keyword used by the
legacy target adapter (its documented extra_body escape hatch remains available).

## Implementation decisions

- Reuse SDK HTTP client factories and HTTPX event hooks for serialized request/
  response capture. Keep native retries; observe each physical attempt instead of
  implementing another provider client or silently changing retry counts.
- Bind events to a trial and role using context-local state; concurrent graders
  share the trial collector without sharing role labels. Preserve native JSON
  bodies and usage fields. Missing responses/usage remain unknown, never zero.
- Capture before dispatch, with optional durable append-only local evidence for
  interrupted requests. An unfinished request does not establish whether the
  provider executed or billed it. Recovery scheduling stays with Inspect/R03.
- Exclude authentication headers and credential query values. Preserve content
  hashes, explicit redaction/capture gaps and native response IDs. These are
  private evaluation artifacts, not a public telemetry feed.
- Supplied/custom clients and unsupported transports must disclose capture gaps.
  SDK-call observations must not masquerade as wire requests. Streaming capture
  must not buffer an unbounded response or change streaming semantics.
- Keep preparation/reliability requests separate from trial requests. Regrading
  must retain parent evidence without implying that old target calls ran again.
- Preserve raw cached-input, cache-creation, reasoning and other provider usage;
  any derived totals must describe their coverage. Dollar estimates are not bills.

## Verification

Use actual installed SDKs with local transports to verify request serialization,
HTTP failures/retries, usage shapes, concurrency, cancellation, and missing usage.
Exercise every runner path and saved JSON round trips. Freeze sources before a
small real Anthropic API experiment; retain raw request/response evidence and all
failed attempts. No accuracy/SoTA claim follows from this instrumentation study.

## Implemented development boundary

Native Anthropic/OpenAI HTTPX hooks now cover built-in targets and text/vision
judges, including native SDK retries. Google sync uses native client arguments;
Google async keeps its upstream transport choice and declares absent wire
observations. An explicitly supplied instrumented async HTTPX client was tested
with Google's native SDK. LiteLLM and arbitrary supplied clients remain coverage
gaps. This boundary avoids substituting a new provider client or retry engine.

Local tests retain full native cache/reasoning/tool usage, missing usage, stream
gaps, concurrent trial identity, sync/async repeat/retry positions and regrade
lineage. A real child-process SIGKILL leaves the pre-dispatch SQLite event intact;
it cannot determine remote execution or billing. Journals validate event digests
and stored identities. Returned and auto-saved report snapshots agree; run
snapshots precede export/gates, while the journal contains later run closure.

Verification before the live protocol: 1,927 tests passed, 5 skipped on Python
3.12; 73 focused checks passed on Python 3.10; all 70 MDX pages parsed. The
18 new provider checks include native SDK retry/serialization, cancellation,
actual SIGKILL, Google transport coverage, vision requests and report snapshots.

R02 remains open: full target configuration beyond observed serialized requests,
unobserved transports and bounded retention are not established by this capture.
The default cost tracker still counts a subset of successful judge responses;
the reconciliation path below is separate. Real billing cannot be inferred from
list-price estimates.

## Declared provider accounting

`account_provider_events` reconciles closed journal events, including targets,
preparation/judges and physical retry attempts. It retains raw usage and maps
provider-specific token dimensions without retokenizing text. Regrade parent
events are excluded by `provider_events(report)`. Missing lifecycle starts/ends,
unobserved operations, request/response mismatch, failure and missing/invalid
usage remain explicit gaps. No declaration can override detected gaps.

`Costs` separates recorded subtotals from a declared complete run-provider scope.
Budget gates reject missing/legacy/partial usage instead of silently skipping;
unknown prices still permit known-token gating but cannot pass dollar gates.
These are post-run gates, not execution spend caps or independent billing audits.

The optional LiteLLM bridge reuses its native response conversion, cache pricing
and model catalog. It snapshots version/hash/tariff evidence and rejects catalog
drift. Standard direct Anthropic Messages/OpenAI Chat Completions text-output
paths are tested; other endpoints and special tariffs remain unknown. Native
Google token normalization is included; native Google pricing is not yet bridged.
Legacy basic-text rates were corrected against official sources and speculative
entries removed; self-hosting is no longer assumed free.

`benchmarks/industrial/reconcile_provider_capture.py` is the frozen offline
reprocessing protocol: use the retained four-call archive, disable remote catalog
fetches, block network before importing LiteLLM, retain pricing provenance, and
check a 68-token / $0.001 post-run budget. Do not rewrite the historical capture
or treat current list-price estimates as historical billing.

Accounting verification: 1,958 tests passed and 18 skipped on Python 3.12
(13 skips are the optional pricing tests exercised separately); 111 focused
checks passed on Python 3.10; 44 accounting/pricing checks passed with actual
LiteLLM 1.101.0 in an isolated environment; all 71 MDX pages parsed. Native
Anthropic cache-window and OpenAI cached/reasoning fixtures agree with the
documented reference arithmetic. The optional workflow runs with a native SDK
local transport, and pricing tests block socket connections.

## Frozen live smoke protocol

`benchmarks/industrial/provider_capture_experiment.py` requires a clean committed
checkout and writes the commit, environment and protocol before requests. It
intends four Anthropic Haiku 4.5 calls: two parallel targets, one synchronous
judge and one asynchronous judge, all with temperature 0.2, max output 16 tokens
and timeout 30 seconds. Native SDK retries may increase physical attempts.
Regrade both saved target outputs with ExactMatch and verify no new requests.
Retain all response statuses, usage objects and unknown outcomes; do not infer
accuracy from the fixed prompts or dollars from incomplete pricing. API keys come
only from the environment; reject evidence containing the supplied credential.

Executed from frozen commit `a11e344`: 4/4 live requests returned HTTP 200 with
native usage; regrading made zero requests. Both strict target checks failed on
terminal punctuation, and remain failed in saved evidence. See the
[validation and evidence archive](../benchmarks/industrial/PROVIDER_EVIDENCE_VALIDATION.md).
This does not close the coverage/accounting gaps above.

The frozen offline reconciliation at `f987489` repriced all four captured calls
with LiteLLM 1.101.0: 48 input / 20 output tokens, $0.000148 list-price estimate,
zero attempted network connections. The
[accounting validation](../benchmarks/industrial/PROVIDER_ACCOUNTING_VALIDATION.md)
retains the pricing assumptions and distinguishes the two standalone judge calls
from the target suite. This is not historical invoice reconciliation.
