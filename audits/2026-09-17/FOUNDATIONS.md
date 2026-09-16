# Evidence and interoperability checkpoint

Development work after release 0.17.0. These changes are not a new PyPI release.
This checkpoint precedes the later [crash/recovery experiment](../../benchmarks/industrial/README.md).

## Implemented

- Strict portable case round trips retain context, expected tools, traces,
  references, IDs, revisions and source groups. Unknown fields are rejected.
- `CaseManifest` snapshots a selected evaluation set and validates identity,
  split assignments and source leakage. It does not implement a dataset engine.
- Hugging Face Datasets supplies native data loading, Arrow/Parquet, streaming,
  transformations and storage. The optional bridge imports bounded selections
  and retains upstream fingerprints and caller-supplied source provenance.
- Reports retain immutable raw trial records across sync, async, parallel,
  repeats, retries and imported outputs. Regrading never calls the target.
- Comparison checks case identity/content, missing or duplicated trial slots,
  detached result headers, and upstream evidence issues. Legacy prompt-only
  comparisons require explicit trust for significance and CI gating.
- Inspect supplies a native execution/log path. The adapter preserves case
  identity, scorer verdicts, skip/error distinctions, actual tool messages,
  epochs, usage references and native sample digests.
- Acceptance policies require named checks, coverage, critical invariants,
  task slices, case/source counts and an explicit retry scope. Decisions have
  accept/reject/indeterminate outcomes and CLI exit codes 0/1/2.

## Verification

| Check | Evidence |
|---|---|
| Full tracked test suite with optional HF/Inspect dependencies, Python 3.12 | 1,538 passed; 4 skipped; 5 existing calibration warnings |
| Foundation, acceptance, HF, Inspect, comparison and documentation tests, Python 3.10 | 131 passed |
| Public comparison/gate CLI after extracting comparison CLI module | 67 passed |
| Real dependencies | `datasets==5.0.1`, `inspect-ai==0.3.263`; dependency check passed |
| Native data interchange | HF Arrow/Parquet read/write, mapping, source leakage, bounded streaming |
| Native execution interchange | Inspect runner, `.eval` disk read, 2 epochs per case, skips/errors, tool messages, policy decisions |
| Runnable documentation | Native offline example, policy/CLI, HF example, Inspect example |
| MDX compilation | 60 pages; no compile errors |
| Lint | New core/adapter modules pass Ruff |
| Package build | Wheel and sdist built in `/tmp`; not uploaded; release version remains 0.17.0 |

The full suite includes the added tests explicitly until they are tracked.
Live-provider tests were excluded. No API credits were spent in this checkpoint.
The final comparison-CLI extraction was tested separately after the full suite.

## Critique and remaining work

Stable IDs, logs, grouped splits and gates are established practices. These
changes improve measurement integrity; they do not establish a moat or a novel
evaluation method. The [reuse decision record](../../plans/reuse-decisions.md)
identifies the upstream foundations and remaining integration work.

Case compatibility is not yet full experiment compatibility. Grader code,
calibration, effective requests, target version, external resources and runtime
configuration need stronger matching before claiming reproducibility.

Multivon-native trials explicitly lack provider request/usage instrumentation.
Inspect retains its own native request/event records, but Multivon judge calls
are not automatically covered by Inspect logging or model cost limits. Do not
infer complete cost accounting from target-model usage alone.

Inspect crash recovery has not yet been exercised through this integration.
Native sample infrastructure errors are not classified into model versus
grader failure by the bridge. The tool transcript projection does not retain
all native timing or multimodal content. Keep the native log with the report.

Case snapshots do not isolate arbitrary callbacks or undo side effects.
Acceptance policies are deterministic contracts, not significance tests or a
guarantee of representative sampling. Regrading individual trials creates
correlated observations; do not treat them as independent source examples.

The industrial document-to-ledger workflow, real API experiments, human review,
multimodal and world-model validation, companion-project updates, and final
website/release delivery remain part of the active implementation program.

## Document workflow checkpoint

Reused official CORD v2 and pdfhell generators through Datasets, with Inspect
owning model/tool execution, logs, offline scoring and native retry. The frozen
study completed 32 development and 156 held-out scored cases. It exposed and
fixed dataclass serialization of native tool errors. Regrading preserved the
malformed original output as a failure and retry preserved 16 completed
executions. Strict acceptance rejected both models; the failure review documents
limitations of using verbatim transcription labels as business outcomes.

- Python 3.12 full tracked suite: **1,548 passed, 4 skipped**, five existing
  calibration fallback warnings (`/tmp/multivon-doc-final-full.log`).
- Python 3.10 focused Inspect/workflow/acceptance checks: **51 passed**.
- Documentation contracts: **16 passed**; **61 MDX pages compiled**.
- Website article: lint, TypeScript and production build passed; headless Chrome
  screenshots inspected at 1440×1000 and 390×844. No page errors or horizontal
  overflow; study link present on the blog index.
- pdfhell companion fix: **170 tracked tests passed**; original PDF bytes and
  renderer identity now protect image cache correctness. Numeric token and
  currency checks reject demonstrated lexical false accepts; prose limitations
  remain explicit. Commit `16d184b` pushed, not yet released to PyPI.

See [study and limitations](../../benchmarks/industrial/DOCUMENT_RESULTS.md).
The broader R01–R17 program remains active; these checkpoints do not establish
customer validation or completion of all execution, review and modality features.

## Published foundation checkpoint

- multivon-eval 0.18.0: retained-trial and recorded-configuration comparison
  safeguards, exact small-sample paired statistics, missing-trace correction,
  Hugging Face/Inspect bridges and explicit acceptance policies.
- pdfhell 0.6.2: scoring and raster-cache fixes published; 170 tests passed on
  Python 3.10 and 3.12 at release.
- MCP 0.4.0: 23 tools, including acceptance from saved reports; original skip/error
  metadata and identity warnings preserved; actual stdio protocol validated.
- eval-action 2.0.1 implementation (local, pending workflow-scoped Git credential):
  real target execution in separate revision processes,
  fail-closed evidence/configuration gates, engine reuse and pinned dependency.
  Container smoke exercised head and Git baseline from the mounted workspace.
- Raw study evidence is downloadable from the 0.18.0 GitHub release. Its checksum
  and offline reproduction recipe are in benchmarks/industrial.

These deliveries do not establish full callback/provider compatibility, customer
usefulness, an independently reviewed dataset, a moat or SoTA performance. The
industrial program remains active for review/calibration, OTel, environment and
multimodal evidence interfaces, world-model experiments and subsequent delivery.

Action publication status correction: GitHub rejected the workflow update and
all associated refs. The prematurely created release was hidden as a draft,
and its old-default-branch tag removed. The implementation and Docker tests
are valid; publication is pending, not complete.
