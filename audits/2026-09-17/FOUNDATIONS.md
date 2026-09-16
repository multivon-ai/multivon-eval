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
