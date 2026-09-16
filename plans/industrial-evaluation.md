# Industrial evaluation program

Owner objective (2026-09-17): complete the proposed industrial/FOSS features,
agentic and multimodal evaluation, and experimental world-model evaluation;
update associated projects, website, docs, and README; run meaningful experiments
with available local resources and authorized provider credits; critique results
and establish a defensible contribution. This document preserves the full scope.

Status: active. No requirement is complete merely because an API or test exists.

Owner constraint added 2026-09-17: do not reinvent established dataset and
infrastructure tools. See [reuse decisions](reuse-decisions.md). Integrations
can satisfy requirements; prefer Hugging Face, Inspect, OpenTelemetry and
Gymnasium at their established boundaries.

## Requirements and completion evidence

| ID | Deliverable | Evidence needed | Status |
|---|---|---|---|
| R01 | Versioned cases/datasets: stable IDs, content revisions, source groups, split provenance | Reordering, duplicate inputs, changed contexts/labels, JSON round trips, split leakage and comparison compatibility tested; migration demonstrated | In progress: case/dataset identities, group splits, strict comparison pairing implemented; recorded grader/engine/calibration and repeat-count drift checked; opaque dependency compatibility remains |
| R02 | Complete immutable trial evidence and regrading | All outputs, traces, grader results, retries/errors, effective requests, usage retained across sync/async/parallel paths; saved output regrading makes no target calls | In progress: per-run/retry snapshots and regrading implemented; provider requests, usage, interrupted attempts and execution configuration remain |
| R03 | Resumable execution and resource controls | Kill/restart experiment preserves completed work; checkpoint compatibility, cancellation, bounded concurrency, deadlines and budget accounting; explicit policy for ambiguous side effects | In progress: Inspect SIGKILL/recovery experiment preserves completed work and rejects duplicate-write control; full compatibility, cancellation/deadline/budget validation remains |
| R04 | Acceptance policies | Required-check coverage, critical invariants, per-slice thresholds, sample requirements, quality/error/indeterminate distinctions and machine-readable decisions verified | Delivered in 0.18.0: policy/CLI, failure-path tests and frozen document-workflow policy; task/domain validity tracked separately |
| R05 | Review and calibration | Label import/export, review disagreements, development-only fitting, held-out evaluation, uncertainty/false-accept reporting and no leakage verified | Development preview: native Label Studio server/browser round trip and saved-score development/held-out source analysis validated with synthetic fixtures; mobile/accessibility limits documented; independent task-label validity remains under R17 |
| R06 | Trace interoperability | OpenTelemetry ingestion/export, documented schema/version handling, actual round trips with supported companion integrations | Development preview: native SDK/OTLP protobuf, actual Collector JSON and published MCP 0.4.0 round trips verified; convention profile and unsupported projections documented; production capture authenticity remains outside the bridge |
| R07 | Task/environment/outcome interfaces | Setup/reset/action/observation/cleanup, isolated repeated episodes, real end-state assertions, forbidden side effects, partial failures and recovery cases | Pending |
| R08 | Failure investigation UI | Trial comparison, evidence references, slices, review, case promotion, local security and accessibility; rendered desktop/mobile verification | Pending |
| R09 | Typed multimodal artifacts | Images/pages/audio/video, content identity, timestamps/regions, actual artifact rendering and grounded verdict references, extraction-versus-perception comparisons | Pending |
| R10 | Controlled robustness suite | Validated invariant-preserving and semantic-changing transformations; reviewed or code-derived answers; no transformations that silently corrupt the oracle | Pending |
| R11 | World-model evaluation (experimental) | Real adapter demo; action responsiveness, horizons, state persistence, uncertainty, planning utility; model and simulator errors separated; reproducible measured results | Pending |
| R12 | FOSS extension/stability contract | Public plugin/environment protocols, versioned report schemas, optional heavy dependencies, contributor fixtures, compatibility matrix and release checks | Pending |
| R13 | Industrial workflow and experiments | Document-to-ledger task, independent assertions, failures/benign controls, matched-budget baselines, held-out splits, actual provider runs, costs and uncertainty; results critique | In progress: CORD/pdfhell study completed with 39 held-out sources, two models, native recovery and explicit task-mapping critique; customer-domain contract validation remains |
| R14 | Research contribution/popularization | Related-work comparison, bounded novelty statement, reproducibility bundle, diagrams when useful, worked demo and evidence-led website narrative | Pending |
| R15 | Associated projects | multivon-mcp, pdfhell, eval-action compatibility and relevant integrations; cross-project end-to-end tests, docs/README, versions and releases as applicable | Foundation delivery: pdfhell 0.6.2 and MCP 0.4.0 published; action implementation tested locally but its push awaits a GitHub credential with workflow scope; later adapters remain tied to R06/R09 |
| R16 | Library/docs/site delivery | Cohesive API/migrations, runnable docs, visual website checks, supported-Python tests, packaging, verified pushes/releases, no overstated claims | Pending |
| R17 | Industrial/customer validity | User-provided target or permissioned real-world cases; independently reviewed usefulness. Synthetic demonstration alone cannot prove production value or demand | Awaiting target; other work proceeds |

## Implementation order

1. Stable identity, manifest compatibility, full trial evidence, effective provider configuration.
2. Durable execution and explicit decision policy; trace exchange and human labels.
3. Environment protocol, document-to-ledger sandbox, typed artifacts and failure explorer.
4. Multimodal robustness and world-model adapters; independent-oracle experiments.
5. Cross-project integrations, experiment critique, public evidence, docs/site, release audit.

Implement in cohesive modules under 500 lines; reduce existing large modules as
interfaces settle. Keep compatibility only where the old behavior can remain
honest; unidentifiable comparisons must disclose uncertainty rather than invent
identity. Every new public feature needs a real application example and failure
behavior, not only a happy-path unit test.

## Experiment discipline

- Credentials remain in the user-authorized key location; never log or commit keys.
- Begin with cheap provider capability probes and deterministic oracle tests.
- Give each paid experiment an explicit spend/call bound, persist usage and output,
  and stop on unexpected errors or invalid labels before scaling.
- Available Claude credits are a resource ceiling, not a target spend.
- Freeze task/source splits and acceptance criteria before final measurements.
- Record model versions, effective requests, package revision, seeds, errors/skips,
  per-trial artifacts, costs, and matched baseline configurations.
- Separate fitting, development, and held-out evaluation. Bootstrap uncertainty at
  the source/task level; do not count variants as independent source examples.
- Review false accepts, false rejects, judge manipulation, ambiguous tasks, and
  valid alternate solutions. Publish negative results and limitations.
- No independent-customer, real-world, or SoTA claims without corresponding evidence.
- Use code-native diagrams for technical architecture; if raster imagery adds value,
  use the user-requested Gemini/Nano Banana path and retain provenance.

## Progress log

2026-09-17: Revalidated clean library main at 8ea3cdc and active goal. Located
multivon-mcp checkout; pdfhell/eval-action need local discovery or isolated clones.
Implementation begins with R01/R02, which underpin all subsequent evidence.

2026-09-17 foundation checkpoint: added portable Dataset manifests and complete
case JSONL round trips; report schema v2 retains per-execution outputs, traces,
grader results, retry history, costs and suite locks. Stable-ID/content pairing
rejects changed definitions and duplicate identities; legacy gating requires
explicit trust. Regrading retains target failures and uses evaluation-time case
metadata; unknown imported latency cannot pass a latency check. Added 36 focused
regression cases and a runnable development guide. Full Python 3.12 suite passed
1,487 tests with 4 skips before the final four focused cases were added; all 84
foundation/comparison/documentation tests then passed on Python 3.10. No API
experiment or new release has occurred at this checkpoint. Main limitations:
request/configuration capture, opaque callbacks, full comparison compatibility,
and real industrial validation remain open. Case IDs are necessary evidence,
not a moat or a claim of scientific novelty.

2026-09-17 reuse checkpoint: owner explicitly requested reuse of credible
upstream work. Renamed unreleased Dataset wrapper to CaseManifest; implemented
a Hugging Face bridge with real Arrow/Parquet and streaming tests. Added
Inspect-native scorer/sample/log interoperability, including tool messages and
epoch grouping. Added required-check/slice acceptance policy and CLI with
three-way decisions and explicit retry scope. These replace planned custom
dataset/execution infrastructure where upstream behavior fits. Inspect
kill/restart, complete provider/judge accounting and industrial experiments
remain required; dependency-level capabilities alone do not complete R03.

2026-09-17 recovery experiment: two synthetic handlers, three cases each,
one SIGKILL after a committed side effect per handler. Inspect preserved two
completed samples and replayed the interrupted sample. An independent SQLite
invariant accepted idempotent final state and rejected the duplicate-write
control. Test exposed omitted historical attempts in a single-log import;
added chronological log-chain import and native UUID deduplication, including
partial-epoch retries. Retained four actual executions per handler. All-attempt
policy stays indeterminate for the interrupted safe run; final-attempt acceptance
requires explicit replay scope and verified state. No inference spend. Claude
credential preflight succeeded via model listing; no credential values logged.


2026-09-17 document study: reused official CORD v2 and existing pdfhell generators.
Completed 32 development and 156 held-out scored cases (39 held-out sources),
with 190 unique model events including two cancelled requests of unknown billing.
Known usage estimate $0.699844; no account-invoice claim. Native tool errors
exposed a bridge dataclass serialization bug, fixed with regression coverage.
Offline regrading preserved a malformed model output as a failure; Inspect retry
preserved 16 completed generations and ran 62 remaining samples. Both models
failed the frozen policy. Review identified transcription-only false rejects for
business semantics alongside wrong numeric values and a missing write. No labels
or prompts were changed after results. Public evidence and critique are in
benchmarks/industrial/DOCUMENT_RESULTS.md; original logs/assets are archived in
Documents/Multivon/document-ledger-2026-09-17. This does not complete customer
validation, broader execution controls, review tooling or the remaining program.

2026-09-17 release checkpoint: published multivon-eval 0.18.0, pdfhell 0.6.2,
multivon-mcp 0.4.0; verified PyPI artifact hashes and Git releases. The tested
eval-action release remains local because GitHub rejected its workflow-file push. The action now runs actual application revisions and fails closed on
missing evidence. MCP retains measurement/identity status and reuses acceptance
policies. Core tests: 1,559 passed on Python 3.12, 1,558 on 3.10 with expected
optional skips; 62 MDX pages compiled. Companion tests: 87 action tests on each
Python version and 14 MCP tests including real stdio. The Docker image built
and exercised actual target/baseline code from a mounted workspace. Website
release information is pushed; broader feature work is still active.

Published the 130,136,236-byte raw document-study archive as a v0.18.0 release
asset, verified its original per-file checksums and GitHub SHA-256 digest, and
reproduced the frozen analysis offline. Public dataset attribution is retained.
This completes the raw-artifact gap for this study; no new model inference or
claims of independent customer/human validation were added.

Release correction: the saved Git token lacks workflow scope and the connected
GitHub app also rejects this write. The action push (including tags) was rejected.
The release API had created v2.0.0 at the old default branch; that release was
returned to draft and the accidental tag removed. No action v2 is claimed as
published. Local 2.0.1 also fixes remote-only PR base refs and has 89 passing
tests. User input is pending for Git credential capability; independent work
continues. Core/library and MCP releases are unaffected.

2026-09-17 review checkpoint: added a native Label Studio JSON bridge, SDK 2.1.1
configuration/annotation validation and runnable offline export/import example.
Synthetic fixture reviews preserve disagreements, missing coverage and exact
trial bindings; they are not independent human labels. Corrected legacy
calibration scripts that substituted midpoint scores after errors and tiny
fixtures after failed downloads. New candidates pin upstream HaluEval revision,
hash actual selected content and retain development evidence. Historical runtime
packs were not regenerated or retroactively validated. Core Python 3.12 checks:
1,604 passed, 4 skipped; focused Python 3.10 checks: 70 passed, 1 optional SDK skip;
63 MDX pages compiled. Deployed review UI and held-out saved-score analysis remain.


2026-09-17 saved-review analysis checkpoint: reuse scikit-learn ROC thresholds
for source-balanced development error cost and SciPy exact intervals for held-out
source-error events. Frozen manifests and review/grader contracts reject leakage
and drift; missing labels/scores cannot fit or silently count as measured. The
12-trial synthetic holdout still has only three source groups. Local Label
Studio 1.23.0 server import/export and desktop browser submission passed; two
annotations remain synthetic and ten unreviewed tasks stay incomplete. Mobile
UI overflow and one non-blocking browser error are disclosed in
benchmarks/industrial/REVIEW_WORKFLOW_VALIDATION.md. The latest focused checks
passed 60 tests on Python 3.12, 59 on Python 3.10 (one optional SDK skip), and
64 MDX pages compiled. No API inference spend or new PyPI release at this checkpoint.


2026-09-17 telemetry checkpoint: reuse OpenTelemetry Python 1.44.0 SDK,
protobuf types and HTTP exporters, plus official Collector Contrib 0.161.0.
The bridge retains original protobuf/JSON bytes and native context while
separating original case identity from observed tool evidence. Actual MCP 0.4.0
stdio calls reproduce accept versus indeterminate for complete versus unasserted
capture. Standard evaluation log events retain trace/span IDs, and missing/error
measurements have no invented score. The Collector's actual JSON output is a
committed fixture with image digest and content hash. Full Python 3.12 tests:
1,642 passed, 4 skipped, 7 warnings; 23 focused tests passed on Python 3.10;
65 MDX pages compiled. No inference calls or new PyPI release. See
benchmarks/industrial/OTEL_VALIDATION.md for scope and upstream limitations.
