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

Owner direction added 2026-09-17: the customers closest to reach are **regulated
enterprise teams**. R14 should evaluate this segment first. This is a customer
segment direction, not evidence of access to a customer target, permissioned
cases or independently validated demand; R17 remains open.

## Requirements and completion evidence

| ID | Deliverable | Evidence needed | Status |
|---|---|---|---|
| R01 | Versioned cases/datasets: stable IDs, content revisions, source groups, split provenance | Reordering, duplicate inputs, changed contexts/labels, JSON round trips, split leakage and comparison compatibility tested; migration demonstrated | Development implemented: case/dataset identities, group splits, strict pairing, grader/engine/calibration/repeat drift, declared opaque dependencies and missing-snapshot rejection; compatibility is bounded by recorded state and caller declarations |
| R02 | Complete immutable trial evidence and regrading | All outputs, traces, grader results, retries/errors, effective requests, usage retained across sync/async/parallel paths; saved output regrading makes no target calls | In progress: native attempts, SQLite SIGKILL retention, target/runner snapshots and declared-coverage accounting with upstream LiteLLM pricing. Hidden target state beyond declarations, unobserved transports, bounded retention and broader tariffs remain; see [provider protocol](provider-evidence.md) |
| R03 | Resumable execution and resource controls | Kill/restart experiment preserves completed work; checkpoint compatibility, cancellation, bounded concurrency, deadlines and budget accounting; explicit policy for ambiguous side effects | In progress: native Inspect recovery, limit/cancellation evidence and declared retry compatibility with preflight and mixed-log detection. Caller declarations cover hidden dependencies; arbitrary mid-agent checkpoint state and production resource guarantees remain unverified. See [control protocol](execution-controls.md) |
| R04 | Acceptance policies | Required-check coverage, critical invariants, per-slice thresholds, sample requirements, quality/error/indeterminate distinctions and machine-readable decisions verified | Delivered in 0.18.0: policy/CLI, failure-path tests and frozen document-workflow policy; task/domain validity tracked separately |
| R05 | Review and calibration | Label import/export, review disagreements, development-only fitting, held-out evaluation, uncertainty/false-accept reporting and no leakage verified | Development preview: native Label Studio server/browser round trip and saved-score development/held-out source analysis validated with synthetic fixtures; mobile/accessibility limits documented; independent task-label validity remains under R17 |
| R06 | Trace interoperability | OpenTelemetry ingestion/export, documented schema/version handling, actual round trips with supported companion integrations | Development preview: native SDK/OTLP protobuf, actual Collector JSON and published MCP 0.4.0 round trips verified; convention profile and unsupported projections documented; production capture authenticity remains outside the bridge |
| R07 | Task/environment/outcome interfaces | Setup/reset/action/observation/cleanup, isolated repeated episodes, real end-state assertions, forbidden side effects, partial failures and recovery cases | Development preview: native Gymnasium lifecycle plus immutable state evidence and outcome checks; 14 actual SQLite environment instances validate separate resources, forbidden history, partial commits and explicit recovery; security isolation and durable execution stay upstream under R03 |
| R08 | Failure investigation UI | Trial comparison, evidence references, slices, review, case promotion, local security and accessibility; rendered desktop/mobile verification | Development preview: saved trial comparison/filtering, native Inspect references, Label Studio consensus-bound promotion and local request protections validated; six browser scans passed with keyboard/download checks; upstream mobile limitations and missing manual screen-reader validation documented |
| R09 | Typed multimodal artifacts | Images/pages/audio/video, content identity, timestamps/regions, actual artifact rendering and grounded verdict references, extraction-versus-perception comparisons | Development preview: content/probe bindings, native HF/Inspect round trips, W3C references, PDF page rendering and actual image/audio/video viewing verified; six Haiku calls on two synthetic pdfhell sources compare PDF/pixels/text. General selectors, video regions, full packet validation and semantic citation validity remain outside the tested profile |
| R10 | Controlled robustness suite | Validated invariant-preserving and semantic-changing transformations; reviewed or code-derived answers; no transformations that silently corrupt the oracle | Development preview: task-bound invariant/counterfactual validation preserves invalid/unknown candidates and source groups; four-source numeric experiment and 200 Hypothesis examples demonstrate independently derived answers and expose consistency false positives. General directional/state/media transformation validation remains outside this profile |
| R11 | World-model evaluation (experimental) | Real adapter demo; action responsiveness, horizons, state persistence, uncertainty, planning utility; model and simulator errors separated; reproducible measured results | Development preview: native CartPole + learned scikit-learn delta models + SciPy planner; 40 held-out forecast sources, 10 planning seeds and preserved action/state-offset/uncertainty/error evidence. Frozen sources and offline replay verified; fully observed numeric-state profile only, not video, hidden-state memory or robotics generalization |
| R12 | FOSS extension/stability contract | Public plugin/environment protocols, versioned report schemas, optional heavy dependencies, contributor fixtures, compatibility matrix and release checks | Development: documented existing Python/Gymnasium/Inspect interfaces, corrected imported replay, packaged report envelope schema, frozen 0.18.0 migration fixture and clean-wheel checks on Python 3.10/3.12. CI improvements prepared; remote workflow delivery and broader dependency combinations remain to verify. See [compatibility audit](foss-compatibility.md) |
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


2026-09-17 environment checkpoint: reuse Gymnasium 1.3.0 lifecycle/spaces and
the existing document-study SQLite posting handler. Capture detached state
observations before/after actions, including writes committed before exceptions;
retain cleanup and execution gaps, and bind versioned outcome checks. Real
FrozenLake/CartPole checks validate native spaces and termination semantics.
The ledger fixture ran 14 environment instances across 12 databases; three
published MCP 0.4.0 calls reproduced accept/reject/indeterminate. The complete
48,069-byte evidence archive includes SQLite state, reports, executed sources
and verified hashes. Exploratory ablation: final text accepted 7 invalid complete
episodes out of 12, final-posting-only checks missed 2 forbidden-change episodes.
This supports the need for the specified state/history contract, not superiority
over a good bespoke checker. Full Python 3.12 checks: 1,661 passed, 4 skipped,
7 warnings; focused Python 3.10: 86 passed. Also fixed regrading to retain prior
capture issues and upstream provenance. No model calls or new PyPI release.
See benchmarks/industrial/ENVIRONMENT_VALIDATION.md; the wider program remains active.


2026-09-17 investigation checkpoint: reuse Inspect View for native logs and
Label Studio for review. Corrected comparison reason pairing for distinct cases
sharing a prompt; expose exact trial evidence and coverage in the existing HTML
report. Added keyboard-operable details, tag/status/ID filters and source-preserving
reviewed development-case promotion. Browser candidate downloads require intact
evidence and never infer an expectation from failed output. Local servers check
Host/Origin, block remote fetching/framing and preserve generated link identity
across directory changes. Six desktop/mobile axe scans passed with no detected
violations or page overflow; native Inspect sample view still overflows on mobile.
Full Python 3.12 checks: 1,674 passed, 5 skipped, 7 warnings; focused Python 3.10:
76 passed, 1 directory-only skip. All 67 MDX pages compiled. No model calls or
new PyPI release. Validation and critique: benchmarks/industrial/INVESTIGATION_VALIDATION.md.


2026-09-17 vision measurement checkpoint: corrected empty-claim perfect scores,
partial/ambiguous/duplicate judge parsing and provider exceptions becoming
quality outcomes. Missing media/claims remain unmeasured; strict valid negatives
remain quality failures. Retain protocol/prompt fingerprints and per-call
thresholds. An actual Anthropic SDK 1.6.0 loopback HTTP check exposed its removed
temperature keyword; the documented extra_body migration now sends the original
setting and preserves authentication failures as judge errors. Full Python 3.12:
1,718 passed, 5 skipped, 7 warnings; focused Python 3.10: 103 passed, 2 warnings;
all 67 MDX pages compiled. No model inference calls or new PyPI release. This is
a measurement audit, not vision accuracy or robustness validation. Typed media
bindings/rendering and full provider evidence remain open. See
benchmarks/industrial/VISION_MEASUREMENT_AUDIT.md and the Inspect/HF/W3C/TIFA/
VQAScore reuse assessment.


2026-09-17 media checkpoint: introduced content/provenance-bound descriptors
without new storage or dataset loading. Reuse Pillow/PDFium/PyAV metadata and
native Inspect content; verify hashes and re-probe properties at use time.
Grounded verdicts retain W3C references with explicit bounds and unknown states;
reference validity never supplies semantic truth. Native HF bytes and saved
Inspect logs round-trip, while changed/missing media reject. Browser checks
verified page pixels and WAV/MP4 playback; native PDF remains a badge and all
four upstream panels overflow on mobile. Six live Haiku calls reused two
synthetic pdfhell development sources: PDF 2/2, pixels 2/2, text 1/2, with known
usage estimate $0.007014. This is not a statistically supported ranking or
independent customer validation. MCP 0.4.0 reproduced the policy rejection over
stdio. The 262,047-byte archive retains 28 checksum-verified files and replays
without target calls; post-run source-snapshot timing is explicitly disclosed.
Full Python 3.12: 1,739 passed, 5 skipped, 7 warnings; after adding codec/orientation
checks, focused Python 3.10 and 3.12 each passed 33 tests; 68 MDX pages compiled.
No new PyPI release. See benchmarks/industrial/MEDIA_VALIDATION.md. R10 controlled
robustness and R11 world-model experiments remain active, alongside R02/R03/R12
execution/evidence/stability requirements and final cross-project delivery.


2026-09-17 controlled robustness checkpoint: corrected mutation label copying and
hardness error accounting. Transformations now produce explicitly unlabelled
candidates; task validators bind derived answers and evidence to immutable pairs
before manifest export. Case JSONL retains stable IDs and source groups. Reused
CheckList's behavioral distinction and Hypothesis 6.168.0 rather than introducing
a new dataset/search engine. Four synthetic numeric sources produced 8 valid,
4 invalid and 4 unknown candidates. Both deterministic parser controls satisfied
all four invariance and all four output-change checks, yet exact correctness was
12/12 versus 0/12. This demonstrates why consistency alone is insufficient, not
industrial or SoTA performance. Two hundred generated property examples passed.
Hardness controls retain unknown baseline/configuration failures. The 435,141-byte
raw bundle contains runtime source frozen before execution and 130 checksummed
files; offline replay reproduces results without target calls. Full Python 3.12:
1,776 passed, 5 skipped; focused Python 3.10: 160 passed; 69 MDX pages compiled.
No provider requests or new PyPI release. Broader state/media transformation
oracles, world-model evaluation and the remaining program continue.


2026-09-17 world-model checkpoint: reused native Gymnasium 1.3.0 CartPole,
scikit-learn 1.9.1 BayesianRidge and SciPy 1.18.1 exhaustive planning. Frozen
runtime commit 69c1bdf trained on 4,413 transitions from 200 episodes. Forty
held-out forecast seeds and ten separate planning seeds expose metric/decision
mismatch: both learned models have near-exact next-angle predictions, while only
the action-conditioned model reaches the 200-step cap (10/10 versus 0/10;
controls average 9.6 steps). Persistence preserves state offsets but fails control.
Step-32 metrics have six observed sources, with 34 explicitly censored. Full
planned actions prevent reference-length leakage identified during development.
The 8,009,403-byte public bundle retains 1,035 checksummed files, pre-run source,
model parameters, raw episodes/predictions/planner costs and both development
runs. Offline replay and published MCP/core accept/reject/indeterminate checks
passed. Full Python 3.12: 1,795 passed, 5 skipped; final focused checks: 20 on
Python 3.10 and 3.12; 70 MDX pages and the runnable guide verified. Stale editable
distribution metadata is disclosed; exact execution is identified by the clean
Git revision and source snapshot. No API calls or new PyPI release. This completes
the bounded R11 development demonstration, not industrial/robotic validation or
the remaining execution, extension, release and contribution requirements.


### Agent-grader integrity follow-up (2026-09-17)

The agent audit reproduced false quality votes after exceptions, unknown-verdict
exclusion, perfect scores with missing tool traces and silent eight-item/200-character
prefix grading. `agent-judgments/v2` requires complete criteria and retains raw
judgments, including failed attempts, in suite trial snapshots. Configurable item
limits now skip the whole measurement before calls; empty tool denominators skip.
ToolCallAccuracy can still assert an observed no-tool outcome. Per-grader judge
configuration, full tool results/prior results and memory references are preserved.

Validation: full tracked Python 3.12 suite 1,878 passed, 5 skipped, 7 warnings;
three additional regression tests subsequently passed with the focused tests.
Python 3.10 execution/concurrency/round-trip subset and all 70 MDX pages checked.
These are offline regression fixtures, not evidence of judge accuracy. No API
requests or PyPI release. Provider usage/request capture and interrupted attempts
remain open; Inspect's native exception logging does not promise these attached
judgment records. Grader dependency contracts remain the next R01 gap.


### R01 dependency-contract follow-up (2026-09-17)

New suite locks bind inherited judge settings, portable configuration, digests of
private settings, type distinctions, local engine Python bytes and native installed
distribution metadata. Reuse `importlib.metadata`, `hashlib` and the existing
`jsonschema` dependency; no package resolver or dataset infrastructure is added.
Custom/opaque graders require `declare_dependencies`, including a caller-managed
contract revision and optional named artifact files rehashed before/after execution.
Private/common credential-like fields are hashed rather than copied verbatim.

Sync, parallel, async, imported-output and regrading paths retain pre-run locks;
observed drift retains the post-run lock and makes comparisons indeterminate.
Malformed/missing dependency records, missing locks and modified lock digests are
explicit unknown evidence. Old snapshots need both comparison sides rerun; the
legacy identity override cannot bypass recorded dependencies. Grading remains
usable without a verifiable contract, but comparison gates cannot silently accept it.

Limits: caller declarations do not prove completeness; local source bytes are not
loaded bytecode, package metadata is not a verified build, and mutable remote model
aliases/undeclared state or transient changes restored between snapshots remain
outside this protocol. All installed distributions are captured conservatively;
unrelated upgrades also invalidate compatibility. Native Inspect logs without this
suite snapshot remain diagnostic for cross-run comparisons. Provider-request and
usage accounting is still R02 work, and this does not complete R03/R12/R16/R17.

Validation for this dependency checkpoint: full tracked Python 3.12 suite
1,909 passed, 5 skipped, 7 warnings; Python 3.10 focused checks 123 passed.
All 70 MDX pages compile. Versioned-evidence/Hugging Face workflows and the custom
class/dependency declaration examples executed locally. Fixtures cover actual file
rehashing, configuration mutation across all runner paths, private-value hashing,
malformed/rehashed records, legacy migration, and schema/callback/judge drift.
The concurrency check measures simultaneous grader execution rather than including
snapshot setup in a wall-clock threshold. No API calls or PyPI release were made.

### R03 execution controls follow-up (2026-09-17)

Source `ad1d015` validates native controls before preparation, drains owned async
tasks on cancellation, and retains Inspect stop/invalidation evidence through
import, error-budget policy and repeated regrading. Existing upstream scheduling
and limits remain authoritative; no new durable runner is introduced.

The frozen offline study uses seven SQLite ledger scenarios plus native
concurrency and cancellation probes. It retains all nine native logs and 35
checksummed artifacts, with zero attempted network connections. Text-only
bounded-task policies accepted all six missing writes; a required independently
queried state invariant rejected them and accepted the completed control.
Full tracked Python 3.12 verification: 2,041 passed / 18 skipped / 7 warnings.
See [execution protocol](execution-controls.md) and its linked result bundle.
Checkpoint compatibility, remote side effects, distributed failure and hard
resource guarantees remain open; no paid requests or PyPI release were made.

### R03 declared retry compatibility (2026-09-17)

Frozen `e62dc7c` binds native Inspect tasks and samples to recorded settings,
existing grader/engine snapshots and caller-declared dependency/file revisions.
It checks compatibility before solver execution and detects mixed imported
evidence after an unguarded retry. The three-case SQLite experiment preserves two
completed UUIDs on compatible retry and makes zero new target calls on rejected
preflight. A changed-rule negative control has 3/3 stale native passes versus
2/3 under fresh grading; the mixed report is indeterminate and the fresh report
rejects. All four native logs and the ledger are retained in a 20-file checksummed
bundle, with zero attempted network connections.

Final regression verification: Python 3.12 2,062 passed / 18 skipped / 7 warnings;
Python 3.10 144 broader checks and 43 final focused checks passed. The guide's
examples execute; all 71 MDX pages parse. See
[retry compatibility study](../benchmarks/industrial/RETRY_CONTRACT_VALIDATION.md).
Caller declarations and observed snapshots do not prove arbitrary restored
agent/sandbox state safe; R03's remaining boundaries are explicit.
