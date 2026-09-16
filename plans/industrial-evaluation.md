# Industrial evaluation program

Owner objective (2026-09-17): complete the proposed industrial/FOSS features,
agentic and multimodal evaluation, and experimental world-model evaluation;
update associated projects, website, docs, and README; run meaningful experiments
with available local resources and authorized provider credits; critique results
and establish a defensible contribution. This document preserves the full scope.

Status: active. No requirement is complete merely because an API or test exists.

## Requirements and completion evidence

| ID | Deliverable | Evidence needed | Status |
|---|---|---|---|
| R01 | Versioned cases/datasets: stable IDs, content revisions, source groups, split provenance | Reordering, duplicate inputs, changed contexts/labels, JSON round trips, split leakage and comparison compatibility tested; migration demonstrated | In progress: case/dataset identities, group splits, strict comparison pairing implemented; full grader/run compatibility remains |
| R02 | Complete immutable trial evidence and regrading | All outputs, traces, grader results, retries/errors, effective requests, usage retained across sync/async/parallel paths; saved output regrading makes no target calls | In progress: per-run/retry snapshots and regrading implemented; provider requests, usage, interrupted attempts and execution configuration remain |
| R03 | Resumable execution and resource controls | Kill/restart experiment preserves completed work; checkpoint compatibility, cancellation, bounded concurrency, deadlines and budget accounting; explicit policy for ambiguous side effects | Pending |
| R04 | Acceptance policies | Required-check coverage, critical invariants, per-slice thresholds, sample requirements, quality/error/indeterminate distinctions and machine-readable decisions verified | Pending |
| R05 | Review and calibration | Label import/export, review disagreements, development-only fitting, held-out evaluation, uncertainty/false-accept reporting and no leakage verified | Pending |
| R06 | Trace interoperability | OpenTelemetry ingestion/export, documented schema/version handling, actual round trips with supported companion integrations | Pending |
| R07 | Task/environment/outcome interfaces | Setup/reset/action/observation/cleanup, isolated repeated episodes, real end-state assertions, forbidden side effects, partial failures and recovery cases | Pending |
| R08 | Failure investigation UI | Trial comparison, evidence references, slices, review, case promotion, local security and accessibility; rendered desktop/mobile verification | Pending |
| R09 | Typed multimodal artifacts | Images/pages/audio/video, content identity, timestamps/regions, actual artifact rendering and grounded verdict references, extraction-versus-perception comparisons | Pending |
| R10 | Controlled robustness suite | Validated invariant-preserving and semantic-changing transformations; reviewed or code-derived answers; no transformations that silently corrupt the oracle | Pending |
| R11 | World-model evaluation (experimental) | Real adapter demo; action responsiveness, horizons, state persistence, uncertainty, planning utility; model and simulator errors separated; reproducible measured results | Pending |
| R12 | FOSS extension/stability contract | Public plugin/environment protocols, versioned report schemas, optional heavy dependencies, contributor fixtures, compatibility matrix and release checks | Pending |
| R13 | Industrial workflow and experiments | Document-to-ledger task, independent assertions, failures/benign controls, matched-budget baselines, held-out splits, actual provider runs, costs and uncertainty; results critique | Pending |
| R14 | Research contribution/popularization | Related-work comparison, bounded novelty statement, reproducibility bundle, diagrams when useful, worked demo and evidence-led website narrative | Pending |
| R15 | Associated projects | multivon-mcp, pdfhell, eval-action compatibility and relevant integrations; cross-project end-to-end tests, docs/README, versions and releases as applicable | Pending |
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
