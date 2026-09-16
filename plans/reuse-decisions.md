# Reuse decisions

2026-09-17 — implementation constraint from the owner: reuse credible projects
and their knowledge; do not rebuild commodity infrastructure.

The purpose of Multivon is to make task-success evidence useful for release
decisions. Dataset loading, telemetry transport, model serving, simulators, and
general workflow scheduling are not differentiators.

| Capability | Upstream foundation | Multivon's boundary | State |
|---|---|---|---|
| Dataset loading, Arrow/Parquet, media, Hub revisions, streaming | [Hugging Face Datasets](https://huggingface.co/docs/datasets/en/loading) | Adapt a bounded evaluation selection; validate case identity and source leakage | Bridge tested with Datasets 5.0.1 on Python 3.10/3.12 |
| Dataset transformation/cache fingerprints | [HF fingerprint semantics](https://huggingface.co/docs/datasets/en/about_cache) | Preserve upstream fingerprint and repository revision; also hash the actual scoring inputs | Implemented in bridge; no new cache engine |
| Grouped split construction | [scikit-learn GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) | Validate supplied assignments and document group-level leakage; no custom random splitter | Decision made; worked example pending |
| Durable agent evaluation, provider configuration, model API logs, retry/resume, sandboxing | [Inspect](https://inspect.aisi.org.uk/eval-logs.html) | Make Multivon graders usable in Inspect and import evidence for acceptance decisions | Native bridge and SIGKILL/recovery tested with Inspect 0.3.263; budget/deadline validation pending |
| Annotation and review | [Label Studio native JSON](https://labelstud.io/guide/export) | Bind labels to exact saved evidence; preserve missing review coverage and disagreement | Development bridge tested with official SDK 2.1.1 and local Community 1.23.0 server/browser; mobile overflow and a browser error documented |
| Reviewed-score threshold analysis | [scikit-learn ROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html), [SciPy binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) | Bind saved scores/reviews and frozen splits; report false accepts, source coverage and leakage | Development implementation with synthetic end-to-end checks; no new learning algorithm or independent-human accuracy claim |
| General log inspection | [Inspect View](https://inspect.aisi.org.uk/log-viewer.html) | Add task/policy-specific views only where existing views are insufficient | Fit assessment pending |
| Telemetry | [OpenTelemetry GenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) | Map to standard spans; preserve schema version and unsupported fields | Adapter pending; conventions marked Development upstream |
| Environment interaction | [Gymnasium Env](https://gymnasium.farama.org/api/env/) | Adapt reset/step/termination; add independent outcome assertions and evidence | Adapter pending |
| World-model simulators | Existing Gymnasium environments and specialist simulators | Measure action-conditioned predictions against simulator state and planning outcomes | Experiment design pending |

## Correction to the first foundation checkpoint

The first development commit named a small snapshot wrapper `Dataset`. It is
now named `CaseManifest` so the API clearly describes its limited role.
It holds the selected evaluation cases, identities, split assignments and source
provenance. It does not implement discovery, download, remote storage, streaming,
joins, transformations, random splitting, or media decoding. Those stay upstream.
The Hugging Face bridge consumes native Dataset/DatasetDict/IterableDataset
objects instead of introducing a competing loader or repository protocol.

Content digests and cache fingerprints serve different purposes. Hugging Face
can assign a random cache fingerprint when a transformation cannot be hashed.
Therefore an upstream cache key alone does not prove two scoring inputs match.
The small case digest is retained for that evaluation-specific check.

## Delivery rule for the remaining program

Before implementing a new subsystem, record: the established alternatives,
the upstream feature used, the actual missing behavior, and a failure test for
the integration. Use public interfaces and optional dependencies. Preserve
license/attribution requirements if source code is copied; currently these
bridges use upstream APIs and original adapter code.

The R01–R17 program remains in scope. Integrations can satisfy a requirement;
it does not have to be implemented in Multivon's native runner. Do not build a
second durable scheduler or general dataset platform merely to check a box.
Test the concrete end-to-end workflows, including errors and missing evidence.

## Novelty critique

Stable IDs, manifests, retained logs, grouped splits and release gates are
established engineering practices. They improve correctness, but they are not
a research contribution or defensible moat by themselves. The hypothesis to
test is the usefulness of independently verified task outcomes, controlled
failure cases, and their connection to industrial release decisions. Publish
negative findings and compare against the chosen upstream baseline.

## Document experiment datasets

Use [CORD v2](https://github.com/clovaai/cord), the official corrected receipt
annotations and Hugging Face release, for the public-data track. Keep its
upstream splits and attribution; Multivon's verbatim-total-to-ledger task is a
separate workflow projection, not the original CORD parsing metric. Use existing
pdfhell generators for controlled cross-modality conflicts. Pin their revision
and validate their numeric oracle independently. The frozen
[protocol](../benchmarks/industrial/DOCUMENT_PROTOCOL.md) records selections,
limits and what the experiment cannot establish.

[DocILE](https://docile.rossum.ai/) is the preferred follow-on for invoice fields
and line items, with its own loader, layout-cluster analyses and official
metrics. Obtain data through its access process rather than assuming a mirror's
license or inventing another invoice corpus. FATURA is another candidate for
layout diversity; its suitability/terms have not yet been verified for inclusion.
