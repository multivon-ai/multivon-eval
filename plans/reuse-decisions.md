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
| General log inspection | [Inspect View](https://inspect.aisi.org.uk/log-viewer.html) | Keep native transcripts/events upstream; extend existing HTML for exact trial identity, coverage and reviewed case promotion | Inspect 0.3.263 sample tabs exercised; its mobile overflow documented. Development Multivon index/report/comparison passed six desktop/mobile axe scans |
| Telemetry | [OpenTelemetry GenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) | Grade retained native OTLP; emit standard evaluation log events, preserving original bytes and convention identity | Development bridge tested with Python SDK 1.44.0, Collector 0.161.0 JSON and published MCP 0.4.0; GenAI conventions remain Development |
| Environment interaction | [Gymnasium Env](https://gymnasium.farama.org/api/env/) | Capture native reset/step/termination/close and independent observed state; no simulator or scheduler replacement | Development adapter tested with Gymnasium 1.3.0, FrozenLake, CartPole and a real SQLite failure/recovery fixture |
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


## Multimodal evidence and grader audit

The 2026-09-17 audit found empty claim extraction could produce a perfect vision
score and malformed judge output could become an ordinary quality measurement.
Correct these before adding modality coverage. The unreleased `vision-qag/v2`
parser distinguishes valid negative judgments from absent/invalid judgments;
this is a measurement fix, not an accuracy improvement claim.

Reuse boundaries for R09:

- [Inspect multimodal content](https://inspect.aisi.org.uk/multimodal.html) owns
  native image/audio/video/PDF transport and log rendering. Do not duplicate its
  provider support or let runtime/model references silently authorize file reads.
- [Hugging Face media features](https://huggingface.co/docs/datasets/package_reference/main_classes)
  own data loading and decoding. Consume bytes or explicitly resolved paths from
  `decode=False`; do not create a competing media dataset/cache abstraction.
- [W3C Web Annotation FragmentSelector](https://www.w3.org/TR/annotation-model/#fragment-selector)
  and [Media Fragments](https://www.w3.org/TR/media-frags/) supply region and time
  reference semantics. The missing Multivon behavior is binding these references
  to exact retained content and validating the evidence used by a verdict.
  These planned bindings are not implemented by the legacy image metadata keys.

Question-based visual scoring already has substantial prior art:
[TIFA](https://github.com/Yushi-Hu/tifa) provides question generation, filtering,
VQA scoring and released human annotations;
[VQAScore](https://github.com/linzhiqiu/t2v_metrics) supplies an established
text-to-visual alignment metric. Multivon's fraction of generated Yes/No
verdicts is not either implementation and cannot inherit their reported results.
For generated-image alignment, evaluate those upstream implementations before
inventing a new metric. Their original task does not validate invoice totals,
state changes or industrial release decisions. No upstream code/data was copied
or a new dataset downloaded for this parser audit.
