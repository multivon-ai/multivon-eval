# Dataset attribution and boundaries

**CORD: A Consolidated Receipt Dataset for Post-OCR Parsing**, NAVER Corp.
Official source: https://github.com/clovaai/cord
License: [Creative Commons Attribution 4.0 International](https://github.com/clovaai/cord/blob/master/LICENSE-CC-BY).
The official repository links `naver-clova-ix/cord-v2` on Hugging Face. This
experiment pins revision `7f0115a4b758a71d6473b8d085751692da2fef98`.

Changes: selected development/held-out rows, extracted the upstream total label
and annotated-word transcript, resized image inputs as specified in the protocol,
and added a local ledger task. Original labels are retained. These derived
annotations and images retain CORD's attribution requirements; they are not
relicensed under the library's Apache-2.0 license. Public result manifests and
failure reviews identify these derivatives. No affiliation or endorsement implied.
The task is not the official CORD parsing benchmark or its original metric.

**pdfhell**, Multivon, [Apache-2.0](https://github.com/multivon-ai/pdfhell/blob/main/LICENSE).
Generators pinned at `16d184b`. Used without generator changes; numeric answers
checked independently with Decimal arithmetic. Page rendering uses pdfhell's
pypdfium2 adapter. New seeds share the same templates; do not claim layout novelty.

**Inspect** owns provider execution, logs, tools, scoring and retry. **Datasets**
owns Hub access, revision selection, Parquet decoding and streaming. **SciPy**
provides the Wilson intervals and exact binomial paired tests in the analysis.
No upstream dataset loader, statistical implementation or scheduler is copied.
