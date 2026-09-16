# Reproduce the document-to-ledger analysis

The [raw evidence archive](https://github.com/multivon-ai/multivon-eval/releases/download/v0.18.0/document-ledger-2026-09-17.tar.gz)
contains original selected CORD images/labels, generated pdfhell fixtures,
Inspect logs (including interrupted and resumed runs), reports, SQLite ledgers,
the protocol, failure review, attribution and per-file checksums.

Archive: 130,136,236 bytes, 167 files. SHA-256:

```text
c32d0b11b315787b6c513293fc3dc23884c022378488c648a01fd96f15ca131e
```

GitHub's uploaded-asset digest matches this hash. The original artifact manifest
was verified before packaging. Media data URIs were decoded for credential-pattern
scanning; no matches remained. Dataset attribution and modifications are recorded
in `THIRD_PARTY.md`; CORD derivatives retain CC BY 4.0 terms.

## Recompute without provider calls

Run from a clone of this repository, with Python 3.12. Download the archive into
the clone's parent directory and extract it there. The commands below use those
relative paths:

```bash
python -m pip install "multivon-eval==0.18.0" "scipy==1.18.1"
python benchmarks/industrial/analyze_documents.py \
  ../document-ledger-2026-09-17/multivon-doc-heldout-haiku-v1 \
  ../document-ledger-2026-09-17/multivon-doc-heldout-sonnet-v1-resumed \
  --output reproduced-heldout.json
```

Compare with `benchmarks/industrial/results/document-ledger-2026-09-17/heldout-analysis.json`.
This reproduces every reported count, interval, paired test and diagnostic using
the saved reports; it does not rerun inference. Log filenames are normalized to
basenames so local directory paths do not change the analysis output.

Inspect can open the `.eval` logs using its own viewer. To run new model calls,
follow `DOCUMENT_PROTOCOL.md`, including pinned source revisions, new output
locations, fixed budgets and task definitions. A rerun is a new experiment;
provider outputs and prices are not guaranteed to repeat.

## Preserve the limitations

The original inference revision and recovery changes are recorded in the
execution and recovery locks. The experiment predates the 0.18.0 release.
Known usage estimates exclude two cancelled requests with unknown billing.
Both models failed the frozen policy; some failures reflect the verbatim-label
contract rather than incorrect business values. The review was performed by
Codex, not an independent human. CORD merchant independence and generalization
to unseen synthetic layouts remain unproven. See `DOCUMENT_RESULTS.md`.
