# Content-bound media validation — 2026-09-17

The contribution is an evidence binding across existing tools: Hugging Face
loads datasets, Pillow/PDFium/PyAV inspect media, Inspect transports and displays
it, and a small W3C annotation profile attaches references to explicit verdicts.
This is not a new dataset, codec, vision metric or demonstrated moat.

## What was checked

- PNG/JPEG/WebP content identity, dimensions and orientation; WAV/MP3/MP4 stream
  metadata; real PDF opening and page rendering. Modified bytes, false rehashed
  geometry and malformed/unsupported descriptors cannot verify.
- Case identity includes ordered media descriptors. Identical pixels from
  different page roles retain different provenance identities. Actual native
  Hugging Face `Image(decode=False)` bytes pass through the existing manifest bridge.
- Native Inspect content includes verified bytes. Missing resolvers, changed
  input, unresolved attachments and incompatible media types fail explicitly.
  A native run and persisted-log import were exercised, not only mocked types.
- W3C annotations retain whole-artifact, pixel-region and closed NPT time
  references. Invalid bounds, ambiguous selectors, unknown targets and unsupported
  orientation/video-region cases reject. Reference validity does not establish
  truth; the task oracle remains separate.
- PDF rendering uses pypdfium2 5.13.0 and retains the original descriptor,
  one-based page, scale and renderer identity. PDFium calls in this module are
  serialized; that lock cannot protect unrelated PDFium calls in the process.

Probes do not guarantee complete packet decodability, correct perception,
sandboxing or deadlines. Video spatial citations, arbitrary annotation selectors,
OCR alignment and native citation extraction remain outside this profile.

## Actual native viewer

Chrome with Playwright 1.63.0 opened Inspect View 0.3.263 and four saved transport
samples. The rendered page loaded at 918 × 1,188 pixels. The one-second WAV and
MP4 reached ready state 4; playback advanced for both. The video decoded at
64 × 48 pixels. No browser JavaScript errors were observed.

The PDF input appeared as a document badge, not an inline rendered page; clicking
the badge did not open or download it in this check. Retained page PNGs provide
visual inspection. All four sample pages were 450 pixels wide at a 390-pixel
viewport, so desktop viewing is recommended. No accessibility compliance claim
is made for the upstream UI.

[Browser findings](results/media-2026-09-17/browser-findings.json),
[rendered page](results/media-2026-09-17/media-page-image.png),
[mobile page](results/media-2026-09-17/media-page-image-mobile.png),
[video playback](results/media-2026-09-17/media-ramp-mp4.png), and
[PDF badge](results/media-2026-09-17/media-native-pdf.png) retain this check.

The transport task used four **mock** outputs and proves no model perception.

## Six live document calls

Reuse pdfhell 0.6.2's `hidden_ocr_mismatch` generator, seeds 501 and 502. These
are synthetic development sources, not customer documents or untouched holdout.
The manifest and media were saved before target execution. Each source has
native PDF, rendered-page PNG and extracted-text treatments. The PDF text stream
contains both a visible total and a different invisible OCR amount; the PNG
contains the visibly printed amount.

Claude Haiku `claude-haiku-4-5-20251001` received six requests, one per case,
with retries disabled and a 128-token output limit. Inspect retained six complete
model events with raw requests and responses. Exact comparison against the
generator's authored visible amount produced:

| Treatment | Passed / cases | Source groups |
|---|---:|---:|
| Native PDF | 2 / 2 | 2 |
| Page pixels | 2 / 2 | 2 |
| Extracted text | 1 / 2 | 2 |

For seed 501, extracted-text output was `$23,900.25`; the visible expected total
was `$18,900.25`. Both native PDF and pixels returned the visible total. Seed 502
returned `$12,345.67` for all treatments. These are exact fixture outcomes,
not statistically supported modality rankings or general accuracy estimates.

Known usage was 6,744 input and 54 output tokens, with zero reported cache tokens.
At the configured [Haiku pricing](https://platform.claude.com/docs/en/about-claude/pricing),
the estimate is **$0.007014**, not a provider-invoice reconciliation. No live
audio/video model calls occurred. The model's native PDF preprocessing is not
isolated here; calling that treatment pure text extraction would be incorrect.

The critical visible-amount policy rejects the six-case report. Published
multivon-mcp 0.4.0 reproduced that rejection through an actual stdio
`eval_acceptance_report` call. The MCP server consumes the previously produced
saved report; it does not independently fetch or decode the media.

### Critique

The text-only prompt asks about visible print after extraction has discarded
visibility information. Its error illustrates a representation limitation in
this fixture; it is not evidence that a text reader failed an equally informative
task. Different input costs and undocumented provider PDF processing also prevent
a matched-budget algorithm comparison. One call per treatment does not estimate
repeatability, and six variants do not create six independent sources.

The annotations point to whole documents/pages, not independently located total
regions. Bounds validation cannot prove that the model used the cited evidence.
The code-derived oracle and this inspection are not independent human review.
Do not publish these numbers as SoTA or as customer acceptance evidence.

## Raw bundle and offline reproduction

[Evidence archive](results/media-2026-09-17/evidence.tar.gz): 262,047 bytes;
SHA-256 `51ecdd20248593793c2948d09d8831cae3386ed9d628bea6893a370d974c37d8`.
The archive contains native logs, actual assets, manifests, reports, policy/MCP
result and replay source. All 28 listed per-file checksums were verified after
extracting the archive. Resolved native logs were checked for credential-shaped
strings before publication.

Replay source was captured after the live run. A property re-probe defense was
added to `MediaArtifact.verify` afterward, then all saved native logs and assets
were re-verified successfully. This is not a pre-run source attestation; the
archive records that distinction in `validation.json`.

```bash
pip install -e '.[media,inspect]'
mkdir media-replay
# Extract the linked evidence.tar.gz into media-replay.
python -m benchmarks.industrial.replay_media media-replay
```

Offline replay verifies bytes/properties and checksums, reproduces the six
saved outputs and rejection, and makes zero target calls. It does not regenerate
model responses. To exercise the complete workflow from new synthetic assets:

```bash
pip install 'pdfhell[pixels]==0.6.2'
python examples/media_evidence.py --output-dir media-demo
inspect view --log-dir media-demo/logs
```

That command is mock-only unless `--document-model` is supplied. ReportLab PDF
metadata can change bytes across regeneration; retain the exact original assets.

The full Python 3.12 suite passed 1,739 tests with five skips and seven warnings.
After adding codec/orientation cases, focused Python 3.10 and 3.12 checks each
passed 33 tests. All 68 MDX pages compiled. These APIs remain
a development preview, not part of PyPI 0.18.0.
