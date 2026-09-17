# Failure investigation validation — 2026-09-17

This validates an investigation workflow, not application quality or customer
usefulness. Fixtures and review annotations are explicitly synthetic. No model
API calls or independent human review were involved.

## Reuse assessment

Inspect View 0.3.263 opened the existing document-study mock log with 16 samples.
We opened one sample's messages, scoring and metadata tabs through the actual
browser. Native log history, conversations, events and media remain in Inspect;
Multivon does not implement a competing generic trace viewer. At a 390-pixel
viewport the tested native sample panel extended to 450 pixels, so desktop use
is recommended for this upstream view. No upstream accessibility certification
or mobile compatibility claim is made.

Label Studio remains the review UI and user store. Its actual server/browser
round trip was validated in [the review workflow check](REVIEW_WORKFLOW_VALIDATION.md).
This checkpoint reuses that native JSON binding, consensus and reviewer-kind
contract for promotion into development cases. It does not invent a new label
format or assert that two reviewer IDs establish independent humans.

## Changes and failure checks

- Saved trial cards expose original case/content identity, run and attempt,
  exact output, grader reasons, errors, origin references and recorded gaps.
  Full retained JSON can be inspected and downloaded without network fetching.
- Tag, status and case-input/ID filters narrow the displayed cases. Overall
  summary metrics remain explicitly identified as full-report metrics.
- Comparison now associates reasons and trial evidence by case identity. Two
  distinct cases sharing a prompt cannot silently receive each other's reasons.
  Unmatched cases and comparison issues are visible.
- Generated directory links bind path identity and file contents. Inserting a
  new earlier-sorting file preserves the target; overwriting the target returns
  HTTP 409 instead of displaying different evidence under the old link.
- Local single-file and directory servers reject foreign Host/Origin and
  cross-site requests, disable framing/remote fetching and use no-store headers.
  Directory discovery excludes symlinked files and paths outside the root.
- Candidate download requires intact trial evidence. Promotion requires bound
  review consensus plus an explicitly authored expectation and rationale,
  preserves source grouping, clears the execution trace and writes a new
  development manifest. Missing/disagreeing/unknown reviews and changed bindings
  cannot promote. The observed failure is never inferred to be the oracle.

Host/origin checks are not authentication against local processes. The server is
read-only and loopback-only; it is not a general network service or sandbox.
Old bare numeric URLs still refer to the current directory index. Exported HTML
contains retained evidence and lacks the HTTP server's response protections.

## Actual browser and command workflow

`examples/investigate_failures.py` produces two reports with three cases and two
trials per case. Two distinct cases share a prompt. A repeated case is flaky,
capture coverage is intentionally incomplete, and hostile script/HTML strings
are retained as text. Two synthetic annotations review one specific failed trial.

Headless Chrome with Playwright 1.63.0 and axe-core Playwright 4.13.0 exercised:

1. Directory selection and baseline/proposal comparison using generated links.
2. Tag filtering and case-ID search, each narrowing the list to one case.
3. Keyboard activation of the case button and native trial details, checking
   `aria-expanded` and the displayed original output.
4. Actual candidate download, parsed back as JSON to verify the selected case,
   required-review flag and original expected answer.
5. The index, expanded report and comparison at 1,440 × 1,000 and 390 × 844.

All six final page/viewport scans had zero detected axe violations and no
page-level horizontal overflow. The Multivon browser sessions had no JavaScript
errors or remote requests, and the hostile fixture did not execute. Initial
checks found contrast, landmark, empty-header and skipped-file disclosure issues;
these were corrected before the final scan. This does not replace manual
screen-reader testing or establish comprehensive accessibility compliance.

[Browser findings](results/investigation-2026-09-17/browser-findings.json),
[desktop report](results/investigation-2026-09-17/report-desktop.png),
[mobile report](results/investigation-2026-09-17/report-mobile.png), and
[mobile comparison](results/investigation-2026-09-17/comparison-mobile.png)
retain the rendered check.

The separate `examples/promote_reviewed_case.py` command consumed the saved
report, original task array, native synthetic annotation export, authored
expectation and rationale. It wrote a new development manifest successfully.
Unit checks reject changed source grouping, tampered tasks, missing or disputed
reviews, detached trial headers, cross-origin access and overwritten evidence.

Full library checks passed 1,674 tests on Python 3.12, with five skips and seven
warnings. Focused Python 3.10 checks passed 76 tests with one directory-only
skip. All 67 MDX pages compiled. These changes remain a development preview.

## Reproduce

```bash
pip install -e .
python examples/investigate_failures.py --output-dir investigation-demo
multivon-eval view --dir investigation-demo --no-browser
```

The example also saves a complete synthetic review and promoted development
case. For actual review files and the separate promotion command, follow the
[guide](../../docs/guides/failure-investigation.mdx). Do not reuse synthetic
annotations as human validation. Inspected held-out sources must not be
presented as untouched validation data after promotion or tuning.

## Contribution critique

Evidence browsers, grouped identities, review workflows and regression cases
are established practices. The useful integration here is maintaining their
bindings across saved trials, native tools and development manifests. This is
not demonstrated research novelty or a defensible moat. Customer workflows,
independent oracle validation and task-specific value remain open requirements.

Sources: [Inspect View](https://inspect.aisi.org.uk/log-viewer.html),
[Label Studio JSON export](https://labelstud.io/guide/export),
[axe-core Playwright](https://github.com/dequelabs/axe-core-npm/tree/develop/packages/playwright).
