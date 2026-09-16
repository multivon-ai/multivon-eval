# Review workflow validation — 2026-09-17

This is an interoperability and failure-behavior check, not a judge benchmark.
All labels and scores used here are explicitly synthetic. No model API calls,
independent human reviews, or production acceptance claims were made.

## Upstream components

- Label Studio Community 1.23.0, served on loopback with a temporary SQLite database.
- Label Studio SDK 2.1.1 for task import, configuration checks and annotation creation.
- scikit-learn 1.9.1 and SciPy 1.18.1 on Python 3.12.
- scikit-learn 1.7.2 and SciPy 1.15.3 on Python 3.10.
- A local headless Chrome session exercised Label Studio's normal login and
  annotation controls. External browser requests were blocked during this check.

Multivon does not replace these projects' dataset, annotation UI, user management,
threshold candidates or interval implementations. Its contribution here is
binding saved scores and reviews to exact evidence, explicit coverage, and frozen
source-disjoint evaluation contracts.

## Actual server and browser check

1. Run `examples/calibrate_reviewed_scores.py` to produce saved reports and
   native Label Studio task files from the synthetic fixture.
2. Configure a Label Studio project with the exported `LABEL_CONFIG` and import
   the 12 development-trial tasks. These cover three sources, two variants and
   two repeats per source.
3. Create one synthetic annotation through the official SDK. Export full JSON
   with `download_all_tasks=true`; verify all 12 original tasks are preserved.
4. Log in through the browser, choose an unannotated task, select Reject, type a
   rationale explicitly identifying browser automation, and submit. The server
   returns HTTP 201.
5. Export again and import through Multivon's bridge with
   `reviewer_kind="synthetic"`. Two task annotations reconcile successfully with
   an explicitly configured minimum of one reviewer. The remaining ten tasks
   are incomplete, not accepted. Original trial/rubric bindings match exactly.

The template uses native radio choices, a visible rationale heading, and an
editable three-line rationale field. The desktop annotation viewport was
1,440 pixels wide with no page-level horizontal overflow.

At a 390-pixel mobile viewport the native UI content remained 905 pixels wide,
so the tested workflow is desktop-oriented. The browser also emitted one
`Illegal invocation` error without a useful stack; it did not prevent project
loading, annotation submission or export. Its cause was not established, and
this check does not claim a clean browser console or comprehensive accessibility.
We did not fork the upstream UI to conceal these limits.

## Statistical behavior

The complete offline demonstration uses a separate, fully annotated synthetic
export for analysis. Development and held-out splits each contain three sources,
with two variants and two repeats per source. The threshold selected on the
invented development scores is 0.8. Held-out evidence has two false-accept trials
from one source: 1/3 sources have at least one false accept, with a two-sided
95% exact source-event interval of approximately [0.0084, 0.9057]. This is a
plumbing check, not an empirical accuracy result.

Regression checks cover changed evidence, missing/cancelled/disagreeing reviews,
invalid numeric labels/scores, duplicate reviewer votes, missing source identity,
source leakage, fitting on a holdout, changed rubric/grader contracts, frozen-fit
mutation, skipped measurements and explicit reject-all selection. Repeating
trials does not increase the independent-source denominator or narrow its CI.

The new focused checks passed 60 tests on Python 3.12 and 59 on Python 3.10,
with the optional Label Studio SDK check skipped on 3.10. The prior full library
run passed 1,604 tests with four skips before the 15 saved-score analysis tests
were added. All 64 MDX pages compiled. See the guides for current release scope:
these features are a development preview and are not in PyPI 0.18.0.

## Remaining limits

Source independence and reviewer identity are assertions, not authenticated facts.
The template displays text and traces; original media needs a suitable native
viewer/project. Recorded grader configuration checks cannot capture undisclosed
opaque dependency changes. Missing reviews can bias the measured subset.
Customer task usefulness and independent label quality remain unvalidated.

Sources: [Label Studio JSON export](https://labelstud.io/guide/export),
[native TextArea behavior](https://labelstud.io/tags/textarea),
[scikit-learn ROC thresholds](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html),
[SciPy exact binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
