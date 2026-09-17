# Industrial workflow experiments

These are synthetic maintainer-run experiments. They test specific failure
mechanisms and integration contracts; they do not establish customer usefulness
or state-of-the-art model performance.

## Financial answer and evidence-record study

The [complete TAT-QA study](results/tatqa-financial-2026-09-17/README.md)
compares answer-only and joint answer-plus-evidence prompts on all 1,663 released
test-gold questions, using Claude Sonnet 5 and the pinned official scorer. The
joint treatment significantly reduced exact match and F1 while increasing cost
and latency. Numeric question-level results, provider accounting, hashes,
context-bootstrap uncertainty and the malformed-output audit are published.

This is a public financial-document reasoning target, not a customer deployment
or a SoTA claim. Its design implication—test answer generation and evidence
binding as separate stages—is a follow-up hypothesis rather than a measured win.

## Process crash after a committed ledger write

The first experiment reuses Inspect 0.3.263 for process recovery and SQLite for
persisted state. There is no custom durable scheduler and no model API call.

```bash
pip install -e '.[inspect]'
python benchmarks/industrial/run_recovery.py --output /tmp/my-fresh-recovery-run
```

Use a fresh output directory. The script launches and kills only its own child
process, then calls Inspect's public `recover_eval_log` and `eval_retry` APIs.
All ledger writes are inside that output directory.

Two completed samples precede a third sample that commits its write and then
waits. The controller sends SIGKILL at that point. The interrupted action is
ambiguous from the runner's perspective: retrying may duplicate a committed
operation. The experiment explicitly permits replay to compare two handlers:

| Handler | Cases | Target invocations | Final ledger entries per invoice | Final-attempt policy |
|---|---:|---|---|---|
| Reused idempotency key | 3 | 1, 1, 2 | 1, 1, 1 | Accept |
| New key per invocation (unsafe control) | 3 | 1, 1, 2 | 1, 1, 2 | Reject |

These are deterministic observed counts for one crash per handler, **n=3 cases
per handler**. No confidence interval or general recovery-success rate is
claimed. The first two completed samples were preserved in both runs. Every
final textual response said `posted`; only the independent ledger invariant
detected the duplicate.

The interrupted report is indeterminate. After recovery, importing the earlier
log retains all four actual executions per handler. Under an all-attempts
policy, the safe handler remains indeterminate because an earlier attempt was
interrupted; the unsafe handler is rejected for the observed duplicate. The
explicit final-attempt policy accepts the safe handler only after the persisted
state assertion passes. It still rejects the unsafe handler.

See [recorded results](results/inspect-recovery-2026-09-17.json). The output folder
also contains native `.eval` logs, their recovered versions, SQLite state,
invocation records, Multivon reports and decisions. No paid inference occurred.

## What the experiment changed

The initial bridge imported a single native log. Inspect's retry log preserves
completed samples but does not itself include every abandoned attempt. Testing
recovery exposed this evidence gap. The importer now accepts
`from_inspect_log(final_log, previous_logs=[recovered_log])`, deduplicates preserved
native sample UUIDs, and retains earlier failed attempts. Partial epoch retries
are tested separately so a preserved epoch is not counted as another execution.

## Limits

- SQLite plus a unique key is one explicit replay-safe mechanism. Arbitrary
  external actions cannot be assumed idempotent. Reconcile an uncertain action
  or stop for review when safe replay cannot be established.
- No OCR, document understanding, real agent, remote provider, network fault,
  distributed worker, or real customer data was exercised here.
- The existing native `EvalSuite` runner has not gained crash recovery from
  this integration; the tested path uses Inspect.
- The bridge cannot infer omitted historical logs. Keep the entire native log
  chain and provide it when judging all attempts.
- A plausible final answer does not prove persisted task success. This is an
  established evaluation principle illustrated by a concrete fault injection,
  not a new scientific contribution.

The follow-up [document-to-ledger study](DOCUMENT_RESULTS.md) now includes
CORD/pdfhell sources, held-out comparisons, actual provider calls, costs and
reviewed failures. [Raw artifacts and offline reproduction](REPRODUCE_DOCUMENT_RESULTS.md)
are public; customer usefulness remains unproven.


## Environment lifecycle validation (development preview)

[Gymnasium/SQLite validation](ENVIRONMENT_VALIDATION.md) reuses the existing
posting handler and standard environment interfaces. The raw archive retains
14 environment instances across 12 databases, including duplicate writes,
forbidden changes followed by restoration and failure after commit. This is
synthetic integration evidence, not model accuracy or production validation.
The offline ablation makes the comparison against simple answer/state checks
explicit and does not claim an advantage over a complete bespoke outcome checker.


## Media evidence validation (development preview)

[Content-bound media validation](MEDIA_VALIDATION.md) records native dataset/log
round trips, actual media playback, a six-call PDF/pixels/text probe on two
synthetic pdfhell sources, and a small raw archive for offline reproduction.
The result demonstrates an evidence/representation boundary, not general model
accuracy, statistically supported superiority or customer validation.
