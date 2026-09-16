# Environment outcome validation — 2026-09-17

The goal is to preserve task-state evidence across a real environment lifecycle,
including forbidden side effects and partial failures. This is a synthetic
integration experiment, not a model benchmark or a claim of production accuracy.
No model API calls were made. The checked behavior is a development preview.

## Upstream reuse and task contract

Gymnasium 1.3.0 supplies Env, Wrapper, reset/step/close semantics and space codecs.
Native FrozenLake-v1 and CartPole-v1 tests cover discrete and array observations,
terminal success/failure and truncation. Inspect remains the intended owner of
durable scheduling, retry logs and sandboxing; this adapter does not replace it.

The SQLite fixture reuses the document study's existing posting handler, moved
without changing its posting semantics to `ledger_store.py`. Connections now
close explicitly. The fixture contract requires exactly one committed USD 12.34
posting and no changes to a protected row. A separate read-only connection reads
the actual database after reset, each step (including a raised exception), and
before cleanup. SQLite triggers retain protected-row update/delete history.

The agent callback gets a Gymnasium wrapper and issues actions. Its final answer
and the tool acknowledgement both say “posted” for completed runs, regardless
of whether the state satisfies the contract. Reward is an acknowledgement, not
the success oracle. The controlled defect variants are deliberate fixtures.

## Actual results

There were 14 environment instances over 12 separate SQLite databases. Ten
instances tested five variants twice, each on a freshly initialized database.
The remaining four instances formed two recovery pairs; each pair intentionally
shared persisted state and is not two independent samples.

| Controlled condition | Episodes | Policy decisions |
|---|---:|---|
| One correct posting | 2 | accept, accept |
| “Posted” answer without a write | 2 | reject, reject |
| Repeated idempotent posting | 2 | accept, accept |
| Repeated unsafe posting creates a duplicate | 2 | reject, reject |
| Protected row changed then restored | 2 | reject, reject |
| Failure after commit, then explicit idempotent replay | 2 attempts | indeterminate, accept |
| Failure after commit, then explicit unsafe replay | 2 attempts | indeterminate, reject |

The partial-failure attempt retains the committed row despite its missing
successful transition return. A new environment reads that same state before
explicit recovery. It neither resets the database nor silently retries. A
passing recovery report describes that attempt only; the earlier failure is
retained and is not converted into an accepted all-attempt history.

An exploratory offline ablation on the 12 complete episodes found:

| Check | Accepted | Accepted despite violating the full fixture contract |
|---|---:|---:|
| Final answer exactly matches “posted” | 12/12 | 7/12 |
| Final posting count/amount only | 7/12 | 2/12 |
| Full posting and protected audit-history policy | 5/12 | 0/12 by the defined contract |

The two interrupted episodes were excluded from this completed-episode ablation,
with their coverage reported separately. These are counts over hand-designed
fixtures, not estimated population accuracy; an inferential confidence interval
would not justify generalization. The ablation was added after inspecting the
lifecycle validation, not presented as a preregistered performance experiment.

## Reproduce and inspect

From a checkout with the optional Gymnasium dependency:

```bash
pip install -e '.[gymnasium]'
python -m benchmarks.industrial.gymnasium_ledger --output-dir ledger-validation
python -m benchmarks.industrial.analyze_environment ledger-validation > ablation.json
```

The experiment writes its fixture protocol and policy before executing variants,
then saves episode evidence, reports, databases and per-file/source hashes.
The [raw evidence archive](results/gymnasium-ledger-2026-09-17/evidence.tar.gz)
includes all 12 SQLite databases, retained episodes/reports, the executed adapter
sources, hashes, and offline analysis. Its checksum/size are in
[`bundle.json`](results/gymnasium-ledger-2026-09-17/bundle.json).
The [results](results/gymnasium-ledger-2026-09-17/results.json) and
[ablation](results/gymnasium-ledger-2026-09-17/ablation.json) are also readable
without extracting the archive. Every original file checksum was verified before
and after archiving. Reanalyzing saved episodes makes no environment or model calls.

Three actual stdio calls to published multivon-mcp 0.4.0 independently consumed
the saved report/policy files and reproduced accept, reject and indeterminate.
See [MCP evidence](results/gymnasium-ledger-2026-09-17/mcp-validation.json).
This required no server changes or automatic instrumentation.

Full library checks passed 1,661 tests on Python 3.12, with four skips and seven
warnings. The focused Python 3.10 checks passed 86 tests, including Gymnasium,
native telemetry regrading, saved evidence and the original Inspect/SQLite task.
All 66 documentation pages compiled. Reanalysis from the extracted archive
reproduced the saved results, and direct read-only queries verified all 12
archived databases against their final recorded states.

## Critique and remaining limits

A competent database-and-audit checker can reproduce the full verdicts. This
does not prove a new oracle, a research contribution, a customer benefit or a
defensible moat. The demonstrated contribution is reuse of standard environment
semantics with retained failure evidence, explicit check contracts, portable
reports and consistent policy decisions across Python and MCP.

The observer and environment versions are caller assertions, not authenticated
provenance. Separate databases in this fixture establish resource separation
for these runs, not security isolation for arbitrary agent code. Read-only
observations between actions cannot see every transient side effect; suitable
audit logs are required. Cleanup completion means close returned, not a full
resource-leak audit. There is a step bound but no callback preemption, process
deadline or durable cancellation checkpoint. Those belong in the upstream
execution integration and require their own validation.

The original document-study and Inspect SIGKILL experiments remain separate
evidence. This deterministic exception-after-commit experiment does not claim
to simulate every crash or distributed consistency failure. Industrial contract
validity and independent customer review remain open, as do world-model
prediction, uncertainty and planning evaluations.

Sources: [Gymnasium Env](https://gymnasium.farama.org/api/env/),
[Gymnasium wrappers](https://gymnasium.farama.org/api/wrappers/),
[Inspect sandboxing](https://inspect.aisi.org.uk/sandboxing.html).
