# Execution control protocol

R03 delegates durable scheduling, sample limits and provider connection management
to Inspect. The native offline runner still needs valid controls, owned task
cleanup and honest evidence when cancellation cannot terminate underlying work.

## Implementation and task contract

- Validate positive integer run/worker/concurrency controls and finite [0, 1]
  quality/error thresholds before preparation or target/judge calls.
- Cancel and await owned asynchronous cases/evaluators on escaping failures or
  cancellation. Leave unrelated application tasks alone. Synchronous thread work
  can outlive a cancelled await; do not claim otherwise.
- Import native Inspect stop limits, invalidations, errors, selected resource and
  generation settings, and measured durations. Preserve them on regrading.
- Default limit-stopped samples to indeterminate acceptance. Named
  `accepted_limits` is an explicit task contract for scoring at a boundary, not
  proof of goal completion. Do not override native errors or invalidations.
- Keep original scores and logs. Do not equate a successful log write/run with a
  successful task or interpret an acknowledgment as evidence of a committed write.

## Offline experiment

Freeze a clean source commit before running
`benchmarks/industrial/execution_controls_experiment.py --output DIR` with the
Inspect extra. Save protocol/runtime versions before execution and block socket
connections during the experiment. Retain all native logs, SQLite ledgers,
strict/bounded-policy reports and a summary.

Seven single-case ledger scenarios include one completed positive control and
six upstream limit stops: time, working, message, token, cost and turn. Each solver
acknowledges a write before it commits. The text grader and a separate SQLite query
intentionally answer different questions. Compare strict import, declared boundary
acceptance with text alone, and a policy requiring the persisted-state invariant.

Additional probes run six samples with max_samples=2 and max_connections=1,
measuring actual overlap, and cancel an evaluation after two active mock provider
calls with a third sample queued. Confirm cleanup, preserved cancellation log,
and an indeterminate acceptance result.

Native mock usage is 4 input + 6 output tokens per completed generation with a
synthetic $1/million tariff. An 8-token or $0.000008 threshold may be exceeded by
an already dispatched generation. The observed Inspect 0.3.263 turn limit also
needs explicit reporting rather than an assumed strict one-call bound.

## Remaining boundaries

This fixture covers cooperative async work, local SQLite and the actual Inspect
runtime. It does not validate arbitrary blocking code, remote side-effect rollback,
provider-side cancellation, hard spend reservations, distributed worker failures,
or checkpoint compatibility after task/solver/grader/environment changes.
Declared native retry history now checks task/grader/dependency bindings as
described below, while arbitrary restored checkpoint state remains outside the
verified profile. Keep R03 open until its remaining boundaries have corresponding
implementation/evidence.

## Regression verification before freezing

Python 3.12: 2,041 passed, 18 skipped, 7 warnings across tracked non-live tests and
the two new control test modules. Python 3.10: 158 focused execution, Inspect,
trial/regrade and acceptance checks passed. All 71 MDX pages parsed. Native
control probes include six stop types, invalidation and repeated regrading,
sample/connection overlap, cancellation with a queued sample, and the documented
synchronous-thread limitation. This verification makes no accuracy or production
reliability claim and made no paid provider requests.

## Frozen result

Final source `ad1d015` passed the full 2,041-test Python 3.12 suite. Its clean
offline protocol retained all nine native logs and seven ledgers with zero
attempted network connections. Six interrupted writes had zero persisted rows;
only the completed control had one. Text-only declared-boundary acceptance
accepted the incomplete acknowledgments, while the independent state invariant
rejected them. Two active calls cleaned up on cancellation, and six concurrency
samples respected the measured 2-solver / 1-model peaks. See the
[study and checksummed evidence](../benchmarks/industrial/EXECUTION_CONTROLS_VALIDATION.md).

The acceptance follow-up keeps stop/invalidation validity separate from a general
error budget, including after regrading. Dedicated final checks passed 76 tests
on Python 3.12 and 52 on Python 3.10; no additional paid requests were made.

## Declared retry compatibility

`bind_inspect_task` now reuses native Task/Sample metadata, existing grader locks,
engine inventory and named-file dependency hashes. It records actual native
input/reference hashes, static dataset identity, selected task settings and
caller-declared solver/environment/configuration revisions. A prior native log
can reject incompatible task reconstruction before solver execution. Native
factory construction/preparation can already have side effects before the guard.

Scoring snapshots retain before/after grader configurations. Imports check
sample/root bindings, observed grading drift, native retry plan/settings and
declared contracts. Legacy chains without declarations remain diagnostic.
Per-trial compatibility issues survive regrading and increased error budgets.

The executable `benchmarks/industrial/retry_contract_experiment.py` uses three
SQLite cases, with two completed cases and one real injected exception after a
committed write. It compares a changed-policy blocked preflight, compatible native
retry, intentionally unguarded mixed retry, and a fresh changed-policy evaluation.
The unguarded native retry preserves two stale passing scores (3/3), while fresh
grading under that same new policy rejects one case (2/3). The declared guard
must add zero target calls; unchanged recovery must preserve sample UUIDs.

This covers the declared static-task/native-log profile. It is not arbitrary
hidden-state discovery, checkpoint signature/authenticity verification, restored
sandbox-state validation or hard resource reservation. Callers must bind source,
image, client and service revisions and provide the complete native log chain.

Executed from frozen `e62dc7c`: four native logs, three imported reports, eight
target invocations and zero attempted network connections. A changed-policy
preflight added zero calls; compatible retry preserved two completed UUIDs and
reran only `c`. The unguarded mixed log's 3/3 native passes became indeterminate,
while fresh changed-policy grading passed 2/3 and rejected the critical invariant.
The [study and 20-file archive](../benchmarks/industrial/RETRY_CONTRACT_VALIDATION.md)
retain those outcomes and the declared boundaries. Final Python 3.12 verification
passed 2,062 tests / 18 skips / 7 warnings; Python 3.10 passed 144 broader checks
and 43 final focused checks. All 71 MDX pages parsed. No PyPI release was made.
