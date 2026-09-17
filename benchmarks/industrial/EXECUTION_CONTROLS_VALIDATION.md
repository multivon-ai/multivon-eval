# Execution controls: native Inspect validation

The frozen offline experiment at `ad1d015f53a45c7d042e46cf73f563caeab31067`
verified six native limit stops, a completed control, sample/model concurrency and
cooperative cancellation. It also demonstrated a task-measurement failure:
an acknowledgment passed a text check even when no ledger transaction existed.
An independent SQLite query distinguished every interrupted fixture from the
completed positive control. This is fault injection, not model accuracy evidence.

## Protocol and reproducibility

- [Executable protocol](execution_controls_experiment.py), frozen before the run.
- [Protocol record](results/execution-controls-2026-09-17/protocol.json) and
  [measured summary](results/execution-controls-2026-09-17/summary.json).
- [Evidence archive](results/execution-controls-2026-09-17/evidence.tar.gz):
  35 files, 77,827 bytes; nine native `.eval` logs, seven SQLite databases,
  fourteen strict/bounded reports, protocol, summary, environment, validation
  and process log. Native log references retain their original temporary paths;
  locate the corresponding relative path in the archive without editing the
  content-addressed report.
- [File manifest](results/execution-controls-2026-09-17/manifest.json) binds
  individual file hashes and archive SHA-256
  `22dd0cbba123b75e5cb3cde00d5e04ee385744dbd07efe818cd28ec72875905f`.

Run with Python 3.12.13 and Inspect 0.3.263. The editable core's module version was
0.18.0 while its installed distribution metadata still reported 0.17.0; both are
recorded. The clean source commit identifies the actual implementation. At that
commit, with the optional Inspect dependency installed:

```bash
python benchmarks/industrial/execution_controls_experiment.py --output /tmp/controls-replay
```

Use a new output directory. Socket connections were blocked during execution;
zero attempted connections and zero paid provider requests were observed. Native
`mockllm` used synthetic usage and tariffs. All nine saved native logs were read,
all seven databases queried, all fourteen report JSON files reloaded, and every
archived member hash verified after packaging.

## Ledger outcomes

Each of seven scenarios has one local invoice. The solver emits `posted` before
attempting the write. The positive control commits one row; six limit scenarios
stop before the commit. ExactMatch checks the acknowledgment; `ledger_committed`
queries persisted state through a separate SQLite connection.

| Scenario | Completed mock generations | Persisted rows | Default text-only acceptance | Declared-boundary text acceptance | Declared-boundary text + required state invariant |
|---|---:|---:|---|---|---|
| Completed control | 0 | 1 | accept | accept | accept |
| Time: 1 second | 0 | 0 | indeterminate | accept | reject |
| Working: 1 second | 0 | 0 | indeterminate | accept | reject |
| Messages: 2 | 0 | 0 | indeterminate | accept | reject |
| Tokens: 8 | 1 | 0 | indeterminate | accept | reject |
| Synthetic cost: $0.000008 | 1 | 0 | indeterminate | accept | reject |
| Turns: 1 | 2 | 0 | indeterminate | accept | reject |

All seven native logs report `success`: Inspect intentionally scores the current
output after a limit stops execution. That log status is not a task-completion
verdict. [Inspect sample limits](https://inspect.aisi.org.uk/setting-limits.html)
describe this behavior. Multivon's default import now retains the stop and blocks
acceptance. Raising `max_error_rate` to 1 did not authorize an undeclared limit.

`accepted_limits=(kind,)` explicitly allows scoring at a named boundary. The
text-only policy then accepts all six incomplete writes: a deliberate negative
control for an inadequate task contract. Adding the persisted-state check rejects
all six, while accepting the completed control. No amount of parser, cancellation
or logging robustness makes a text acknowledgment sufficient proof of task success.

The token/cost fixtures declare 4 input + 6 output tokens per mock generation and
a synthetic $1/million input/output tariff. The first generation therefore reaches
10 tokens and $0.000010, exceeding the thresholds of 8 and $0.000008. This is not
billing data. These are observed-usage thresholds, not hard spend reservations.
The tested `turn_limit=1` path made **two** mock generations before stopping;
the native log reports value 2, limit 1. Preserve this observed upstream behavior
rather than claiming a strict one-request ceiling.

## Concurrency and cancellation

Six samples ran with `max_samples=2`, `max_connections=1`. Counters inside actual
native solver and mock-model calls measured a solver peak of 2 and a model peak
of 1, with six completed generations and no remaining active calls. These reuse
[Inspect's distinct sample and model concurrency controls](https://inspect.aisi.org.uk/parallelism.html).

The cancellation probe planned three samples and cancelled after two active
mock-model calls started, with the third queued. Both active calls cleaned up;
the queued call never started. `CancelledError` propagated and the native log
retained two errored samples with overall status `cancelled`. Imported acceptance
remained indeterminate. Remote execution/billing after real request cancellation
was not tested by this mock-provider probe.

## Implementation regression evidence

The final tracked non-live Python 3.12 suite passed **2,041 tests**, with 18 skips
and 7 warnings. The Python 3.10 execution/evidence subset passed 158 tests; after
the error-budget follow-up, 52 relevant tests passed again on 3.10 and 76 relevant
tests passed on 3.12. All 71 MDX pages parsed.

Regression fixtures verify validation before preparation/calls, actual native
run-wide concurrency, child/parent cancellation, sibling evaluator cleanup,
preserved journal closures, invalidation, report round trips and repeated regrading.
A separate thread fixture proves that a cancelled async await can finish while
the synchronous grader thread remains active. No thread-termination guarantee is
claimed. Invalidated scores do not become valid measurements through an increased
error budget, a limit declaration or a new saved-output grade.

## Limits and critique

These are seven authored ledger fixtures and two resource-control probes, not
independent customer tasks, a statistical performance estimate or a new benchmark
dataset. Model responses and usage are synthetic. Time limits cover upstream
execution semantics; scoring, cleanup, blocked threads and external resources
must not be confused with a hard wall-clock cap on the entire process.

The library fixes hang/cancellation and evidence-loss paths and reuses upstream
execution infrastructure. It does not establish full checkpoint compatibility
after changes to tasks, solvers, graders or environments. Distributed failures,
remote side-effect ambiguity, security isolation and hard spend control still
need separate evidence. R03 remains open; this study does not establish a moat
or production readiness.
