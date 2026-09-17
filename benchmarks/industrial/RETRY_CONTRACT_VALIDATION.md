# Retry compatibility: stale scores after a task change

A frozen native Inspect experiment reproduced a misleading release result:
retrying after a business-rule change preserved two stale passing scores and
produced a 3/3 passing native log. A fresh evaluation under the changed rule
passed 2/3. Multivon's declared task preflight rejected the incompatible retry
before any additional target call, and its importer diagnosed the mixed evidence
when that guard was deliberately omitted.

These are three synthetic SQLite cases and deterministic checks, not model
accuracy measurements or independent customer validation.

## Frozen protocol and artifacts

- Source commit: `e62dc7cedf4a4367f38a865b34b08483895ec724`.
- [Driver](retry_contract_experiment.py) and [registered native task](inspect_contract_task.py).
- [Protocol](results/retry-contract-2026-09-17/protocol.json) and
  [observed summary](results/retry-contract-2026-09-17/summary.json).
- [Evidence archive](results/retry-contract-2026-09-17/evidence.tar.gz):
  20 files, 107,123 bytes, including four native logs, three imported reports,
  the SQLite ledger, invocation history, source files, policy inputs,
  environment and validation records.
- [Manifest](results/retry-contract-2026-09-17/manifest.json) with per-file hashes
  and archive SHA-256:
  `cb1fb709689dfd49dd1fde838fa5f36bd509918235c936b52125289decbdfb6a`.

Runtime: Python 3.12.13, Inspect 0.3.263, local core module 0.18.0. The editable
distribution metadata still reported 0.17.0; the protocol retains both values.
The clean source commit identifies the executed implementation. With that commit
and its optional Inspect dependency installed:

```bash
python benchmarks/industrial/retry_contract_experiment.py --output /tmp/retry-contract-replay
```

Use a new output directory. The protocol is written before evaluation, and socket
connections are blocked during execution. There were zero attempted network
connections and zero paid requests. The fixture sets local solver outputs directly;
it makes no mock or real provider generations. Original native paths are retained
in logs/reports; the manifest and summary map their relative archive locations.

## Task and intervention

The task has invoices `a`, `b` and `c`, with amounts 100, 200 and 300. The solver
uses idempotent SQLite inserts. A separate grader queries the persisted amount
and compares it with a policy file. ExactMatch checks the `posted` acknowledgment.

The initial policy requires the original amounts. The initial run completes `a`
and `b`, then encounters an injected exception after committing `c` but before
returning its answer. This is an actual solver exception and native error log;
it is not a new SIGKILL experiment. The earlier
[recovery protocol](run_recovery.py) separately tests process death.

The intervention changes only the required amount for `a`, from 100 to 999.
The fixed solver still writes 100. Dataset case IDs and the text acknowledgment
remain unchanged. Policy content hashes and grader configuration therefore
matter even when native sample IDs are stable.

| Path | Additional target calls | Native result | Multivon acceptance |
|---|---:|---|---|
| Changed policy, preflight guard | 0 | Reconstruction rejected | No new evaluation |
| Original policy restored, guarded retry | 1 (`c`) | 3/3 passing; `a` and `b` UUIDs preserved | `final_attempt`: accept; all attempts: indeterminate |
| Changed policy, guard deliberately omitted | 1 (`c`) | 3/3 passing; two old scores preserved | indeterminate: mixed contracts/graders |
| Fresh evaluation with changed policy | 3 | 2/3 passing | reject: required persisted amount for `a` is wrong |

There were eight target invocations total: the initial three plus 0, 1, 1 and 3
in the paths above. All paths use the same ledger and idempotent insert rule;
the final database contains exactly the original three rows and amounts.
The fresh evaluation is new grading/execution, not a fresh database. That state
is intentional: the changed rule requires a different persisted outcome.

Native sample preservation is an execution policy, not a claim that a modified
grading contract is comparable. [Inspect's retry documentation](https://inspect.aisi.org.uk/eval-logs.html#eval-retries)
describes reusing completed samples. This study addresses the additional
compatibility requirements of a release decision, rather than treating native
preservation itself as a bug.

## What the integration verifies

`bind_inspect_task` binds native Task/Sample metadata to static sample identities,
native input/reference hashes, recorded task settings, prepared grader
fingerprints, engine inventory, and caller-declared dependency/file revisions.
Reconstruction can compare against the prior native log before solver execution.
The unchanged retry preserves native UUIDs and reruns only the interrupted case.

Scorers retain configuration before/after grading. Import detects a sample bound
to a different declared task, altered native inputs or references, observed grader
drift, and incompatible native plans/settings across supplied retry logs. The
unguarded negative control is blocked even when imported without its earlier log,
because its retained sample bindings disagree with the new root declaration.
General error budgets and saved-output regrading do not clear those failures.

Validation reread all four native logs, verified preserved sample UUIDs in both
retry paths, queried SQLite, reloaded all three reports and reproduced their
accept/indeterminate/reject decisions. Policy byte files were reconstructed from
the frozen protocol values and verified against the original native declaration
hashes. The archived task source matches its declared hash. Every archive member
was hashed again after packaging.

The final Python 3.12 regression suite passed **2,062 tests**, with 18 skips and
7 warnings. A broader Python 3.10 subset passed 144 tests; after adding malformed
header handling, all 43 focused native contract/limit tests passed on 3.10.
All 71 MDX pages parsed, and the integration guide's Python examples execute as
part of the regression suite.

## Boundaries and critique

The contract is a caller assertion about hidden solver arguments, services,
clients, code and environment revisions. Named files are rehashed; hashes are
not signatures or proof of authentic execution. Installed package metadata is
not a verified build. Unrelated environment changes can conservatively invalidate
compatibility. Changes restored between observations are not detected.

Task construction and grader preparation happen before binding and can already
have side effects. Native execution overrides are not all known to the factory;
declare them consistently and retain the complete native log chain. Legacy
chains without declarations are diagnostic, not retroactively verified.

This verifies declared static-task retry compatibility. It does not establish
safe restoration of arbitrary mid-agent memory, sandbox processes, mutable remote
models or external side effects. It adds no checkpoint store, dataset system or
replacement scheduler. The result supports a concrete reliability feature; three
authored cases establish neither a scientific novelty claim nor a durable moat.
