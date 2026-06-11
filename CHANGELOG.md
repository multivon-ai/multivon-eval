# Changelog

All notable changes to `multivon-eval`. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as of 0.7.0.

## [0.11.1] — 2026-06-11

Robustness hardening from an adversarial audit that ran the staleness /
provenance / scanner / bootstrap surface against malformed inputs, symlink
tricks, unicode edge cases, and concurrent writers. The theme: every failure
the audit found was a place where the tool either crashed with a raw
traceback or — worse — reported something *false*. Both violate the same
contract: honest UNKNOWN over confident wrong.

### Fixed

- **A syntax-broken file no longer reads as REMOVED.** The scanner silently
  returned zero records for files it couldn't parse (syntax errors, non-UTF8
  encodings), so staleness reported every baselined site in them as REMOVED —
  and `--fail-on removed` failed CI with a misleading verdict. Unscannable
  files now surface as a distinct UNSCANNABLE tier ("file exists but could
  not be parsed — verdict unknown, NOT removed"), a warning line names each
  file with its reason in all three renderers, JSON gains `skipped_files`,
  and `--fail-on removed` no longer trips. Skipped files are a report-time
  concept — never written into baselines.
- **Symlinks resolving outside the repo root are skipped, not recorded** —
  previously they wrote machine-specific absolute paths into the baseline,
  producing false REMOVED+ADDED churn on every other checkout.
- **Fingerprints are NFC-normalized** (`SCANNER_VERSION` 3 → 4) — composed
  vs decomposed unicode ("é" as one codepoint vs e+combining-accent) is an
  editor/OS artifact, not a prompt change; it previously fingerprinted as
  drift. Old baselines print the standing "rescan recommended" warning.
- **`match`-statement capture patterns disqualify module constants** —
  `case PROMPT:` rebinds via a str field the scanner didn't see, letting a
  rebound constant read as static (a false "static" poisons every verdict).
- **Clean errors instead of tracebacks**: `staleness stamp` on malformed
  JSONL (file:line in the message), `staleness baseline` on a nonexistent
  path or missing `--out` dir, `bootstrap` on a malformed traces file, and
  `--site …#xx` with a non-integer position — all exit 2 with actionable
  messages. `multivon-eval … | head` no longer dumps a BrokenPipeError.
- **`attribution scan /typo/path` exits 2** instead of a green "No SDK
  prompt call sites found" — a typo'd CI path looked permanently passing.
- **The documented 10K trace cap is now enforced** with a loud truncation
  warning, and a malformed *final* trace line (the normal shape of an
  interrupted streamed dump) skips with a warning while malformed interior
  lines stay a hard error.
- **Bootstrap artifacts are emitted atomically** (temp dir + rename) — a
  Ctrl-C mid-emission can no longer leave a half-written `eval_suite.py`
  that looks complete. `schema_version: true` no longer passes the int
  check (bool ⊂ int).

34 new tests across the touched surface; 1038 green.

## [0.11.0] — 2026-06-11

**The answer to the 20.9% ceiling.** The determinacy gate (0.10.1) measured scanner v3 against five real repos: 20.9% of call sites are statically resolvable — the rest build prompts dynamically and are statically unbridgeable *by construction*. The runtime prompt recorder (designed in [#9](https://github.com/multivon-ai/multivon-eval/issues/9), promoted from the 0.10.0 deferred list by the gate result) is the honest path past it: during an eval run, an opt-in interceptor records the **rendered** prompt text per call site, fingerprinted with the same `fingerprint_text` the static scanner uses. A `**kwargs` unpack the scanner can only report as UNKNOWN is, at call time, real kwargs with real text.

The honesty discipline survives the new power — **three labeled trust tiers, never collapsed**:

1. **static** — the scan proves the prompt *text*;
2. **runtime** — recordings prove only the renderings *observed*, not all renderings (variable renderings per site are a fingerprint SET, and every verdict speaks in "current recordings matched **k of N** previously observed renderings" — a site is never called fresh because one rendering matched);
3. **templates / external prompts** — deferred, unverifiable.

### Added

- **`multivon_eval.recorder`** — opt-in runtime prompt recorder. `record_prompts()` context manager (non-pytest) or `pytest --record-prompts` (plus `--record-prompts-out`, `--record-text`). Method-level wrapping of exactly the three SDK surfaces the static scanner knows — anthropic `Messages.create`, openai `chat.completions.create`, `litellm.completion`/`acompletion` — save original, wrap, restore byte-identical on exit (inherited attributes restored by `delattr`, so `__dict__`s end exactly as found). Zero overhead when off: importing multivon_eval performs NO patching, pinned by a fresh-interpreter subprocess test. Recordings stay local in `prompt_recordings.jsonl`; **fingerprints only by default**, rendered text only behind explicit `--record-text`. Append-safe storage: duplicate (anchor, role, fingerprint) keys merge counts/case_uids on write, atomic rewrite.
- **Case binding by observation** — a contextvar carries the active `case_uid`; `EvalSuite` binds it per case from `_provenance.case_uid` (one None-check when recording is off) and the pytest plugin binds the test nodeid per test. Recordings carry the case_uids observed per (anchor, role, fingerprint) — the run *knows* which sites fired for which case.
- **`multivon-eval staleness baseline --merge-recordings [FILE]`** — merges recordings into `prompt_baseline.json` as `source:"runtime"` records with **fingerprint SETS**, stored under a separate `runtime_records` key. Merge-only: never rescans, NEVER touches static records; a static rescan never discards the runtime tier; re-merging the same recordings file is idempotent.
- **OBSERVED report tier** — runtime-sourced sites render distinctly in text/json/markdown: compared recordings-vs-recordings (runtime-only sites *cannot* be compared against a static scan, and the report says so), always in the k/N language. The determinacy headline gains a third clause: "K sites observed at runtime." The standing footer now states all three trust tiers verbatim, next to the blind-spots list.
- **`multivon-eval staleness stamp --from-recordings [FILE]`** — prints observed case→site bindings as **proposals** (case_uid → anchor + fingerprint with observation counts); writes only with explicit `--apply --cases F.jsonl` (targets land as `source:"runtime"`, `bound:"observed"`). Observation removes the fabrication objection that blocked auto-binding in the 0.10.0 adversarial review — the human confirmation stays. Runtime-bound targets are verified against recordings, never against the static scan (reported `unverifiable [runtime]` there, by rule), and never enter the static coverage denominator.

### Fixed

- **`add_check` QAG question generation no longer invents stricter sub-requirements.** "Response should mention the return policy" generated questions about return *procedures* and *eligibility* the criterion never asked for, scoring a plainly-correct answer 0.33 FAIL (reproduced 3/3 trials). The generation prompt now requires every question be answerable "Yes" by any response satisfying the criterion as stated; the same answer now scores 1.0 (and the evasive control still fails). Found by a fresh-user E2E audit on the quickstart's own example.
- **Keyless demo picks an Ollama model you actually have** — `python -m multivon_eval` used hardcoded `llama3` and reported "detected but unreachable" when the *server* was fine and the *model* wasn't pulled. It now asks `/api/tags` for an installed model (text models preferred), honors `DEMO_MODEL`, and the failure message distinguishes "model not available" from "server unreachable".
- **Bootstrap creates the output dir before any paid LLM call** — a typo'd or read-only `--output` previously failed *after* ~$0.12 of judge spend. Progress lines now print to stderr as each LLM stage starts (the ~4-minute wait was previously silent; docs no longer claim "under 60 seconds").
- **Staleness `what changed:` hint works on a dirty tree** — `git diff <sha>..HEAD -- file` printed nothing for uncommitted edits (the most common moment to run staleness); the hint now uses `git diff <sha> -- file`.
- Markdown staleness reports no longer end with a `_exit N_` debug line (the exit code stays in the text renderer and JSON payload); `install-skills --dry-run` says "would install … (dry-run — nothing was written)" instead of "installed".

### Notes

- Capture scope v1 (honest): string `system=` kwargs and string `content` entries in `messages=` lists. Content-block lists (vision, tool results) and calls anchored outside the repo root are skipped, not guessed at. The caller anchor comes from a stack walk to the first repo-relative frame; `line` is an advisory hint, never a matching input.
- 30 new tests (`tests/test_recorder.py`): patch-and-restore byte-identity, zero-overhead-off, fingerprint parity across all three SDKs (stubbed, no network), `**kwargs` rendered-text capture, contextvar case binding (including through `EvalSuite.run`), idempotent JSONL merge, static-records-untouched baseline merge, k/N OBSERVED rendering, and propose-only stamping. 168 green across the touched surface; zero new failures elsewhere.

## [0.10.1] — 2026-06-11

Scanner v3 — the determinacy gate (spec test-plan #14) run against five real repos (aider, gpt-researcher, open-interpreter, letta, pr-agent) found that **4 of 5 reported zero call sites**: not because they have no prompts, but because the scanner was silently blind to how real code calls LLMs. v3 fixes detection and honestly reports what it still cannot read.

### Fixed

- **Aliased litellm imports detected** — `from litellm import acompletion` then bare `acompletion(...)` (pr-agent's shape) now matches. Star imports and function-local imports stay out of scope.
- **`**kwargs`-unpacked calls surface as UNKNOWN** — `litellm.completion(**kwargs)` (aider's shape) now emits an honest `<dynamic:KwargsUnpack>` record instead of vanishing.
- **`messages=<variable>` surfaces as UNKNOWN** — the most common real-world shape (messages list built elsewhere) now emits one dynamic record per call site instead of nothing. A literal empty `messages=[]` correctly emits nothing (statically known empty).
- `SCANNER_VERSION` bumped to 3; baselines written by v2 print a "rescan recommended" warning instead of fake drift.

### Measured (the determinacy gate, public on the epic)

Honest detection changed the denominator: 73 → 278 sites across the five repos, and static resolvability is **20.9%** — below the 50% gate. Conclusion recorded publicly: real-world prompt traffic is mostly dynamic construction; static analysis tracks call-site add/remove for all of it but can verify text drift only for prompts-as-constants codebases (letta-style: 58 static sites). The runtime recorder (epic) is now the priority path for the rest. The staleness report's determinacy headline makes this exact ratio visible per-repo — by design.

## [0.10.0] — 2026-06-11

**Evals drift as code changes — this release ships the detection layer.** Prompts evolve, eval suites go stale, and nobody notices until a regression sails through. 0.10.0 adds prompt-drift staleness detection: a committed baseline snapshot of every prompt call site in your repo, a read-only report that tells you exactly which prompts changed since your cases were authored, and an opt-in provenance layer binding cases to the prompts they exercise. The design went through a 3-design × 2-adversarial-critic review before a line was written; the design rule that survived every round: **the tool never overclaims what static analysis can know.** Every report opens with a determinacy headline ("N of M call sites statically resolvable") and closes with a standing blind-spots footer.

### Added

- **`multivon-eval staleness [PATH]`** — read-only drift report. Diffs a live `attribution` scan against the committed `prompt_baseline.json`: **CHANGED** (prompt text differs — with before/after fingerprints, bound cases, and a `git diff` pointer), **REMOVED** (always with the three-way caveat: feature removed / renamed+edited / moved beyond static reach), **ADDED** (new prompts with no covering cases), **UNKNOWN** (dynamic prompts — never guessed at). `--format text|json|markdown`, `--fail-on changed,removed,added` for CI (exit 0 report-only by default; markdown format drops straight into `$GITHUB_STEP_SUMMARY`).
- **`multivon-eval staleness baseline [PATH]`** — writes/refreshes the baseline snapshot, printing the diff before writing. Bootstrap writes one automatically.
- **`multivon-eval staleness stamp`** — binds hand-written JSONL cases to the prompt call sites they exercise (`--site 'file.py::qualname.role'`). Raw-line-preserving rewrite (never round-trips through `load_jsonl`, which would drop `expected_tool_calls`); idempotent restamps are byte-identical; refuses ambiguous sites instead of guessing.
- **`multivon_eval.provenance`** — `metadata["_provenance"]` schema (case_uid, authored_at/stamped_at, git context, prompt-fingerprint targets) + a `stamp()` helper for Python-inline cases. Stamping **never perturbs `suite.lock`** (cases_hash excludes metadata by design) — pinned by a regression test.
- **Attribution scanner v2** — one-hop module-level constant resolution (`SYSTEM_PROMPT = "..."` then `system=SYSTEM_PROMPT` now resolves to real text instead of a dynamic placeholder; conditional/cross-module names honestly stay dynamic) + `loose_fingerprint` (whitespace-collapsed) so formatting-only prompt changes are labeled as such — flagged, never suppressed.
- **Bootstrap integration** — `--repo` flag; generated cases are stamped `authored_by="bootstrap"` with the repo SHA (honest "authored against this state" provenance — bindings are never fabricated), and `prompt_baseline.json` is written alongside the suite.

### Notes

- Matching is content-first: line numbers and git SHAs are display-only, never matching inputs — a whitespace refactor or rebase produces zero false staleness.
- Dynamic prompts gate FIRST: a prompt the scanner can't statically read is UNKNOWN forever rather than fake-fresh. The runtime recorder that closes this gap is tracked as future work.
- 51 new tests (staleness 27, provenance 24) + 26 extended attribution tests. 178 green across the touched surface; zero new failures elsewhere.

## [0.9.8] — 2026-06-03

Post-iter-3 documentation + DX polish pass. The headline ship is a one-command installer for the bundled Claude Code skills so the wheel doesn't just contain them, it wires them up. Plus a README rewrite that leads with the credibility story (κ=0.03, F1 0.830 held-out, the 0.9.4 → 0.9.7 self-correction sequence) instead of bootstrap CLI feature copy. The release sequence IS the audit trail — the README opening now says so explicitly.

### Added

- **`multivon-eval install-skills` CLI subcommand** — symlinks the three bundled Claude Code skills (eval-bootstrap, eval-audit, eval-explain) from the installed wheel into `~/.claude/skills/`. Defaults to symlinks (so `pip install -U multivon-eval` propagates SKILL.md edits without re-running install); falls back to `shutil.copytree` on Windows or where directory symlinks are refused. Supports `--dry-run` to preview and `--force` to replace existing entries. One command, no `ln -sf` shell incantation.

### Changed

- **README rewrite — credibility story leads.** The opening now leads with the κ=0.03 three-framework disagreement finding, the F1 0.830 [0.70–0.92] cross-distribution held-out result, and the 0.9.4 → 0.9.5 → 0.9.6 → 0.9.7 self-correction sequence as the credibility narrative. Bootstrap is now a feature paragraph, not the hook. All existing content preserved — only the order changed.
- **Unified the DeepEval F1 comparison number.** Several paragraphs cited slightly different values (0.79, 0.804, 0.787) across the README. All now consistently report **F1 0.804 [0.71–0.88]** vs DeepEval **F1 0.586 [0.48–0.68]** — the value in `benchmarks/results/hallucination.json` and `benchmarks/README.md`.
- **"As of May 2026" → "as of June 2026"** in the comparison-table footnote.
- **SKILL.md frontmatter cleanup** for all three bundled skills. Removed non-spec keys (`trigger_phrases`, `provides`, `requires`) that aren't part of the Anthropic skill schema; folded that info into the `description` body so it's still discoverable. Added the correct `allowed-tools` per skill: eval-bootstrap = Bash/Read/Edit/Write/Glob; eval-audit = Bash/Read/Grep/Edit; eval-explain = Read/Grep/WebFetch.
- **eval-bootstrap local-judge fallback** no longer hard-codes `qwen2.5:14b`. The skill now instructs the agent to run `ollama list` to detect what is pulled and pick the strongest instruction-tuned model available; common picks documented (`qwen2.5:72b`, `llama3.3:70b-instruct`, `deepseek-r1:32b`).

### Fixed

- **`benchmarks/README.md` "Planned benchmarks"** — Benchmark 5 (SummEval) is written; marked the planned-benchmark item complete and removed the stale "run to fill TBD above" leftover.
- **Broken Marketplace URL** in `multivon_eval/_skills/eval-bootstrap/SKILL.md` replaced with a pointer to the [multivon-ai/eval-action GitHub repo](https://github.com/multivon-ai/eval-action) until the Marketplace listing is published.
- **Dead Anthropic skills doc link** in `multivon_eval/_skills/README.md` (was `docs.anthropic.com/claude/skills`, 404) updated to `docs.claude.com/en/docs/agents-and-tools/agent-skills`.

---

## [0.9.7] — 2026-06-03

Iter-3 confirmation hotfix. The ML researcher persona caught a more subtle inconsistency in the 0.9.5 held-out test: the run was reporting threshold 0.7 in its print output (the init-time default of `Hallucination()`) but never asserted that the calibrated threshold 0.55 was being applied. Without an explicit `JudgeConfig` argument, `Hallucination()` falls back to the default threshold instead of looking up the calibrated value for Haiku in `_calibration_data/v2.json`. The result is a different F1 on the same data — 0.852 at threshold 0.7 vs 0.830 at the actually-calibrated threshold 0.55. Only the 0.830 figure is defensible as "held-out at the calibrated threshold."

### Fixed

- **`benchmarks/run_truly_held_out.py` now passes explicit `JudgeConfig(provider='anthropic', model='claude-haiku-4-5-20251001')`** to `Hallucination(...)` so the v2-calibrated threshold (0.55) is applied. Reports the post-resolve threshold (the value actually used at runtime), not the init-time default.
- **`benchmarks/results/hallucination_held_out.json` updated** with the corrected run: F1 0.830 [0.70–0.92] (was 0.852 [0.73–0.94] at the wrong threshold). Raw counts: TP=22, FP=1, FN=8, TN=29.
- **`benchmarks/README.md` Benchmark 4** rewritten to cite the correct threshold (0.55) and the corrected F1. Added an explicit reproducibility note explaining the default-vs-calibrated threshold gotcha so future contributors don't repeat it.
- **multivon.ai/eval tile** updated to show F1 0.83 [0.70–0.92] with the corrected sublabel.

### Discipline note

The framework's whole pitch is "what we do is what we say." Catching this gotcha in iter-3 review and publishing the correction is the same discipline 0.9.5 demonstrated on the bigger contamination flag. Three same-day releases (0.9.5 fixing the held-out framing, 0.9.6 fixing the bootstrap template runtime, 0.9.7 fixing the threshold-vs-default mismatch) is the cost of shipping during a public launch review — and the cost is the right kind of public if the fixes land within hours and the historical record stays intact.

---

## [0.9.6] — 2026-06-03

Round-2 review hotfix: r/Python persona caught three runtime blockers in the bootstrap-generated `eval_suite.py` that v0.9.4 shipped. Anyone who ran `multivon-eval bootstrap` then `python eval_suite.py` would have hit a TypeError on the third line.

### Fixed

- **Generated `eval_suite.py` used non-existent `suite.run(cases=...)` kwarg.** `EvalSuite.run` takes `(model_fn, runs=..., ...)` — cases go through `suite.add_cases(...)` before the run call. Template now calls `suite.add_cases(cases)` then `suite.run(stub_model, runs=args.runs)`.
- **Generated `eval_suite.py` called `report.print_summary()` which doesn't exist.** Replaced with inline printing of `report.pass_rate`, `report.passed`, `report.total`, `report.failed`, `report.errors` — all real EvalReport public methods.
- **`stub_model` signature was `(case: EvalCase)` but `EvalSuite.run` expects `Callable[[str], str]`.** Fixed to `stub_model(prompt: str) -> str`.
- **`multivon_eval/discover.py:_call_judge` rejected local providers.** Same pattern as the `auto.py:_call_judge_raw` fix in 0.9.4 — `ollama` and `litellm` now route through `make_judge_call`. The bootstrap pipeline now genuinely runs end-to-end on a local judge, not just at the CLI argparse level.

### Tested

End-to-end smoke test: generated a real `eval_suite.py` template, dropped a 2-case `seed_cases.jsonl` next to it, ran `python eval_suite.py --runs 1`. Suite executes, summary prints, exit code reflects pass rate. The framework's own "judge availability" warning fires correctly when no API key is loaded — exactly the user-facing signal you want when the env is misconfigured.

---

## [0.9.5] — 2026-06-03

Same-day correction for 0.9.4. The round-2 peer review (ML researcher persona) caught that the "held-out HaluEval-Sum F1 0.783" claim in 0.9.4 was actually in-distribution: the Faithfulness evaluator's Haiku threshold is itself calibrated on HaluEval-Sum, so testing Faithfulness on HaluEval-Sum reproduces the calibration F1 by construction. 0.9.4 patched the most visible dunk from round 1 (the HaluEval-QA contamination) and accidentally repackaged the same contamination on HaluEval-Sum. We caught it within hours.

### Added

- **`benchmarks/run_truly_held_out.py`** — the actually-held-out reproducer. Runs the **Hallucination** evaluator (which IS calibrated on HaluEval-QA, threshold 0.7, never seen summarization data) against HaluEval-**Sum**. Result: F1 = 0.852 [0.73–0.94] on n=60. Different task family, different evaluator-of-record for this dataset, calibration set ↮ test set. This is the cross-distribution generalization figure the previous release was trying to make.
- **`benchmarks/results/hallucination_held_out.json`** — raw numbers for the corrected held-out test: TP=23, FP=1, FN=7, TN=29. Wilson CIs on precision/recall + bootstrap CI on F1 included.

### Changed

- **`benchmarks/README.md` Benchmark 3 carries a correction note at the top.** The framing changed from "held-out: threshold calibrated on HaluEval-QA" (wrong) to "in-distribution — corrected" (right). The data didn't change; the label did. Both numbers stay in the README; only the framing changes. The summary table's footnote ² now points at the new Benchmark 4 (genuinely held-out) instead of falsely labeling Benchmark 3 as held-out.
- **Summary table's "cross-distribution" row** is now F1 0.852 [0.73–0.94] (Hallucination evaluator on HaluEval-Sum), not F1 0.783 (which was the in-distribution Faithfulness number relabeled).

### Notes

The original v0.9.4 release is left on PyPI. The historical record matters: we shipped a wrong claim and corrected it within a few hours. Yanking the prior release would erase that. Anyone who pulled 0.9.4 and re-reads can do `pip install --upgrade multivon-eval` to get the corrected framing.

---

## [0.9.4] — 2026-06-03

Launch-prep release driven by the 7-persona launch simulation (HN top-commenter, HN early adopter, r/LocalLLaMA, r/MachineLearning, r/MLOps, r/Python, CTO procurement). Closes the cross-persona convergent findings: load-traces silent skips, half-done bootstrap output, cloud-only judge in the bootstrap CLI, in-distribution-only headline F1, and missing CIs on shipped numbers.

### Added

- **`benchmarks/_add_cis.py`** — walks `benchmarks/results/*.json` and writes Wilson CIs on precision/recall + bootstrap CIs (1000 resamples, stable seed 20260603) on F1. Idempotent. Adds `precision_ci_lo/hi`, `recall_ci_lo/hi`, `f1_ci_lo/hi` fields on every metrics block that has TP/FP/FN/TN counts. Closes the "framework preaches CIs but doesn't ship them on its own published numbers" eat-own-dogfood violation.
- **Held-out faithfulness benchmark on HaluEval-Sum** — re-ran `benchmarks/run_faithfulness_benchmark.py` with the v2-calibrated threshold *frozen* (no re-tuning on the held-out split). New result: **F1 0.783 [0.68–0.88]** on n=60. Discharges the "you tuned thresholds on the same set you tested on" criticism for cross-task generalization. See `benchmarks/README.md` Benchmark 3 for the full update.
- **`multivon_eval.discover.load_traces` accepts field aliases.** LangSmith (`query`/`answer`/`retrieved_context`), LangFuse (`prompt`/`completion`), Phoenix (`input`/`output`) all auto-rename to the canonical shape. Loud one-line summary to stderr: "loaded N/M traces · renamed K input/output/context fields · skipped X rows with no input/query/prompt field." Silent skip behavior removed — the previous failure mode was real users thinking their dump was empty when 198/200 rows had used `query` instead of `input`. Pass `verbose=False` to suppress.
- **Bootstrap-generated `eval_suite.py` is now runnable end-to-end.** Previously emitted a literal `TODO: add your cases here` comment and stopped. Now scaffolds an `argparse` CLI, a `load_cases()` helper that reads `seed_cases.jsonl` by default (or `--cases path/to/real.jsonl`), and a `stub_model()` placeholder with an obvious replace-me prompt. `python eval_suite.py --runs 1` runs cleanly after bootstrap — only thing the user has to swap is the stub model function.
- **`multivon-eval bootstrap --judge-provider` accepts `ollama` and `litellm`.** The SDK judge layer always supported local providers (judge.py respects `OLLAMA_HOST`, injects dummy keys for OpenAI-shim servers), but the bootstrap CLI argparse hard-restricted to cloud. Now `--judge-provider ollama --judge-model qwen2.5:14b` works end-to-end. Added `--judge-base-url` for vLLM / LM Studio / custom Ollama endpoints. `auto.py::_call_judge_raw` (the adversarial seed generator) routes local providers through `make_judge_call` instead of its bespoke switch, so it stops hard-failing on non-cloud providers.
- **`skills/` directory** — three Claude Code skills (`eval-bootstrap`, `eval-audit`, `eval-explain`) shipping with the framework. SKILL.md files following the Anthropic skill model. Install via symlink into `~/.claude/skills/`. See `skills/README.md`.

### Changed

- **`benchmarks/README.md`** — headline Hallucination F1 (0.804) now labeled in-distribution with a footnote citing the calibration dataset hash. Benchmark 3 (Faithfulness) updated with the held-out HaluEval-Sum result and a "what this discharges vs doesn't" explanation. All F1 numbers in the summary table now carry their 95% bootstrap CI. Limitations section opens with the in-distribution caveat and the post-hoc threshold-sweep caveat — the framework's own page now applies the standard the framework preaches.

---

## [Unreleased]

(reserved for in-flight work — empty)

---

### Phase 1 attribution (carried forward from 0.9.4)

- `multivon_eval.attribution` — public API: `scan(repo_root)`, `diff_records(base, head)`, `render_markdown(diffs)`. Descriptive only; causal attribution intentionally not shipped (see the Phase 2 sidecar design doc).
- `multivon-eval attribution scan <repo>` — text or JSON output.
- `multivon-eval attribution diff <base> <head>` — markdown / text / JSON output.

## [0.9.3] — 2026-05-28

Patch release: cross-platform fix for lock-file verification.

### Fixed

- **`EvalSuite.verify_lock(<json string>)` no longer crashes on Linux.** The method probed its argument with `Path(s).exists()` to tell a file path from an inline JSON payload, but a lock JSON string is longer than the OS filename limit, so on Linux that raised `OSError: [Errno 36] File name too long` (macOS silently returned `False`). The filesystem probe is now guarded and falls back to parsing the string as JSON. Affects anyone passing a lock payload as a string — including the `eval-action` `lockfile:` input running on Linux runners.

## [0.9.2] — 2026-05-27

Patch release hardening the zero-setup paths. `python -m multivon_eval` and the local-judge flow (Ollama / LM Studio / vLLM) now work out of the box, even on a machine where a local server is running but no model is pulled.

### Fixed

- **Local OpenAI-compatible judges no longer fail with "Missing credentials."** When `JudgeConfig.base_url` is set (Ollama on `:11434`, LM Studio on `:1234`, vLLM, any OpenAI-compatible endpoint) and `OPENAI_API_KEY` is unset, the OpenAI SDK refused to even construct the client. The judge call now supplies a placeholder key for local endpoints, so the documented local-judge path works without a cloud key. Applies to both the sync and async judge paths.
- **`python -m multivon_eval` never exits with a traceback.** Previously, a local server listening on `:11434` with no model pulled was auto-detected, an LLM-judge evaluator was added, and the resulting judge error propagated out of `suite.run()` as an uncaught `JudgeUnavailable` (exit 1 + stack trace) — contradicting the "30 seconds, no API key" promise. The demo now liveness-probes the detected judge before enabling it (and prints an honest header when it's unreachable), with a safety net around `suite.run()` that falls back to the deterministic tier if a judge dies mid-run.

### Added

- **`multivon-eval --version`** prints the installed version, for parity with the rest of the CLI surface.

### Docs

- Package metadata `Documentation` URL fixed (`evaldocs.multivon.ai` → `docs.multivon.ai`; the old host did not resolve).
- Contributing clone instructions corrected (`cd llm-evals` → `cd multivon-eval`).

## [0.9.1] — 2026-05-24

Patch release driven by the pdfhell mini-v4 eval-pipeline post-mortem (see `multivon-ai/pdfhell/pdfhell/research/CORRECTION_NOTICE.md`). Same Anthropic API change that silently broke every Opus 4-7 call in the pdfhell leaderboard would have broken any multivon-eval consumer using Opus 4-7 as a judge — this release closes that gap upstream.

### Fixed

- **`AnthropicAdapter` now omits `temperature` for the reasoning tier.** Anthropic's `claude-opus-4-7` and the `claude-opus-5+` family reject the parameter with a 400. The adapter detects those models by name prefix and drops the field; older models (`claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-3-5-*`, etc.) still receive `temperature` unchanged. Same fix applied to `multivon_eval.discover._call_judge`. New helper `AnthropicAdapter._supports_temperature()` exposes the decision for subclassers.

### Added

- **`multivon_eval.vision`** — vision-call dispatch wrapped behind a single `call_vision(prompt, sources, judge, max_tokens)` function. Previously lived in `pdfhell.vision`; promoted here so any multivon-eval consumer (pdfhell, image-graded benchmarks, multimodal RAG audits) can grade documents/images without re-implementing per-provider content-block conversions. Providers: `anthropic`, `openai`, `google`, `ollama`. PDFs are rasterised via `pypdfium2` for ollama (which expects images, not PDFs).

- **`ollama:` as a first-class `JudgeConfig` provider.** Previously you had to pass `provider="litellm"` and `model="ollama/llama3.2"`. Now `JudgeConfig(provider="ollama", model="llama3.2")` resolves to the same litellm-backed call internally. Matches the colon convention used by the rest of the SDK (`anthropic:`, `openai:`, `google:`). Both sync and async judge paths support it. `OLLAMA_HOST` env var sets the base URL (default `http://127.0.0.1:11434`).

### Compatibility

- No breaking changes. `AnthropicAdapter`'s constructor signature is unchanged — `temperature` is still accepted, just silently dropped for the reasoning tier. Existing pinned dependents (incl. `pdfhell>=0.5.0`) work without modification.
- Tests: 19 new unit tests covering the temperature-fix matrix. Pre-existing test suite (`test_beginner_friendly.py`) has two pre-existing unrelated failures from a model-name issue independent of this release.

## [0.9.0] — 2026-05-23

Field-report release. A fresh-user dogfood pass + a 5-app SDK deepdive surfaced a cluster of UX gaps and one parallel-execution regression. Most of the work is making the SDK behave the way its docs already promised.

### Added

- **`multivon-eval doctor`** — new pre-flight CLI that verifies Python version, API keys (Anthropic / OpenAI / Google), live provider pings, optional deps (presidio, opentelemetry, datasets), and `~/.multivon` writability. Rich-rendered by default; `--json` for CI consumption. Exits 0 / 1 / 2 for ok / error / warn so CI gates can branch on the outcome.
- **`multivon-eval bootstrap --validate`** — optional N-shot judge-noise filter on the generated seed cases via the existing `validate_adversarial_cases` primitive. Drops cases outside the (0.5, 1.0) hardness band against a stub-refusal baseline. Adds a `hardness_report.jsonl` alongside the filtered `seed_cases.jsonl` so a reviewer can audit what was thrown out.
- **`Evaluator._skipped(reason)` helper** — base-class method that returns `EvalResult(score=1.0, passed=True, reason="[skipped] ...", metadata={"skipped": True})`. Used by every evaluator that previously returned `score=0.0` when the case shape didn't fit (see Fixed below).
- **Skip propagation across the evaluator catalog.** When the case shape doesn't fit an evaluator — no `context` for a RAG metric, no `expected_output` for an exact-match metric, no `agent_trace` for a tool metric — the evaluator now returns a skipped pass instead of a 0.0-score failure. Applies to `Faithfulness`, `Hallucination`, `ContextPrecision`, `AnswerAccuracy`, summarization checks, `ToolCallAccuracy`, `ToolCallNecessity`, `ToolArgumentAccuracy`, `TrajectoryEfficiency`, `PlanQuality`, `StepFaithfulness`, `AgentMemoryEval`, `ExactMatch` + 3 other deterministic, 2 text-metric evaluators, and all 4 conversation evaluators.
- **Refusal detection on Faithfulness + Hallucination.** Short responses (<240 chars) starting with one of the known refusal prefixes ("I don't know", "I cannot", "Sorry", etc.) now skip these metrics — a correct refusal doesn't make substantive claims, so faithfulness is N/A.
- **`ToolCallAccuracy` three-shape semantics.** `expected_tool_calls=None` skips. `expected_tool_calls=[]` + no calls = PASS ("Correctly called no tools"); `expected_tool_calls=[]` + any tool = FAIL. `expected_tool_calls=[...]` requires `agent_trace` to evaluate (skips if absent).
- **Parallel execution by default.** `EvalSuite.run(workers=...)` now auto-picks `min(8, len(cases))` when no tracer is supplied (was 1). A 10-case × 6-evaluator RAG suite drops from ~167s serial to ~17s with workers=8.
- **`PIIEvaluator` rewritten for full standards coverage** with checksum validation and per-pattern citations to source standards:
  - HIPAA Safe Harbor (45 CFR § 164.514(b)(2)) — all 18 identifier categories where regex is feasible (13/18). MRN widened to 4–15 digits. PERSON_NAME via high-precision context-led pattern catches "Patient John Smith" / "Mr. Doe" / "Dr. Wilson". age_over_89 detection. Stricter admission/discharge/death-date patterns.
  - GDPR (Reg EU 2016/679, Art.4(1)) — UK NI Number, NHS Number (Mod-11 validated), Spain DNI/NIE, Italy Codice Fiscale, France NIR, Germany Steuer-IdNr, Netherlands BSN, Poland PESEL, Sweden Personnummer, Denmark CPR, Ireland PPSN, Finland HETU, IBAN (Mod-97 validated per ISO 13616).
  - DPDP India (Act 22 of 2023) — Aadhaar with Verhoeff checksum, PAN with structural validation, GSTIN, IFSC, Voter ID (EPIC), +91 mobile, Indian Driving License, Indian Passport, Vehicle Registration, Ration Card.
  - CCPA (Cal. Civ. Code § 1798.140(o)) — context-anchored bank account, California Driver's License.
  - New `strict=True` (default) runs Luhn / Verhoeff / Mod-97 / Mod-11 / structural validators to drop false positives on transaction IDs and order numbers.
  - New `use_ner=True` lazy-imports `presidio_analyzer` for partial coverage of HIPAA categories regex can't reach (unprefixed names, free-form addresses). Silent no-op when Presidio isn't installed.
- **Bootstrap discovery report deterministic from final evaluator list.** "Why this mix" prose is now generated from the committed `EvaluatorRecommendation[]` list, not a separate LLM prose pass. Eliminates the previous drift where the report said "we skip Hallucination" while the suite included it, or "add PIIEvaluator" while the suite omitted it. The LLM proposer's notes are moved to a clearly-labeled "Proposer notes (advisory)" footer. When traces contain PII but the suite omits `PIIEvaluator`, the report surfaces a `⚠ PII gap` callout with the exact `suite.add_evaluators(PIIEvaluator(...))` snippet to add.
- **Calibration N-warning.** Bootstrap writes a stderr warning when `n_traces < 20` explaining that p25 over a small sample has wide CIs and the resulting thresholds shouldn't be treated as authoritative.
- **`EvalReport` API reference docs page** (`docs/reference/eval-report.mdx`). Every public attribute and method documented with type + one-line description, plus a "common gotchas" section for the `cases` vs `case_results` / `summary` JSON-vs-attr / `passed_by_evaluator` method-vs-attr drifts.

### Fixed

- **CRITICAL: `_run_parallel` silently dropped all but the first case.** Regression introduced when parallel-by-default landed: a single `parent_ctx = contextvars.copy_context()` was reused across every `ThreadPoolExecutor` submission. Per Python docs, a single `Context` cannot be entered concurrently — `parent_ctx.run()` raises `RuntimeError` when another thread is already inside it. Threads 1..N silently captured the error into `CaseResult.actual_output` (`"[ERROR: cannot enter context: ... is already entered]"`) and the user saw all-but-the-first case appear to fail with empty results. Fixed by per-submission `contextvars.copy_context().run()` — each thread gets its own Context snapshot.
- **`EvalGateFailure` now inherits from `Exception` AND `SystemExit`.** Previously a `SystemExit`-only base meant library users couldn't catch it with `except Exception:` — a common pattern in notebooks, test harnesses, and Jupyter. Dual-inherit keeps CI exit semantics (uncaught instances still exit non-zero cleanly without traceback noise) while making `except Exception as e:` work for budget gate handling.
- **Loud stderr warning when most cases hit a judge error.** When `judge_error >= max(2, total/2)` after a run, `suite.run()` writes a block at end-of-run naming the first error verbatim. Catches the "I forgot `pip install multivon-eval[google]`" footgun — previously the user saw `pass_rate=0%, cost=$0, calls=0` and assumed the model failed; now they see the `JudgeUnavailable` message at the top of the block.
- **Bootstrap `eval_suite.py` no longer truncates rationale into inline comments.** Full rationale lives in the module docstring; inline comments carry only the tier tag.

### Notes

- Existing 0.8.x users with custom `JudgeRetry` policies, custom adapters, or downstream code calling `report.assert_budget()` are unaffected by the `EvalGateFailure` base-class change.
- The `--validate` flag on `bootstrap` is opt-in; default behavior is unchanged.
- The skip-propagation change affects aggregate pass-rate numbers on existing suites — cases that previously failed at 0.0 because the data shape didn't fit will now appear as a passing-but-skipped result. Use `cr.results[i].metadata.get("skipped")` to filter when analyzing.
- App 1–5 of the SDK deepdive (~$0.05 of real API spend across Anthropic / OpenAI / Gemini judges) serve as the integration smoke tests for the changes above.

## [0.8.2] — 2026-05-20

Second dogfood pass after 0.8.1 surfaced a UX paper cut: `EvalSuite.for_rag()` auto-includes `ContextRecall`, but a RAG case without `expected_output` made it return `score=0.0, passed=False` with a "Requires …" reason — looking like a quality failure when the data shape just didn't support the metric.

### Fixed

- **`ContextRecall` now skips cleanly when `expected_output` is missing.** Returns `score=1.0, passed=True` with `reason="[skipped] Requires both case.context and case.expected_output — add expected_output to your case to enable ContextRecall."` and `metadata.skipped=True`. Users can filter on `[skipped]` to see what was skipped vs. genuinely passed.

### Known issues

- A similar "returns 0.0 when input shape doesn't match" pattern exists in ~20 other evaluators (`AnswerAccuracy`, `ExactMatch`, `Contains`, `BLEU`, `ROUGE`, agent evaluators when no agent_trace, conversation evaluators when no conversation, etc.). These will get the same skip-semantics treatment in 0.9.0. For now, only `ContextRecall` is fixed because it's auto-included by `EvalSuite.for_rag()` and was the most-visible footgun.

### Tests

- 3 new tests in `tests/test_context_recall_skip.py` cover all three "missing input" paths.
- Full suite: 835 passed, 13 skipped (was 832/13 at 0.8.1).

## [0.8.1] — 2026-05-20

Fixes a launch-blocking UX bug surfaced by a critical-user dogfood pass.

### Fixed

- **`run_with_anthropic` / `run_with_openai` / `run_with_litellm` now auto-inject `EvalCase.context` into the system prompt.** Previously, every RAG case run via these one-line helpers silently dropped its context — Claude/GPT got the question with no grounding, faithfulness/hallucination evaluators scored 0/N against the empty-context reality, and users had no signal that the helper wasn't doing what its name implied. Adapter contract extended via a new optional `_call_with_case(case)` method that the suite uses when available; existing custom adapters (string-only `__call__`) are unaffected. List-valued contexts are formatted with `[chunk i]` markers so the model sees the boundaries.

### Tests

- 14 new tests in `tests/test_adapter_context_injection.py` cover: `_format_context_block` helper, AnthropicAdapter / OpenAIAdapter context injection, system-prompt composition with both user-supplied and RAG prefixes, list-valued context, suite routing to `_call_with_case` when available + fallback for plain callables.
- Full suite: 832 passed, 13 skipped (was 818/13 at 0.8.0).

## [0.8.0] — 2026-05-20

The intelligent-eval release. Two new public surfaces solve the "I don't know what to eval" cold-start problem for teams shipping LLM products: a CLI bootstrap command that proposes a tuned EvalSuite from a product description + sample traces, and a `multivon_eval.auto` module that exposes the underlying primitives (case-shape inference, LLM-driven adversarial generation, N-shot judge-noise aggregation) for users who want to compose their own pipelines.

### Added

- **`multivon-eval bootstrap` CLI** — cold-start eval generator. Takes `--product PRODUCT.md --traces TRACES.jsonl` and emits four artifacts to `--output DIR`: `eval_suite.py` (runnable suite with 4-6 evaluators), `seed_cases.jsonl` (30 adversarial seed cases), `thresholds.yaml` (calibrated from your traces at p25 of baseline scores), and `DISCOVERY_REPORT.md` (a forwardable eval design review). Single Claude Haiku call for metric proposal, capped trace-sample calibration, deterministic safety net via `auto_evaluators`. Cost target ≈$0.12 per bootstrap, hard ceiling $0.15.
- **PII / secret redaction before any LLM call.** Bootstrap runs a high-confidence local scan (AWS / Anthropic / OpenAI / GitHub / Google / Stripe / JWT / private key / SSN / email / Luhn-valid credit card) and redacts detections before traces are sent upstream. Three policies via `--pii-policy`: `redact` (default), `strict` (abort on detection), `allow` (raw, with explicit confirmation prompt).
- **`multivon_eval.auto` module** — intelligent-eval primitives:
  - `auto_evaluators(case)` — pure-heuristic, infers the recommended evaluator set from an `EvalCase` shape. Supports `task_type=` override, `strict_mode`, `include_pii`, `include_safety`. Zero-cost, microseconds.
  - `generate_adversarial_cases(seed_text, mode, n)` — LLM-generates cases targeting one of 10 failure modes (ungrounded_claim, off_topic, format_violation, jailbreak, tool_misuse, numeric_edge, prompt_injection_direct, prompt_injection_indirect, tool_injection, pii_leakage_invitation). Stress-test labels embedded in metadata so downstream tools can route automatically.
  - `generate_unicode_obfuscation_cases(base_strings, kinds)` — deterministic homoglyph / zero-width / RTL-override transforms, no LLM call.
  - `validate_adversarial_cases(cases, baseline, n_shots=3, hardness_band=(0.5, 1.0))` — runs each case N times against a baseline + evaluator, computes failure_rate per case, filters by hardness band. Validated live this release: +0.80 mean failure-rate separation between weak (always-confabulate) and strong (always-refuse) baselines on `ungrounded_claim` cases, with judge noise correctly filtered out at the per-shot level.
- **Top-level exports:** `bootstrap`, `BootstrapResult`, `RecommendedEvaluator`, `TraceSummary`, `infer_product_shape`, `summarize_traces`, `load_traces`, `auto_evaluators`, `EvaluatorRecommendation`, `AmbiguousCaseShape`, `generate_adversarial_cases`, `generate_unicode_obfuscation_cases`, `validate_adversarial_cases`, `HardnessReport`.

### Fixed

- **`rag_eval.ipynb`** — `Experiment.add_run("name", report)` doesn't exist; corrected to `exp.record(report, run_id="name")`. `suite.prepare()` doesn't exist; corrected to `evaluator.prepare()` per-CheckEvaluator. Cells also switched from private (`_criterion`, `_evaluators`) to public accessors (`criterion`, `evaluators`) to match the quickstart notebook's style.

### Tests

- 33 new tests in `tests/test_discover.py` cover the bootstrap pipeline (PII scan + redact, shape inference, LLM response parsing + safety-net merge, threshold calibration math, end-to-end with mocked LLM).
- 19 tests in `tests/test_auto_validate_adversarial.py` cover N-shot aggregation, hardness_band filtering, baseline / evaluator crash resilience, and degenerate n_shots=1 fallback.
- 19 tests in `tests/test_auto_evaluators.py` cover the heuristic surface (RAG / QA / agent / conversation / multimodal / structured-output paths, ambiguity scoring, strict_mode).
- 7 tests in `tests/test_auto_unicode_obfuscation.py` cover the deterministic Unicode transforms.
- Full suite: 818 passed, 13 skipped (was 745/12 at 0.7.3).

## [0.7.3] — 2026-05-17

The trust release: the single most-cited bug across a 26-voice strategy deliberation was the silent calibration fallback. Fixed, with a public escape hatch for legacy callers who depended on it. Two experimental multimodal evaluators land as the seed for a forthcoming document-AI benchmark (see [`pdfhell`](https://github.com/multivon-ai/pdfhell)).

### Fixed

- **Silent calibration fallback is now loud.** `calibrated_threshold(evaluator, judge)` previously fell back to `0.7` silently when the `(evaluator, judge_model)` pair was missing from `_calibration_data/v2.json` — the strategy deliberation flagged this as the single most-cited trust bug (5+ persona voices including a Series-A CTO who called it "deceitful code"). The default behaviour is now `"warn"`: a `UserWarning` fires once per pair, then the call returns `0.7`. Pre-0.7.3 silent behaviour is opt-in via `set_calibration_fallback_policy("silent")` for back-compat. For procurement/audit deployments call `set_calibration_fallback_policy("strict")` (raises `CalibrationMissing`). The `MULTIVON_CALIBRATION_FALLBACK={silent,warn,strict}` env var overrides at process start.

### Added

- **`set_calibration_fallback_policy(policy)`** exported at top level. Module-level switch; per-call `strict=True` still wins.
- **`MULTIVON_CALIBRATION_FALLBACK` env var** — set to `silent`, `warn`, or `strict` to override the in-process default without code changes.
- **Multimodal evaluators (experimental)** — first multimodal capabilities shipped:
  - **`VQAFaithfulness`** — image-grounded faithfulness. Generates 3 QAG claims about an image, verifies each. Reads image from `case.metadata['image_url' | 'image_path' | 'images']`.
  - **`DocumentGrounding`** — multi-page document-agent grounding. Three QAG questions per case: claim support, entity invention, exception handling. Seed evaluator for the Document Agent Acceptance Protocol v0.1.
  - Vision dispatch wired for `anthropic` (Claude 3.5+ + 4.x), `openai` (GPT-4o+), `google` (Gemini 1.5+). Raises `JudgeUnavailable` with a friendlier hint when a text-only judge is mis-wired.
  - Both classes flagged experimental: no calibration rows shipped yet (so the new "warn" default fires on first use until thresholds are calibrated).

### Tests

- 21 new tests in `tests/test_multimodal.py` exercise the public surface (image-metadata parsing, error paths, parse helpers) without provider API calls.
- Full suite: 745 passed, 12 skipped (was 724/12).

## [0.7.2] — 2026-05-16

Gemini lands as a first-class judge provider. The 5-judge multi-judge agreement benchmark re-ran with 250 LLM calls and ships in the website.

### Added

- **`provider="google"` for `JudgeConfig`** — backed by the official `google-genai` SDK. Default model: `gemini-2.5-flash`. Install with `pip install 'multivon-eval[google]'` (the extra pulls `google-genai>=1.0.0`). Sync + async paths wired. Auth via `GOOGLE_API_KEY` (matches Google's own docs); the standard "missing key" `JudgeUnavailable` setup hint now mentions where to get one.
- **Pricing for Gemini** in `_cost_models.py` — `gemini-2.5-flash` $0.075/$0.30 per million in/out, `gemini-2.5-flash-lite` $0.0375/$0.15, `gemini-2.5-pro` $1.25/$5.00, plus 1.5-series. Per-token usage is recorded the same way as Anthropic and OpenAI, so `report.costs.total_cost_usd` is correct out of the box.
- **Multi-judge benchmark refreshed with 5 judges**. `benchmarks/results/multi_judge_agreement.json` now reports pairwise Cohen's κ 0.60–0.80 (substantial agreement on most pairs) and per-judge accuracy/precision/F1 on HaluEval QA, N=50. `gemini-2.5-flash` leads on every metric in this run (accuracy 0.860, precision 0.950, F1 0.844). Numbers surface on the website and in the new "Why multivon-eval" docs page.

## [0.7.1] — 2026-05-16

Pre-public-launch hardening: two real bugs the new benchmarks surfaced, plus calibration around the numbers we put on the website.

### Fixed

- **`workers > 1` lost every CostTracker record.** `_run_parallel` submitted work via `ThreadPoolExecutor` without copying the caller's `contextvars`, so each worker started in an empty context and the active CostTracker was invisible. `report.costs.total_cost_usd` came back `$0.00` whenever you parallelised. Wrapped each `submit()` call in `contextvars.copy_context().run(...)`. Verified — `workers=4` on 4 Hallucination calls now reports `$0.00022` instead of `$0`.
- **`set_cache(JudgeCache(...))` was a silent no-op.** The cache only activated when `JudgeConfig(cache=True)` was passed explicitly to each evaluator. Installing a cache globally — the most natural way to opt in — did nothing because `JudgeConfig.cache` defaults to `False`. Added `cache_is_user_opted_in()`; `set_cache(non_none)` now flips it; `JudgeConfig.resolve()` honors it. Verified — sequential rep1→rep2 cache hit goes from 2.9s/4-calls to 0ms/0-calls (~2,271× faster) without any per-evaluator config.
- **`EvalSuite.run(save_json=..., save_junit_xml=...)` already in 0.7.0 — added documentation here.** Both keyword args write the report BEFORE `fail_threshold` raises `EvalGateFailure`, so a failing gate still leaves an artifact for `multivon-eval view` / `compare`.

### Added (benchmarks)

- `benchmarks/run_multi_judge_agreement_benchmark.py` — pairwise Cohen's κ across `claude-haiku-4-5`, `claude-sonnet-4-6`, `gpt-4o-mini`, `gpt-4o` on the same hallucination cases. Output at `benchmarks/results/multi_judge_agreement.json`. Numbers ship on the website.
- `benchmarks/run_cost_latency_benchmark.py` — 50 HaluEval cases × 4 LLM-judge evaluators with `workers=1`, real Anthropic billing. `cost_latency.json` reports $0.00127/case, 17 judge calls/case, and a linear $6.35 extrapolation to 5,000 cases.
- `benchmarks/run_reproducibility_benchmark.py` — 10 cases × 10 reps, cache on/off. Surfaces (a) the cache-miss bug fixed in this release and (b) ~3% irreducible stdev at `temperature=0` across reps of `claude-haiku-4-5`. The cache fix turned this into the 2,271× speedup published on the site.
- `docs/sample-audit-package.zip` (5.5 KB) — a real `audit-package` zip from the `regulated` template. Linked from the website's Compliance Bundle CTA so buyers can see what an auditor actually receives.

### Changed (docs / README)

- README hero example: `your_llm.generate` placeholder replaced with a real `anthropic.Anthropic()` snippet that runs after `pip install`.
- README + docs claims aligned to reality: calibration F1 range corrected from "0.76–0.98" to the actual shipped 0.66–1.00; the `2.9× run_async` and `4,700× cache` website claims (the latter conservatively true now but unsupported in the repo) are replaced with the linkable benchmark numbers above.
- New docs page: [Why multivon-eval](https://evaldocs.multivon.ai/why-multivon-eval) with head-to-head benchmark tables.



## [0.7.0] — 2026-05-16

The trust release: explicit error classification so a transient judge outage no longer masquerades as a model regression, plus the first major batch of community-facing usability work — JUnit CI integration, a local HTML report viewer, classical similarity metrics, repaired examples and notebooks.

### Fixed (pre-release audit, 0.7.0)

- **Headline trust feature now actually works.** Every LLM-judge evaluator (`Faithfulness`, `Hallucination`, `Relevance`, `ContextPrecision`, `CustomRubric`, `GEval`, `CheckEvaluator`) plus the agent evaluators (`ToolArgumentAccuracy`, `ToolCallNecessity`, `TaskCompletion`, `StepFaithfulness`) and `SelfConsistency` had bare `except Exception:` blocks that silently swallowed `JudgeUnavailable` and re-classified the case as a quality failure (`score=0.0`). This defeated the entire `CaseResult.status` distinction the release advertises. Each judge call now re-raises `JudgeUnavailable` so `suite.run()` routes the case to `EvalStatus.JUDGE_ERROR` and `pass_rate` excludes it correctly.
- **`fail_threshold` no longer reports "Eval failed: pass rate 0.0%" when every case errored.** When `evaluated == 0` and `errors > 0`, `suite.run()` raises `EvalGateFailure` with the underlying error message (e.g. "Missing credentials … export OPENAI_API_KEY=sk-…") instead of a misleading quality gate failure.
- **`rag` init template no longer hangs ~45s** when no API key is set and Ollama isn't running. The template now probes Ollama with a 0.5s timeout (matching the `regulated` template) and falls back to a JudgeConfig whose `suite.run()` call surfaces an actionable setup hint at first use.
- **`__all__` re-exports** — `CaseResult`, `EvalResult`, `EvalReport`, `EvalStatus`, `EVALUATION_STATUSES`, `ERROR_STATUSES`, `Costs`, `CostTracker`, `ProviderUsage`, `ModelPricing`, `register_pricing`, `SuiteLock`, `EvaluatorFingerprint`, `LockMismatch`, `build_suite_lock`, `fingerprint_evaluator`, `verify_suite_against_lock`, `build_audit_package`, `assert_evaluators`, `EvaluatorFailure` were imported at module top-level but missing from `__all__`. `from multivon_eval import *` and IDE introspection now see them.
- **README PII example** had four missing commas between `add_evaluators(...)` arguments — pasted-as-is was a `SyntaxError`. Fixed.
- **`docs/evaluators/deterministic.mdx`** `StartsWith("```json")` example used a literal triple-backtick inside a triple-backtick fence, closing the outer code block early in Mintlify. Switched the outer fence to four backticks.

### Added

#### Foundation primitives

- **`CaseResult.status`** — new property returning an `EvalStatus` enum (`passed`, `failed_quality`, `model_error`, `judge_error`, `evaluator_error`, `timeout`, `skipped`). Surfaces *what kind* of outcome the case had, not just pass/fail. Status fields (`judge_error`, `evaluator_error`, `skipped`, `agent_trace`) added to `CaseResult` directly.
- **`EvalReport.evaluated`, `.errors`, `.errors_by_kind`, `.skipped`** — counts that distinguish quality outcomes from infrastructure failures.
- **Per-evaluator error isolation** — when one evaluator raises `JudgeUnavailable`, the rest of the case's evaluators still run. The failing evaluator's result records a clear "judge unavailable" reason in `EvalResult.metadata["error_kind"]`, and the case is tagged `EvalStatus.JUDGE_ERROR`. A non-`JudgeUnavailable` exception in an evaluator is tagged `EvalStatus.EVALUATOR_ERROR` (distinct, so retry logic can target judge outages without masking real bugs). Both sync and async (`run_async`) paths honor this.
- **`CaseResult.agent_trace`** — captured agent traces now surface on the result (not only on the input case), so notebooks can iterate steps from the report without reaching back into the suite.
- **Multi-run aggregation propagates error fields** — when `runs > 1` and any run errors, the aggregate `CaseResult` keeps the first error of each kind. `pass_count` uses `cr.passed` (status-aware), so SPRT early-stop and flaky-detection don't count error runs as successes.

#### CI integration

- **`EvalReport.to_junit_xml()` + `.save_junit_xml(path)`** — render the report as JUnit XML. GitHub Actions, GitLab CI, CircleCI, Jenkins all render JUnit XML natively in their PR/job summary UI. Quality failures emit `<failure>`, plumbing failures emit `<error>` (distinct so CI can route them differently), skipped cases emit `<skipped>`. XML 1.0-invalid control characters are stripped at the serialization boundary so strict CI consumers accept the document.
- **`multivon-eval report results.json --junit out.xml`** flag.
- **`multivon-eval view <report.json>` CLI** — local HTTP server with the HTML dashboard. `--port`, `--no-browser` flags. `TemporaryDirectory` + `SIGTERM` handler so the temp dir is cleaned up on Ctrl-C, `docker stop`, or exception. Port collision produces a clean error, not a traceback.

#### Public API surface

- **Top-level imports**: `CaseResult`, `EvalReport`, `EvalResult`, `EvalStatus`, `EVALUATION_STATUSES`, `ERROR_STATUSES` are now importable from `multivon_eval` directly. Saves users from reaching into `multivon_eval.result`.

#### Evaluators

- **`Levenshtein`** — character edit-distance similarity. Score = 1 − dist / max(len). Pure-Python (no extra deps). `threshold`, `case_sensitive` kwargs.
- **`ChrfScore`** — character n-gram F-beta (Popović 2015), standard sacreBLEU aggregation: average precision per order, average recall per order, then F-beta on the averages. Defaults match sacreBLEU's chrF (`max_n=6`, `beta=2`, whitespace stripped). `include_whitespace=True` for the count-spaces variant.

#### Onboarding (from 0.6.2, surfaced here)

- **`multivon-eval init`** — scaffold a starter project in under 5 minutes. Templates: `quickstart` (offline, no API key), `rag`, `agent`, `regulated`. `--ci github` generates a GitHub Actions workflow. `--force` to overwrite a non-empty target.
- **`EvalReport.assert_budget(...)`** — opt-in cost / token / latency gate. Raises `EvalGateFailure` on violation. All thresholds opt-in; missing pricing data surfaces a clear actionable error.

#### CI hardening

- **`.github/workflows/test.yml`** — pytest matrix on Python 3.10/3.11/3.12, every PR.
- **`.github/workflows/install-smoke.yml`** — builds the wheel, installs in a clean venv WITHOUT the dev extras or pytest, verifies bare import, verifies the public API, runs the quickstart notebook headlessly with a placeholder API key (auth errors are expected; `AttributeError`/`TypeError` from API mismatches → regression). The project shipped 0.6.0 with no CI at all; both workflows close that gap.

#### Enterprise / compliance (later 0.7.0 additions)

- **Immutable audit-record provenance** — every `ComplianceReporter.record()` row now carries a `provenance` block with `package_version`, `package_git_sha` + `package_git_dirty` (when running from a git workspace), `host` (python/platform/machine — no PII), full `suite_lock` (evaluator + judge + calibration + per-evaluator config fingerprint + cases hash), and a `suite_lock_status` field that distinguishes "absent" (synthetic report) from "ok" and "serialization_failed". The block is part of the SHA-256 hash chain, so tampering with provenance is detected by `reporter.verify()`. Marcus persona's compliance-grade blocker.
- **Evaluator config in the fingerprint** — `SuiteLock.evaluators[].extra.config` now captures the JSON-safe public attributes (`WordCount.min_words`, `Contains.substrings`, `RegexMatch.pattern`, etc.) so two suites with the same evaluator name + threshold but different config produce different `suite_hash` values. `diff()` surfaces config-level changes.
- **Calibration version pinning** — `load_calibration(version="v1")`, `calibrated_threshold(..., version=)`, and `threshold_table(version=)` take an explicit version label. `MULTIVON_CALIBRATION_VERSION` env var pins globally for CI. `calibration_versions()` lists shipped labels. Unknown versions raise `FileNotFoundError` loudly — silent fallback would defeat the purpose of pinning for reproducibility. Sarah persona ask.
- **HTML report status badges** — six pill variants surface the 0.7.0 EvalStatus enum: PASS, FAIL, FLAKY, MODEL ERR, JUDGE ERR, EVAL ERR, SKIPPED. Distinct colors (green/red/yellow/orange/slate) so a judge outage isn't visually confused with a model regression. Each error pill carries a tooltip explaining which subsystem to investigate. Errors and Skipped counts surface as summary cards when present. Priya persona ask.
- **Conversation template** for `multivon-eval init` — fifth template (`init -t conversation`) demonstrating multi-turn dialogue eval with `ConversationRelevance` + `KnowledgeRetention` + `TurnConsistency`. Closes the gap noted in the examples audit (no template demoed the conversation API).
- **Calibration version pinned through audit-package replay** — `SuiteLock` gains a top-level `calibration_version` field populated unconditionally from `effective_calibration_version()` at lock-build time. The label flows through suite lock → audit log provenance → `build_audit_package()`, which now extracts the version from the FIRST log record and bundles the matching `calibration_v{label}.json`. Manifest gains `calibration_version` + `calibration_source` ("logged" vs "default"). An unshipped pin (`MULTIVON_CALIBRATION_VERSION=v_doesnotexist`) raises `FileNotFoundError` at `suite.run` time instead of silently writing `suite_lock=None` and defeating the pin. Fixes a real Marcus-persona replay-fidelity bug: previously a v1-pinned audit packaged on a v2-default install would silently bundle v2.
- **Per-case retry on transient judge errors** — new `JudgeRetry` policy + `suite.run(..., judge_retry=JudgeRetry(...))` opt-in. Cases whose status is in `policy.retry_on` (default: `judge_error`) are re-evaluated up to `max_attempts` times with exponential backoff (`base_backoff * factor ** (attempt - 2)`), symmetric jitter, and a `max_backoff` cap. Quality failures, model errors, and evaluator bugs are NOT retried — those are signal. `CaseResult` gains `retry_attempts` (count of retries actually performed) and `retry_errors` (the error per failed attempt that prompted a retry; `len == retry_attempts`). Sync, async (`run_async` — uses `asyncio.sleep`), and parallel-workers paths all honor the policy. JSON round-trip preserves retry history. Sarah persona ask — a 10k-case weekend cron no longer needs Monday triage when one 429 trips one case.
- **Native agent framework integrations** (D16) — two new templates with real-framework tracers:
  - `multivon-eval init -t agent-langgraph` — `StateGraph` + `MessagesState` + `ToolNode` + `tools_condition`, instrumented via the new `LangGraphTracer`. Uses run_id-keyed metadata + `langgraph_node` + `graph:step:N` tags. **One AgentStep per LLM turn** (not per graph node) so a ReAct's `tools` node aggregates with its preceding decision. Parallel tools within one node are correctly attributed; subgraph metadata is preserved.
  - `multivon-eval init -t agent-openai-sdk` — real `Agent` + `function_tool` + `Runner.run_sync`, instrumented via the new `OpenAIAgentsTracer`. **Two integration paths**: post-hoc `tracer.capture(result)` parses `RunResult.new_items` (default, no global state); live `tracer.run_hooks()` + `tracer.merge(hooks)` uses isolated `RunHooksBase` buffers (no leakage across concurrent runs). Idempotent `merge`. Known SDK item types (`CompactionItem`, `ToolApprovalItem`, MCP / ComputerCall / CodeInterpreter / ToolSearch items) preserved as visible markers rather than silently dropped.
  - Both templates ship 5 cases including negative trajectories (already-refunded, not-found, processing). New `ToolCallAccuracy(penalize_unexpected=True)` makes the negatives actually fail when the agent over-calls.
  - Pyproject extras: `[langgraph]`, `[openai-agents]`. README "Pick your path" table extended.
- **Beginner-friendly onboarding pass** (D15 from OSS-adoption audit):
  - README quickstart flipped to `init -t quickstart` (offline, no API key) instead of `init -t rag` (needed key). New "Pick your path" table makes the right entry obvious.
  - Agent template (`init -t agent`) now runs OFFLINE by default with deterministic `ToolCallAccuracy`. LLM-judge evaluators (`ToolArgumentAccuracy`, `TrajectoryEfficiency`, `TaskCompletion`) auto-activate when `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / local Ollama is detected. Previously: silent 0-scores when no key → looked like the agent failed.
  - **`JudgeUnavailable` carries a setup hint** when the underlying exception is auth- or connection-shaped. Concrete next steps: `export ANTHROPIC_API_KEY=...`, `ollama pull`, or `init -t quickstart`. Generic API errors (BadRequest, APIError, prompt-too-long, invalid model id) get clean messages without the hint — real bugs aren't drowned in setup advice.
  - **`AgentTracer.format_trace()` + `print_trace()`** for agent debugging: pretty-print a captured `list[AgentStep]` from a notebook or CLI without reaching into the dataclasses.
  - **Public accessors**: `EvalSuite.evaluators`, `EvalSuite.cases`, `CheckEvaluator.criterion`. Notebooks no longer teach `_evaluators` / `_criterion` private internals.
  - Local Ollama probe added to `_auto_judge()` in the `agent`, `regulated`, and `conversation` templates so the README's "no API key needed (Ollama works)" claim is honored everywhere.
  - Quickstart notebook version pin bumped to `>=0.7.0` (was stale `>=0.6.1`).
- **`multivon-eval compare baseline.json proposal.json`** — answer "did my prompt change help?" in one command. Pairs cases by `case_input` (sequential within duplicates), reports pass-rate / avg-score / errors / flaky deltas, per-case regressions and improvements, and a McNemar p-value over paired cases (None when no valid pairs). SKIPPED on either side is excluded from direction + McNemar so a not-evaluated case isn't falsely scored as a regression. CLI: `--regressions-only`, `--markdown` (PR-comment format), `--json`, `--fail-on-regression` (CI gate). Python: `compare_reports()`, `EvalReport.compare(other)`, `ReportDiff`, `CaseDiff`.

### Changed (BREAKING — minor version bump)

- **`EvalReport.pass_rate` excludes error cases from the denominator.** A run with 2 passed + 3 judge-error cases now reports `pass_rate = 1.0` (2/2 evaluated), not `0.4` (2/5 total). Use `EvalReport.errors` to surface infrastructure problems independently. **This is the headline behavior change.**
- **`EvalReport.avg_score`** excludes error cases from the average.
- **`EvalReport.failed`** counts *quality failures only* (cases with `EvalStatus.FAILED_QUALITY`). Use `EvalReport.errors` for the rest.
- **`EvalReport.pass_rate_ci()`** uses `evaluated` as the denominator to match `pass_rate`. Pre-0.7.0 callers reading `RunRecord.total` for the z-test denominator should now read `evaluated` (legacy records default to `total` for backward compatibility).
- **`CaseResult.passed`** is defined as `status == EvalStatus.PASSED`, so a case with no evaluator results or in any error state returns `False` even if individual `EvalResult.passed` values were `True`.

### Fixed

Carry-over from the 0.6.1 + 0.6.2 patch series (which never reached PyPI; all changes are part of 0.7.0):

- **`import multivon_eval` no longer requires pytest.** The pytest plugin import is guarded; users who don't have pytest installed get a clear `ImportError` only when they actually call `assert_evaluators()`.
- **All 4 QAG-based agent evaluators** (`PlanQuality`, `TaskCompletion`, `TrajectoryEfficiency`, `AgentMemoryEval`) now pass `judge` to `_qag_eval`. Previously raised `TypeError` on every real invocation.
- **All 4 conversation evaluators** — same `_qag_eval` fix.
- **`Contains.match_any`** — added as a keyword-only argument so `Contains([...], False, 0.75)` keeps `0.75` as `threshold`.
- **`WordCount(min=, max=)`** alias kwargs.
- **`audit-package` CLI** bundles the calibration version actually in use (`v2.json` preferred over `v1.json`).
- **Notebook auto-detects judge** from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` env vars rather than hard-coding local Ollama. Colab now works without setup.
- **`TrajectoryEfficiency` recovery scoring** uses the per-evaluator judge instead of the global default.
- **`run_on_cases()`** applies the same per-evaluator isolation as the live run path.
- **Calibration reconciliation** — `v2.json` extends `v1.json` with new judges (`gpt-5.5`) but preserves v1 thresholds for every existing judge × evaluator combination. Eliminates the silent threshold drift between 0.5.x and 0.6.0.

### Examples + notebooks repaired

- `examples/ci_eval.py` — removed dead post-`fail_threshold` code, added JUnit XML output, distinct exit code 2 for infrastructure errors.
- `examples/basic_eval.py` — simplified evaluator setup; added `Levenshtein` for short-string similarity.
- `examples/eu_ai_act_eval.py` — tamper-detect demo now asserts the verifier raises (the contract); previously silently succeeded.
- All `examples/*.py` — added `if __name__ == "__main__"` guards so importing them doesn't auto-run an LLM eval.
- `notebooks/agent_eval.ipynb` — fixed cells 7 and 10 that referenced `cr.trace.steps` (never existed); now use `cr.agent_trace` directly.
- All notebooks: install pins bumped to `multivon-eval>=0.7.0`.

### Migration notes

Most callers don't need any code changes for 0.7.0. The behavior change is concentrated in `EvalReport.pass_rate` and `.avg_score`:

- **CI thresholds that gate on `pass_rate`** become more sensitive — error cases no longer drag the metric down. What used to be a 60% pass rate (6 pass / 4 errors out of 10) is now `pass_rate = 1.0` with `errors = 4`. If you want CI to fail on errors too, check `report.errors == 0` explicitly.

Old:
```python
report = suite.run(fn)
if report.pass_rate < 0.8:
    sys.exit(1)
```

New (recommended):
```python
report = suite.run(fn)
if report.errors > 0:
    sys.exit(2)   # infrastructure problem — caller should retry
if report.pass_rate < 0.8:
    sys.exit(1)   # quality regression
```

The shipped `multivon-eval init --template rag` template uses this pattern.

## [0.6.x] — never published to PyPI

The 0.6.1 and 0.6.2 wheels were built but not published; their contents (bug fixes, init scaffolder, budget gates) ship as part of 0.7.0 above.

## [0.6.0] — 2026-05-13

Initial public release with the full feature surface: deterministic + LLM-judge evaluators, agent + conversation eval, hash-chained compliance audit log, calibrated thresholds per (judge × evaluator) with shipped F1 evidence, pytest plugin, async runner, suite lockfile for drift detection.

See [the v2 benchmark blog post](https://multivon.ai/blog/benchmark-v2-cross-dataset) for the cross-dataset F1 numbers.
