# Changelog

All notable changes to `multivon-eval`. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as of 0.7.0.

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
