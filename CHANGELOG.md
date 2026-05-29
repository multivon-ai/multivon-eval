# Changelog

All notable changes to `multivon-eval`. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as of 0.7.0.

## [Unreleased]

Phase 1 of prompt-regression attribution. Adds `multivon_eval.attribution`, a small package that walks a Python repo for LLM SDK call sites, fingerprints prompt literals, and emits a structured diff across two refs. **Descriptive only — no causal attribution claims.** The hardened calibration spike of 2026-05-30 showed Haiku-based causal attribution failing catastrophically on mixed-cause regressions; the design doc at `multivon-strategy/positioning/feature_prompt_attribution_phase2_sidecar_design_2026_05_30.md` plans the v2 sidecar signal that closes that hole structurally.

### Added

- **`multivon_eval.attribution`** module — public API: `scan(repo_root)`, `diff_records(base, head)`, `render_markdown(diffs)`. Detects `anthropic.messages.create`, `openai.chat.completions.create`, and `litellm.completion`/`acompletion` call sites via suffix-based matching that handles both `client.messages.create(...)` and `anthropic.Anthropic().messages.create(...)`. Extracts string-literal `system=` kwargs and the `content` field of each `messages=[…]` entry. f-strings with zero runtime interpolation are treated as literals; runtime-interpolated f-strings and `Name` references are flagged `is_dynamic=True` with a stable placeholder. Skips `.venv`, `node_modules`, `__pycache__`, build directories.
- **`multivon-eval attribution scan <repo>`** — list every detected prompt call site in a repo, with `--format text` (default) or `--format json`.
- **`multivon-eval attribution diff <base> <head>`** — structured diff between two checkouts, with `--format markdown` (default; PR-comment-ready), `--format text`, or `--format json`.

### Notes

- Causal attribution (which prompt change caused which regression) is deliberately not shipped in v1. See the Phase 2 design doc for the sidecar plan that gates Haiku attribution behind a non-prompt-change detector to avoid HIGH-confidence-and-wrong failures on mixed-cause PRs.

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
