# multivon-eval

[![PyPI](https://img.shields.io/pypi/v/multivon-eval.svg)](https://pypi.org/project/multivon-eval)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/multivon-eval)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/multivon-eval/month)](https://pepy.tech/project/multivon-eval)
[![Tests](https://github.com/multivon-ai/multivon-eval/actions/workflows/test.yml/badge.svg)](https://github.com/multivon-ai/multivon-eval/actions/workflows/test.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multivon-ai/multivon-eval/blob/main/notebooks/quickstart.ipynb)

**[Docs](https://docs.multivon.ai)** · [Website](https://multivon.ai) · [PyPI](https://pypi.org/project/multivon-eval) · [Changelog](CHANGELOG.md) · [Benchmark vs DeepEval + RAGAS](https://github.com/multivon-ai/eval-framework-benchmark)

**AI evaluation for teams that ship models to production.** The popular eval frameworks disagree with *each other* on 56% of hallucination verdicts — Cohen's **κ = 0.03**, agreement no better than a coin flip, measured on the same data with the same labels ([raw data + code](https://github.com/multivon-ai/eval-framework-benchmark)). multivon-eval is the framework that measures itself first.

## Quickstart — 30 seconds, no API key

```bash
pip install multivon-eval
python -m multivon_eval                       # runs a demo eval — no setup
multivon-eval init -t quickstart -d my-eval   # scaffold your own (offline)
cd my-eval && python eval.py
```

Or as code — fully offline, and `runs=5` records five trials per task:

```python
from multivon_eval import EvalSuite, EvalCase, Contains, NotEmpty

suite = EvalSuite("smoke", purpose="capability")
suite.add_cases([EvalCase(input="What is 2+2?", expected_output="4")])
suite.add_evaluators(NotEmpty(), Contains(["4"]))

if __name__ == "__main__":  # guard so `multivon-eval validate` never runs your model at import time
    report = suite.run(lambda prompt: "2+2 = 4", runs=5)  # swap the lambda for your model fn
    print(report.pass_rate, report.pass_rate_ci())  # 1.0, 95% CI [0.21, 1.0] — one task, and the CI says so
    print(report.pass_hat_k(3))                     # P(a task passes all 3 of 3 trials), with CI
```

The `quickstart` template sticks to deterministic evaluators (`NotEmpty`, `Contains`, `WordCount`), so the first run needs no API key at all. The "no API key" promise is scoped to that template: the `python -m multivon_eval` demo will emit LLM-judge scores too if it detects a key or a local server (Ollama on `:11434`, LM Studio on `:1234`, or `OPENAI_BASE_URL`), so a running local model can show judge output under this banner. The template stays deterministic-only regardless.

### Pick your path

| You're… | Run this | Needs API key? |
|---|---|---|
| Brand new — just kicking the tires | `python -m multivon_eval` | No (LLM judges activate if a key is set) |
| Beginner writing your first eval | `multivon-eval init -t quickstart` | **No** — fully offline |
| Building an agent (hand-rolled or any framework) | `multivon-eval init -t agent` | **No** for default eval, optional for richer judging |
| Building a **LangGraph** agent | `multivon-eval init -t agent-langgraph` | Yes (or local Ollama via `ChatOpenAI(base_url=...)`) |
| Building an agent with the **OpenAI Agents SDK** | `multivon-eval init -t agent-openai-sdk` | Yes (OpenAI) |
| Building a RAG / QA system | `multivon-eval init -t rag` | Yes (or local Ollama) |
| Working a regulated domain | `multivon-eval init -t regulated` | Yes (or local Ollama) |
| Multi-turn dialogue eval | `multivon-eval init -t conversation` | Yes (or local Ollama) |

LLM-judge evaluators auto-activate when `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or a local server (Ollama on `:11434`, LM Studio on `:1234`, or `OPENAI_BASE_URL`) is detected — but every template runs without one in some form.

### Why we exist

**The eval tools don't agree with each other.** We ran the three popular ones (multivon-eval, DeepEval, RAGAS) over the same data with the same labels. On a simple yes/no hallucination call, they disagree on 56% of cases. Cohen's **κ = 0.03** — agreement no better than a coin flip. So when your CI gate flips after you switch frameworks, that's the tool arguing with itself, not your model getting worse. Raw data and code: [eval-framework-benchmark](https://github.com/multivon-ai/eval-framework-benchmark).

**We test ourselves the hard way.** We calibrate the Hallucination evaluator on one dataset (HaluEval-QA), then score it on a different one (HaluEval-Sum, n=60) without re-tuning. It gets **F1 0.830 [0.70–0.92]**. On the in-distribution comparison, our worst case (CI lower bound 0.71) still beats DeepEval's best case (upper bound 0.68): F1 0.804 [0.71–0.88] vs 0.586 [0.48–0.68]. Full method and raw counts: [`benchmarks/README.md`](benchmarks/README.md) Benchmark 4.

**When the measurement catches us, we publish it.** Three times, newest first:

- We added a pixels-only mode to our PDF benchmark, and the leaderboard nearly flipped. Every PDF leader dropped (GPT-5 94.7% → 67.6%, Haiku 91.2% → 58.2%) and every laggard rose (Opus 79.4% → 85.9%). The benchmark had been measuring each provider's text-extraction pipeline as much as the model.
- That same pixels mode then caught a bug in *our own* benchmark on its first run: two trap families rendered a visible tofu box (■) instead of the invisible character we claimed they used. We redesigned them, footnoted the affected rows, and added a glyph-level gate so it can't ship again.
- We set a 50% bar for our prompt-drift detector, measured real traffic, and hit **20.9%**. We published the failed gate and shipped the honest design (a runtime recorder in its own trust tier) instead of the claim we couldn't back.

Earlier, the release run 0.9.4 → 0.9.5 → 0.9.6 → 0.9.7 was the same discipline at smaller scale: a review caught a "held-out" claim that wasn't, plus a threshold mismatch that had inflated the held-out F1 from 0.830 to 0.852, plus three runtime blockers — four releases in a day, all still on PyPI. We hold the framework to the same standard it asks of your models.

multivon-eval runs structured evals over model outputs: string checks, LLM-judge scoring, agent traces, multi-turn conversations. Python API, terminal and HTML reports, CI hooks.

## Demystifying evals, operationalized

Anthropic's [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) lays out how their teams build agent evals. Each recommendation maps to a runnable multivon-eval feature. (Vocabulary map, stated once: a **task** is an `EvalCase`, a **trial** is one of `--runs N`, a **grader** is an `Evaluator`, the **transcript** is `agent_trace`, the **outcome** is the graded verdict. The API names predate the post and stay for stability.)

| The post recommends | In multivon-eval |
|---|---|
| Keep a reference solution per task — a 0% pass rate usually means a broken task or grader, not an incapable agent | `multivon-eval validate eval.py` runs your graders against each task's reference output (`BROKEN_TASK_OR_GRADER` exits 1); every report flags zero-pass suspects (`EvalReport.zero_pass_cases`) |
| Report pass@k *and* pass^k when agents are non-deterministic | `suite.run(model_fn, runs=N)` prints a reliability panel; `report.pass_at_k(k)` / `report.pass_hat_k(k)` with CIs; gate CI on `report.assert_pass_hat_k(k, min_ci_low)` |
| Graduate saturated capability evals into regression suites | saturation warning with the Wilson floor + minimum detectable regression; `EvalSuite(purpose="regression")` flips the warning's direction |
| Read the transcripts | `multivon-eval view --dir runs/` — per-case judge reasons, diff two runs with McNemar p |
| Give the judge an explicit "Unknown" way out | hedged judge verdicts parse as UNKNOWN — excluded from the QAG score denominator and disclosed; all-UNKNOWN raises `JUDGE_ERROR`, never a silent 0.0 |
| Grade each dimension with an isolated judge | QAG scoring — binary per-dimension questions instead of one 1–10 rating |
| Calibrate the judge against human labels | `suite.calibrate(labeled_pairs)` + shipped per-(judge × evaluator) thresholds with provenance ([`_calibration_data/v2.json`](multivon_eval/_calibration_data/v2.json)) |

The first three rows and the judge-UNKNOWN row shipped in 0.16.0 (see [What's new in 0.16.0](#whats-new-in-0160)); every row above is on PyPI today.

**Index:**
[Quickstart](#quickstart--30-seconds-no-api-key) ·
[Why we exist](#why-we-exist) ·
[Demystifying evals](#demystifying-evals-operationalized) ·
[What's new 0.16.0](#whats-new-in-0160) ·
[What's new 0.10–0.15](#whats-new-in-010015) ·
[Ecosystem](#the-multivon-ecosystem) ·
[Why multivon-eval](#why-multivon-eval) ·
[Install](#install) ·
[Core concepts](#core-concepts) ·
[Compliance & privacy](#compliance--privacy) ·
[Statistical rigor](#statistical-rigor) ·
[Synthetic data](#synthetic-dataset-generation) ·
[Experiments](#experiment-tracking) ·
[CLI](#cli) ·
[CI/CD](#cicd-integration) ·
[Architecture](#architecture) ·
[Examples](#examples) ·
[Tests](#tests) ·
[Roadmap](#roadmap)

## What's new in 0.16.0

> Released on PyPI 2026-07-13. Full detail in [CHANGELOG.md](CHANGELOG.md) under *0.16.0*. Built from the same [Demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) checklist mapped above.

- **pass@k and pass^k, estimated honestly.** The same `--runs N` data answers two different questions: `report.pass_at_k(k)` — "does at least one of k trials pass?" (capability) — and `report.pass_hat_k(k)` — "do all k trials pass?" (reliability: what a user hitting the feature k times experiences). Unbiased combinatorial / exact hypergeometric estimators — never the upward-biased `(c/n)**k` plug-in — with cluster-bootstrap CIs that resample tasks, not trials (trials within a task are correlated; resampling them fakes precision). When `k > runs` the answer is an honest **UNKNOWN** with a rerun hint — the framework does not extrapolate past the data. `report.lottery_cases()` names the tasks driving the pass@k↔pass^k gap; `report.assert_pass_hat_k(k, min_ci_low)` gates CI on the pass^k CI *lower bound* and fails loudly (never silently passes) when the metric is UNKNOWN.

- **`multivon-eval validate` — grade your graders before blaming the model.** Runs every grader against each task's reference output (`EvalCase.reference_output`, falling back to `expected_output`) without ever calling the model under test. A reference that fails its own graders is `BROKEN_TASK_OR_GRADER` — the task is unsolvable or the grader miscalibrated; either way the agent is innocent. Contrast-twin tasks get a zero-LLM-call discrimination check (`NO_DISCRIMINATION` when a grader passes both the reference and the known-bad twin — zero information); tasks without a reference are listed `UNVALIDATABLE`, never dropped. LLM-judge graders are skipped by default; `--judges` opts in and reports the spend. Exits 1 only on BROKEN. Companion signal in every report: `EvalReport.zero_pass_cases` flags tasks that failed every trial — 0% pass usually means a broken task or grader, not an incapable agent.

- **Saturation monitor.** A suite at 100% can no longer detect improvement, and the report now says so instead of celebrating: `EvalReport.saturated` (all *evaluated* tasks passed — a 100% built on judge outages doesn't count) plus `EvalReport.min_detectable_regression`, with a warning that quantifies the Wilson floor and recommends graduating the suite. `EvalSuite(purpose="capability")` vs `purpose="regression"` flips the warning's direction (a regression suite *below* ceiling warns instead). Always a warning, never a gate.

- **Judge integrity.** Hedged judge replies ("I cannot say yes or no…") no longer parse as YES — they are UNKNOWN, excluded from the QAG score denominator and disclosed in the reason; judge exceptions surface as `JUDGE_ERROR` / `EVALUATOR_ERROR` statuses instead of laundering into 0.0 quality scores; and a new `max_error_rate=` budget on the CI gate stops 90 judge errors + 10 passes from gating green (raises `EvalGateFailure`: "Eval gate INDETERMINATE"). Unset, a `fail_threshold` gate still warns loudly on stderr at ≥ 10% errors.

## What's new in 0.10–0.15

- **Prompt-drift staleness + case provenance (0.10.0).** Your code changes; your eval cases quietly rot. `multivon-eval staleness` diffs a committed `prompt_baseline.json` against a live scan of every prompt call site and names which prompts changed since your cases were written: `CHANGED` (before/after fingerprints, plus the cases bound to that prompt), `REMOVED`, `ADDED`, and `UNKNOWN` for dynamic prompts it refuses to guess at. `staleness stamp` binds cases to call sites. `--fail-on changed,removed` gates CI. Every report opens with a determinacy headline ("N of M call sites statically resolvable") and ends with a blind-spots footer listing what static analysis cannot see.

- **Scanner v3 and a failed gate (0.10.1).** Before claiming drift coverage, we measured how much real prompt traffic static analysis can actually read, across five real repos (aider, gpt-researcher, open-interpreter, letta, pr-agent). We set ourselves a 50% bar. The result was 20.9%, and it's published as-is on [#4](https://github.com/multivon-ai/multivon-eval/issues/4): most real-world prompts are built dynamically, so static analysis tracks call-site add/remove for everything but can verify text drift only where prompts live as constants. That failure decided what 0.11.0 had to be.

- **Runtime prompt recorder (0.11.0, [#9](https://github.com/multivon-ai/multivon-eval/issues/9)).** The way past that 20.9% ceiling: `pytest --record-prompts` (or the `record_prompts()` context manager) intercepts anthropic/openai/litellm calls during an eval run and records the rendered prompt per call site. A `**kwargs` unpack the scanner can only call UNKNOWN is, at call time, real kwargs with real text. Recordings get their own trust tier and stay there: the static scan proves prompt text, recordings prove only the renderings actually observed (reports say "matched k of N previously observed renderings", never "fresh"), and template/external prompts remain honestly out of scope. Fingerprints only by default; nothing leaves your machine. Merge with `staleness baseline --merge-recordings`.

- **Robustness hardening (0.11.1).** We threw malformed inputs, symlink tricks, and unicode edge cases at the staleness/scanner/bootstrap surface. Every failure found was either a crash or, worse, a false report. So: a syntax-broken file now surfaces as `UNSCANNABLE` instead of falsely `REMOVED`, fingerprints are NFC-normalized (`SCANNER_VERSION` 3 → 4), `match`-statement rebinding disqualifies module constants from static resolution, and CLI errors exit 2 with a usable message instead of a traceback. The rule behind all of them: an honest UNKNOWN beats a confident wrong answer.

- **Persona simulator + scaled case generation (0.12.0, [#10](https://github.com/multivon-ai/multivon-eval/issues/10)/[#11](https://github.com/multivon-ai/multivon-eval/issues/11)).** Static multi-turn test scripts break the moment your model answers differently. `multivon-eval simulate` drives the conversation live instead: a persona LLM with a profile, a goal, and a temper talks to your `model_fn`, adapting each turn, and the transcript gets scored by the conversation evaluators plus a goal judge. Every output is labeled "simulated personas — measures behavior under synthetic users, not real traffic", and there's a hard `--budget` ceiling. Separately, `bootstrap --n-seed-cases` now scales to 500 cases behind duplicate and hardness gates, and the report accounts for every reject: "generated 500, accepted 431 — dropped 38 duplicates, 12 malformed".

- **Generation toolkit (0.13.0, [#13](https://github.com/multivon-ai/multivon-eval/issues/13)).** Five ways to make eval data, two of them free. `mutate_cases` applies deterministic robustness mutations (typo and whitespace noise, unicode confusables, punctuation strip, a conservative negation flip) and records whether each mutant should hold the old label or flip it. `cases_from_template` expands a parametric grid over named axes, full product or greedy pairwise. `generate_contrast_pairs` writes a minimally-edited unfaithful twin per case and only keeps it if a judge confirms the verdict actually flipped. Span-grounded doc-QA records the source offsets behind every generated question and can mix in refusal-bait questions whose right answer is "I don't know". And `simulate --export-cases` turns persona transcripts into conversation cases. Every generator stamps provenance, runs through the dedupe gates, and reports its rejects. The `generate` CLI picks up `--mutate`, `--template`/`--axes`/`--sample`, and `--contrast`/`--no-verify`.

- **Input-quality gate (0.14.0, [#14](https://github.com/multivon-ai/multivon-eval/issues/14)).** Garbage in is a quiet failure: a thin or duplicative trace dump still produces a confident-looking suite. `assess_input()` and `multivon-eval assess` run a free, deterministic preflight over four signals — trace count, per-field completeness, near-duplicate ratio, and PII/secret density — and reuse machinery the rest of the framework already trusts, so there are no new dependencies and no LLM call. There is deliberately no 0-100 score, which is the vanity metric the gate exists to prevent. It warns rather than blocks: a clean input passes silently, a flagged one prints a determinacy headline ("2 of 4 signals flagged"), one line per flag, and a footer naming what it did not check. A WARN can't break your CI. The gate runs as a preflight inside `bootstrap` and `generate` before the first paid call; `--skip-input-gate` turns it off but still leaves one line on stderr, so suppression is never silent.

- **`view --dir` report browser (0.15.0, [#15](https://github.com/multivon-ai/multivon-eval/issues/15)).** Point `multivon-eval view --dir runs/` at a folder of report JSONs and get a sortable index of every run — suite, model, when, n, pass rate with a Wilson CI bar, error and flaky badges, cost. Click through to any report rendered exactly as `view` already renders a single file, or diff two runs: pass-rate and avg-score deltas, McNemar p with a significance label, and the regressed cases stacking both runs' judge reasons so you can read why a verdict flipped. It's read-only and runs on the same stdlib server `view` already uses — no new dependencies, fully offline. Single-file `view <report.json>` still works unchanged.

- **`view --dir` fix for Python 3.10/3.11 (0.15.1).** The index renderer used f-strings with quotes and backslashes inside the `{}` expression, which is valid on 3.12+ but a `SyntaxError` on 3.10/3.11 — so `view` broke on the lower half of the supported range (the package minimum is 3.10). A fresh-install check on the CI matrix caught it; the nested markup is now a module constant and `view --dir` works across every supported version.

<details>
<summary><strong>What's new in 0.9.x (older)</strong></summary>

### What's new in 0.9.x

- **`multivon-eval install-skills`** (new in 0.9.8) — one-command installer for the three bundled Claude Code skills (`eval-bootstrap`, `eval-audit`, `eval-explain`). The wheel ships them under `multivon_eval/_skills/`; this CLI symlinks them into `~/.claude/skills/` so `pip install -U multivon-eval` automatically propagates SKILL.md edits.

  ```bash
  multivon-eval install-skills              # symlinks the three skills
  multivon-eval install-skills --dry-run    # preview without touching anything
  multivon-eval install-skills --force      # replace existing entries

  ls ~/.claude/skills/
  # eval-audit  eval-bootstrap  eval-explain
  ```

  See [`multivon_eval/_skills/README.md`](multivon_eval/_skills/README.md) for the full skill catalog and what each one does. Pairs with `multivon-eval bootstrap` (which `eval-bootstrap` wraps as a Claude Code workflow) and the `eval-action` GitHub Action (which `eval-audit` complements on the pre-PR side).

- **Bootstrap CLI expansions** —
  - `--judge-provider ollama` + `--judge-provider litellm` for fully-local bootstrap (was cloud-only before 0.9.4).
  - `--judge-base-url` (0.9.4) for vLLM / LM Studio / custom Ollama endpoints — injects a placeholder API key when paired with `--judge-provider openai`, so OpenAI-compatible servers work without a real key.
  - `--validate` (0.9.0) runs the N-shot judge-noise filter (`auto.validate_adversarial_cases`) on the generated seed cases — drops anything outside the (0.5, 1.0) hardness band. Adds ~$0.03 but removes 20–40% of synthetic noise.
  - `--validate-n-shots` controls the rerun count for `--validate` (default 3).

- **`multivon-eval doctor`** (new in 0.9.0) — preflight your setup. Reports detected API keys, local-judge availability (Ollama / LM Studio / OpenAI-compat base URL), Python + package versions, `~/.multivon/` writeability. `--json` for CI consumers, exit codes `0 / 1 / 2` for hard/soft failures.

- **Self-correction audit trail (0.9.4 → 0.9.7)** — the four-release cadence that produced the F1 0.830 [0.70–0.92] held-out number is documented release-by-release in [CHANGELOG.md](CHANGELOG.md). 0.9.5 corrected the "held-out" framing on a Faithfulness number that was actually in-distribution. 0.9.6 fixed three runtime blockers in the bootstrap-generated template. 0.9.7 caught a threshold-vs-default mismatch that inflated the held-out F1 from 0.830 (calibrated 0.55) to 0.852 (init-time default 0.7) — only the 0.830 figure is defensible as "held-out at the calibrated threshold." See [`benchmarks/README.md`](benchmarks/README.md) Benchmark 4 for the reproducibility note on resolving thresholds at runtime.

### Carried forward from 0.8.x

- **`multivon-eval bootstrap`** — cold-start eval generator. Describe your LLM product + hand over a JSONL of sample traces, get back a runnable `EvalSuite`, 30 adversarial seed cases, thresholds calibrated from your data, and a forwardable `DISCOVERY_REPORT.md`. A few minutes and a few cents per run. PII / secrets redacted locally before any LLM call. Best documented path is the [bootstrap guide](https://docs.multivon.ai/guides/bootstrap).

  ```bash
  multivon-eval bootstrap --product PRODUCT.md --traces TRACES.jsonl --output ./eval-bootstrap/
  ```

- **`multivon_eval.auto` module** — the programmatic primitives the bootstrap CLI composes:
  - `auto_evaluators(case)` — pure-heuristic, infers the recommended evaluator set from `EvalCase` shape. 0 LLM cost, microseconds.
  - `generate_adversarial_cases(seed, mode, n)` — LLM-generated stress cases across 10 named failure modes (`ungrounded_claim`, `jailbreak`, `prompt_injection_direct/indirect`, `tool_injection`, `pii_leakage_invitation`, etc.).
  - `validate_adversarial_cases(cases, baseline, n_shots=3)` — N-shot judge-noise filter. Validated +0.80 mean failure-rate separation between weak vs strong baselines.

- **Reproducible head-to-head** — multivon-eval **F1 0.804 [0.71–0.88]** vs DeepEval **F1 0.586 [0.48–0.68]** on HaluEval-QA, same N=100, same labels, same judge family. The lower bound of our CI clears DeepEval's upper bound. RAGAS errored on the same input. Run it yourself: [eval-framework-benchmark](https://github.com/multivon-ai/eval-framework-benchmark).

### Carried forward from 0.7.x

- **`CaseResult.status` enum** distinguishes `judge_error` / `model_error` / `evaluator_error` from quality failures. `pass_rate` excludes errors from the denominator.
- **Per-evaluator error isolation** — one judge outage no longer crashes the case.
- **JUnit XML output** + `multivon-eval view <report.json>` HTML dashboard + `multivon-eval init` starter templates + `EvalReport.assert_budget(...)` cost/latency gates.

</details>

See [CHANGELOG.md](CHANGELOG.md) for the complete release history.

## The Multivon ecosystem

Five public + one early-access package, all built on a shared evaluation engine:

| Repo | What it is |
|---|---|
| **multivon-eval** (you are here) | Python SDK — 44 evaluators (2 experimental) + `bootstrap` CLI + `multivon_eval.auto` |
| [pdfhell](https://github.com/multivon-ai/pdfhell) | Adversarial PDFs that break AI document readers — procedural ground truth, not LLM-as-judge |
| [multivon-mcp](https://github.com/multivon-ai/multivon-mcp) | MCP server exposing 22 evaluation tools to Claude / Cursor / Cline / OpenCode |
| [eval-action](https://github.com/multivon-ai/eval-action) | GitHub Action — run a suite on every PR, post a comment, gate the merge on regressions |
| [eval-framework-benchmark](https://github.com/multivon-ai/eval-framework-benchmark) | Reproducible head-to-head benchmark vs DeepEval + RAGAS |
| multivon-guard *(early access)*⁺ | Local proxy that catches LLM coding agents leaking secrets / PII before the request hits the wire. [`hello@multivon.ai`](mailto:hello@multivon.ai). |

⁺ multivon-guard is a closed early-access product, not open source — listed for completeness, no public repo yet.

### When NOT to use multivon-eval

| You want… | Use |
|---|---|
| To call evals from inside Claude Code via SKILL files | bundled Claude Code skills — `multivon-eval install-skills` |
| To call evals from Cursor / Cline / Claude Desktop mid-edit | [multivon-mcp](https://github.com/multivon-ai/multivon-mcp) |
| To gate every PR on eval regressions automatically | [eval-action](https://github.com/multivon-ai/eval-action) |
| Adversarial PDF benchmarking with code-based ground truth | [pdfhell](https://github.com/multivon-ai/pdfhell) |
| To see how multivon-eval stacks up against DeepEval / RAGAS | [eval-framework-benchmark](https://github.com/multivon-ai/eval-framework-benchmark) |
| Frontier-model capability studies on a research-grade harness | [Inspect AI](https://inspect.aisi.org.uk/) (UK AI Security Institute) — the reference research harness; if you're publishing model-capability results, start there |
| Tracing dashboards / production observability | [Arize Phoenix](https://phoenix.arize.com/) or [LangSmith](https://www.langchain.com/langsmith) — observability platforms. They trace, we gate — use one *with* multivon-eval, not instead of it |
| Quick agent-trajectory matching inside LangChain, no statistics needed | [agentevals](https://github.com/langchain-ai/agentevals) — trajectory matchers without CIs or significance tests; lighter if that's all you need |
| Just to gate on a single LLM judge call without a suite | call `Faithfulness(...).evaluate(case, output)` directly — overkill to spin up an `EvalSuite` |

> Claude Code skills run inside Claude Code; the MCP server works with any MCP client (Cursor, Cline, Claude Desktop, OpenCode); the GitHub Action runs on every PR. All three call the same evaluators against the same calibration table. The only difference is where the agent lives.

---

```python
# pip install multivon-eval anthropic
# export ANTHROPIC_API_KEY=sk-ant-...

import anthropic
from multivon_eval import EvalSuite, EvalCase

client = anthropic.Anthropic()

def support_bot(prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

suite = EvalSuite("Support Bot Eval")
suite.add_check("Response explains how to resolve the issue")
suite.add_check("Tone is professional and not defensive", threshold=0.8)
suite.add_cases([
    EvalCase(
        input="How do I reset my password?",
        context="Users can reset their password by clicking 'Forgot Password' on the login page.",
    ),
])
report = suite.run(support_bot)
```

```
─────────────────────── Support Bot Eval ───────────────────────
  #  Input                      Output                   Score  Status    Latency
  1  How do I reset my pas...   Click 'Forgot Passwor…   0.92   PASS      843ms

                           By Evaluator
  Evaluator           Avg Score    Pass Rate
  response_explains      0.92        100%
  tone_is_profess…       0.88         88%

╭────────────────────────────────── Summary ───────────────────────────────────╮
│ Total: 1   Passed: 1   Failed: 0                                              │
│ Pass Rate: 100% [20%–100% 95% CI]   Avg Score: 0.90 [0.82–0.96]             │
╰──────────────────────────────────────────────────────────────────────────────╯
  ⚡ Power warning: 1 case(s) — minimum detectable change at 80% power is ~100%.
  Add ≥291 cases to reliably detect a 10pp shift.
```

---

## Why multivon-eval

The question every team eventually hits: did this change make the model better or worse?

| Feature | multivon-eval | DeepEval | RAGAS | Promptfoo |
|---|:---:|:---:|:---:|:---:|
| Plain-English checks (`add_check`) | ✓ | — | — | — |
| Multi-run + flakiness detection | ✓ | — | — | — |
| CI on every report (Wilson + bootstrap) | ✓ | — | — | — |
| Multiple-comparison correction (BH) | ✓ | — | — | — |
| Power warning + dataset size guidance | ✓ | — | — | — |
| Judge calibration against human labels | ✓ | — | — | — |
| QAG scoring (binary questions, not 1-10) | ✓ | — | — | — |
| Agent-native evaluators (8 metrics) | ✓ | ✓ | partial | — |
| LangChain / LangSmith integration | ✓ | ✓ | ✓ | partial |
| Compliance audit trail (EU AI Act / NIST) | ✓ | — | — | — |
| Local PII detection (zero API calls) | ✓ | partial | — | — |
| HTML reports (self-contained, shareable) | ✓ | — | — | — |
| Local-first, no account needed | ✓ | ✓ | ✓ | ✓ |
| Synthetic data generation | ✓ | ✓ | ✓ | — |
| Open source (Apache 2.0) | ✓ | ✓ | ✓ | ✓ |

> Comparison based on each project's public documentation (last reviewed 2026-07-13; revisit every minor release). Inspect AI, Phoenix/LangSmith, and agentevals are different categories — research harness, observability platforms, trajectory matchers — so they're routed in [When NOT to use multivon-eval](#when-not-to-use-multivon-eval) rather than scored in this table. We host these benchmarks open: see [`benchmarks/`](benchmarks/) for code + datasets and [`benchmarks/results/`](benchmarks/results/) for the raw output JSON. Found something wrong? [Open an issue](https://github.com/multivon-ai/multivon-eval/issues) — we'll fix it.

### Numbers, not adjectives

Hallucination detection, HaluEval QA, N=100, claude-haiku-4-5 judge, human labels:

| Evaluator | Precision | False positives | F1 |
|---|---:|---:|---:|
| **multivon-eval (QAG)** | **0.788** | **11** | **0.804** |
| DeepEval (GPT-4o-mini)  | 0.456 | 49 | 0.586 |
| Simple LLM judge (1-10) | 0.617 | 31 | 0.763 |
| Keyword overlap         | 0.605 | 15 | 0.523 |

Multi-judge agreement on the same task, N=50, all judges temperature=0:

| Judge | Accuracy vs human | Precision | F1 |
|---|---:|---:|---:|
| **gemini-2.5-flash**  | **0.860** | **0.950** | **0.844** |
| gpt-4o-mini           | 0.820 | 0.900 | 0.800 |
| claude-haiku-4-5      | 0.800 | 0.895 | 0.773 |
| gpt-4o                | 0.780 | 0.792 | 0.776 |
| claude-sonnet-4-6     | 0.720 | 0.720 | 0.720 |

Pairwise Cohen's κ across the 5 judges: 0.60–0.80 (substantial on most pairs). Calibration provenance + per-(judge × evaluator) thresholds ship in [`multivon_eval/_calibration_data/v2.json`](multivon_eval/_calibration_data/v2.json). `gemini-2.5-flash` leads on every metric in this run; `claude-haiku-4-5` and `gpt-4o-mini` are close seconds with cheaper tokens. Pick by your latency / cost / sovereignty constraints — all three are first-class providers.

**Cost / latency** ([`benchmarks/results/cost_latency.json`](benchmarks/results/cost_latency.json)) — 50 HaluEval QA cases × 4 LLM-judge evaluators with `claude-haiku-4-5`, `workers=1`:

| Metric | Value |
|---|---|
| Cost per case (4 evaluators) | **$0.00127** |
| Total cost for the run | $0.0635 |
| Judge calls per case | 17.1 (QAG produces 3 questions × 4 evaluators + verification) |
| Wall clock for 50 cases | 15 min |
| Linear extrapolation to 5,000 cases | $6.35 |

**Cache hit speedup** ([`benchmarks/results/reproducibility.json`](benchmarks/results/reproducibility.json)) — same suite, sequential reruns with `set_cache(JudgeCache(...))` installed:

| Run | Wall clock | Judge calls |
|---|---|---|
| Rep 1 (cold) | 2.9 s | 4 |
| Rep 2 (hot)  | 0 ms | 0 |

Cache speedup on the rep-1→rep-2 transition: **2,271×** — read that as "paid API calls vs local cache hits (4 calls → 0)", so a large ratio is expected by construction, not a model-quality claim. The point is that cache hits produce identical scores by construction — flake-proof reruns. `set_cache()` auto-enables caching for every subsequent `JudgeConfig`; no need to thread `cache=True` through every evaluator.

### What makes `multivon-eval` different

| | What it is | One-line why |
|---|---|---|
| **QAG scoring** | Binary yes/no questions instead of 1-10 ratings | Eliminates scale ambiguity, fully auditable — every score traces to specific questions that passed or failed |
| **Plain-English checks** | `suite.add_check("Response explains the return policy")` | No evaluator class to pick, no prompt to craft. Questions auto-generated; pin them for reproducible CI |
| **Bootstrap CLI** | `multivon-eval bootstrap` (new in 0.8.0) | Cold-start from product description + traces → tuned suite in 60s |
| **Agent-native** | Tool-call accuracy, plan quality, step faithfulness, task completion | Works with traces from any framework (LangChain, LlamaIndex, OpenAI Agents SDK, custom) |
| **Four tiers** | Deterministic / LLM-judge / agent-trace / conversation | Mix freely; pay for LLM calls only where they matter |
| **Reliability + flakiness** | `suite.run(runs=5)` + pass@k / pass^k + statistical significance | Detect cases that pass sometimes and fail others; tells you regressions from noise; pass^k reports what a user hitting the feature k times experiences |
| **Statistical rigor** | Wilson CIs, bootstrap, p10/p50/p90, power warnings, BH correction | NAACL 2025: single-run eval scores are unreliable. CIs ship by default |
| **No cold-start** | `generate_from_file("docs/")` synthesises cases | No labeled data required to start |
| **Local-first compliance** | `PIIEvaluator` + `SchemaEvaluator` + `ComplianceReporter` | Hash-chained audit trails, EU AI Act / NIST AI RMF mappings, `EvalSuite.eu_ai_act_high_risk()` factory |
| **Experiment tracking** | `Experiment.record(report)` + `compare(a, b)` | p-values, CIs, McNemar across runs |
| **Cache** | `set_cache(JudgeCache(...))` — once | Reruns are free by construction (4 judge calls → 0; measured 2,271× on rep-2), identical scores guaranteed |

---

## Install

```bash
pip install multivon-eval
```

```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

### Claude Code skills (optional)

If you use [Claude Code](https://claude.com/claude-code), wire up the three bundled skills with one command:

```bash
multivon-eval install-skills        # symlinks eval-bootstrap / eval-audit / eval-explain into ~/.claude/skills/
```

What each one does:

- **`eval-bootstrap`** — auto-invoked when Claude Code detects an LLM-touching codebase without an eval directory. Wraps the bootstrap CLI in a Claude Code workflow that fills in the stub model from the project's existing call sites.
- **`eval-audit`** — auto-invoked between `/review` and `/ship` on diffs touching prompts / model calls / tool defs. Runs only the eval cases that stress the changed surface, blocks safety-class regressions.
- **`eval-explain`** — auto-invoked after `/eval-bootstrap` (and on phrases like "why did multivon pick X"). Answers in three sentences using the DISCOVERY_REPORT.md rationale.

Full details in [`multivon_eval/_skills/README.md`](multivon_eval/_skills/README.md). Run `multivon-eval install-skills --help` for the `--dry-run` / `--force` flags.

---

## Core concepts

Three primitives, one runner:

```python
from multivon_eval import EvalSuite, EvalCase, Faithfulness, NotEmpty

case = EvalCase(
    input="What caused the 2008 financial crisis?",
    expected_output="Subprime mortgage collapse...",
    context="The 2008 crisis was triggered by widespread mortgage defaults...",
    tags=["finance"],
)

suite = EvalSuite("My eval")
suite.add_cases([case])
suite.add_evaluators(NotEmpty(), Faithfulness(threshold=0.7))

# Serial / parallel / async / multi-run — pick what fits
report = suite.run(model_fn, fail_threshold=0.85)
report = suite.run(model_fn, workers=8)
report = suite.run(model_fn, runs=5)                 # flakiness detection
report = await suite.run_async(model_fn, concurrency=10)

report.save_json("results.json")    # also save_csv, save_html, save_junit_xml
```

Agent cases use `agent_trace=[AgentStep(...)]` + `expected_tool_calls=[...]`. Conversation cases use `conversation=[{"role": ..., "content": ...}]`. Load existing datasets with `load("cases.jsonl")` or `load("cases.csv")`.

In the vocabulary of Anthropic's [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents): an `EvalCase` is a *task*, each of `runs=N` is a *trial*, an `Evaluator` is a *grader*, and `agent_trace` is the *transcript* — see the [mapping table](#demystifying-evals-operationalized) above.

> **ToolCallAccuracy three-shape semantics** (0.9.0): `expected_tool_calls=None` skips the case (no expectation set), `expected_tool_calls=[]` asserts "no tools should have been called" (and a non-empty trace fails), and `expected_tool_calls=[...]` checks the trace contains the named calls in order. The skip variant is treated as `skipped-pass` in the report, not `0.0` — see the [`integrations/`](multivon_eval/integrations/) tracers (`LangGraphTracer`, `OpenAIAgentsTracer`, `ManualTracer`) for how each tracer populates `agent_trace`.

### Evaluators — 44 across 7 tiers

| Tier | Examples | Cost |
|---|---|---|
| **Deterministic** | `NotEmpty`, `ExactMatch`, `Contains`, `RegexMatch`, `JSONSchemaEval`, `WordCount`, `BLEU`, `ROUGE`, `Latency`, `BERTScore`, `Levenshtein`, `ChrfScore` | Free, instant |
| **LLM-judge (QAG)** | `Faithfulness`, `Hallucination`, `Relevance`, `Coherence`, `Toxicity`, `Bias`, `AnswerAccuracy`, `ContextPrecision`, `ContextRecall`, `CustomRubric`, `GEval`, `CheckEvaluator` | ~$0.001 / case |
| **Agent-trace** | `ToolCallAccuracy`, `ToolArgumentAccuracy`, `ToolCallNecessity`, `TrajectoryEfficiency`, `AgentMemoryEval`, `PlanQuality`, `TaskCompletion`, `StepFaithfulness` | LLM-judge subset |
| **Compliance** | `PIIEvaluator` (zero API calls, multi-jurisdiction), `SchemaEvaluator` (Pydantic + JSON Schema) | Free |
| **Conversation** | `ConversationRelevance`, `KnowledgeRetention`, `ConversationCompleteness`, `TurnConsistency` | LLM-judge |
| **Multimodal** *(experimental)* | `VQAFaithfulness`, `DocumentGrounding` | LLM-judge |
| **Consistency** | `SelfConsistency` | LLM-judge |

The 2 multimodal evaluators are counted in the 44 but are still experimental — they have not been through the calibration pipeline the text evaluators use. **Full reference + signatures + examples per evaluator:** [docs.multivon.ai/evaluators](https://docs.multivon.ai/evaluators).

---

## Compliance & privacy

For regulated industries (healthcare, finance, legal) where traces can't leave your environment.

- **`PIIEvaluator`** — local regex-only detection across GDPR, CCPA, HIPAA, DPDP (India), PIPEDA jurisdictions. Email, phone, SSN, credit card (Luhn), passport, IBAN, Aadhaar (Verhoeff), PAN. `redact=True` masks in the report. Zero LLM calls.
- **`SchemaEvaluator`** — validates outputs against Pydantic models or JSON Schema with per-field failures. Based on StructEval (2025): GPT-4 fails complex structured extraction ~12% of the time even with explicit format instructions.
- **`ComplianceReporter`** — append-only hash-chained NDJSON audit log (`prev_hash` linked, SHA-256). The chain detects accidental edits; for tamper *evidence* against a deliberate rewrite, anchor the head hash externally (e.g. in a signed commit or timestamping service). Each result annotated with EU AI Act articles (9(2)(b), 10, 15) or NIST AI RMF subcategories. `reporter.coverage(suite)` surfaces uncovered controls before you ship. `EvalSuite.eu_ai_act_high_risk()` factory + `for_regulated(jurisdiction="hipaa")`.

```python
from multivon_eval import EvalSuite, ComplianceReporter

suite = EvalSuite.eu_ai_act_high_risk(jurisdiction="gdpr")
reporter = ComplianceReporter(output_dir="./audit", framework="eu-ai-act")
reporter.record(suite.run(model_fn, runs=5))
reporter.verify(suite.name)  # hash-chain check — detects accidental edits; anchor the head hash externally for tamper evidence
```

**Full reference:** [docs.multivon.ai/compliance](https://docs.multivon.ai/compliance) — jurisdictions, Article mappings, audit-pack generation, sample-audit-pack download.

---

## Statistical rigor

Backed by NAACL 2025: single-run eval scores are unreliable — variance is large enough to reverse model rankings.

```
Pass Rate: 80% [69%–89% 95% CI]   Avg Score: 0.82 [0.74–0.90]
Score distribution  p10:0.41  p50:0.88  p90:0.96
⚡ Power warning: 12 cases — minimum detectable change at 80% power is ~45%.
```

What ships by default in every report:

- **Wilson 95% CI** on pass rate · **bootstrap 95% CI** on avg score
- **pass@k / pass^k** (0.16.0) — `report.pass_at_k(k)` (capability: at least one of k trials) and `report.pass_hat_k(k)` (reliability: all k trials) from `runs=N` data, unbiased estimators with cluster-bootstrap CIs; honest UNKNOWN when `k > runs`; `assert_pass_hat_k(k, min_ci_low)` gates on the CI lower bound
- **Saturation monitor** (0.16.0) — a 100% suite reports its Wilson floor and minimum detectable regression instead of celebrating; `EvalSuite(purpose="regression")` for graduated suites
- **p10 / p50 / p90 percentiles** — exposes bimodal distributions that `avg_score` hides
- **Power warning** when your test set is too small to detect the shift you care about
- **`runs_needed(delta=0.10)` + `min_detectable_effect(n=50)`** for sample-size sizing
- **Benjamini-Hochberg correction** auto-applied in `exp.compare()` for multi-evaluator runs
- **Judge calibration** — `suite.calibrate(labeled_pairs)` reports F1 vs human labels per evaluator. Shipped calibration table in [`_calibration_data/v2.json`](multivon_eval/_calibration_data/v2.json) with per-(judge × evaluator) thresholds (F1 0.58–1.00 range)
- **Judge reliability check** — `JudgeConfig(reliability_check=True)` flags non-determinism in the judge itself

**Full reference:** [docs.multivon.ai/guides/statistical-rigor](https://docs.multivon.ai/guides/statistical-rigor).

---

## Synthetic dataset generation

No labeled data? Point `generate_from_file()` at your docs:

```python
from multivon_eval import generate_from_file, generate_hallucination_pairs

cases = generate_from_file("docs/faq.md", n=20, task="qa")
cases = generate_from_file("docs/whitepaper.txt", n=10, task="summarization")
pairs = generate_hallucination_pairs(my_docs, n=20)
```

CLI: `multivon-eval generate --from docs/faq.md --n 20 --task qa --output cases.jsonl`.

For more sophisticated cold-start, the **`multivon-eval bootstrap`** CLI composes generation + heuristic anchoring + N-shot judge-noise filtering into one command — see [What's new in 0.9.x](#whats-new-in-09x) above for the full flag set (including 0.9.4's `--judge-base-url` and 0.9.0's `--validate`) and the [bootstrap guide](https://docs.multivon.ai/guides/bootstrap). Run `multivon-eval bootstrap --help` for the canonical flag reference.

---

## Experiment tracking

Record every run, compare across model / prompt versions, surface regressions before they ship. Stored locally in `~/.multivon/experiments/` — no cloud, no account.

```python
from multivon_eval import Experiment

exp = Experiment("rag-pipeline")
run_a = exp.record(suite.run(old_model_fn), tags={"prompt_v": "2"})
run_b = exp.record(suite.run(new_model_fn), tags={"prompt_v": "3"})
exp.compare(run_a, run_b)  # prints CIs + McNemar p + BH-corrected per-evaluator deltas
```

CLI: `multivon-eval experiments list / history / compare`.

**Full reference:** [docs.multivon.ai/guides/experiments](https://docs.multivon.ai/guides/experiments).

---

## CLI

```bash
multivon-eval init -t <template> -d <dir>     # scaffold a starter eval suite (templates: quickstart, agent, rag, regulated, conversation, agent-langgraph, agent-openai-sdk)
multivon-eval run eval.py                     # execute an eval file
multivon-eval validate eval.py                # grade your graders: run evaluators against each task's reference output — exits 1 on BROKEN_TASK_OR_GRADER (0.16.0)
multivon-eval report results.json             # print a saved JSON report
multivon-eval view results.json [--open]      # render the JSON as an HTML dashboard
multivon-eval view --dir runs/                # browse a folder of reports — sortable index, open any, diff two
multivon-eval compare a.json b.json           # diff two reports, McNemar + BH-corrected per-evaluator deltas
multivon-eval generate --from docs/ --n 20    # synthetic case generation from a file/dir
multivon-eval generate --mutate cases.jsonl   # deterministic robustness mutations (also --template/--axes, --contrast)
multivon-eval assess traces.jsonl             # free preflight: trace count, completeness, near-dups, PII — before you spend
multivon-eval bootstrap --product PRODUCT.md --traces TRACES.jsonl   # cold-start a tuned suite
multivon-eval doctor [--json]                 # preflight: API keys, local judges, versions, dirs
multivon-eval install-skills [--dry-run] [--force]    # symlink the three Claude Code skills
multivon-eval experiments list | history <name> | compare <run_a> <run_b>
multivon-eval attribution scan <repo> | diff <base> <head>   # Phase 1 prompt-fingerprint diff
multivon-eval staleness . [baseline|stamp]    # which prompts changed since your cases were authored — drift report / bless a baseline / bind cases to call sites
multivon-eval simulate --model-cmd model.py --personas p.jsonl   # persona-driven adaptive multi-turn eval, scored by the conversation evaluators
```

`multivon-eval --help` enumerates every flag. Each subcommand has its own `--help` with examples.

---

## CI/CD integration

```python
# eval.py
report = suite.run(model_fn, fail_threshold=0.85)  # exits 1 if < 85% pass
```

```yaml
# .github/workflows/eval.yml
- name: Run evals
  run: python eval.py
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

## Architecture

```
EvalSuite.run(model_fn)
  → for each case: model_fn(case.input) → output
  → for each evaluator: deterministic | LLM-judge (QAG) | agent-trace | conversation
  → EvalReport (CaseResults + per-evaluator scores + CIs + rich terminal report)
  → save_json / save_csv / save_html / save_junit_xml
```

**Judges:** `claude-haiku-4-5` by default (configurable via `JUDGE_MODEL` + `JUDGE_PROVIDER`). Local + self-hosted models supported via `OPENAI_BASE_URL` (Ollama, LM Studio, vLLM, any OpenAI-compatible server). Per-(judge × evaluator) thresholds calibrated against human-labeled benchmarks — see [`_calibration_data/v2.json`](multivon_eval/_calibration_data/v2.json) for the shipped table with provenance.

---

## Examples

| File | What it shows |
|------|--------------|
| [`basic_eval.py`](examples/basic_eval.py) | Deterministic evaluators only — zero API cost, instant sanity check |
| [`rag_eval.py`](examples/rag_eval.py) | Faithfulness + hallucination for RAG pipelines |
| [`ci_eval.py`](examples/ci_eval.py) | CI/CD integration — `fail_threshold` exits 1 on regression |
| [`check_eval.py`](examples/check_eval.py) | `add_check()` — write criteria in English, no evaluator class needed |
| [`agent_eval.py`](examples/agent_eval.py) | Agent tool call accuracy with `ManualTracer` — surfaces flaky tool selection |

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full shipped + in-flight list. The headline open items: LlamaIndex / CrewAI tracers, `@eval_case` pytest decorator, multi-LLM bootstrap consultation, tiered cost optimizer. File an issue if you want one prioritized.

---

## Contributing

Issues and PRs welcome.

**Small changes** (docs, bug fixes): open a PR directly.
**Large changes** (new evaluators, architecture): open an issue first.

```bash
git clone https://github.com/multivon-ai/multivon-eval
cd multivon-eval
pip install -e ".[dev]"
pytest tests/
```

---

## License

Apache 2.0 — built by [Multivon](https://multivon.ai)
