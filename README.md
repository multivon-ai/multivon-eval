# multivon-eval

[![PyPI](https://img.shields.io/pypi/v/multivon-eval.svg)](https://pypi.org/project/multivon-eval)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/multivon-eval)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/multivon-eval/month)](https://pepy.tech/project/multivon-eval)
[![Tests](https://github.com/multivon-ai/multivon-eval/actions/workflows/test.yml/badge.svg)](https://github.com/multivon-ai/multivon-eval/actions/workflows/test.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multivon-ai/multivon-eval/blob/main/notebooks/quickstart.ipynb)

**[Docs](https://docs.multivon.ai)** · [Website](https://multivon.ai) · [PyPI](https://pypi.org/project/multivon-eval) · [Changelog](CHANGELOG.md) · [Benchmark vs DeepEval + RAGAS](https://github.com/multivon-ai/eval-framework-benchmark)

**AI evaluation for teams that ship models to production.**

Run structured evals over your AI outputs — from simple string checks to LLM-as-judge scoring to agent trace validation — with a clean Python API, beautiful terminal reports, and CI/CD integration out of the box. **New in 0.8.x:** `multivon-eval bootstrap` proposes a tuned eval suite from your product description + sample traces, in 60 seconds.

## Quickstart — 30 seconds, no API key

```bash
pip install multivon-eval
python -m multivon_eval                       # runs a demo eval — no setup
multivon-eval init -t quickstart -d my-eval   # scaffold your own (offline)
cd my-eval && python eval.py
```

That's it. The `quickstart` template uses only deterministic evaluators (`NotEmpty`, `Contains`, `WordCount`) so the first eval runs without an API key.

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

## What's new in 0.8.x

- **`multivon-eval bootstrap`** — cold-start eval generator. Describe your LLM product + hand over a JSONL of sample traces, get back a runnable `EvalSuite` + 30 adversarial seed cases + thresholds calibrated from your data + a forwardable `DISCOVERY_REPORT.md`. ~60 seconds, ~$0.12 per run. PII / secrets redacted locally before any LLM call. Best documented path is the [bootstrap guide](https://docs.multivon.ai/guides/bootstrap).

  ```bash
  multivon-eval bootstrap --product PRODUCT.md --traces TRACES.jsonl --output ./eval-bootstrap/
  ```

- **`multivon_eval.auto` module** — the programmatic primitives the bootstrap CLI composes:
  - `auto_evaluators(case)` — pure-heuristic, infers the recommended evaluator set from `EvalCase` shape. 0 LLM cost, microseconds.
  - `generate_adversarial_cases(seed, mode, n)` — LLM-generated stress cases across 10 named failure modes (`ungrounded_claim`, `jailbreak`, `prompt_injection_direct/indirect`, `tool_injection`, `pii_leakage_invitation`, etc.).
  - `validate_adversarial_cases(cases, baseline, n_shots=3)` — N-shot judge-noise filter. Validated +0.80 mean failure-rate separation between weak vs strong baselines.

- **Reproducible head-to-head** — multivon-eval F1 **0.79** vs DeepEval **0.0** at default thresholds, **0.85** vs **0.59** at best-tuned thresholds, RAGAS errored. Run it yourself: [eval-framework-benchmark](https://github.com/multivon-ai/eval-framework-benchmark).

### Carried forward from 0.7.x

- **`CaseResult.status` enum** distinguishes `judge_error` / `model_error` / `evaluator_error` from quality failures. `pass_rate` excludes errors from the denominator.
- **Per-evaluator error isolation** — one judge outage no longer crashes the case.
- **JUnit XML output** + `multivon-eval view <report.json>` HTML dashboard + `multivon-eval init` starter templates + `EvalReport.assert_budget(...)` cost/latency gates.

See [CHANGELOG.md](CHANGELOG.md) for the complete release history.

## The Multivon ecosystem

Five public + one early-access package, all built on a shared evaluation engine:

| Repo | What it is |
|---|---|
| **multivon-eval** (you are here) | Python SDK — 44 evaluators + `bootstrap` CLI + `multivon_eval.auto` |
| [pdfhell](https://github.com/multivon-ai/pdfhell) | Adversarial PDFs that break AI document readers — procedural ground truth, not LLM-as-judge |
| [multivon-mcp](https://github.com/multivon-ai/multivon-mcp) | MCP server exposing 22 evaluation tools to Claude / Cursor / Cline / OpenCode |
| [eval-action](https://github.com/multivon-ai/eval-action) | GitHub Action — run a suite on every PR, post a comment, gate the merge on regressions |
| [eval-framework-benchmark](https://github.com/multivon-ai/eval-framework-benchmark) | Reproducible head-to-head benchmark vs DeepEval + RAGAS |
| multivon-guard *(early access)* | Local proxy that catches LLM coding agents leaking secrets / PII before the request hits the wire. [`hello@multivon.ai`](mailto:hello@multivon.ai). |

### When NOT to use multivon-eval

| You want… | Use |
|---|---|
| To call evals from inside Claude Code / Cursor mid-edit | [multivon-mcp](https://github.com/multivon-ai/multivon-mcp) |
| To gate every PR on eval regressions automatically | [eval-action](https://github.com/multivon-ai/eval-action) |
| Adversarial PDF benchmarking with code-based ground truth | [pdfhell](https://github.com/multivon-ai/pdfhell) |
| To see how multivon-eval stacks up against DeepEval / RAGAS | [eval-framework-benchmark](https://github.com/multivon-ai/eval-framework-benchmark) |
| Just to gate on a single LLM judge call without a suite | call `Faithfulness(...).evaluate(case, output)` directly — overkill to spin up an `EvalSuite` |

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

Every team building AI products hits the same problem: **how do you know if your model is getting better or worse?**

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

> Comparison based on each project's public documentation as of May 2026. We host these benchmarks open: see [`benchmarks/`](benchmarks/) for code + datasets and [`benchmarks/results/`](benchmarks/results/) for the raw output JSON. Found something wrong? [Open an issue](https://github.com/multivon-ai/multivon-eval/issues) — we'll fix it.

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

Cache speedup on the rep-1→rep-2 transition: **2,271×**. Cache hits also produce identical scores by construction — flake-proof reruns. `set_cache()` auto-enables caching for every subsequent `JudgeConfig`; no need to thread `cache=True` through every evaluator.

### What makes `multivon-eval` different

| | What it is | One-line why |
|---|---|---|
| **QAG scoring** | Binary yes/no questions instead of 1-10 ratings | Eliminates scale ambiguity, fully auditable — every score traces to specific questions that passed or failed |
| **Plain-English checks** | `suite.add_check("Response explains the return policy")` | No evaluator class to pick, no prompt to craft. Questions auto-generated; pin them for reproducible CI |
| **Bootstrap CLI** | `multivon-eval bootstrap` (new in 0.8.0) | Cold-start from product description + traces → tuned suite in 60s |
| **Agent-native** | Tool-call accuracy, plan quality, step faithfulness, task completion | Works with traces from any framework (LangChain, LlamaIndex, OpenAI Agents SDK, custom) |
| **Four tiers** | Deterministic / LLM-judge / agent-trace / conversation | Mix freely; pay for LLM calls only where they matter |
| **Reliability + flakiness** | `suite.run(runs=5)` + statistical significance | Detect cases that pass sometimes and fail others; tells you regressions from noise |
| **Statistical rigor** | Wilson CIs, bootstrap, p10/p50/p90, power warnings, BH correction | NAACL 2025: single-run eval scores are unreliable. CIs ship by default |
| **No cold-start** | `generate_from_file("docs/")` synthesises cases | No labeled data required to start |
| **Local-first compliance** | `PIIEvaluator` + `SchemaEvaluator` + `ComplianceReporter` | Hash-chained audit trails, EU AI Act / NIST AI RMF mappings, `EvalSuite.eu_ai_act_high_risk()` factory |
| **Experiment tracking** | `Experiment.record(report)` + `compare(a, b)` | p-values, CIs, McNemar across runs |
| **Cache** | `set_cache(JudgeCache(...))` — once | 2,271× speedup on rep-2 (4 judge calls → 0), identical scores guaranteed |

---

## Install

```bash
pip install multivon-eval
```

```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

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

### Evaluators — 44 across 7 tiers

| Tier | Examples | Cost |
|---|---|---|
| **Deterministic** | `NotEmpty`, `ExactMatch`, `Contains`, `RegexMatch`, `JSONSchemaEval`, `WordCount`, `BLEU`, `ROUGE`, `Latency`, `BERTScore`, `Levenshtein`, `ChrfScore` | Free, instant |
| **LLM-judge (QAG)** | `Faithfulness`, `Hallucination`, `Relevance`, `Coherence`, `Toxicity`, `Bias`, `AnswerAccuracy`, `ContextPrecision`, `ContextRecall`, `CustomRubric`, `GEval`, `CheckEvaluator` | ~$0.001 / case |
| **Agent-trace** | `ToolCallAccuracy`, `ToolArgumentAccuracy`, `ToolCallNecessity`, `TrajectoryEfficiency`, `AgentMemoryEval`, `PlanQuality`, `TaskCompletion`, `StepFaithfulness` | LLM-judge subset |
| **Compliance** | `PIIEvaluator` (zero API calls, multi-jurisdiction), `SchemaEvaluator` (Pydantic + JSON Schema) | Free |
| **Conversation** | `ConversationRelevance`, `KnowledgeRetention`, `ConversationCompleteness`, `TurnConsistency` | LLM-judge |
| **Multimodal** | `VQAFaithfulness`, `DocumentGrounding` | LLM-judge |
| **Consistency** | `SelfConsistency` | LLM-judge |

**Full reference + signatures + examples per evaluator:** [docs.multivon.ai/evaluators](https://docs.multivon.ai/evaluators).

---

## Compliance & privacy

For regulated industries (healthcare, finance, legal) where traces can't leave your environment.

- **`PIIEvaluator`** — local regex-only detection across GDPR, CCPA, HIPAA, DPDP (India), PIPEDA jurisdictions. Email, phone, SSN, credit card (Luhn), passport, IBAN, Aadhaar (Verhoeff), PAN. `redact=True` masks in the report. Zero LLM calls.
- **`SchemaEvaluator`** — validates outputs against Pydantic models or JSON Schema with per-field failures. Based on StructEval (2025): GPT-4 fails complex structured extraction ~12% of the time even with explicit format instructions.
- **`ComplianceReporter`** — hash-chained NDJSON audit log (`prev_hash` linked, SHA-256). Each result annotated with EU AI Act articles (9(2)(b), 10, 15) or NIST AI RMF subcategories. `reporter.coverage(suite)` surfaces uncovered controls before you ship. `EvalSuite.eu_ai_act_high_risk()` factory + `for_regulated(jurisdiction="hipaa")`.

```python
from multivon_eval import EvalSuite, ComplianceReporter

suite = EvalSuite.eu_ai_act_high_risk(jurisdiction="gdpr")
reporter = ComplianceReporter(output_dir="./audit", framework="eu-ai-act")
reporter.record(suite.run(model_fn, runs=5))
reporter.verify(suite.name)  # tamper-evident chain check
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
- **p10 / p50 / p90 percentiles** — exposes bimodal distributions that `avg_score` hides
- **Power warning** when your test set is too small to detect the shift you care about
- **`runs_needed(delta=0.10)` + `min_detectable_effect(n=50)`** for sample-size sizing
- **Benjamini-Hochberg correction** auto-applied in `exp.compare()` for multi-evaluator runs
- **Judge calibration** — `suite.calibrate(labeled_pairs)` reports F1 vs human labels per evaluator. Shipped calibration table in [`_calibration_data/v2.json`](multivon_eval/_calibration_data/v2.json) with per-(judge × evaluator) thresholds (F1 0.66–1.00 range)
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

For more sophisticated cold-start, the **`multivon-eval bootstrap`** CLI composes generation + heuristic anchoring + N-shot judge-noise filtering into one command — see [What's new in 0.8.x](#whats-new-in-08x) above and the [bootstrap guide](https://docs.multivon.ai/guides/bootstrap).

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
multivon-eval run eval.py
multivon-eval report results.json
```

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

See [ROADMAP.md](ROADMAP.md) for the full shipped + in-flight list. The headline open items: LlamaIndex / CrewAI tracers, pytest plugin, LiteLLM adapter, tiered cost optimizer, agent simulation. File an issue if you want one prioritized.

---

## Contributing

Issues and PRs welcome.

**Small changes** (docs, bug fixes): open a PR directly.
**Large changes** (new evaluators, architecture): open an issue first.

```bash
git clone https://github.com/multivon-ai/multivon-eval
cd llm-evals
pip install -e ".[dev]"
pytest tests/
```

---

## License

Apache 2.0 — built by [Multivon](https://multivon.ai)
