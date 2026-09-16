# multivon-eval

[![PyPI](https://img.shields.io/pypi/v/multivon-eval.svg)](https://pypi.org/project/multivon-eval)
[![Python](https://img.shields.io/badge/python-3.10–3.14-blue.svg)](https://pypi.org/project/multivon-eval)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/multivon-ai/multivon-eval/actions/workflows/test.yml/badge.svg)](https://github.com/multivon-ai/multivon-eval/actions/workflows/test.yml)

**[Docs](https://docs.multivon.ai/)** · [Examples](examples/README.md) · [Benchmarks](benchmarks/README.md) · [Changelog](CHANGELOG.md) · [Website](https://multivon.ai)

**Know whether your AI product actually got better.** multivon-eval is an
open-source Python framework for evaluating LLM applications, RAG systems,
agents, conversations, and document AI. It validates automated graders,
quantifies uncertainty, and produces evidence your team can inspect locally.

> **Current release: 0.17.0 — September 17, 2026.** Python 3.10–3.14, Apache 2.0,
> no hosted account and no telemetry. See [what changed](#current-release--0170).

## Start in 30 seconds

```bash
pip install multivon-eval
python -m multivon_eval
```

That runs a real deterministic evaluation without an API key. To scaffold an
editable project:

```bash
multivon-eval init -t quickstart -d my-eval
cd my-eval
python eval.py
```

Or use the Python API directly:

```python
from multivon_eval import Contains, EvalCase, EvalSuite, NotEmpty

suite = EvalSuite("support-smoke", purpose="regression")
suite.add_cases([
    EvalCase(input="What is 2+2?", expected_output="4"),
])
suite.add_evaluators(NotEmpty(), Contains(["4"]))

if __name__ == "__main__":
    report = suite.run(lambda prompt: "2+2 = 4", runs=5)
    print(report.pass_rate, report.pass_rate_ci())
    print(report.pass_hat_k(3))
```

Replace the lambda with your model function. The report includes confidence
intervals, repeated-trial reliability, failure reasons, and exportable JSON,
CSV, HTML, and JUnit artifacts.

## Why teams choose it

Most evaluation libraries grade model output. multivon-eval also checks whether
the measurement itself deserves trust.

| Need | What multivon-eval does |
|---|---|
| **Validate the graders** | `multivon-eval validate eval.py` runs graders against reference answers before the model is blamed. |
| **Separate change from noise** | Wilson and bootstrap confidence intervals, power warnings, pass@k/pass^k, flakiness detection, and paired comparisons ship with the report. |
| **Audit LLM judges** | QAG breaks broad ratings into binary questions; thresholds can be calibrated against human labels, with provenance attached. |
| **Keep evidence portable** | Runs stay local and export to inspectable artifacts, including hash-linked compliance logs and offline-verifiable audit packages. |
| **Evaluate more than chat** | The same suite covers RAG, agent traces, multi-turn conversations, structured output, PII, images, and PDFs. |

The design rule is simple: an honest **UNKNOWN** is better than a confident
wrong answer. Judge outages are reported as infrastructure errors, insufficient
samples trigger power warnings, and unsupported statistical questions do not
silently become scores.

## Pick your path

| You are evaluating… | Start here | API key? |
|---|---|---|
| A first offline smoke test | `multivon-eval init -t quickstart` | No |
| A RAG or QA system | `multivon-eval init -t rag` | Judge key or local model |
| A framework-agnostic agent | `multivon-eval init -t agent` | Optional |
| A **LangGraph** agent | `multivon-eval init -t agent-langgraph` | Provider key or local model |
| An **OpenAI Agents SDK** agent | `multivon-eval init -t agent-openai-sdk` | OpenAI |
| A multi-turn conversation | `multivon-eval init -t conversation` | Judge key or local model |
| A regulated workflow | `multivon-eval init -t regulated` | Judge key or local model |

Anthropic, OpenAI, Gemini, Ollama, LM Studio, vLLM, and other
OpenAI-compatible judge endpoints are supported. Agent integrations include
LangChain, LangGraph, LangSmith, the OpenAI Agents SDK, and manual traces.

## How a run works

```text
cases + reference answers
        ↓
your model, one or many trials per case
        ↓
deterministic checks + calibrated judges + trace evaluators
        ↓
validated report: verdicts + reasons + uncertainty + errors
        ↓
local HTML / JSON / CSV / JUnit / CI gate / audit package
```

Three objects make up the core API:

- `EvalCase` describes the input, context, reference output, agent trace, or
  conversation to evaluate.
- `Evaluator` grades one dimension and explains its verdict.
- `EvalSuite` runs cases, coordinates repeated trials, and produces an
  `EvalReport`.

```python
from multivon_eval import EvalCase, EvalSuite, Faithfulness, Relevance

suite = EvalSuite("rag-regression", purpose="regression")
suite.add_evaluators(Faithfulness(), Relevance())
suite.add_cases([
    EvalCase(
        input="What is the renewal period?",
        context="The agreement renews annually unless terminated.",
    )
])

report = suite.run(my_rag_model, runs=5, fail_threshold=0.95)
report.save_html("report.html")
report.save_junit_xml("junit.xml")
```

## Evaluators — 44 across 7 tiers

| Group | Examples | Judge required? |
|---|---|---|
| Deterministic | `ExactMatch`, `Contains`, `JSONSchemaEval`, `BLEU`, `ROUGE`, `Latency` | No |
| LLM judge / QAG | `Faithfulness`, `Hallucination`, `Relevance`, `AnswerAccuracy`, `GEval` | Yes |
| Agent trace | `ToolCallAccuracy`, `ToolArgumentAccuracy`, `PlanQuality`, `TaskCompletion` | Some |
| Conversation | `KnowledgeRetention`, `ConversationCompleteness`, `TurnConsistency` | Yes |
| Compliance | `PIIEvaluator`, `SchemaEvaluator` | No |
| Multimodal | `VQAFaithfulness`, `DocumentGrounding` | Yes |
| Consistency | `SelfConsistency` | Yes |

See the [evaluator reference](https://docs.multivon.ai/evaluators) for signatures,
inputs, costs, and examples. The two multimodal evaluators are experimental and
are not yet covered by the text-evaluator calibration pipeline.

For agent cases, `expected_tool_calls=None` means no expectation was supplied,
`[]` asserts that no tool should be called, and `[...]` lists the expected
calls. `require_order=True` checks them as an **ordered subsequence**, so
unrelated calls may still occur between expected ones.

## Reliability and CI

Every report carries the context needed to interpret its headline score:

- Wilson 95% confidence interval for pass rate and bootstrap interval for the
  average score.
- Power warning and sample-size guidance when the suite cannot detect the
  change you care about.
- pass@k for capability and pass^k for repeated-use reliability.
- Explicit `JUDGE_ERROR` / `EVALUATOR_ERROR` accounting and an optional
  `max_error_rate` gate.
- Paired run comparison with McNemar tests and Benjamini–Hochberg correction.

```python
report = suite.run(model_fn, runs=5, fail_threshold=0.90, max_error_rate=0.05)
report.assert_pass_hat_k(k=3, min_ci_low=0.80)
```

For the estimators, assumptions, and interpretation, read the
[statistical-rigor guide](https://docs.multivon.ai/guides/statistical-rigor) and
[reliability guide](https://docs.multivon.ai/guides/reliability-metrics).

## Evidence, not just claims

The repository includes the benchmark code, datasets, configurations, and raw
results behind its published evaluator claims.

- Held-out hallucination evaluation: **F1 0.830 [0.70–0.92]** after calibrating
  on a different HaluEval split.
- In-distribution HaluEval QA: multivon-eval **F1 0.804 [0.71–0.88]** versus
  DeepEval **0.586 [0.48–0.68]** under the disclosed configurations.
- When a prompt-drift experiment missed its own 50% determinacy gate and scored
  **20.9%**, the failed result was published and the design was changed.

Inspect the [benchmark methodology and raw artifacts](benchmarks/README.md), or
read the project’s public [track record](https://multivon.ai/track-record).

## Local-first compliance

`PIIEvaluator` performs local detection for GDPR, CCPA, HIPAA, DPDP (India),
and PIPEDA patterns. `ComplianceReporter` writes a hash-linked NDJSON record;
anchor the head hash externally when you need evidence against deliberate
rewriting. Audit packages include the run, cases, calibration provenance,
manifest, and offline verifier.

```python
from multivon_eval import ComplianceReporter, EvalSuite

suite = EvalSuite.eu_ai_act_high_risk(jurisdiction="gdpr")
report = suite.run(model_fn, runs=5)
ComplianceReporter("./audit", framework="eu-ai-act").record(report)
```

See the [compliance guide](https://docs.multivon.ai/compliance) and
[audit-package format](https://multivon.ai/security).

## Useful commands

```bash
multivon-eval validate eval.py                 # grade the graders
multivon-eval view --dir runs/                 # browse and compare local reports
multivon-eval compare before.json after.json   # paired regression analysis
multivon-eval bootstrap --product PRODUCT.md --traces TRACES.jsonl
multivon-eval generate --from docs/faq.md --n 20
multivon-eval assess traces.jsonl              # free input-quality preflight
multivon-eval staleness .                      # detect prompt/case drift
multivon-eval doctor --no-ping --json          # offline configuration check
```

Run `multivon-eval --help` for the complete CLI. The
[documentation](https://docs.multivon.ai/) carries the exhaustive flag and API
reference so this README can stay focused on orientation. `doctor` exits 0
when clean, 2 when it finds warnings, and 1 when it finds an error; warnings
are useful diagnostics, not a failed installation.

## Ecosystem

Four public packages plus one closed early-access product:

| Project | Role |
|---|---|
| **multivon-eval** | Core Python evaluation engine |
| [pdfhell](https://github.com/multivon-ai/pdfhell) | Adversarial document-AI benchmark with code-based ground truth |
| [multivon-mcp](https://github.com/multivon-ai/multivon-mcp) | 22 evaluation tools for MCP-compatible agents |
| [eval-action](https://github.com/multivon-ai/eval-action) | GitHub Action for PR evaluation and regression gates |
| multivon-guard *(closed early access)* | Local outbound safety proxy for coding agents |

Use multivon-eval beside an observability platform when you need both production
traces and release gates. For research-grade model capability studies, consider
[Inspect AI](https://inspect.aisi.org.uk/). The longer selection guide is in
[the docs](https://docs.multivon.ai/why-multivon-eval).

## Current release — 0.17.0

Released September 17, 2026:

- Skipped graders no longer count as successful measurements or inflate scores.
- Quality gates reject unmeasured cases and default to zero infrastructure errors;
  set `max_error_rate=` explicitly to permit an error budget.
- Apology prefixes cannot bypass factuality checks. Ambiguous verdicts stay UNKNOWN.
- Comparisons separate infrastructure recovery from model improvements.
- Judge timeouts reach the provider; reliability checks bypass cached verdicts.

See the [migration notes](docs/guides/migration-0-17.mdx) for the stricter gate and
skip semantics. Published benchmark numbers describe historical configurations;
this correctness release does not claim new benchmark accuracy.

Read the complete [changelog](CHANGELOG.md) for release history and migration
details.

## Contributing

Issues and pull requests are welcome. For substantial new evaluators or API
changes, open an issue first so the measurement contract can be reviewed.

```bash
git clone https://github.com/multivon-ai/multivon-eval
cd multivon-eval
pip install -e ".[dev]"
python -m pytest tests/ --ignore=tests/test_integrations_live.py
```

## License

Apache 2.0 — built by [Multivon](https://multivon.ai)
