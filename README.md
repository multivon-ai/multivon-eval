# multivon-eval

[![PyPI](https://img.shields.io/pypi/v/multivon-eval.svg)](https://pypi.org/project/multivon-eval)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/multivon-eval)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/multivon-ai/multivon-eval/actions/workflows/test.yml/badge.svg)](https://github.com/multivon-ai/multivon-eval/actions/workflows/test.yml)

**Did your AI application get better—or did the measurement change?**

multivon-eval is a Python library for testing LLM applications, RAG systems,
and agents. Define task-specific cases, grade outputs, compare changes, and
inspect quality failures separately from errors and missing evidence.
Runs and reports stay local; LLM graders call the judge provider you configure.
No hosted account is required.

[Documentation](https://docs.multivon.ai/) · [Examples](examples/README.md) ·
[Benchmarks](benchmarks/README.md) · [Changelog](CHANGELOG.md)

**Current release: 0.18.0 — September 17, 2026.** Python 3.10+, Apache 2.0.
[Migration notes](docs/guides/migration-0-18.mdx).

[Case manifests and trial evidence](docs/guides/versioned-evidence.mdx)
support safer comparisons and regrading. Use Hugging Face for dataset operations,
[Inspect for execution](docs/guides/inspect-integration.mdx), and Multivon's
[acceptance policies](docs/guides/acceptance-policies.mdx) for required checks and
task slices. These features ship in 0.18.0. The
[implementation program](plans/industrial-evaluation.md) tracks unfinished work;
[reuse decisions](plans/reuse-decisions.md) keep the integration boundaries explicit.

## Start in 30 seconds

```bash
pip install multivon-eval
```

Run this complete example. It needs no API key:

```python
from multivon_eval import EvalCase, EvalSuite, ExactMatch

suite = EvalSuite("capital lookup")
suite.add_cases([
    EvalCase(input="Capital of France?", expected_output="Paris"),
    EvalCase(input="Capital of Japan?", expected_output="Tokyo"),
])
suite.add_evaluators(ExactMatch())

# Replace this fixture with your application: a function from str to str.
answers = {"Capital of France?": "Paris", "Capital of Japan?": "Tokyo"}
report = suite.run(
    answers.__getitem__,
    fail_threshold=1.0,
    save_json="results.json",
    save_html="results.html",
    verbose=False,
)
print(f"{report.passed}/{report.evaluated} passed; {report.errors} errors")
# 2/2 passed; 0 errors
```

Open `results.html` to inspect each verdict. The example verifies a small
lookup fixture; it does not establish the quality of a real AI application.
Start your own suite by [defining task success](docs/guides/task-success.mdx).

For development builds, the [Label Studio review bridge](docs/guides/review-labels.mdx)
exchanges saved text/trace trials and preserves review disagreements and missing
coverage. [Saved-score calibration](docs/guides/review-calibration.mdx) reuses
scikit-learn and SciPy for development fitting and held-out source analysis.
[OpenTelemetry interoperability](docs/guides/otel-evidence.mdx) grades retained
OTLP traces and emits standard evaluation events through your existing SDK.
[Environment outcome checks](docs/guides/environment-outcomes.mdx) reuse Gymnasium
and independently observed state to catch missing writes, duplicate writes and
forbidden changes. The [failure investigation workflow](docs/guides/failure-investigation.mdx)
connects saved trial comparison to Label Studio review and development regression
cases. The [vision grader audit](docs/evaluators/multimodal.mdx) corrects empty-claim
perfect scores, invalid-judgment handling and Anthropic SDK 1.x compatibility.
These previews and fixes are not included in PyPI 0.18.0.

## A real workflow example

The [document-to-ledger study](benchmarks/industrial/DOCUMENT_RESULTS.md) reuses
CORD receipts, pdfhell generators and Inspect execution. Models write to SQLite;
independent checks verify the saved state. Both models fail the frozen policy
on 39 held-out source documents. The report separates wrong amounts, missing
posts, transcription-only mismatches and integration errors, with raw-log hashes,
cost accounting and a reproducible protocol. It is a sandbox study, not proof
of production readiness or a new dataset.

## Why use it

| Need | Available today |
|---|---|
| Catch regressions | Deterministic checks, LLM graders, and saved baseline/proposal comparisons. |
| Detect missing evidence | Separate quality failures, model/judge errors, and skipped checks. Active quality gates block errors and skipped coverage by default. |
| Check the graders | Validate reference answers, measure agreement with reviewed labels, and inspect grader reasons. |
| Measure variability | Repeated trials, flakiness, pass@k/pass^k, and confidence intervals with documented assumptions. |
| Keep results portable | Local JSON, HTML, CSV, JUnit, and optional audit artifacts. |

## Pick your path

| Task | Starting point |
|---|---|
| First offline suite | `multivon-eval init -t quickstart -d my-eval` |
| RAG / question answering | `multivon-eval init -t rag` |
| Agent tool use | `multivon-eval init -t agent` |
| LangGraph agent | `multivon-eval init -t agent-langgraph` |
| OpenAI Agents SDK agent | `multivon-eval init -t agent-openai-sdk` |
| Multi-turn conversations | `multivon-eval init -t conversation` |
| Existing logs | [Score recorded outputs](https://docs.multivon.ai/guides/score-logged-outputs) |
| Unsure what to measure | [Bootstrap a starter suite](https://docs.multivon.ai/guides/bootstrap) |

Bootstrap suggests evaluators and synthetic cases. Its p25 threshold suggestions
are provisional score summaries, not calibration against human acceptance labels.
Review them before using them as release criteria.

## Add an LLM judge

Use deterministic checks for exact requirements and judges for qualities that
need interpretation. Install the relevant provider SDK and set its API key;
configure a local judge if you want to avoid hosted calls.

```python
from multivon_eval import EvalCase, EvalSuite, Faithfulness, JudgeConfig, configure

configure(JudgeConfig(provider="anthropic", model="claude-haiku-4-5"))
suite = EvalSuite("policy answers")
suite.add_cases([EvalCase(
    input="What is the refund window?",
    context="Refunds are available within 30 days of purchase.",
)])
suite.add_evaluators(Faithfulness())

# your_app(prompt) must call your application, including its retrieval step.
report = suite.run(your_app, fail_threshold=0.90, save_json="policy-results.json")
```

Judge providers include Anthropic, OpenAI, Google, Ollama, and LiteLLM;
OpenAI-compatible endpoints support local servers. See
[judge configuration](https://docs.multivon.ai/evaluators/llm-judge) for setup,
historical threshold packs, and the current temperature-forwarding limitation.
A judge score is an estimate to validate on your task, not ground truth.

## Evaluators — 44 across 7 tiers

| Family | Examples | Judge calls? |
|---|---|---|
| Deterministic | `ExactMatch`, `Contains`, `JSONSchemaEval`, `BLEU`, `ROUGE`, `Latency` | No |
| LLM judge | `Faithfulness`, `Hallucination`, `Relevance`, `AnswerAccuracy`, `GEval` | Yes |
| Agent trace | `ToolCallAccuracy`, `ToolArgumentAccuracy`, `TaskCompletion` | Some |
| Conversation | `KnowledgeRetention`, `ConversationCompleteness`, `TurnConsistency` | Yes |
| Compliance checks | `PIIEvaluator`, `SchemaEvaluator` | No |
| Multimodal | `VQAFaithfulness`, `DocumentGrounding` (experimental) | Yes |
| Consistency | `SelfConsistency` | Yes |

[Browse the evaluator reference](https://docs.multivon.ai/evaluators/deterministic).
For tool expectations, `None` means unspecified, `[]` means no calls expected,
and `require_order=True` checks an **ordered subsequence**. A matching trace
alone does not prove the task changed external state correctly.

## Use it in CI

`fail_threshold` checks absolute quality. In 0.17.0, an active gate returns:

| Exit | Meaning |
|---|---|
| `0` | The configured gate passed. |
| `1` | Completed measurements failed the quality threshold. |
| `2` | Evidence is indeterminate: errors, empty runs, or skipped coverage. |

Set `max_error_rate=` explicitly to allow an error budget. Pass `save_json=`,
`save_html=`, or `save_junit_xml=` into `suite.run()` so reports are written
**before** a failing gate raises.

For a saved baseline/proposal comparison:

```bash
multivon-eval compare baseline.json proposal.json --fail-on-regression
```

This also blocks incomplete or unmatched comparisons. It flags case regressions
without waiting for statistical significance; no detected regression is not
proof of equivalence. See [CI/CD](https://docs.multivon.ai/guides/ci-cd) and
[statistical assumptions](https://docs.multivon.ai/guides/statistical-rigor).

## Evidence and limitations

The repository publishes [benchmark scripts and historical results](benchmarks/README.md).
One cross-task measurement reported F1 **0.830** on 60 HaluEval summarization
outputs from 30 source examples, using a QA-selected threshold. It is a small,
maintainer-run result with generated hallucination labels; it does not establish
state-of-the-art accuracy. Historical live benchmarks have not been rerun after
the 0.17.0 grader changes.

Confidence intervals do not correct biased labels, correlated cases, or grader
mistakes. Validate your success criteria and inspect errors, skips, and important
task slices. Optional compliance reports organize evidence; they do not certify
legal compliance.

A case with one measured check and another skipped check still counts as evaluated.
The built-in gate blocks wholly skipped cases; enforce per-check coverage separately
when every evaluator is required.

## Useful commands

```bash
multivon-eval validate eval.py                 # check reference outputs against graders
multivon-eval view --dir runs/                 # browse saved reports
multivon-eval bootstrap --product PRODUCT.md --traces TRACES.jsonl
multivon-eval generate --from docs/faq.md --n 20
multivon-eval assess traces.jsonl              # inspect input quality locally
multivon-eval staleness .                      # inspect prompt/case drift
multivon-eval doctor --no-ping --json          # check configuration offline
```

`doctor` exits 0 when clean, 2 when it finds warnings, and 1 when it finds an error.
Use `multivon-eval --help` for all commands.

## Current release — 0.18.0

- Reuse Hugging Face Datasets for loading; retain versioned case manifests and source groups.
- Retain individual outputs, grader results, retries and traces for inspection and regrading.
- Use Inspect for execution, native provider logs and crash recovery.
- Apply required-check, coverage and slice contracts with accept/reject/indeterminate decisions.
- Block paired significance when identity, retained trials or recorded grader settings are incompatible.
- Preserve missing trace observations and use exact small-sample McNemar tests.

See [migration notes](docs/guides/migration-0-18.mdx), the [worked document study](benchmarks/industrial/DOCUMENT_RESULTS.md),
and the [changelog](CHANGELOG.md). This release does not complete provider accounting,
opaque callback compatibility, customer validation or the remaining industrial program.

## Related tools and contributing

Use [pdfhell](https://github.com/multivon-ai/pdfhell) for adversarial document
fixtures, [multivon-mcp](https://github.com/multivon-ai/multivon-mcp) for MCP access,
and [eval-action](https://github.com/multivon-ai/eval-action) for GitHub workflows.
The library can sit alongside your existing tracing system.

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
setup and testing. Apache 2.0 — [Multivon](https://multivon.ai).
