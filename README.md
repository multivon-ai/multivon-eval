# multivon-eval

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![PyPI](https://img.shields.io/pypi/v/multivon-eval.svg)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multivon-ai/multivon-eval/blob/main/notebooks/quickstart.ipynb)

**[Documentation](https://evaldocs.multivon.ai)** · [Website](https://multivon.ai) · [PyPI](https://pypi.org/project/multivon-eval)

**AI evaluation for teams that ship models to production.**

Run structured evals over your AI outputs — from simple string checks to LLM-as-judge scoring to agent trace validation — with a clean Python API, beautiful terminal reports, and CI/CD integration out of the box.

---

```python
from multivon_eval import EvalSuite, EvalCase, Relevance, Faithfulness, NotEmpty

suite = EvalSuite("Support Bot Eval")
suite.add_cases([
    EvalCase(
        input="How do I reset my password?",
        context="Users can reset their password by clicking 'Forgot Password' on the login page.",
    ),
])
suite.add_evaluators(NotEmpty(), Relevance(), Faithfulness())
report = suite.run(my_model_fn)
```

```
─────────────────────── Support Bot Eval ───────────────────────
  #  Input                      Output                   Score  Status    Latency
  1  How do I reset my pas...   Click 'Forgot Passwor…   0.92   PASS      843ms

                           By Evaluator
  Evaluator       Avg Score    Pass Rate
  not_empty          1.00        100%
  relevance          0.88         88%
  faithfulness       0.87         87%

╭─────────────────────── Summary ───────────────────────╮
│ Total: 1   Passed: 1   Failed: 0   Pass Rate: 100%   │
╰────────────────────────────────────────────────────────╯
```

---

## Why multivon-eval

Every team building AI products hits the same problem: **how do you know if your model is getting better or worse?**

| Feature | multivon-eval | DeepEval | RAGAS | Promptfoo |
|---|:---:|:---:|:---:|:---:|
| Multi-run + flakiness detection | ✓ | — | — | — |
| Wilson CI + power analysis | ✓ | — | — | — |
| Statistical significance in comparisons | ✓ | — | — | — |
| QAG scoring (binary questions, not 1-10) | ✓ | — | — | — |
| Agent-native evaluators (8 metrics) | ✓ | ✓ | partial | — |
| LangChain / LangSmith integration | ✓ | ✓ | ✓ | partial |
| Compliance audit trail (EU AI Act / NIST) | ✓ | — | — | — |
| Local PII detection (zero API calls) | ✓ | partial | — | — |
| HTML reports (self-contained, shareable) | ✓ | — | — | — |
| Local-first, no account needed | ✓ | ✓ | ✓ | ✓ |
| Synthetic data generation | ✓ | ✓ | ✓ | — |
| Open source (Apache 2.0) | ✓ | ✓ | ✓ | ✓ |

`multivon-eval` is different:

**QAG scoring** — Instead of asking a judge "rate this 1-10", we generate yes/no questions about the output and score by the fraction answered correctly. Binary questions are easier for LLMs to get right, fully auditable, and cheaper. [Benchmarked at 65% fewer false positives](https://github.com/multivon-ai/multivon-eval/tree/main/benchmarks) than numeric scoring.

**Agent-native** — Built-in evaluators for tool call accuracy, plan quality, step faithfulness, and task completion. Covers agent traces from any framework (LangChain, LlamaIndex, custom).

**Four tiers** — Deterministic (free, instant), LLM-judge (QAG), agent-trace, and conversation evaluators. Mix and match; pay for LLM calls only where it matters.

**No cold-start** — Generate eval cases from your docs with `generate_from_file()`. No labeled data required to get started.

**Reliability & flakiness detection** — LLMs are non-deterministic. Run each case N times with `suite.run(runs=5)` to detect cases that pass sometimes and fail others. Statistical significance in experiment comparison tells you whether a regression is real or noise.

**Statistical rigor built in** — Wilson score confidence intervals on pass rates. `runs_needed(delta=0.05)` tells you how many test cases you need before a 5% improvement is statistically detectable. Backed by NAACL 2025: single-run eval scores are unreliable.

**Agent trajectory evaluation** — Beyond "did the task complete?": evaluate whether tool calls were necessary, whether the agent took the optimal number of steps, and whether it recovered correctly from tool failures. Plus `AgentMemoryEval` for multi-session agents.

**Local-first compliance** — `PIIEvaluator` detects PII in outputs using local regex patterns (zero API calls). `SchemaEvaluator` validates structured outputs against Pydantic models or JSON Schema with per-field failure breakdowns. `ComplianceReporter` writes tamper-evident NDJSON audit trails mapped to EU AI Act Article 9 and NIST AI RMF controls.

**Experiment tracking** — Record every run, compare across model versions, catch regressions before they reach users. p-values, confidence intervals, and power hints included.

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

### `EvalCase` — A test case

```python
from multivon_eval import EvalCase

case = EvalCase(
    input="What caused the 2008 financial crisis?",          # required
    expected_output="Subprime mortgage collapse...",          # for ExactMatch, Contains
    context="The 2008 crisis was triggered by...",           # for Faithfulness, Hallucination
    tags=["finance", "history"],                             # for filtering reports
    metadata={"source": "test_set_v2", "difficulty": "hard"},
)
```

For agent evals:

```python
from multivon_eval import EvalCase, AgentStep, ToolCall

case = EvalCase(
    input="search for recent AI papers and summarize",
    agent_trace=[
        AgentStep(tool_calls=[ToolCall(name="search", arguments={"query": "AI papers 2025"})]),
        AgentStep(tool_calls=[ToolCall(name="summarize")]),
    ],
    expected_tool_calls=["search", "summarize"],
)
```

### Evaluators

Four tiers — pick what fits your use case.

#### Tier 1: Deterministic (free, instant, no LLM needed)

| Evaluator | What it checks |
|-----------|---------------|
| `NotEmpty` | Response is non-empty |
| `ExactMatch` | Response matches `expected_output` exactly |
| `Contains(substrings)` | Response contains all required strings |
| `RegexMatch(pattern)` | Response matches a regex pattern |
| `JSONSchemaEval(schema)` | Response is valid JSON matching a schema |
| `WordCount(min, max)` | Word count within range |
| `Latency(max_ms)` | Response time under limit |
| `BLEU(n)` | BLEU-n score vs expected output |
| `ROUGE` | ROUGE-L F1 vs expected output |
| `StartsWith(prefix)` | Response starts with prefix |

#### Tier 2: LLM-as-judge (QAG scoring)

| Evaluator | What it measures | Requires `context` |
|-----------|-----------------|-------------------|
| `Faithfulness` | Response is grounded in context | Yes |
| `Hallucination` | Response doesn't invent facts | Yes |
| `Relevance` | Response addresses the question | No |
| `Coherence` | Response is clear and well-structured | No |
| `Toxicity` | Response is safe and non-harmful | No |
| `Bias` | Response is free of demographic bias | No |
| `Summarization` | Summary captures key points faithfully | Yes |
| `AnswerAccuracy` | Factual correctness vs expected | No |
| `ContextPrecision` | Relevant context retrieved | Yes |
| `ContextRecall` | All needed context retrieved | Yes |
| `CustomRubric` | Your own yes/no criteria | Optional |
| `GEval` | Holistic numeric quality score | Optional |

```python
from multivon_eval import Faithfulness, CustomRubric

CustomRubric(
    name="support_quality",
    criteria=[
        ("Does the response acknowledge the customer's problem?", True),
        ("Does the response provide a concrete next step?", True),
        ("Does the response use apologetic or defensive language?", False),
    ],
    threshold=0.8,
)
```

#### Tier 3: Agent trace evaluators

| Evaluator | What it checks |
|-----------|---------------|
| `ToolCallAccuracy` | Expected tools called (ordered or unordered) |
| `ToolArgumentAccuracy` | Quality of tool arguments (LLM judge) |
| `ToolCallNecessity` | Were tool calls actually needed, or redundant? |
| `TrajectoryEfficiency` | Optimal step count + error recovery quality |
| `AgentMemoryEval` | Multi-session memory: retrieval, forgetting, consistency |
| `PlanQuality` | Plan logic, completeness, efficiency |
| `TaskCompletion` | Final output satisfies the task goal |
| `StepFaithfulness` | Each step follows logically from prior |

```python
from multivon_eval import ToolCallAccuracy, ToolCallNecessity, AgentMemoryEval

ToolCallAccuracy(require_order=True)  # strict ordering
ToolCallAccuracy(require_order=False) # set match (default)

# Multi-session memory eval
case = EvalCase(
    input="What did I ask you to prioritize last week?",
    context="Prior session: User set priority to shipping the auth module first.",
)
suite.add_evaluators(AgentMemoryEval())
```

#### Tier 4: Conversation evaluators

| Evaluator | What it checks |
|-----------|---------------|
| `ConversationRelevance` | Each response stays on topic |
| `KnowledgeRetention` | Model remembers earlier context |
| `ConversationCompleteness` | Conversation resolves the original goal |
| `TurnConsistency` | No contradictions across turns |

```python
from multivon_eval import EvalCase

case = EvalCase(
    input="Is this product available in blue?",
    conversation=[
        {"role": "user", "content": "I need a new laptop"},
        {"role": "assistant", "content": "I can help you find a laptop. What's your budget?"},
        {"role": "user", "content": "Around $1000"},
        {"role": "assistant", "content": "Here are some options around $1000..."},
    ],
)
```

### `EvalSuite` — The runner

```python
from multivon_eval import EvalSuite

suite = EvalSuite("My Eval", model_id="gpt-4o")
suite.add_cases(cases)
suite.add_evaluators(NotEmpty(), Relevance(), Faithfulness(threshold=0.7))

# Serial
report = suite.run(model_fn, verbose=True, fail_threshold=0.8)

# Parallel (thread-based)
report = suite.run(model_fn, workers=8)

# Multi-run: detect flaky cases, get score confidence intervals
report = suite.run(model_fn, runs=5)
print(report.flaky_count)       # cases that sometimes pass, sometimes fail
print(report.stability_score)   # 1.0 = fully consistent

for cr in report.case_results:
    print(cr.run_pass_rate)  # e.g. 0.6 = passed 3/5 runs
    print(cr.score_std)      # score variance across runs
    print(cr.is_flaky)       # True if inconsistent

# Async
import asyncio
report = asyncio.run(suite.run_async(model_fn, concurrency=10))
```

### Loading datasets

```python
from multivon_eval import load

cases = load("tests/dataset.jsonl")  # auto-detects format
cases = load("tests/dataset.csv")
```

**JSONL format:**
```json
{"input": "What is the capital of France?", "expected_output": "Paris", "tags": ["factual"]}
{"input": "Summarize this document.", "context": "Document text here...", "tags": ["summarization"]}
```

**CSV format:**
```
input,expected_output,context,tags
What is 2+2?,4,,math
Summarize this.,,Long text here,summarization
```

### Exporting results

```python
report.save_json("results.json")
report.save_csv("results.csv")
```

---

## Compliance & privacy evaluators

For regulated industries (healthcare, finance, legal) where traces can't leave your environment.

### PII Detection (zero API calls)

```python
from multivon_eval import PIIEvaluator

suite.add_evaluators(
    PIIEvaluator()                        # all patterns, all jurisdictions
    PIIEvaluator(jurisdiction="gdpr")     # GDPR-specific extensions
    PIIEvaluator(jurisdiction="ccpa")     # California CCPA
    PIIEvaluator(redact=True)             # mask PII in the report
    PIIEvaluator(patterns={               # custom patterns
        "employee_id": r"EMP-\d{6}",
    })
)
```

Detects: email, phone, SSN, credit card, IP address, IBAN, date of birth, passport numbers, physical addresses. Reports per-type with examples. Zero LLM calls — regex only.

### Structured output validation

```python
from pydantic import BaseModel
from multivon_eval import SchemaEvaluator

class ExtractedInvoice(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str

# Validate every output against your schema
suite.add_evaluators(SchemaEvaluator(ExtractedInvoice))

# Or use JSON Schema directly
suite.add_evaluators(SchemaEvaluator({
    "type": "object",
    "required": ["vendor", "amount"],
    "properties": {
        "vendor": {"type": "string"},
        "amount": {"type": "number"},
    }
}))
```

Per-field failures reported. Based on StructEval (2025): GPT-4 fails complex structured extraction ~12% of the time even with explicit format instructions.

### Audit trail generation

```python
from multivon_eval import ComplianceReporter

reporter = ComplianceReporter(
    output_dir="./audit-logs",
    framework="eu-ai-act",   # or "nist-ai-rmf" or "none"
)
report = suite.run(model_fn)
record_id = reporter.record(report)

# Verify integrity of the audit log
reporter.verify("My Eval Suite")
```

Produces append-only NDJSON audit records, SHA-256 hashed. Each evaluator result is annotated with the relevant EU AI Act Article 9 / NIST AI RMF control category.

---

## Statistical rigor

Backed by [NAACL 2025](https://arxiv.org/abs/2502.01775): single-run benchmark scores are unreliable — variance is large enough to reverse model rankings.

### Confidence intervals on pass rates

```python
from multivon_eval import wilson_interval

# 95% Wilson score CI for 80 passing out of 100 cases
lo, hi = wilson_interval(80, 100)
print(f"95% CI: [{lo:.1%}, {hi:.1%}]")   # → [71.1%, 86.7%]

# experiment.compare() now shows CIs automatically:
#   95% CI (before): [71.4%, 89.3%]
#   95% CI (after):  [83.5%, 96.2%]
```

### Minimum test cases calculator

```python
from multivon_eval import runs_needed

# How many test cases do you need to detect a 10% improvement?
n = runs_needed(delta=0.10)          # → 291
n = runs_needed(delta=0.05)          # → 1248
n = runs_needed(delta=0.10, power=0.90)  # higher power → more cases
```

If `exp.compare()` finds a non-significant difference, it now automatically suggests the minimum number of test cases needed to confirm or rule out that effect.

---

## Synthetic dataset generation

No labeled data? No problem. Point `generate_from_file()` at your docs and get eval cases ready to run in seconds.

```python
from multivon_eval import generate_from_file

# Generate QA pairs from your docs
cases = generate_from_file("docs/faq.md", n=20, task="qa")

# Generate summarization cases
cases = generate_from_file("docs/whitepaper.txt", n=10, task="summarization")

suite.add_cases(cases)
report = suite.run(my_model_fn)
```

From raw text:

```python
from multivon_eval import generate_from_text

cases = generate_from_text(my_knowledge_base, n=50, task="qa")
```

Build a hallucination benchmark from your own content:

```python
from multivon_eval import generate_hallucination_pairs

pairs = generate_hallucination_pairs(my_docs, n=20)
# Returns: [{question, context, faithful_answer, hallucinated_answer}, ...]
```

CLI:

```bash
multivon-eval generate --from docs/faq.md --n 20 --task qa --output cases.jsonl
```

---

## Experiment tracking

Record every suite run and compare results across model versions, prompt changes, or time. Stored locally in `~/.multivon/experiments/` — no cloud, no account.

```python
from multivon_eval import Experiment

exp = Experiment("rag-pipeline")

# Run A — baseline
report_a = suite.run(old_model_fn)
run_a = exp.record(report_a, tags={"model": "gpt-4o", "prompt_v": "2"})

# Run B — new version
report_b = suite.run(new_model_fn)
run_b = exp.record(report_b, tags={"model": "gpt-4o", "prompt_v": "3"})

# Compare
exp.compare(run_a, run_b)
```

```
  ============================================================
  Experiment comparison: a1b2c3d4 → e5f6g7h8
  ============================================================

  Metric                   Before           After
  ------------------------------------------------------------
  Model                    gpt-4o           gpt-4o
  Pass rate                  84.0%  →   91.0%  ↑   +7.0%
  Avg score                 0.8210  →   0.8890  ↑  +0.0680
  Passed                        42  →       46
  Failed                         8  →        4

  Evaluator scores         Before           After
  ------------------------------------------------------------
  faithfulness             0.7800  →   0.8600  ↑  +0.0800
  relevance                0.9100  →   0.9300  ↑  +0.0200

  Verdict: IMPROVED — pass rate up +7.0%
```

CLI:

```bash
multivon-eval experiments list
multivon-eval experiments history rag-pipeline
multivon-eval experiments compare rag-pipeline a1b2c3d4 e5f6g7h8
```

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
     │
     ├── for each EvalCase:
     │     ├── call model_fn(case.input) → output
     │     └── for each Evaluator:
     │           ├── Deterministic → no LLM, instant
     │           ├── LLM Judge → QAG yes/no questions → fraction score
     │           ├── Agent → trace inspection + LLM judge
     │           └── Conversation → multi-turn analysis
     │
     └── EvalReport
           ├── CaseResult × N
           ├── per-evaluator scores
           ├── terminal report (rich)
           └── export → JSON / CSV
```

**Judge model:** Configured via `JUDGE_MODEL` and `JUDGE_PROVIDER` env vars. Defaults to `claude-sonnet-4-6`. The model under test and the judge model can be different providers.

---

## Examples

| File | What it shows |
|------|--------------|
| [`basic_eval.py`](examples/basic_eval.py) | Deterministic evaluators, no LLM judge |
| [`rag_eval.py`](examples/rag_eval.py) | Faithfulness + hallucination for RAG systems |
| [`ci_eval.py`](examples/ci_eval.py) | CI/CD integration with pass threshold |

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Roadmap

- [x] Deterministic evaluators (BLEU, ROUGE, regex, JSON schema, latency)
- [x] LLM-as-judge with QAG scoring
- [x] Agent trace evaluators (tool call accuracy, plan quality)
- [x] Agent trajectory efficiency + necessity scoring
- [x] Multi-session agent memory evaluation
- [x] Conversation evaluators
- [x] PII detection (local, zero API calls)
- [x] Schema validation (Pydantic + JSON Schema)
- [x] Compliance audit trail (EU AI Act / NIST AI RMF)
- [x] Wilson score confidence intervals on pass rates
- [x] Minimum test cases calculator (`runs_needed`)
- [x] Parallel + async runners
- [x] CLI (`multivon-eval run`, `multivon-eval report`, `--html`, `--json`)
- [x] HTML report export (self-contained, shareable)
- [x] Framework integrations (LangChain, LangSmith, ManualTracer)
- [ ] LlamaIndex / CrewAI integrations
- [ ] Pytest plugin (`@eval_case` decorator)
- [ ] Effect size (Cohen's h) + min-detectable-effect in experiment comparison
- [ ] Tiered eval cost optimizer (heuristic → local model → frontier)
- [ ] Agent simulation / adversarial user testing

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
