# llm-evals

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

**Practical LLM evaluation for teams that ship to production.**

Run structured evals over your model outputs — from simple string checks to LLM-as-judge scoring — with a clean Python API, beautiful terminal reports, and CI/CD integration out of the box.

---

```python
from llm_evals import EvalSuite, EvalCase, Relevance, Faithfulness, NotEmpty

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

## Why llm-evals

Every team building LLM-powered products hits the same problem: **how do you know if your model is getting better or worse?**

Existing tools have real limitations:
- **DeepEval** — powerful but LLM-as-judge for everything is expensive, slow, and hard to audit
- **RAGAS** — excellent, but RAG-only
- **Promptfoo** — YAML-driven, feels rigid for Python teams

`llm-evals` is different in one important way: **QAG scoring** (Question-Answer Generation). Instead of asking a judge "rate this 1-10" — which introduces its own hallucination risk — we generate a set of yes/no questions about the output and score by the fraction answered correctly. This approach is:

- **More reliable** — binary questions are easier for LLMs to get right than numeric ratings
- **Auditable** — you can see exactly which questions passed and failed
- **Cheaper** — shorter judge prompts, fewer tokens

---

## Install

```bash
pip install llm-evals
```

```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

---

## Core concepts

### `EvalCase` — A test case

```python
from llm_evals import EvalCase

case = EvalCase(
    input="What caused the 2008 financial crisis?",          # required
    expected_output="Subprime mortgage collapse...",          # for ExactMatch, Contains
    context="The 2008 crisis was triggered by...",           # for Faithfulness, Hallucination
    tags=["finance", "history"],                             # for filtering reports
    metadata={"source": "test_set_v2", "difficulty": "hard"},
)
```

### Evaluators

Three tiers — pick what fits your use case.

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

```python
from llm_evals import ExactMatch, Contains, JSONSchemaEval

# Validate structured output
JSONSchemaEval({
    "type": "object",
    "properties": {"sentiment": {"type": "string"}, "score": {"type": "number"}},
    "required": ["sentiment", "score"],
})

# Check for required content
Contains(["refund policy", "contact us"], threshold=1.0)
```

#### Tier 2: LLM-as-judge (QAG scoring)

| Evaluator | What it measures | Requires `context` |
|-----------|-----------------|-------------------|
| `Faithfulness` | Response is grounded in context | Yes |
| `Hallucination` | Response doesn't invent facts | Yes |
| `Relevance` | Response addresses the question | No |
| `Coherence` | Response is clear and well-structured | No |
| `Toxicity` | Response is safe and non-harmful | No |
| `CustomRubric` | Your own criteria | Optional |

```python
from llm_evals import Faithfulness, Hallucination, CustomRubric

# Custom rubric for a specific use case
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

### `EvalSuite` — The runner

```python
from llm_evals import EvalSuite

suite = EvalSuite("My Eval", model_id="gpt-4o")
suite.add_cases(cases)
suite.add_evaluators(NotEmpty(), Relevance(), Faithfulness(threshold=0.7))

report = suite.run(
    model_fn=my_model,        # any callable: str -> str
    verbose=True,             # print terminal report
    fail_threshold=0.8,       # exit(1) in CI if pass rate < 80%
)
```

### Loading datasets

```python
from llm_evals import load

# Auto-detects format from extension
cases = load("tests/dataset.jsonl")
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
report.save_json("results.json")   # full detail
report.save_csv("results.csv")     # one row per evaluator per case
```

---

## CI/CD integration

Run evals as a quality gate in your pipeline:

```python
# eval.py
report = suite.run(model_fn, fail_threshold=0.85)  # exits with code 1 if < 85% pass
```

```yaml
# .github/workflows/eval.yml
- name: Run LLM evals
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
     │           └── LLM Judge → QAG yes/no questions → fraction score
     │
     └── EvalReport
           ├── CaseResult × N
           ├── per-evaluator scores
           ├── terminal report (rich)
           └── export → JSON / CSV
```

**Judge model:** Configured via `JUDGE_MODEL` and `JUDGE_PROVIDER` env vars. Defaults to `claude-sonnet-4-6`. The model being evaluated and the judge model can be different providers.

---

## Examples

| File | What it shows |
|------|--------------|
| [`basic_eval.py`](examples/basic_eval.py) | Deterministic evaluators, no LLM judge |
| [`rag_eval.py`](examples/rag_eval.py) | Faithfulness + hallucination for RAG systems |
| [`ci_eval.py`](examples/ci_eval.py) | CI/CD integration with pass threshold |

```bash
python examples/basic_eval.py
python examples/rag_eval.py
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Roadmap

- [ ] Async evaluation runner for faster parallel evals
- [ ] HTML report export
- [ ] Pytest plugin (`@eval_case` decorator)
- [ ] Model comparison mode — run same cases against two models, diff results
- [ ] Eval versioning — track scores over time, detect regressions
- [ ] Built-in Langfuse integration for eval tracing

---

## Contributing

Issues and PRs welcome.

**Small changes** (docs, bug fixes): open a PR directly.
**Large changes** (new evaluators, architecture): open an issue first.

```bash
git clone https://github.com/OmniTensorLabs/llm-evals
cd llm-evals
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT — built by [OmniTensorLabs](https://omnitensorlabs.com)
