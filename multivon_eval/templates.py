"""Project templates for ``multivon-eval init``.

Each template renders to a small, runnable project that produces a successful
first eval in under 5 minutes. Keep these compact — they're scaffolding, not
tutorials. The README inside each generated project teaches the next step.

Add a new template by appending to ``TEMPLATES`` with the same shape — a dict
of ``{relative_path: file_contents_str}``. ``cli.cmd_init`` will dump those
files into the target dir verbatim.

Templates currently shipped:

  - ``quickstart``  — deterministic-only eval, no API keys, no LLM calls.
                      Hello-world that works offline. Good first run.
  - ``rag``         — Faithfulness + Hallucination on a tiny knowledge base,
                      with a budget gate that fails the run if cost or
                      latency drift. Needs OPENAI_API_KEY or ANTHROPIC_API_KEY.
  - ``agent``       — Tool-calling support agent with a manual AgentTracer,
                      ToolCallAccuracy + TrajectoryEfficiency. Needs a key.
  - ``regulated``   — RAG with ComplianceReporter, hash-chained NDJSON audit
                      log, and an ``audit-package`` step to produce the
                      auditor-attachable zip. Needs a key.
"""
from __future__ import annotations


# Shared building blocks — small files used across multiple templates.
_DOTENV_EXAMPLE = """\
# Copy to .env and fill in. One of the two is enough.
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
"""

_GITIGNORE = """\
.env
__pycache__/
*.pyc
*.log
audit-logs/
eval-reports/
.venv/
"""


# Common GitHub Actions CI workflow — drops into .github/workflows/eval.yml.
# Runs on every PR + on main. Tiered: smoke (no API) on every push; full
# (with secrets) only on main + nightly cron. Mirrors the structure GPT-5
# called out as Sarah's blocker.
def _ci_workflow(template_name: str) -> str:
    return f"""\
name: eval

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: "0 7 * * *"   # nightly 07:00 UTC

jobs:
  smoke:
    name: smoke (no API)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {{ python-version: "3.12" }}
      - run: pip install -r requirements.txt
      - name: Import + run nothing — proves the suite compiles
        run: python -c "import eval"

  evaluate:
    name: evaluate (live judge)
    # Live runs cost money — only on main pushes and nightly cron.
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    needs: smoke
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {{ python-version: "3.12" }}
      - run: pip install -r requirements.txt
      - name: Run eval
        env:
          ANTHROPIC_API_KEY: ${{{{ secrets.ANTHROPIC_API_KEY }}}}
          OPENAI_API_KEY: ${{{{ secrets.OPENAI_API_KEY }}}}
        run: python eval.py
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-report-{template_name}
          # `audit-logs/` is only populated by the regulated template; the
          # action ignores missing dirs so this is safe everywhere.
          path: |
            eval-reports/
            audit-logs/
          if-no-files-found: ignore
"""


def _requirements(extras: list[str] | None = None) -> str:
    """Generate a requirements.txt pinning the library + named extras."""
    extras_str = f"[{','.join(extras)}]" if extras else ""
    return f"multivon-eval{extras_str}>=0.6.1\n"


# ─────────────────────────────────────────────────────────────────────────────
# Template: quickstart — deterministic-only, no API key needed
# ─────────────────────────────────────────────────────────────────────────────
_QUICKSTART_EVAL = '''\
"""Hello-world eval — runs offline, no API key, no LLM calls.

This template demonstrates the deterministic evaluator surface:
NotEmpty, Contains, WordCount. Use it as your sanity check that
the library is installed correctly.

When you're ready for LLM-judge evals (Faithfulness, Hallucination,
Relevance), see ``multivon-eval init --template rag``.
"""
from multivon_eval import (
    EvalSuite, EvalCase,
    NotEmpty, Contains, WordCount,
)


def my_model_fn(input_text: str) -> str:
    """Stand-in for your real model. Replace with your LLM call."""
    if "hello" in input_text.lower():
        return "Hi there! How can I help?"
    if "weather" in input_text.lower():
        return "I can't fetch live weather, but I can suggest a service."
    return "I'm not sure how to help with that yet."


suite = EvalSuite("quickstart")
suite.add_cases([
    EvalCase(input="Hello"),
    EvalCase(input="What's the weather?"),
    EvalCase(input="Anything else?"),
])
suite.add_evaluators(
    NotEmpty(),
    Contains(["help", "weather"], match_any=True),
    WordCount(min=2, max=40),
)

if __name__ == "__main__":
    import os
    report = suite.run(my_model_fn)
    os.makedirs("eval-reports", exist_ok=True)
    # save_json() writes a multivon-eval report — view later with
    # `multivon-eval report eval-reports/quickstart.json`.
    report.save_json("eval-reports/quickstart.json")
'''

_QUICKSTART_README = """\
# multivon-eval quickstart

A deterministic-only eval — no API key, no LLM calls, runs offline.

## 3-command flow

```bash
pip install -r requirements.txt
python eval.py
multivon-eval report eval-reports/quickstart.json
```

Expected: 3 cases, all PASS, pass rate 100% (your `my_model_fn` is hand-tuned to satisfy the checks).

## Next steps

- Replace `my_model_fn` with your real LLM call.
- Try LLM-judge evaluators: `multivon-eval init --template rag` in a new dir.
- Add cost + latency budgets: see `EvalReport.assert_budget()` in the docs.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Template: rag — Faithfulness + Hallucination + budget gate
# ─────────────────────────────────────────────────────────────────────────────
_RAG_EVAL = '''\
"""RAG faithfulness eval — Faithfulness + Hallucination + budget gate.

Needs ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment, or a
local Ollama running on :11434.
"""
import os
import sys

# Load .env if present so the keys set up via `cp .env.example .env`
# are picked up automatically.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from multivon_eval import (
    EvalSuite, EvalCase, JudgeConfig, configure,
    Faithfulness, Hallucination,
)


# Auto-detect judge: cloud key wins; falls back to local Ollama if present.
def _auto_judge() -> JudgeConfig:
    if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-") and \\
       "..." not in os.getenv("ANTHROPIC_API_KEY", ""):
        return JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0)
    if os.getenv("OPENAI_API_KEY", "").startswith("sk-") and \\
       "..." not in os.getenv("OPENAI_API_KEY", ""):
        return JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.0)
    # Local fallback — assumes Ollama on default port with llama3 pulled.
    # The OpenAI SDK requires a non-empty api_key even for local endpoints,
    # so set a sentinel value; the local server ignores it.
    os.environ.setdefault("OPENAI_API_KEY", "ollama-local-no-auth")
    return JudgeConfig(
        provider="openai", model="llama3",
        base_url="http://localhost:11434/v1",
    )


configure(_auto_judge())


# Tiny knowledge base — what the retriever returns for each query.
KB = {
    "vacation": "Employees get 15 vacation days/year (1.25/month). 2 weeks advance notice required.",
    "expenses": "Receipts go to finance@company within 30 days. Reimbursed every 2 weeks.",
}


def rag_model(input_text: str) -> str:
    """Stub RAG model — replace with your retriever + generator pipeline."""
    if "vacation" in input_text.lower():
        return "Employees get 15 vacation days per year, accrued at 1.25 days/month."
    if "expense" in input_text.lower():
        return "Submit receipts to finance@company; reimbursed every 2 weeks."
    return "I don't have information on that yet."


cases = [
    EvalCase(input="How many vacation days do I get?", context=KB["vacation"]),
    EvalCase(input="How do I expense a flight?", context=KB["expenses"]),
    EvalCase(input="What's the deadline for vacation requests?", context=KB["vacation"]),
]


suite = EvalSuite("rag-faithfulness")
suite.add_cases(cases)
suite.add_evaluators(Faithfulness(), Hallucination())


if __name__ == "__main__":
    # fail_threshold=0.7 makes the process exit 1 when pass-rate drops
    # below 70%, so CI catches eval failures — not just code errors.
    report = suite.run(rag_model, fail_threshold=0.7)

    # Budget gate — fail CI if costs blow up. Tune these for your suite.
    try:
        report.assert_budget(
            max_total_cost_usd=0.10,
            max_avg_cost_per_case_usd=0.05,
            max_p95_latency_ms=10_000,
        )
    except SystemExit as exc:
        print(f"\\n[BUDGET GATE FAILED] {exc}", file=sys.stderr)
        raise

    os.makedirs("eval-reports", exist_ok=True)
    report.save_json("eval-reports/rag.json")
'''

_RAG_README = """\
# multivon-eval — RAG template

Faithfulness + Hallucination evaluation with a cost/latency budget gate.

## 3-command flow

```bash
pip install -r requirements.txt
cp .env.example .env && edit .env   # add ANTHROPIC_API_KEY or OPENAI_API_KEY
python eval.py
```

## What this template demonstrates

- **Auto-judge detection** — uses your Anthropic key, then OpenAI, falling back to a local Ollama on `localhost:11434` if neither is set.
- **Two LLM-judge evaluators** — Faithfulness checks that the answer is supported by the context; Hallucination flags claims not in the context.
- **Budget gate** — `report.assert_budget()` fails the run if total cost > $0.10, avg cost/case > $0.05, or p95 latency > 10s. Wire this into CI to prevent runaway eval bills.

## Next steps

- Add more cases — see `cases = [...]` in `eval.py`.
- Tune thresholds — every evaluator takes `threshold=` for stricter/looser pass criteria. Library defaults are calibrated per-judge.
- Generate cases from docs: `multivon-eval generate --from docs/faq.md --n 20 --task qa --output cases.jsonl`.
- Save HTML reports: `multivon-eval report eval-reports/rag.json --html out.html`.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Template: agent — tool-calling with AgentTracer + agent evaluators
# ─────────────────────────────────────────────────────────────────────────────
_AGENT_EVAL = '''\
"""Agent eval — toy support agent with ToolCallAccuracy + TrajectoryEfficiency.

The agent has 2 tools (lookup_order, refund_order). Cases assert which
tools the agent SHOULD call. AgentTracer captures the call sequence.

Needs ANTHROPIC_API_KEY or OPENAI_API_KEY for the judge.
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from multivon_eval import (
    EvalSuite, EvalCase, JudgeConfig, configure,
    AgentTracer, AgentStep, ToolCall,
    ToolCallAccuracy, ToolArgumentAccuracy, TrajectoryEfficiency,
)


def _auto_judge() -> JudgeConfig:
    if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-") and \\
       "..." not in os.getenv("ANTHROPIC_API_KEY", ""):
        return JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0)
    return JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.0)


configure(_auto_judge())


# ── Tools the agent could call ──
ORDERS = {"O-101": {"status": "shipped", "total": 49.99}}


def lookup_order(order_id):
    return ORDERS.get(order_id, {"error": "not found"})


def refund_order(order_id, amount):
    if order_id not in ORDERS:
        return {"error": "order not found"}
    return {"refund_id": f"R-{order_id}", "status": "approved", "amount": amount}


# ── Tracer — captures the agent's steps so evaluators can score the trajectory ──
class HandRolledTracer(AgentTracer):
    def __init__(self): self._steps: list[AgentStep] = []
    def reset(self): self._steps = []
    def get_trace(self): return list(self._steps)
    def instrument(self, fn):
        def wrapped(input_text):
            self.reset()
            return fn(input_text, self._steps)
        return wrapped


def support_agent(input_text: str, steps: list[AgentStep]) -> str:
    lower = input_text.lower()
    if "where is order" in lower or "status of order" in lower:
        order_id = next((tok for tok in input_text.split() if tok.startswith("O-")), None)
        result = lookup_order(order_id) if order_id else {}
        steps.append(AgentStep(
            thought=f"Look up status of {order_id}.",
            tool_calls=[ToolCall(name="lookup_order", arguments={"order_id": order_id}, result=result)],
            output=f"Order {order_id} is {result.get('status', 'not found')}.",
        ))
        return f"Order {order_id} is {result.get('status', 'not found')}."
    if "refund" in lower:
        order_id = next((tok for tok in input_text.split() if tok.startswith("O-")), None)
        order = lookup_order(order_id) if order_id else {}
        steps.append(AgentStep(
            thought="Verify the order before refunding.",
            tool_calls=[ToolCall(name="lookup_order", arguments={"order_id": order_id}, result=order)],
        ))
        refund = refund_order(order_id, order.get("total", 0)) if order.get("status") else {"error": "no order"}
        steps.append(AgentStep(
            thought="Process the refund.",
            tool_calls=[ToolCall(name="refund_order",
                                  arguments={"order_id": order_id, "amount": order.get("total", 0)},
                                  result=refund)],
            output=f"Refund {refund.get('refund_id', '—')} approved.",
        ))
        return f"Refund {refund.get('refund_id', '—')} approved."
    return "I can't help with that."


cases = [
    EvalCase(
        input="Where is order O-101?",
        expected_tool_calls=["lookup_order"],
    ),
    EvalCase(
        input="Refund order O-101.",
        expected_tool_calls=["lookup_order", "refund_order"],
    ),
]


tracer = HandRolledTracer()
suite = EvalSuite("support-agent")
suite.add_cases(cases)
suite.add_evaluators(
    ToolCallAccuracy(),
    ToolArgumentAccuracy(),
    TrajectoryEfficiency(),
)


if __name__ == "__main__":
    import os
    report = suite.run(support_agent, tracer=tracer, fail_threshold=0.7)
    os.makedirs("eval-reports", exist_ok=True)
    report.save_json("eval-reports/agent.json")
'''

_AGENT_README = """\
# multivon-eval — Agent template

Tool-calling agent eval with `ToolCallAccuracy`, `ToolArgumentAccuracy`, and `TrajectoryEfficiency`.

## 3-command flow

```bash
pip install -r requirements.txt
cp .env.example .env && edit .env   # add ANTHROPIC_API_KEY or OPENAI_API_KEY
python eval.py
```

## What this template demonstrates

- **Hand-rolled tracer** (`HandRolledTracer`) — implements `AgentTracer.instrument`, so any agent loop (LangChain, AutoGen, custom) can be evaluated without framework lock-in.
- **`expected_tool_calls`** on each `EvalCase` — declares which tools the agent *should* call. `ToolCallAccuracy` scores actual vs expected.
- **`TrajectoryEfficiency`** — judges whether the agent took the optimal number of steps and recovered from any failed tool calls.

## Next steps

- Replace the toy `support_agent` with your real loop (LangChain `Runnable`, AutoGen `Agent`, etc.).
- Add `TaskCompletion` to score the final output, not just the trajectory.
- Add `AgentMemoryEval` for multi-session memory tests — see `multivon-eval --help` and the docs.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Template: regulated — RAG + ComplianceReporter + audit-package
# ─────────────────────────────────────────────────────────────────────────────
_REGULATED_EVAL = '''\
"""Regulated eval — RAG with hash-chained audit log + auditor-attachable zip.

Demonstrates the compliance pipeline:
  1. Run the eval suite as usual.
  2. ComplianceReporter writes an append-only NDJSON audit log to
     ./audit-logs/, with every record linked into a SHA-256 hash chain.
  3. After the run, optionally bundle the log + calibration + verifier
     into a single zip for an auditor:
        multivon-eval audit-package --logs audit-logs --suite eu-ai-act-eval \\
                                    --framework eu-ai-act --out evidence.zip

Needs ANTHROPIC_API_KEY or OPENAI_API_KEY for the judge.
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from multivon_eval import (
    EvalSuite, EvalCase, JudgeConfig, configure,
    Faithfulness, Hallucination,
    ComplianceReporter,
)


def _auto_judge() -> JudgeConfig:
    if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-") and \\
       "..." not in os.getenv("ANTHROPIC_API_KEY", ""):
        return JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0)
    return JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.0)


configure(_auto_judge())


KB = {
    "data_handling": "Personal data is encrypted at rest (AES-256) and in transit (TLS 1.3). Access is role-based and audited.",
}


def regulated_model(input_text: str) -> str:
    if "data" in input_text.lower():
        return ("Personal data is encrypted at rest with AES-256 and in transit "
                "with TLS 1.3. Access controls are role-based and audited.")
    return "I don't have information on that."


cases = [
    EvalCase(input="How is personal data protected at rest?", context=KB["data_handling"]),
    EvalCase(input="Is data encrypted in transit?", context=KB["data_handling"]),
]


SUITE_NAME = "eu-ai-act-eval"
suite = EvalSuite(SUITE_NAME)
suite.add_cases(cases)
suite.add_evaluators(Faithfulness(), Hallucination())


if __name__ == "__main__":
    reporter = ComplianceReporter("audit-logs", framework="eu-ai-act")
    report = suite.run(regulated_model, fail_threshold=0.7)

    reporter.record(report, tags={"system": "regulated-template-demo", "version": "0.1.0"})

    # Save report under eval-reports/ so the CI artifact upload picks it up
    # alongside the audit log.
    os.makedirs("eval-reports", exist_ok=True)
    report.save_json("eval-reports/regulated.json")

    print("\\nAudit log written to ./audit-logs/")
    print("Bundle for an auditor with:")
    print(f"  multivon-eval audit-package \\\\")
    print(f"      --logs audit-logs --suite {SUITE_NAME} \\\\")
    print(f"      --framework eu-ai-act --out evidence.zip")
'''

_REGULATED_README = """\
# multivon-eval — Regulated template

End-to-end compliance flow: RAG eval → hash-chained audit log → auditor-attachable zip.

## 3-command flow

```bash
pip install -r requirements.txt
cp .env.example .env && edit .env   # add ANTHROPIC_API_KEY or OPENAI_API_KEY
python eval.py
multivon-eval audit-package --logs audit-logs --suite eu-ai-act-eval --framework eu-ai-act --out evidence.zip
```

## What this template demonstrates

- **`ComplianceReporter`** — writes every eval run to an append-only NDJSON file under `audit-logs/`, with every record linked into a SHA-256 hash chain. Detecting mid-log tampering is end-to-end.
- **EU AI Act mapping** — the `framework="eu-ai-act"` argument tags each record with the Articles it exercises (Art. 9, 10, 15). Use `framework="hipaa"` or `framework="nist-ai-rmf"` for other regimes.
- **`audit-package` CLI** — bundles the log + the calibration table that drove the decisions + a standalone verifier script into a single zip an auditor can run offline.

## Verifying the package

```bash
unzip evidence.zip -d evidence/
cd evidence/compliance-evidence-*
python verify.py   # exits 0 if every file's hash matches and the chain is intact
```

## Next steps

- Replace `KB` and `regulated_model` with your real RAG pipeline.
- Add more controls — `PIIEvaluator` for HIPAA Safe Harbor, `SchemaEvaluator` for structured-output checks.
- Use `EvalSuite.eu_ai_act_high_risk()` or `EvalSuite.hipaa_safe_harbor()` for pre-built compliance suites.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Registry — what `--template` accepts
# ─────────────────────────────────────────────────────────────────────────────

_CONVERSATION_EVAL = '''\
"""Conversation eval — multi-turn dialogue quality with QAG judges.

Tests a customer-support chatbot across a realistic 3-turn dialogue.
ConversationRelevance + KnowledgeRetention + TurnConsistency catch the
common failure modes: drifting off-topic, forgetting earlier facts,
contradicting an earlier response.

Needs ANTHROPIC_API_KEY or OPENAI_API_KEY.
"""
from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from multivon_eval import (
    EvalSuite, EvalCase, JudgeConfig, configure,
    ConversationRelevance, KnowledgeRetention, TurnConsistency,
)


def _auto_judge() -> JudgeConfig:
    if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-") and \\
       "..." not in os.getenv("ANTHROPIC_API_KEY", ""):
        return JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0)
    return JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.0)


configure(_auto_judge())


# A realistic multi-turn case. The ``conversation`` field is the
# history; ``input`` is the final user turn; the model output is the
# assistant's response. Evaluators inspect the FULL conversation.
cases = [
    EvalCase(
        input="Will it ship by Friday?",
        conversation=[
            {"role": "user", "content": "I ordered SKU-12 yesterday — order O-451."},
            {"role": "assistant", "content": "I see order O-451 with SKU-12. Standard shipping takes 3-5 business days."},
            {"role": "user", "content": "Will it ship by Friday?"},
        ],
        tags=["shipping"],
    ),
    EvalCase(
        input="So I should return the broken one and you'll refund the second?",
        conversation=[
            {"role": "user", "content": "My headphones (order O-789) broke after one week."},
            {"role": "assistant", "content": "Sorry to hear that. We can either replace order O-789 free of charge or refund you."},
            {"role": "user", "content": "Send a replacement please."},
            {"role": "assistant", "content": "Got it — replacement for O-789 is on its way; tracking will arrive in your inbox."},
            {"role": "user", "content": "So I should return the broken one and you'll refund the second?"},
        ],
        tags=["returns"],
    ),
]


# Stub model — replace with your real chatbot. The signature is still
# str → str (the latest user turn); evaluators see the full conversation
# via case.conversation.
def chatbot(latest_user_turn: str) -> str:
    if "ship by friday" in latest_user_turn.lower():
        return (
            "Standard shipping is 3-5 business days from yesterday, so "
            "your order O-451 should arrive by Friday."
        )
    if "return the broken" in latest_user_turn.lower():
        return (
            "No — keep the broken pair; you'll have it picked up by our "
            "carrier. We're sending a replacement (no refund) for O-789."
        )
    return "Could you clarify what you'd like me to help with?"


suite = EvalSuite("conversation-eval")
suite.add_cases(cases)
suite.add_evaluators(
    ConversationRelevance(),
    KnowledgeRetention(),
    TurnConsistency(),
)


if __name__ == "__main__":
    report = suite.run(chatbot, fail_threshold=0.7)

    try:
        report.assert_budget(
            max_total_cost_usd=0.10,
            max_avg_cost_per_case_usd=0.05,
        )
    except SystemExit as exc:
        print(f"\\n[BUDGET GATE FAILED] {exc}", file=sys.stderr)
        raise

    os.makedirs("eval-reports", exist_ok=True)
    report.save_json("eval-reports/conversation.json")
'''

_CONVERSATION_README = """\
# multivon-eval — Conversation template

Multi-turn dialogue eval with `ConversationRelevance`, `KnowledgeRetention`, and `TurnConsistency`.

## 3-command flow

```bash
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY or OPENAI_API_KEY
python eval.py
```

## What this template demonstrates

- **Multi-turn `EvalCase`** — the `conversation` field carries the dialogue history (`[{"role": "user|assistant", "content": "..."}]`). Evaluators inspect the whole transcript, not just the last response.
- **Three conversation evaluators:**
  - `ConversationRelevance` flags responses that ignore the prior context.
  - `KnowledgeRetention` catches forgetting facts established earlier (the order ID, the customer's choice).
  - `TurnConsistency` flags self-contradictions across turns.
- **Budget gate** — same `report.assert_budget(...)` pattern as the other templates.

## Next steps

- Swap `chatbot()` for your real implementation. The signature stays `str → str` (the latest user turn); evaluators see the full conversation via `case.conversation`.
- Add `ConversationCompleteness` to assert the user's overall goal was addressed by the end of the dialogue.
- For more elaborate scenarios: `multivon-eval simulate_users` (programmatic synthetic dialogues).
"""


TEMPLATES: dict[str, dict[str, str]] = {
    "quickstart": {
        "eval.py": _QUICKSTART_EVAL,
        "README.md": _QUICKSTART_README,
        "requirements.txt": _requirements(),
        ".gitignore": _GITIGNORE,
    },
    "rag": {
        "eval.py": _RAG_EVAL,
        "README.md": _RAG_README,
        "requirements.txt": _requirements(),
        ".env.example": _DOTENV_EXAMPLE,
        ".gitignore": _GITIGNORE,
    },
    "agent": {
        "eval.py": _AGENT_EVAL,
        "README.md": _AGENT_README,
        "requirements.txt": _requirements(),
        ".env.example": _DOTENV_EXAMPLE,
        ".gitignore": _GITIGNORE,
    },
    "conversation": {
        "eval.py": _CONVERSATION_EVAL,
        "README.md": _CONVERSATION_README,
        "requirements.txt": _requirements(),
        ".env.example": _DOTENV_EXAMPLE,
        ".gitignore": _GITIGNORE,
    },
    "regulated": {
        "eval.py": _REGULATED_EVAL,
        "README.md": _REGULATED_README,
        "requirements.txt": _requirements(),
        ".env.example": _DOTENV_EXAMPLE,
        ".gitignore": _GITIGNORE,
    },
}


def list_templates() -> list[str]:
    """Return the available template names in display order."""
    return ["quickstart", "rag", "agent", "conversation", "regulated"]


def render(template: str, *, with_ci: str | None = None) -> dict[str, str]:
    """Return the file map for ``template``, optionally adding a CI workflow.

    Args:
        template: Name of a template in :data:`TEMPLATES`.
        with_ci:  If set to ``"github"``, also includes a GitHub Actions
                  workflow at ``.github/workflows/eval.yml``. Other CI
                  flavors can be added here later.

    Raises:
        ValueError: If ``template`` is unknown or ``with_ci`` is unsupported.
    """
    if template not in TEMPLATES:
        raise ValueError(
            f"Unknown template: {template!r}. "
            f"Available: {', '.join(list_templates())}"
        )

    files = dict(TEMPLATES[template])

    if with_ci is not None:
        if with_ci != "github":
            raise ValueError(f"Unsupported CI flavor: {with_ci!r}. Supported: 'github'")
        files[".github/workflows/eval.yml"] = _ci_workflow(template)

    return files
