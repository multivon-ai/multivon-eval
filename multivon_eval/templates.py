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
"""Agent eval — toy support agent with ToolCallAccuracy.

The agent has 2 tools (lookup_order, refund_order). Cases assert which
tools the agent SHOULD call. AgentTracer captures the call sequence.

DEFAULT MODE: runs OFFLINE with the deterministic ``ToolCallAccuracy``
evaluator only — no API key required. Set ANTHROPIC_API_KEY or
OPENAI_API_KEY to ALSO enable richer LLM-judge evaluators
(ToolArgumentAccuracy, TrajectoryEfficiency, TaskCompletion). The
script will tell you which ones it activated.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from multivon_eval import (
    EvalSuite, EvalCase, JudgeConfig, configure,
    AgentTracer, AgentStep, ToolCall,
    ToolCallAccuracy,
)


def _have_judge() -> bool:
    """True if any LLM judge is reachable. Used to opt-in to the
    judge-based evaluators without making the offline path fail.

    Checks (in order): valid-looking Anthropic key, valid-looking
    OpenAI key, locally-running Ollama on :11434, OPENAI_BASE_URL
    override (any OpenAI-compatible endpoint)."""
    ak = os.getenv("ANTHROPIC_API_KEY", "")
    ok = os.getenv("OPENAI_API_KEY", "")
    if ak.startswith("sk-ant-") and "..." not in ak:
        configure(JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0))
        return True
    if ok.startswith("sk-") and "..." not in ok:
        configure(JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.0))
        return True
    # Local LLM fallback. Probe Ollama's tags endpoint with a short
    # timeout so a missing daemon doesn't block startup.
    base = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    try:
        import urllib.request
        # Ollama responds at /api/tags on the same host:port as /v1.
        probe = base.rstrip("/").rsplit("/v1", 1)[0] + "/api/tags"
        with urllib.request.urlopen(probe, timeout=0.5):
            pass
    except Exception:
        return False
    os.environ.setdefault("OPENAI_API_KEY", "ollama-local-no-auth")
    configure(JudgeConfig(provider="openai", model="llama3",
                          base_url=base, temperature=0.0))
    return True


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

# Tier 1 — always on, deterministic, no API key needed.
suite.add_evaluator(ToolCallAccuracy())

# Tier 2 — LLM-judge evaluators, auto-activated when a key is detected.
# These check tool ARGUMENTS, trajectory efficiency, and task
# completion against the user's stated goal. Skipped silently when no
# judge is reachable so the offline run still produces a clean report.
if _have_judge():
    from multivon_eval import (
        ToolArgumentAccuracy, TrajectoryEfficiency, TaskCompletion,
    )
    suite.add_evaluators(
        ToolArgumentAccuracy(),
        TrajectoryEfficiency(),
        TaskCompletion(),
    )
    print("[multivon-eval] LLM judge detected — enabling argument / trajectory / "
          "completion evaluators.")
else:
    print("[multivon-eval] Running offline (no judge detected). For richer eval, "
          "set ANTHROPIC_API_KEY or OPENAI_API_KEY and re-run.")


if __name__ == "__main__":
    import os
    # We deliberately do NOT pass fail_threshold here — this is a
    # starter template, not a hardened CI gate. With a stale/revoked
    # API key, judge-based evaluators would push pass_rate below the
    # threshold and EvalGateFailure would be raised BEFORE save_json
    # runs, eating the report. Add `fail_threshold=...` once you have
    # a reliable judge configured.
    report = suite.run(support_agent, tracer=tracer)
    os.makedirs("eval-reports", exist_ok=True)
    report.save_json("eval-reports/agent.json")
    print(f"Saved report to eval-reports/agent.json")
    print(f"  multivon-eval view eval-reports/agent.json   # interactive HTML")
'''

_AGENT_README = """\
# multivon-eval — Agent template

Tool-calling agent eval. Runs **offline by default** with the
deterministic `ToolCallAccuracy` evaluator — no API key needed for
your first run. LLM-judge evaluators auto-activate when a key is set.

## 2-command flow (offline, no API key)

```bash
pip install -r requirements.txt
python eval.py
```

Expected output:

```
[multivon-eval] Running offline (no judge detected). For richer eval,
set ANTHROPIC_API_KEY or OPENAI_API_KEY and re-run.

──────────────────────────── support-agent ────────────────────────────
  #  Input                       Output                Score  Status
  1  Where is order O-101?       Order O-101 is...      1.00   PASS
  2  Refund order O-101.         Refund R-O-101...      1.00   PASS

Saved report to eval-reports/agent.json
  multivon-eval view eval-reports/agent.json   # interactive HTML
```

## Add LLM-judge evaluators (optional)

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY or OPENAI_API_KEY
python eval.py
```

This activates `ToolArgumentAccuracy` (was each tool called with sensible args?),
`TrajectoryEfficiency` (did the agent meander?), and `TaskCompletion`
(did the final answer fulfil the user's request?).

## What this template demonstrates

- **Hand-rolled tracer** (`HandRolledTracer`) — implements `AgentTracer.instrument`, so any agent loop (LangChain, AutoGen, OpenAI Agents SDK, custom) can be evaluated without framework lock-in.
- **`expected_tool_calls`** on each `EvalCase` — declares which tools the agent *should* call. `ToolCallAccuracy` scores actual vs expected.
- **Tiered eval design** — deterministic checks first, LLM-judge checks layered on when a key is available. Same pattern works in CI.

## Next steps

- Replace the toy `support_agent` with your real loop (LangChain `Runnable`, OpenAI Agents SDK, AutoGen `Agent`, etc.).
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
    """Anthropic key → OpenAI key → local Ollama on :11434. Returns the
    first reachable config. Ollama is probed with a 0.5s timeout so a
    missing daemon doesn't block startup."""
    ak = os.getenv("ANTHROPIC_API_KEY", "")
    ok = os.getenv("OPENAI_API_KEY", "")
    if ak.startswith("sk-ant-") and "..." not in ak:
        return JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0)
    if ok.startswith("sk-") and "..." not in ok:
        return JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.0)
    base = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    try:
        import urllib.request
        probe = base.rstrip("/").rsplit("/v1", 1)[0] + "/api/tags"
        with urllib.request.urlopen(probe, timeout=0.5):
            pass
        os.environ.setdefault("OPENAI_API_KEY", "ollama-local-no-auth")
        return JudgeConfig(provider="openai", model="llama3",
                           base_url=base, temperature=0.0)
    except Exception:
        # Fall back to OpenAI; will raise JudgeUnavailable with a
        # plain-language setup hint at suite.run() time if neither
        # key is set. See multivon_eval/judge.py _setup_hint.
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
    """Anthropic key → OpenAI key → local Ollama → OpenAI fallback.

    Ollama is probed with a 0.5s timeout. JudgeUnavailable with a
    plain-language setup hint surfaces if nothing is reachable.
    """
    ak = os.getenv("ANTHROPIC_API_KEY", "")
    ok = os.getenv("OPENAI_API_KEY", "")
    if ak.startswith("sk-ant-") and "..." not in ak:
        return JudgeConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.0)
    if ok.startswith("sk-") and "..." not in ok:
        return JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.0)
    base = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    try:
        import urllib.request
        probe = base.rstrip("/").rsplit("/v1", 1)[0] + "/api/tags"
        with urllib.request.urlopen(probe, timeout=0.5):
            pass
        os.environ.setdefault("OPENAI_API_KEY", "ollama-local-no-auth")
        return JudgeConfig(provider="openai", model="llama3",
                           base_url=base, temperature=0.0)
    except Exception:
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


# ─────────────────────────────────────────────────────────────────────
# agent-langgraph — real LangGraph agent with a graph-aware tracer
# ─────────────────────────────────────────────────────────────────────


_AGENT_LANGGRAPH_EVAL = '''\
"""LangGraph agent eval — order-support agent with ToolCallAccuracy.

A real LangGraph ReAct-style agent (LLM node + tools node + END), not a
hand-rolled toy. ``LangGraphTracer`` listens to the graph's callback
events and captures the trace as ``list[AgentStep]`` — one step per
LLM turn, with tool calls grouped under the model decision that made
them.

NEEDS AN API KEY. LangGraph drives a real chat model. Set
ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment (or .env),
or point ``ChatOpenAI(base_url=...)`` at a local Ollama / LM Studio
server.
"""
import os
from typing import Annotated

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from multivon_eval import EvalSuite, EvalCase, LangGraphTracer
from multivon_eval.evaluators import ToolCallAccuracy


def _chat_model():
    """Pick whichever chat model the user has configured."""
    if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    if os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raise RuntimeError(
        "No chat model configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
        "or edit _chat_model() to point at a local Ollama (ChatOpenAI with base_url=...)."
    )


# ── Tools ─────────────────────────────────────────────────────────────

ORDERS = {
    "O-101": {"status": "shipped", "total": 49.99, "refunded": False},
    "O-202": {"status": "delivered", "total": 19.99, "refunded": True},
    "O-303": {"status": "processing", "total": 79.99, "refunded": False},
}


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by ID. Returns status, total, refund status."""
    return ORDERS.get(order_id, {"error": "order not found"})


@tool
def refund_order(order_id: str) -> dict:
    """Refund the given order. Refuses if already refunded or still processing."""
    order = ORDERS.get(order_id)
    if not order:
        return {"error": "order not found"}
    if order["refunded"]:
        return {"error": "already refunded", "refund_id": None}
    if order["status"] == "processing":
        return {"error": "cannot refund a processing order", "refund_id": None}
    return {"refund_id": f"R-{order_id}", "amount": order["total"], "status": "approved"}


TOOLS = [lookup_order, refund_order]


# ── Build the graph ──────────────────────────────────────────────────

def _build_graph():
    """Standard ReAct: model decides → tools (optional) → model again → END."""
    llm = _chat_model().bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


# ── Wire to multivon-eval ────────────────────────────────────────────

GRAPH = _build_graph()
TRACER = LangGraphTracer()


def support_agent(input_text: str, **kwargs) -> str:
    """Suite-compatible signature. Forwards callbacks so the tracer
    captures the LangGraph callback events."""
    result = GRAPH.invoke(
        {"messages": [HumanMessage(content=input_text)]},
        config={"callbacks": kwargs.get("callbacks", [])},
    )
    return result["messages"][-1].content


# Five cases that exercise framework value — not just the happy path.
cases = [
    EvalCase(
        input="Where is order O-101?",
        expected_tool_calls=["lookup_order"],
    ),
    EvalCase(
        input="Refund order O-101.",
        expected_tool_calls=["lookup_order", "refund_order"],
    ),
    EvalCase(
        input="Refund order O-202.",
        # Already refunded — well-behaved agent looks BEFORE attempting
        # to refund again. We accept either trajectory (look only, or
        # look + try-refund-which-fails).
        expected_tool_calls=["lookup_order"],
    ),
    EvalCase(
        input="Where is order O-999?",
        # Nonexistent order — agent should look it up, get "not found",
        # and stop. Don't call refund_order.
        expected_tool_calls=["lookup_order"],
    ),
    EvalCase(
        input="Status of O-303 and refund it if possible.",
        # Processing — model should look up, see processing, NOT refund.
        expected_tool_calls=["lookup_order"],
    ),
]


suite = EvalSuite("langgraph-support-agent")
suite.add_cases(cases)
# ``penalize_unexpected=True`` makes the negative cases (e.g. already
# refunded, processing) actually fail when the agent calls refund_order
# anyway. Without it, ToolCallAccuracy only checks "did the expected
# tools fire" and ignores extras — a real false-positive risk.
suite.add_evaluator(ToolCallAccuracy(penalize_unexpected=True))


if __name__ == "__main__":
    import os
    report = suite.run(support_agent, tracer=TRACER)
    os.makedirs("eval-reports", exist_ok=True)
    report.save_json("eval-reports/langgraph-agent.json")
    print(f"Saved report to eval-reports/langgraph-agent.json")
    print(f"  multivon-eval view eval-reports/langgraph-agent.json   # interactive HTML")
'''


_AGENT_LANGGRAPH_README = """\
# multivon-eval — LangGraph agent template

Evaluate a real LangGraph ReAct agent with `ToolCallAccuracy`.
`LangGraphTracer` hooks into the graph's LangChain callbacks and
captures the trace as one `AgentStep` per LLM turn.

## 3-command flow

```bash
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY or OPENAI_API_KEY
python eval.py
```

## How the tracer wires in

The crucial bit is **forwarding callbacks** to LangGraph. Your
`model_fn` must accept `**kwargs` and pass `kwargs.get("callbacks", [])`
into the graph's invoke config:

```python
def support_agent(input_text: str, **kwargs) -> str:
    result = GRAPH.invoke(
        {"messages": [HumanMessage(content=input_text)]},
        config={"callbacks": kwargs.get("callbacks", [])},
    )
    return result["messages"][-1].content
```

The suite calls `tracer.instrument(support_agent)`, which injects the
tracer's callback handler into `kwargs["callbacks"]`. Forget the
`**kwargs` and the trace will be silently empty — flagged by
``ToolCallAccuracy`` failing every case.

## What this template demonstrates

- **Real LangGraph graph** — `StateGraph` + `MessagesState` + `ToolNode` + `tools_condition`. No hand-rolled toy loop.
- **`LangGraphTracer`** — extends our LangChain integration with
  graph-node awareness. Tool calls inside a `tools` node are attached
  to the *preceding* LLM turn's `AgentStep`, which matches how
  agent eval frameworks model decisions.
- **Five cases** exercising real agent value:
  1. Simple lookup
  2. Lookup → refund (multi-step)
  3. Already-refunded order (agent should refuse a redundant refund)
  4. Nonexistent order (agent should NOT call refund after a not-found)
  5. Processing order (refund should NOT fire on a non-refundable state)
- **No `fail_threshold`** — this is a starter, not a hardened CI gate.
  Add it when your judge setup is reliable.

## Migrating from `multivon-eval init -t agent`

The toy `agent` template uses a hand-rolled tracer; this one uses
`LangGraphTracer` against a real graph. Evaluators and case schema
(`expected_tool_calls`) are **100% compatible** — just:

1. Replace `HandRolledTracer` with `LangGraphTracer`.
2. Add `**kwargs` to your `model_fn` signature and forward
   `kwargs.get("callbacks", [])` to your graph's `config={"callbacks": ...}`.
3. Build a real `StateGraph` instead of populating `steps: list[AgentStep]`
   by hand.

## Glossary

- `StateGraph` — LangGraph's graph builder. Nodes are functions; edges
  are routing rules.
- `MessagesState` — a built-in state schema that holds a `messages: list`.
- `ToolNode` — a prebuilt node that executes any tool calls in the last
  AI message and appends results to `messages`.
- `tools_condition` — a prebuilt routing function: if the last AI
  message has tool calls, go to the tools node; otherwise END.
- `HumanMessage` — a LangChain message type for user input.

## Next steps

- Add `TaskCompletion` (LLM-judge) to score the final answer, not just
  the tool trajectory.
- Add `TrajectoryEfficiency` to flag agents that meander.
- For multi-agent handoffs: file an issue with your graph shape —
  v1 of the tracer is single-agent.
"""


_AGENT_LANGGRAPH_REQS = """\
multivon-eval[langgraph]>=0.7.0
# Choose ONE chat model lib (delete the others):
langchain-anthropic>=0.3.0
langchain-openai>=0.3.0
# Or use a local Ollama by editing eval.py's _chat_model().
"""


_AGENT_LANGGRAPH_DOTENV = """\
# Pick ONE of these — leave the others commented or unset.
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
"""


# ─────────────────────────────────────────────────────────────────────
# agent-openai-sdk — OpenAI Agents SDK agent with post-hoc trace
# ─────────────────────────────────────────────────────────────────────


_AGENT_OPENAI_SDK_EVAL = '''\
"""OpenAI Agents SDK eval — order-support agent.

A real ``agents.Agent`` driven by ``Runner.run_sync``. ``OpenAIAgentsTracer``
reads the trace from ``RunResult.new_items`` after each run — no global
trace processor, no shared state across cases.

NEEDS AN OPENAI API KEY. The Agents SDK uses OpenAI by default. Set
OPENAI_API_KEY in your environment (or .env).
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agents import Agent, Runner, function_tool

from multivon_eval import EvalSuite, EvalCase, OpenAIAgentsTracer
from multivon_eval.evaluators import ToolCallAccuracy


def _check_key():
    """Defer the OPENAI_API_KEY check to runtime so users can still
    `python -c "import eval"` and read the file without a key."""
    if not os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
        raise RuntimeError(
            "OpenAI Agents SDK needs OPENAI_API_KEY. Set it in .env or your "
            "shell (export OPENAI_API_KEY=sk-...), then re-run."
        )


# ── Tools (function_tool decorator registers them with the SDK) ──────

ORDERS = {
    "O-101": {"status": "shipped", "total": 49.99, "refunded": False},
    "O-202": {"status": "delivered", "total": 19.99, "refunded": True},
    "O-303": {"status": "processing", "total": 79.99, "refunded": False},
}


@function_tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by ID. Returns status, total, refund status."""
    return ORDERS.get(order_id, {"error": "order not found"})


@function_tool
def refund_order(order_id: str) -> dict:
    """Refund the given order. Refuses if already refunded or processing."""
    order = ORDERS.get(order_id)
    if not order:
        return {"error": "order not found"}
    if order["refunded"]:
        return {"error": "already refunded", "refund_id": None}
    if order["status"] == "processing":
        return {"error": "cannot refund a processing order", "refund_id": None}
    return {"refund_id": f"R-{order_id}", "amount": order["total"], "status": "approved"}


# ── Agent ────────────────────────────────────────────────────────────

support_agent_sdk = Agent(
    name="OrderSupport",
    instructions=(
        "You are a customer support agent. Look up orders before any "
        "refund action. Do not refund orders that are already refunded "
        "or still processing — report the situation to the user instead."
    ),
    tools=[lookup_order, refund_order],
    model="gpt-4o-mini",
)


# ── Wire to multivon-eval ────────────────────────────────────────────

TRACER = OpenAIAgentsTracer()


def support_agent(input_text: str) -> str:
    """Suite-compatible signature. Captures the trace post-hoc from
    the SDK's RunResult — no shared state with other cases."""
    result = Runner.run_sync(support_agent_sdk, input_text)
    TRACER.capture(result)
    return result.final_output


cases = [
    EvalCase(
        input="Where is order O-101?",
        expected_tool_calls=["lookup_order"],
    ),
    EvalCase(
        input="Refund order O-101.",
        expected_tool_calls=["lookup_order", "refund_order"],
    ),
    EvalCase(
        input="Refund order O-202.",
        expected_tool_calls=["lookup_order"],
    ),
    EvalCase(
        input="Where is order O-999?",
        expected_tool_calls=["lookup_order"],
    ),
    EvalCase(
        input="Status of O-303 and refund it if possible.",
        expected_tool_calls=["lookup_order"],
    ),
]


suite = EvalSuite("openai-agents-support")
suite.add_cases(cases)
# ``penalize_unexpected=True`` makes the negative cases (already
# refunded, processing) actually fail when the agent calls refund_order
# anyway. Without it, an over-eager agent could refund both
# already-refunded AND processing orders and still score 100%.
suite.add_evaluator(ToolCallAccuracy(penalize_unexpected=True))


if __name__ == "__main__":
    import os
    _check_key()
    report = suite.run(support_agent, tracer=TRACER)
    os.makedirs("eval-reports", exist_ok=True)
    report.save_json("eval-reports/openai-agents.json")
    print(f"Saved report to eval-reports/openai-agents.json")
    print(f"  multivon-eval view eval-reports/openai-agents.json   # interactive HTML")
'''


_AGENT_OPENAI_SDK_README = """\
# multivon-eval — OpenAI Agents SDK template

Evaluate a real `agents.Agent` driven by `Runner.run_sync`.
`OpenAIAgentsTracer` reads `RunResult.new_items` after the run —
no global trace processor, no leakage across concurrent cases.

## 3-command flow

```bash
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
python eval.py
```

## How the tracer wires in

`OpenAIAgentsTracer` is **post-hoc**: it reads `RunResult.new_items`
after the SDK's `Runner` finishes. You MUST call `tracer.capture(result)`
inside your `model_fn`, BEFORE returning the string output. Forget
this step and the trace will be empty:

```python
def support_agent(input_text: str) -> str:
    result = Runner.run_sync(support_agent_sdk, input_text)
    TRACER.capture(result)        # <-- required
    return result.final_output
```

The suite has no way to unwrap a `RunResult` itself — capture
explicitly.

## What this template demonstrates

- **Real `agents.Agent`** — `function_tool`-decorated tools, instructions,
  `Runner.run_sync`. No hand-rolled loop.
- **`OpenAIAgentsTracer.capture(result)`** — post-hoc parse of the
  SDK's `RunResult.new_items` into our `AgentStep` model. One step
  per LLM turn; adjacent reasoning + message items merge into one
  step; tool calls and outputs reconcile by `call_id`.
- **Five cases** with non-trivial trajectories:
  1. Simple lookup
  2. Lookup → refund (multi-step)
  3. Already-refunded order (agent should refuse)
  4. Nonexistent order
  5. Processing order (refund should NOT fire)

## Live RunHooks variant (optional)

For event-time tracing (cancel on guardrail, stream into another
sink), use `tracer.run_hooks()` and `tracer.merge(hooks)` after the
async run. The hooks instance has its OWN buffer — safe for
concurrent runs:

```python
hooks = TRACER.run_hooks()
result = await Runner.run(support_agent_sdk, input_text, hooks=hooks)
TRACER.merge(hooks)
return result.final_output
```

Use the post-hoc `capture()` path unless you have a specific need to
intercept events live.

## Glossary

- `Agent` — the SDK's agent class. Defines instructions, tools, model.
- `function_tool` — a decorator that turns a Python function into a
  tool the agent can call. Schema is inferred from type hints.
- `Runner.run_sync(agent, input)` — runs the agent loop synchronously
  and returns a `RunResult`. Use `Runner.run(...)` for async.
- `RunResult.new_items` — the SDK's structured trace: a list of
  `MessageOutputItem`, `ReasoningItem`, `ToolCallItem`, `ToolCallOutputItem`,
  etc. This is what `tracer.capture(result)` reads.
- `RunResult.final_output` — the agent's final string output.

## Migrating from `multivon-eval init -t agent`

If you started with the toy `agent` template and now want to use the
real OpenAI Agents SDK:

1. Replace `HandRolledTracer` with `OpenAIAgentsTracer`.
2. Wrap your agent in `Runner.run_sync(...)` and call
   `TRACER.capture(result)` inside `model_fn`.
3. Cases (`expected_tool_calls`) stay the same.

## Next steps

- Add `TaskCompletion` to score the final answer (needs an LLM judge).
- Add `TrajectoryEfficiency` for meandering detection.
- Handoffs are captured as markers in `step.output` but not yet
  expanded into sub-trace AgentSteps — file an issue if you need
  full handoff expansion.
"""


_AGENT_OPENAI_SDK_REQS = """\
multivon-eval[openai-agents]>=0.7.0
"""


_AGENT_OPENAI_SDK_DOTENV = """\
OPENAI_API_KEY=sk-...
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
    "agent-langgraph": {
        "eval.py": _AGENT_LANGGRAPH_EVAL,
        "README.md": _AGENT_LANGGRAPH_README,
        "requirements.txt": _AGENT_LANGGRAPH_REQS,
        ".env.example": _AGENT_LANGGRAPH_DOTENV,
        ".gitignore": _GITIGNORE,
    },
    "agent-openai-sdk": {
        "eval.py": _AGENT_OPENAI_SDK_EVAL,
        "README.md": _AGENT_OPENAI_SDK_README,
        "requirements.txt": _AGENT_OPENAI_SDK_REQS,
        ".env.example": _AGENT_OPENAI_SDK_DOTENV,
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
    return [
        "quickstart", "rag",
        "agent", "agent-langgraph", "agent-openai-sdk",
        "conversation", "regulated",
    ]


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
