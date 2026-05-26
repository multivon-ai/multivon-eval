"""
python -m multivon_eval          # run the demo (default)
python -m multivon_eval demo     # same
python -m multivon_eval --help

The demo runs a self-contained customer-support eval with no setup required.
It auto-detects available LLM backends and adds judge evaluators when found:

  Tier 1 — always:   deterministic checks (NotEmpty, WordCount)
  Tier 2 — API key:  LLM judge via ANTHROPIC_API_KEY or OPENAI_API_KEY
  Tier 3 — local:    same LLM checks via OPENAI_BASE_URL or Ollama on :11434
"""
from __future__ import annotations

import os
import socket
import sys


# ── Tier detection ─────────────────────────────────────────────────────────────

def _port_open(port: int, host: str = "localhost") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def _detect_judge() -> tuple[str, str, str]:
    """Return (provider, model, base_url). Empty strings = no LLM available."""
    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Cloud APIs
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-haiku-4-5-20251001", ""
    if os.getenv("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini", ""

    # Explicit custom base URL (any OpenAI-compatible server)
    if os.getenv("OPENAI_BASE_URL"):
        model = os.getenv("DEMO_MODEL", "llama3")
        return "openai", model, os.getenv("OPENAI_BASE_URL", "")

    # Ollama running locally on :11434
    if _port_open(11434):
        model = os.getenv("DEMO_MODEL", "llama3")
        return "openai", model, "http://localhost:11434/v1"

    # LM Studio on :1234
    if _port_open(1234):
        model = os.getenv("DEMO_MODEL", "local-model")
        return "openai", model, "http://localhost:1234/v1"

    return "", "", ""


def _judge_reachable(cfg) -> tuple[bool, str]:
    """Liveness-probe the detected judge with one tiny call.

    The "no setup" demo must never crash. A backend can be *detected* yet
    unusable — e.g. Ollama is listening on :11434 but the model isn't pulled,
    or a stale OPENAI_BASE_URL points at a dead server. Probing first lets the
    demo fall back to the deterministic tier with a clear note instead of
    dumping a traceback. Returns (ok, first_line_of_reason).
    """
    from multivon_eval.judge import make_judge_call
    try:
        make_judge_call("Reply with the single word: ok", cfg)
        return True, ""
    except Exception as exc:  # a demo probe must never raise
        text = str(exc).strip()
        return False, (text.splitlines()[0] if text else type(exc).__name__)


# ── Demo ───────────────────────────────────────────────────────────────────────

_RESPONSES: dict[str, str] = {
    "return":   "You can return any item within 30 days of purchase for a full refund.",
    "password": "Click 'Forgot Password' on the login page to receive a reset link by email.",
    "shipping": "Yes, we offer free standard shipping on all orders over $50.",
    "hours":    "",   # intentional empty — NotEmpty will catch it
    "located":  "Please reach out to our support team and they can assist you.",  # evasive
    "cancel":   "Orders can be cancelled within 24 hours of placement via your account page.",
}


def _demo_model(question: str) -> str:
    q = question.lower()
    for key, response in _RESPONSES.items():
        if key in q:
            return response
    return "Please contact our support team for assistance."


_DEMO_CASES_DATA = [
    ("What is your return policy?",   "30 days"),
    ("How do I reset my password?",   "reset link"),
    ("Do you offer free shipping?",   "$50"),
    ("What are your business hours?", "hours"),
    ("Where are you located?",        "address"),
    ("Can I cancel my order?",        "24 hours"),
]


def _run_demo() -> None:
    from multivon_eval import (
        EvalSuite, EvalCase, configure, JudgeConfig,
        NotEmpty, WordCount,
    )

    cases = [EvalCase(input=q, expected_output=exp) for q, exp in _DEMO_CASES_DATA]

    provider, model_name, base_url = _detect_judge()
    has_llm = bool(provider)
    judge_down_reason = ""

    suite = EvalSuite("multivon-eval demo · customer support bot")
    suite.add_cases(cases)

    # Tier 1 — always
    suite.add_evaluators(NotEmpty(), WordCount(min_words=5))

    # Tier 2 / 3 — LLM judge. Probe the detected backend first so a
    # detected-but-unreachable judge degrades to deterministic-only instead of
    # crashing the "no setup" demo with a traceback.
    if has_llm:
        cfg = JudgeConfig(provider=provider, model=model_name, base_url=base_url)
        ok, judge_down_reason = _judge_reachable(cfg)
        if ok:
            from multivon_eval import Relevance
            configure(cfg)
            suite.add_evaluators(Relevance())
            suite.add_check("Response directly answers the customer's question")
        else:
            has_llm = False

    # Header
    _sep = "─" * 56
    print(f"\n  {_sep}")
    print("  multivon-eval · demo")
    print(f"  {_sep}")
    print("  6 customer-support questions · simulated model\n")

    if has_llm:
        src = base_url if base_url else f"{provider}"
        print(f"  LLM judge : {model_name}  [{src}]")
        print(f"  Tier 1    : NotEmpty, WordCount  (deterministic)")
        print(f"  Tier 2    : Relevance, add_check  (LLM-as-judge)")
    elif judge_down_reason:
        src = base_url if base_url else provider
        print(f"  LLM judge : detected at [{src}] but unreachable — {judge_down_reason}")
        print("  Tier 1    : NotEmpty, WordCount  (deterministic only)")
        print()
        print("  Running the deterministic tier only. Fix the judge above")
        print("  (e.g. `ollama pull qwen2.5:14b`) to enable LLM evaluators.")
    else:
        print("  LLM judge : not detected")
        print("  Tier 1    : NotEmpty, WordCount  (deterministic only)")
        print()
        print("  To enable LLM evaluators, set one of:")
        print("    ANTHROPIC_API_KEY   — Anthropic API")
        print("    OPENAI_API_KEY      — OpenAI API")
        print("    OPENAI_BASE_URL     — any OpenAI-compatible endpoint")
        print("    (or start Ollama on localhost:11434)")

    print(f"\n  {_sep}\n")

    try:
        suite.run(_demo_model)
    except Exception as exc:
        # The "no setup" demo must never end in a traceback. The liveness probe
        # catches a dead judge up front, but a judge that passed the probe can
        # still fail mid-run (transient outage, a local model that answers the
        # one-word probe but errors on the longer QAG prompt). suite.run() calls
        # evaluator.prepare() in a bare loop (suite.py) so that failure would
        # otherwise propagate. Drop to the deterministic tier and finish clean.
        if not has_llm:
            raise  # deterministic-only never calls a judge — a real bug, don't mask it
        text = str(exc).strip()
        reason = text.splitlines()[0] if text else type(exc).__name__
        print(f"  LLM judge failed mid-run ({reason}).")
        print("  Re-running the deterministic tier only.\n")
        det = EvalSuite("multivon-eval demo · customer support bot (deterministic only)")
        det.add_cases(cases)
        det.add_evaluators(NotEmpty(), WordCount(min_words=5))
        det.run(_demo_model)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else "demo"

    # No-arg call keeps the legacy "run the demo" behavior so `pip install
    # multivon-eval && python -m multivon_eval` stays a 1-liner.
    if cmd in ("demo", ""):
        _run_demo()
        return

    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return

    # Anything else routes to the CLI so `python -m multivon_eval init`
    # works identically to the `multivon-eval` console-script entry.
    from . import cli
    cli.main()


if __name__ == "__main__":
    main()
