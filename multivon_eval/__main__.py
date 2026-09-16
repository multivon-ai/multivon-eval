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
import warnings


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
        model = os.getenv("DEMO_MODEL") or _first_ollama_model() or "llama3"
        return "openai", model, "http://localhost:11434/v1"

    # LM Studio on :1234
    if _port_open(1234):
        model = os.getenv("DEMO_MODEL", "local-model")
        return "openai", model, "http://localhost:1234/v1"

    return "", "", ""


def _first_ollama_model() -> str:
    """Name of an installed Ollama model (instruction-tuned text models
    preferred over vision/embedding ones), or "" if listing fails.

    Hardcoding a default like "llama3" makes the keyless demo fail for
    anyone who pulled different models — ask the server what it has.
    """
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=2
        ) as resp:
            names = [m.get("name", "") for m in json.load(resp).get("models", [])]
    except Exception:
        return ""
    names = [n for n in names if n]
    deprioritized = ("embed", "vision", "vl", "moondream", "llava")
    text_first = sorted(names, key=lambda n: any(d in n.lower() for d in deprioritized))
    return text_first[0] if text_first else ""


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


def _calibration_advisory(caught: list) -> str:
    """Turn a captured calibration-fallback UserWarning into a demo ⚠ line.

    Preserves the warning's information (the default threshold + validation note),
    just reformatted to match the demo's other advisory lines. Returns "" when
    no calibration warning was captured. Any warning is included exactly once.
    """
    for w in caught:
        msg = str(w.message)
        if "calibrat" in msg.lower() and "uncalibrated" in msg.lower():
            return (
                "  ⚠ Uncalibrated judge threshold: no calibration row for this "
                "judge model,\n"
                "    falling back to 0.7 (accuracy on your task is unknown).\n"
                "    Demo scores are indicative only — validate thresholds on "
                "held-out task data,\n"
                "    or set the fallback policy to \"strict\" to "
                "fail closed."
            )
    return ""


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

    # The calibration fallback warning (no threshold row for this judge model,
    # common with a freshly pulled local model) is real information, but a raw
    # Python UserWarning printed above the clean banner reads like a crash.
    # Capture it here and re-emit it through the demo's own ⚠ advisory channel
    # (below the banner) with its message intact. Scoped to the demo only — the
    # global warning policy is untouched for library callers.
    calibration_advice = ""

    # Tier 2 / 3 — LLM judge. Probe the detected backend first so a
    # detected-but-unreachable judge degrades to deterministic-only instead of
    # crashing the "no setup" demo with a traceback.
    if has_llm:
        cfg = JudgeConfig(provider=provider, model=model_name, base_url=base_url)
        ok, judge_down_reason = _judge_reachable(cfg)
        if ok:
            from multivon_eval import Relevance
            configure(cfg)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                suite.add_evaluators(Relevance())
                suite.add_check("Response directly answers the customer's question")
            calibration_advice = _calibration_advisory(caught)
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
        print("  Tier 1    : NotEmpty, WordCount  (deterministic)")
        print("  Tier 2    : Relevance, add_check  (LLM-as-judge)")
    elif judge_down_reason:
        src = base_url if base_url else provider
        lowered = judge_down_reason.lower()
        if "not found" in lowered or "404" in lowered:
            print(f"  LLM judge : server at [{src}] is up, but model "
                  f"'{model_name}' isn't available there")
            print("  Tier 1    : NotEmpty, WordCount  (deterministic only)")
            print()
            print(f"  Pull it (`ollama pull {model_name}`) or point the demo at")
            print("  a model you have: `DEMO_MODEL=<name> python -m multivon_eval`")
        else:
            print(f"  LLM judge : detected at [{src}] but the probe call "
                  f"failed — {judge_down_reason}")
            print("  Tier 1    : NotEmpty, WordCount  (deterministic only)")
            print()
            print("  Running the deterministic tier only. Fix the judge above")
            print("  (or set DEMO_MODEL=<name>) to enable LLM evaluators.")
    else:
        print("  LLM judge : not detected")
        print("  Tier 1    : NotEmpty, WordCount  (deterministic only)")
        print()
        print("  To enable LLM evaluators, set one of:")
        print("    ANTHROPIC_API_KEY   — Anthropic API")
        print("    OPENAI_API_KEY      — OpenAI API")
        print("    OPENAI_BASE_URL     — any OpenAI-compatible endpoint")
        print("    (or start Ollama on localhost:11434)")

    if calibration_advice:
        print()
        print(calibration_advice)

    print(f"\n  {_sep}\n")

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            suite.run(_demo_model)
        # A judge whose threshold is resolved lazily at run time (rather than at
        # prepare time) emits the same calibration warning here; surface it the
        # same way if it wasn't already shown above.
        if not calibration_advice:
            late_advice = _calibration_advisory(caught)
            if late_advice:
                print(late_advice)
        # Re-emit any non-calibration warnings so nothing is silently swallowed.
        for w in caught:
            if "calibrat" not in str(w.message).lower():
                warnings.warn_explicit(
                    w.message, w.category, w.filename, w.lineno
                )
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
