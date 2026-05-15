"""D15: beginner-friendly fixes.

Locks in the contracts from the OSS-adoption audit:
  - `multivon-eval init -t agent` runs OFFLINE without an API key
  - Public accessors on EvalSuite (`evaluators`, `cases`) and on
    CheckEvaluator (`criterion`) so users don't reach for private
    underscored attrs
  - JudgeUnavailable carries an actionable setup hint when the
    underlying exception looks like missing-credentials or
    can't-reach-server
  - `AgentTracer.format_trace` / `print_trace` for agent debugging
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from multivon_eval import (
    AgentStep, AgentTracer, EvalCase, EvalSuite, ToolCall,
)
from multivon_eval.evaluators.llm_judge import CheckEvaluator
from multivon_eval.judge import _looks_like_auth_or_connection_error, _wrap_provider_error


# ─────────────────────────────────────────────────────────────────────────────
# Public accessors — Suite.evaluators / .cases / CheckEvaluator.criterion
# ─────────────────────────────────────────────────────────────────────────────

def test_suite_evaluators_is_public_and_returns_copy():
    suite = EvalSuite("test")
    suite.add_check("Response is concise")
    suite.add_case(EvalCase("hi"))
    listed = suite.evaluators
    assert len(listed) == 1
    # Mutating the return value must NOT mutate the suite (returns a copy).
    listed.clear()
    assert len(suite.evaluators) == 1


def test_suite_cases_is_public_and_returns_copy():
    suite = EvalSuite("test")
    suite.add_case(EvalCase("a"))
    suite.add_case(EvalCase("b"))
    listed = suite.cases
    assert [c.input for c in listed] == ["a", "b"]
    listed.clear()
    assert len(suite.cases) == 2


def test_check_evaluator_criterion_is_public():
    ev = CheckEvaluator("Response explains the refund policy")
    assert ev.criterion == "Response explains the refund policy"


# ─────────────────────────────────────────────────────────────────────────────
# Actionable judge-missing error
# ─────────────────────────────────────────────────────────────────────────────

def test_looks_like_auth_or_connection_error_matches_common_signatures():
    """Match the most common SDK-side exception messages a beginner
    hits: missing API key, 401, connection refused."""
    class _Fake401(Exception): ...
    class _Conn(Exception): ...

    # By exception name
    assert _looks_like_auth_or_connection_error(
        type("AuthenticationError", (Exception,), {})("...")
    )
    assert _looks_like_auth_or_connection_error(
        type("APIConnectionError", (Exception,), {})("...")
    )
    # By message content
    assert _looks_like_auth_or_connection_error(_Fake401("Error code: 401 unauthorized"))
    assert _looks_like_auth_or_connection_error(_Conn("could not connect to host"))
    assert _looks_like_auth_or_connection_error(Exception("missing api_key"))


def test_looks_like_auth_does_not_match_real_quality_bugs():
    """A real ValueError about prompt formatting must NOT be flagged as
    auth. Otherwise we'd drown real bugs in 'check your key' noise."""
    assert not _looks_like_auth_or_connection_error(
        ValueError("expected list[dict], got str at line 12")
    )
    assert not _looks_like_auth_or_connection_error(
        RuntimeError("model returned empty completion")
    )


def test_looks_like_auth_does_not_match_generic_api_errors():
    """Codex round-1: BadRequestError and APIError used to trigger the
    auth hint by NAME alone. But those classes fire for prompt-too-long,
    invalid model id, unsupported params, schema errors, etc. — real
    bugs the user needs to see UNADORNED, not drowned in 'check your
    key' advice. Narrow the whitelist."""
    BadRequest = type("BadRequestError", (Exception,), {})
    api_err = type("APIError", (Exception,), {})
    # Plain class-name match no longer counts.
    assert not _looks_like_auth_or_connection_error(
        BadRequest("Invalid 'model': unknown model id 'gpt-5'")
    )
    assert not _looks_like_auth_or_connection_error(
        api_err("Prompt length exceeds 128k token limit")
    )
    # ...but if the same exception's MESSAGE explicitly mentions auth,
    # we still surface the hint — content-based signals are targeted.
    assert _looks_like_auth_or_connection_error(BadRequest("api_key not set"))


def test_looks_like_auth_does_not_match_missing_or_not_found_in_message():
    """Codex round-1: 'missing' / 'not found' in a message are NOT
    auth-specific — they fire for missing keys in JSON, file-not-found,
    etc. Removed from the message-content signature."""
    assert not _looks_like_auth_or_connection_error(
        KeyError("missing required field 'context' in case")
    )
    assert not _looks_like_auth_or_connection_error(
        FileNotFoundError("eval-reports/baseline.json: not found")
    )


def test_wrap_provider_error_includes_setup_hint_on_auth():
    """Auth-shaped exception wrapped as JudgeUnavailable carries a
    concrete next-step block in its message."""
    class _Auth(Exception): ...
    _Auth.__name__ = "AuthenticationError"  # mimic the SDK class name
    wrapped = _wrap_provider_error("anthropic", "claude-haiku-4-5", _Auth("401"))
    msg = str(wrapped)
    assert "ANTHROPIC_API_KEY" in msg
    assert "console.anthropic.com" in msg
    assert "ollama" in msg.lower()
    assert "deterministic" in msg.lower() or "quickstart" in msg.lower()


def test_wrap_provider_error_skips_hint_for_real_bugs():
    """A real bug (e.g. JSON-shape error from the SDK) must NOT trip
    the hint path — drowning real bugs in setup advice would be worse
    than no hint."""
    wrapped = _wrap_provider_error(
        "openai", "gpt-4o-mini",
        ValueError("expected list[dict], got str at line 12"),
    )
    msg = str(wrapped)
    assert "expected list[dict]" in msg
    assert "OPENAI_API_KEY" not in msg
    assert "ollama" not in msg.lower()


def test_wrap_provider_error_picks_provider_specific_hint():
    """The setup hint mentions the provider that actually failed —
    not a generic both-keys block — so the user reaches for the right
    fix."""
    class _Auth(Exception): ...
    _Auth.__name__ = "AuthenticationError"

    msg_anthropic = str(_wrap_provider_error("anthropic", "claude", _Auth("401")))
    msg_openai = str(_wrap_provider_error("openai", "gpt-4o-mini", _Auth("401")))
    assert "ANTHROPIC_API_KEY" in msg_anthropic
    assert "OPENAI_API_KEY" not in msg_anthropic.split("OR run")[0]  # not in pre-fallback
    assert "OPENAI_API_KEY" in msg_openai


# ─────────────────────────────────────────────────────────────────────────────
# AgentTracer.format_trace / print_trace
# ─────────────────────────────────────────────────────────────────────────────

class _StubTracer(AgentTracer):
    """Bare-bones tracer for testing the new format helpers."""
    def instrument(self, fn): return fn
    def get_trace(self): return list(self._steps)


def test_format_trace_handles_empty_trace():
    assert AgentTracer.format_trace([]) == "(no trace captured)"
    assert AgentTracer.format_trace(None) == "(no trace captured)"


def test_format_trace_renders_steps_and_tool_calls():
    steps = [
        AgentStep(
            thought="Look up the order",
            tool_calls=[ToolCall(name="lookup_order",
                                  arguments={"order_id": "O-101"},
                                  result={"status": "shipped"})],
            output="Order shipped.",
        ),
    ]
    formatted = AgentTracer.format_trace(steps)
    assert "Step 1" in formatted
    assert "thought: Look up the order" in formatted
    assert "lookup_order(order_id='O-101')" in formatted
    assert "'status': 'shipped'" in formatted
    assert "output: Order shipped." in formatted


def test_format_trace_multiple_steps_numbered():
    steps = [
        AgentStep(thought="step a", tool_calls=[], output="a"),
        AgentStep(thought="step b", tool_calls=[], output="b"),
        AgentStep(thought="step c", tool_calls=[], output="c"),
    ]
    formatted = AgentTracer.format_trace(steps)
    assert "Step 1" in formatted
    assert "Step 2" in formatted
    assert "Step 3" in formatted


def test_print_trace_uses_get_trace_when_no_arg():
    """Called with no argument, prints the tracer's OWN captured steps —
    useful for interactive debugging right after running an agent."""
    t = _StubTracer()
    t._steps = [AgentStep(thought="hi", tool_calls=[], output="ok")]
    buf = io.StringIO()
    with redirect_stdout(buf):
        t.print_trace()
    assert "Step 1" in buf.getvalue()
    assert "thought: hi" in buf.getvalue()


def test_print_trace_accepts_explicit_steps():
    """Called with an explicit trace (e.g. a CaseResult's agent_trace),
    prints THAT trace — useful for inspecting failures from a saved
    report without re-running the agent."""
    t = _StubTracer()
    t._steps = []  # the tracer's own state is empty
    explicit = [AgentStep(thought="from CR", tool_calls=[], output="x")]
    buf = io.StringIO()
    with redirect_stdout(buf):
        t.print_trace(explicit)
    out = buf.getvalue()
    assert "from CR" in out


# ─────────────────────────────────────────────────────────────────────────────
# Agent template scaffolds AND runs without API key
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_template_runs_offline(tmp_path):
    """End-to-end: `multivon-eval init -t agent -d X` then `python eval.py`
    succeeds with NO API key set. This is the biggest beginner footgun
    we fixed — previously the judge-based evaluators silently 0-scored
    and the user thought their agent broke."""
    target = tmp_path / "agent_proj"
    # Scaffold
    rc = subprocess.run(
        [sys.executable, "-m", "multivon_eval", "init", "-t", "agent", "-d", str(target)],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, rc.stderr
    assert (target / "eval.py").exists()

    # Run with NO API key set
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    env["PYTHONPATH"] = ""  # don't accidentally pick up dev sources

    rc = subprocess.run(
        [sys.executable, "eval.py"],
        cwd=str(target), env=env,
        capture_output=True, text=True, timeout=60,
    )
    # Exit code is determined by fail_threshold inside eval.py — but
    # the offline run should NOT crash with a NameError or ImportError.
    assert rc.returncode in (0, 1), (
        f"agent eval crashed: rc={rc.returncode}\nstdout:\n{rc.stdout}\nstderr:\n{rc.stderr}"
    )
    # The "running offline" hint must show
    assert "Running offline" in rc.stdout or "Running offline" in rc.stderr, rc.stdout
    # And a report must be produced
    assert (target / "eval-reports" / "agent.json").exists()


def test_agent_template_eval_passes_offline(tmp_path):
    """A stricter assertion: the deterministic ToolCallAccuracy
    evaluator alone IS enough for the toy agent's cases to pass with
    score 1.0 — proving the offline path produces a meaningful PASS,
    not just 'didn't crash.'"""
    target = tmp_path / "agent_proj_strict"
    subprocess.run(
        [sys.executable, "-m", "multivon_eval", "init", "-t", "agent", "-d", str(target)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    rc = subprocess.run(
        [sys.executable, "eval.py"],
        cwd=str(target), env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert "PASS" in rc.stdout, f"no PASS found:\n{rc.stdout}"
    # Read the saved report and assert pass_rate == 1.0
    import json
    report = json.loads((target / "eval-reports" / "agent.json").read_text())
    assert report["summary"]["pass_rate"] == 1.0
