"""D16: LangGraphTracer — graph-node-aware tracer for LangGraph agents.

Tests are MOCKED — they feed synthesized LangChain callback events to
the tracer's handler directly, without requiring langgraph to be
installed. Verified contract: one AgentStep per LLM turn, tool calls
attached to the preceding LLM turn, last step preserved (codex D16
cycle 2 finding), subgraph and parallel-tool safe.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from multivon_eval import AgentStep, ToolCall

# Skip module entirely if langchain-core isn't installed. Most CI matrices
# include it via the [langgraph] extra; local dev installs it transitively.
pytest.importorskip("langchain_core")

from multivon_eval.integrations.langgraph import (
    LangGraphTracer, _parse_tool_args, _stringify_tool_result, _extract_llm_text,
)


def _llm_response(text: str):
    """Build a minimal object that mimics LangChain's ChatResult so
    ``_extract_llm_text`` can find the assistant message."""
    msg = SimpleNamespace(content=text)
    gen = SimpleNamespace(message=msg, text=text)
    return SimpleNamespace(generations=[[gen]])


def _node_meta(name: str, step: int = 1, ns: str = ""):
    return {
        "langgraph_node": name,
        "langgraph_step": step,
        "langgraph_checkpoint_ns": ns,
    }


def _drive_react_turn(handler, *, agent_text: str, tool_name: str,
                       tool_input: str, tool_output: str,
                       agent_run_id=None, llm_run_id=None,
                       tool_run_id=None, tool_node_run_id=None):
    """Simulate one ReAct turn: agent node → on_llm → on_tool → end-of-tools-node."""
    agent_run_id = agent_run_id or uuid4()
    llm_run_id = llm_run_id or uuid4()
    tool_node_run_id = tool_node_run_id or uuid4()
    tool_run_id = tool_run_id or uuid4()
    # Agent node start
    handler.on_chain_start(
        {"name": "agent"}, {}, run_id=agent_run_id, parent_run_id=uuid4(),
        tags=["graph:step:1"], metadata=_node_meta("agent"),
    )
    # LLM call inside the agent node
    handler.on_chat_model_start({"name": "llm"}, [], run_id=llm_run_id)
    handler.on_llm_end(_llm_response(agent_text), run_id=llm_run_id)
    handler.on_chain_end({"output": agent_text}, run_id=agent_run_id)
    # Tools node — fires AFTER the agent decided to call
    handler.on_chain_start(
        {"name": "tools"}, {}, run_id=tool_node_run_id, parent_run_id=uuid4(),
        tags=["graph:step:2"], metadata=_node_meta("tools", step=2),
    )
    handler.on_tool_start({"name": tool_name}, tool_input, run_id=tool_run_id)
    handler.on_tool_end(tool_output, run_id=tool_run_id)
    handler.on_chain_end({}, run_id=tool_node_run_id)
    return agent_run_id, tool_run_id


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_tool_args_handles_json_and_plain():
    """JSON-shaped inputs parse to dicts; plain strings wrap in {input: ...}."""
    assert _parse_tool_args('{"order_id": "O-101"}') == {"order_id": "O-101"}
    assert _parse_tool_args("not json") == {"input": "not json"}
    assert _parse_tool_args("") == {}
    assert _parse_tool_args("{malformed") == {"input": "{malformed"}


def test_stringify_tool_result_keeps_strings_jsonifies_structures():
    assert _stringify_tool_result("hi") == "hi"
    assert _stringify_tool_result({"a": 1}) == '{"a": 1}'
    assert _stringify_tool_result(42) == "42"
    # Non-JSON-serializable falls back to str()
    obj = object()
    out = _stringify_tool_result(obj)
    assert isinstance(out, str) and "object" in out


def test_extract_llm_text_handles_str_content_and_block_list():
    """Plain string content + list-of-blocks content both extract cleanly."""
    str_resp = _llm_response("simple")
    assert _extract_llm_text(str_resp) == "simple"

    block_resp = SimpleNamespace(generations=[[SimpleNamespace(
        message=SimpleNamespace(content=[
            {"type": "text", "text": "block1"},
            {"type": "text", "text": "block2"},
        ]),
        text="",
    )]])
    out = _extract_llm_text(block_resp)
    assert "block1" in out and "block2" in out


def test_extract_llm_text_returns_empty_on_garbage():
    """Unfamiliar response shapes don't crash — return ''."""
    assert _extract_llm_text(SimpleNamespace()) == ""
    assert _extract_llm_text(None) == ""


# ─────────────────────────────────────────────────────────────────────────────
# single-turn ReAct
# ─────────────────────────────────────────────────────────────────────────────

def test_single_turn_react_produces_one_step():
    """Agent decides → calls tool → tool result. After the LAST turn,
    get_trace() must flush the in-flight step (codex D16 cycle 2)."""
    tracer = LangGraphTracer()
    handler = tracer._build_handler()
    _drive_react_turn(
        handler,
        agent_text="Looking up the order.",
        tool_name="lookup_order",
        tool_input='{"order_id": "O-101"}',
        tool_output='{"status": "shipped"}',
    )
    steps = tracer.get_trace()
    assert len(steps) == 1, f"expected 1 step, got {len(steps)}"
    s = steps[0]
    assert "Looking up the order" in s.thought
    assert len(s.tool_calls) == 1
    tc = s.tool_calls[0]
    assert tc.name == "lookup_order"
    assert tc.arguments == {"order_id": "O-101"}
    assert tc.result == '{"status": "shipped"}'


def test_two_turn_react_produces_two_steps():
    """Two LLM turns (lookup → refund) = two steps. The tools from each
    turn attach to its preceding LLM turn."""
    tracer = LangGraphTracer()
    handler = tracer._build_handler()

    _drive_react_turn(
        handler,
        agent_text="Will look it up.",
        tool_name="lookup_order",
        tool_input='{"order_id": "O-101"}',
        tool_output='{"status": "shipped"}',
    )
    _drive_react_turn(
        handler,
        agent_text="Order found. Refunding.",
        tool_name="refund_order",
        tool_input='{"order_id": "O-101"}',
        tool_output='{"refund_id": "R-O-101"}',
    )
    # Final assistant turn after the tool returns
    final_id = uuid4()
    agent_id = uuid4()
    handler.on_chain_start(
        {"name": "agent"}, {}, run_id=agent_id, parent_run_id=uuid4(),
        tags=["graph:step:3"], metadata=_node_meta("agent", step=3),
    )
    handler.on_chat_model_start({"name": "llm"}, [], run_id=final_id)
    handler.on_llm_end(_llm_response("Refund R-O-101 approved."), run_id=final_id)
    handler.on_chain_end({}, run_id=agent_id)

    steps = tracer.get_trace()
    # 3 turns: lookup-then-tool, refund-then-tool, final answer (no tools)
    assert len(steps) == 3, f"expected 3 steps, got {len(steps)} ({[(s.thought, [t.name for t in s.tool_calls]) for s in steps]})"
    assert steps[0].tool_calls[0].name == "lookup_order"
    assert steps[1].tool_calls[0].name == "refund_order"
    assert steps[2].tool_calls == []  # final turn, no tools
    assert "Refund R-O-101 approved" in steps[2].thought


# ─────────────────────────────────────────────────────────────────────────────
# Codex D16 cycle 2 regressions
# ─────────────────────────────────────────────────────────────────────────────

def test_last_step_is_not_lost_after_only_one_llm_turn():
    """Cycle 2 ISSUE 1: get_trace() must flush the in-flight step,
    not just return ``self._steps``. A single-turn run was previously
    silently dropped."""
    tracer = LangGraphTracer()
    handler = tracer._build_handler()
    # ONE LLM call, no follow-up — previously lost.
    run_id = uuid4()
    handler.on_chat_model_start({"name": "llm"}, [], run_id=run_id)
    handler.on_llm_end(_llm_response("Hi there."), run_id=run_id)
    assert len(tracer.get_trace()) == 1
    assert tracer.get_trace()[0].thought == "Hi there."


def test_parallel_tool_calls_in_one_node_attach_to_same_step():
    """Two tools firing in parallel from one ``tools`` node should
    both attach to the preceding LLM turn's step. Codex flagged
    concurrent tool tracking via run_id."""
    tracer = LangGraphTracer()
    handler = tracer._build_handler()
    # Agent decides; on_llm_start opens the step.
    llm_id = uuid4()
    handler.on_chat_model_start({"name": "llm"}, [], run_id=llm_id)
    handler.on_llm_end(_llm_response("Look up two orders in parallel."), run_id=llm_id)
    # Tools node starts.
    tools_node_id = uuid4()
    handler.on_chain_start(
        {"name": "tools"}, {}, run_id=tools_node_id, parent_run_id=uuid4(),
        tags=["graph:step:1"], metadata=_node_meta("tools"),
    )
    # Two parallel tool calls — interleaved start events.
    t1_id, t2_id = uuid4(), uuid4()
    handler.on_tool_start({"name": "lookup_order"}, '{"order_id": "O-1"}', run_id=t1_id)
    handler.on_tool_start({"name": "lookup_order"}, '{"order_id": "O-2"}', run_id=t2_id)
    # Outputs arrive in REVERSE order.
    handler.on_tool_end('{"status": "shipped"}', run_id=t2_id)
    handler.on_tool_end('{"status": "processing"}', run_id=t1_id)
    handler.on_chain_end({}, run_id=tools_node_id)

    steps = tracer.get_trace()
    assert len(steps) == 1
    tools = steps[0].tool_calls
    assert len(tools) == 2
    # Each tool's result matches its OWN call_id (codex worried about
    # interleaving clobbering this).
    by_order = {t.arguments["order_id"]: t.result for t in tools}
    assert by_order["O-1"] == '{"status": "processing"}'
    assert by_order["O-2"] == '{"status": "shipped"}'


def test_empty_routing_node_does_not_pollute_trace():
    """A node that opens and immediately ends without any LLM or tool
    event should NOT produce an empty step in the trace."""
    tracer = LangGraphTracer()
    handler = tracer._build_handler()
    rid = uuid4()
    handler.on_chain_start(
        {"name": "router"}, {}, run_id=rid, parent_run_id=uuid4(),
        tags=["graph:step:1"], metadata=_node_meta("router"),
    )
    handler.on_chain_end({}, run_id=rid)
    assert tracer.get_trace() == []


def test_tool_error_records_error_marker():
    """A tool raising shouldn't crash the trace; the error is recorded
    in the tool call's result so evaluators can see it."""
    tracer = LangGraphTracer()
    handler = tracer._build_handler()
    llm_id = uuid4()
    handler.on_chat_model_start({"name": "llm"}, [], run_id=llm_id)
    handler.on_llm_end(_llm_response("call the tool"), run_id=llm_id)
    tool_id = uuid4()
    handler.on_tool_start({"name": "lookup_order"}, '{}', run_id=tool_id)
    handler.on_tool_error(RuntimeError("boom"), run_id=tool_id)
    steps = tracer.get_trace()
    assert len(steps) == 1
    assert "[ERROR" in steps[0].tool_calls[0].result
    assert "boom" in steps[0].tool_calls[0].result


def test_reset_between_cases_does_not_leak():
    """Behavioral guarantee: after ``reset()``, the trace is empty AND
    a stale handler from before the reset can't write into the next
    case's trace. Codex D16 cycle 4 — test behavior, not the
    ``_live_handler is None`` private detail."""
    tracer = LangGraphTracer()
    stale_handler = tracer._build_handler()
    rid = uuid4()
    stale_handler.on_chat_model_start({}, [], run_id=rid)
    stale_handler.on_llm_end(_llm_response("first"), run_id=rid)
    assert len(tracer.get_trace()) == 1

    tracer.reset()
    # New case starts: build a fresh handler and run a different LLM turn.
    fresh_handler = tracer._build_handler()
    rid2 = uuid4()
    fresh_handler.on_chat_model_start({}, [], run_id=rid2)
    fresh_handler.on_llm_end(_llm_response("second"), run_id=rid2)

    # Behavior check: get_trace() must NOT include the stale "first"
    # text. The stale handler is GC-eligible but even if held alive,
    # its in-flight step must not bleed into the new case.
    trace = tracer.get_trace()
    assert all("first" not in s.thought for s in trace), (
        f"stale handler leaked into new case: {[s.thought for s in trace]}"
    )
    assert any("second" in s.thought for s in trace), (
        f"fresh handler didn't write through: {[s.thought for s in trace]}"
    )


def test_serial_branches_do_not_cross_contaminate():
    """Codex D16 cycle 4 ISSUE 1: the tracer keeps a SINGLE
    ``_current_step``. If LangGraph fires two LLM turns serially
    (typical: model → tools → model), the second turn's start must
    cleanly close the first — no cross-attribution of tools.

    True interleaved parallel branches (``Send`` API, parallel
    subgraphs) are a known v1 limitation, documented in the tracer's
    class docstring. This test locks the SERIAL contract.
    """
    tracer = LangGraphTracer()
    handler = tracer._build_handler()

    # Turn 1: model decides, tool fires, result returns.
    llm1 = uuid4()
    tool1 = uuid4()
    handler.on_chat_model_start({}, [], run_id=llm1)
    handler.on_llm_end(_llm_response("Use tool A."), run_id=llm1)
    handler.on_tool_start({"name": "toolA"}, '{}', run_id=tool1)
    handler.on_tool_end("resultA", run_id=tool1)

    # Turn 2: NEW model turn closes Turn 1's step.
    llm2 = uuid4()
    tool2 = uuid4()
    handler.on_chat_model_start({}, [], run_id=llm2)
    handler.on_llm_end(_llm_response("Use tool B."), run_id=llm2)
    handler.on_tool_start({"name": "toolB"}, '{}', run_id=tool2)
    handler.on_tool_end("resultB", run_id=tool2)

    steps = tracer.get_trace()
    assert len(steps) == 2, f"serial turns must produce 2 steps, got {len(steps)}"
    # Strict attribution: toolA belongs to step 1 (says "Use tool A"),
    # toolB to step 2 (says "Use tool B"). No cross-contamination.
    assert "Use tool A" in steps[0].thought
    assert [t.name for t in steps[0].tool_calls] == ["toolA"]
    assert "Use tool B" in steps[1].thought
    assert [t.name for t in steps[1].tool_calls] == ["toolB"]
