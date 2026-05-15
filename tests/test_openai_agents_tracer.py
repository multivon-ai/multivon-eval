"""D16: OpenAIAgentsTracer — captures traces from the OpenAI Agents SDK.

Tests are MOCKED — they construct fake ``RunResult.new_items`` lists
with the same shape the SDK produces, without requiring the SDK to be
installed. The tracer's parser uses class-name string matching for
items (codex D16 cycle 2 deferred the isinstance approach for v1).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from multivon_eval import AgentStep, ToolCall, OpenAIAgentsTracer
from multivon_eval.integrations.openai_agents import (
    _items_to_steps, _extract_item_text, _extract_tool_call,
    _extract_tool_output, _stringify,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fake item types — class names matter (the parser matches on them)
# ─────────────────────────────────────────────────────────────────────────────

class MessageOutputItem(SimpleNamespace): pass
class ReasoningItem(SimpleNamespace): pass
class ToolCallItem(SimpleNamespace): pass
class ToolCallOutputItem(SimpleNamespace): pass
class HandoffCallItem(SimpleNamespace): pass
class HandoffOutputItem(SimpleNamespace): pass


def _msg(text: str, *, cls=MessageOutputItem):
    """Build an item that looks like a Responses-API output message."""
    raw = SimpleNamespace(content=[SimpleNamespace(text=text)])
    return cls(raw_item=raw)


def _tool_call(*, call_id: str, name: str, args_json: str):
    return ToolCallItem(raw_item=SimpleNamespace(
        call_id=call_id, id=call_id, name=name, arguments=args_json,
    ))


def _tool_output(*, call_id: str, output):
    item = ToolCallOutputItem(
        raw_item=SimpleNamespace(call_id=call_id),
        output=output,
    )
    return item


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_item_text_from_responses_api_shape():
    item = _msg("Hello world.")
    assert _extract_item_text(item) == "Hello world."


def test_extract_item_text_handles_dict_blocks():
    """Some SDK versions emit content as dict blocks."""
    raw = SimpleNamespace(content=[
        {"type": "text", "text": "block1"},
        {"type": "text", "text": "block2"},
    ])
    item = MessageOutputItem(raw_item=raw)
    out = _extract_item_text(item)
    assert "block1" in out and "block2" in out


def test_extract_item_text_returns_empty_on_garbage():
    """Don't crash on unfamiliar shapes."""
    assert _extract_item_text(MessageOutputItem(raw_item=None)) == ""
    assert _extract_item_text(MessageOutputItem()) == ""


def test_extract_tool_call_parses_json_args():
    item = _tool_call(call_id="c1", name="lookup_order",
                     args_json='{"order_id": "O-101"}')
    cid, name, args = _extract_tool_call(item)
    assert cid == "c1"
    assert name == "lookup_order"
    assert args == {"order_id": "O-101"}


def test_extract_tool_call_handles_non_json_args():
    """If args isn't valid JSON, wrap as {input: ...} not crash."""
    item = _tool_call(call_id="c2", name="raw_str_tool",
                     args_json="not json at all")
    _, _, args = _extract_tool_call(item)
    assert args == {"input": "not json at all"}


def test_stringify_keeps_strings_and_jsons_structures():
    assert _stringify("hi") == "hi"
    assert _stringify({"a": 1}) == '{"a": 1}'
    assert _stringify(42) == "42"


# ─────────────────────────────────────────────────────────────────────────────
# _items_to_steps — the core post-hoc parser
# ─────────────────────────────────────────────────────────────────────────────

def test_simple_one_turn_one_tool():
    """Message → tool call → tool output → final message = 2 steps
    (one per LLM turn). Tool attaches to step 1."""
    items = [
        _msg("Let me look that up."),
        _tool_call(call_id="c1", name="lookup_order",
                  args_json='{"order_id": "O-101"}'),
        _tool_output(call_id="c1", output={"status": "shipped"}),
        _msg("Order O-101 is shipped."),
    ]
    steps = _items_to_steps(items)
    assert len(steps) == 2, f"expected 2 steps, got {len(steps)}"
    assert steps[0].tool_calls[0].name == "lookup_order"
    assert steps[0].tool_calls[0].result == '{"status": "shipped"}'
    assert "shipped" in steps[1].thought


def test_adjacent_reasoning_and_message_merge_into_one_step():
    """Codex D16 cycle 2 ISSUE 5: a ReasoningItem followed by a
    MessageOutputItem in the SAME turn (no tool between) must NOT
    over-segment into two steps."""
    items = [
        _msg("Think first.", cls=ReasoningItem),
        _msg("Then respond.", cls=MessageOutputItem),
        _tool_call(call_id="c1", name="lookup", args_json="{}"),
        _tool_output(call_id="c1", output="ok"),
    ]
    steps = _items_to_steps(items)
    # Reasoning + message = ONE step (one LLM turn). Tool attaches.
    assert len(steps) == 1
    assert "Think first" in steps[0].thought
    assert "Then respond" in steps[0].thought
    assert steps[0].tool_calls[0].name == "lookup"


def test_orphan_output_before_call_reconciles():
    """Codex D16 cycle 2 ISSUE 6: a ToolCallOutputItem arriving BEFORE
    its matching ToolCallItem must reconcile when the call shows up,
    not produce a bogus ``<unknown>`` tool."""
    items = [
        _msg("Working."),
        _tool_output(call_id="c1", output={"status": "shipped"}),
        _tool_call(call_id="c1", name="lookup_order", args_json="{}"),
    ]
    steps = _items_to_steps(items)
    tools = [t for s in steps for t in s.tool_calls]
    # Exactly one tool, and it's the named one with the result attached.
    assert len(tools) == 1
    assert tools[0].name == "lookup_order"
    assert tools[0].result == '{"status": "shipped"}'


def test_first_item_is_tool_call_does_not_crash():
    """Edge case: a tool item arrives before any message. The parser
    must open an implicit step so the call has somewhere to land."""
    items = [
        _tool_call(call_id="c1", name="lookup", args_json="{}"),
        _tool_output(call_id="c1", output="result"),
    ]
    steps = _items_to_steps(items)
    assert len(steps) == 1
    assert steps[0].tool_calls[0].name == "lookup"


def test_multi_turn_tool_loop_produces_one_step_per_llm_turn():
    """A typical multi-turn agent (decide → call → see-result → decide
    → call → see-result → answer) produces one step per LLM turn, not
    per tool."""
    items = [
        _msg("Look up O-101"),                                              # turn 1
        _tool_call(call_id="c1", name="lookup", args_json='{"id":"O-101"}'),
        _tool_output(call_id="c1", output={"status": "shipped"}),
        _msg("Now refund it"),                                              # turn 2
        _tool_call(call_id="c2", name="refund", args_json='{"id":"O-101"}'),
        _tool_output(call_id="c2", output={"refund_id": "R-O-101"}),
        _msg("Refund R-O-101 approved."),                                   # turn 3 (final)
    ]
    steps = _items_to_steps(items)
    assert len(steps) == 3
    assert steps[0].tool_calls[0].name == "lookup"
    assert steps[1].tool_calls[0].name == "refund"
    assert steps[2].tool_calls == []
    assert "approved" in steps[2].thought


def test_handoff_items_record_marker_in_output():
    """Handoffs aren't expanded into sub-trace AgentSteps yet (v1
    limitation, documented). They DO appear as markers in the current
    step's output so the user can see something happened."""
    items = [
        _msg("Transferring you to specialist."),
        HandoffCallItem(raw_item=SimpleNamespace(target="specialist")),
        HandoffOutputItem(
            raw_item=SimpleNamespace(),
            source_agent="main", target_agent="specialist",
        ),
    ]
    steps = _items_to_steps(items)
    assert len(steps) == 1
    assert "handoff" in steps[0].output


def test_orphan_output_with_no_matching_call_becomes_synthetic_at_end():
    """If an output arrives whose call never shows up, surface it as a
    synthetic ToolCall on the final step — better to expose the
    inconsistency than silently drop the data."""
    items = [
        _msg("Hello."),
        _tool_output(call_id="c1", output="orphan"),
        _msg("Goodbye."),
    ]
    steps = _items_to_steps(items)
    # The trailing orphan appears as a synthetic tool with "orphan output" name.
    found = [t for s in steps for t in s.tool_calls if "orphan output" in t.name]
    assert found, f"expected orphan synthetic, got {[t.name for s in steps for t in s.tool_calls]}"
    assert found[0].result == "orphan"


def test_other_item_types_are_skipped_silently_but_act_as_turn_boundary():
    """Truly unknown item types (custom subclasses, undocumented SDK
    types we haven't catalogued) don't crash and don't appear in the
    output. They DO act as turn boundaries — a subsequent message
    starts a new step rather than merging — because an unknown event
    might be a turn separator we just don't recognize. Codex D16
    cycle 5 raised this precise design question and we picked
    'unknown = turn boundary' over 'unknown = ignored'."""
    class _Unknown(SimpleNamespace): pass
    items = [
        _msg("hi"),
        _Unknown(),                  # silently skipped, BUT closes turn
        _msg("bye"),                 # NEW step
    ]
    steps = _items_to_steps(items)
    assert len(steps) == 2, (
        f"unknown items must act as turn boundaries, got {len(steps)}"
    )
    assert "hi" in steps[0].thought
    assert "bye" in steps[1].thought
    # And the unknown itself does NOT leak into the trace text.
    flat = " ".join(s.thought + " " + (s.output or "") for s in steps)
    assert "_Unknown" not in flat


# ─────────────────────────────────────────────────────────────────────────────
# Codex D16 cycle 4: real-SDK-shape fixtures + dict-shaped raw_item
# ─────────────────────────────────────────────────────────────────────────────

def test_reasoning_item_with_real_sdk_shape_summary():
    """Codex cycle 4 ISSUE 2: real ``ResponseReasoningItem`` exposes
    text via ``summary: list[Summary]`` where each Summary has a
    ``.text``. Previous fixture used the message-style ``content``,
    which masked this code path."""
    reasoning = ReasoningItem(raw_item=SimpleNamespace(
        summary=[
            SimpleNamespace(text="Step 1: think.", type="summary_text"),
            SimpleNamespace(text="Step 2: respond.", type="summary_text"),
        ],
        content=None,
    ))
    items = [reasoning, _msg("Done thinking.")]
    steps = _items_to_steps(items)
    assert len(steps) == 1, "reasoning + message in same turn → one step"
    assert "Step 1: think." in steps[0].thought
    assert "Step 2: respond." in steps[0].thought
    assert "Done thinking." in steps[0].thought


# ─────────────────────────────────────────────────────────────────────────────
# Codex D16 cycle 5 — adversarial findings
# ─────────────────────────────────────────────────────────────────────────────

def test_known_unhandled_items_preserved_as_markers():
    """Codex cycle 5 ISSUE 2: known SDK item classes like
    ``ToolSearchCallItem`` / ``ToolApprovalItem`` / MCP items /
    ``CompactionItem`` shouldn't silently disappear from the trace.
    They're marked in step.output until the parser learns to model
    them fully."""
    class ToolApprovalItem(SimpleNamespace): pass
    class ToolSearchCallItem(SimpleNamespace): pass
    class CompactionItem(SimpleNamespace): pass
    items = [
        _msg("Looking things up."),
        ToolSearchCallItem(),
        _msg("Found it."),
        ToolApprovalItem(),
        CompactionItem(),
    ]
    steps = _items_to_steps(items)
    flat_output = " ".join(s.output or "" for s in steps)
    assert "ToolSearchCallItem" in flat_output
    assert "ToolApprovalItem" in flat_output
    assert "CompactionItem" in flat_output


def test_merge_is_idempotent():
    """Codex cycle 5 ISSUE 4: merge() must clear hooks.steps so a
    second call is a no-op, instead of duplicating the trace."""
    pytest.importorskip("agents")
    tracer = OpenAIAgentsTracer()
    hooks = tracer.run_hooks()
    hooks.steps.append(AgentStep(thought="once"))
    tracer.merge(hooks)
    assert len(tracer.get_trace()) == 1
    # Second merge of the same hooks → still 1, not 2.
    tracer.merge(hooks)
    assert len(tracer.get_trace()) == 1, (
        "merge() duplicated trace on second call — must clear hooks.steps"
    )


def test_coerce_args_handles_json_dict_and_garbage():
    """Codex cycle 5 ISSUE 3: the new _coerce_args helper underlies
    RunHooks args extraction. It must handle JSON strings, plain
    strings, dicts, and None."""
    from multivon_eval.integrations.openai_agents import _coerce_args
    assert _coerce_args('{"a": 1}') == {"a": 1}
    assert _coerce_args({"a": 1}) == {"a": 1}
    assert _coerce_args(None) == {}
    assert _coerce_args("") == {}
    assert _coerce_args("not json") == {"input": "not json"}
    assert _coerce_args(42) == {"input": "42"}


def test_tool_call_with_dict_shaped_raw_item():
    """Codex cycle 4 ISSUE 3: some SDK paths serialize raw_item as a
    dict. The parser must read call_id / name / arguments by key,
    not just attribute."""
    items = [
        ToolCallItem(raw_item={
            "call_id": "c1",
            "name": "lookup_order",
            "arguments": '{"order_id": "O-101"}',
        }),
        ToolCallOutputItem(raw_item={"call_id": "c1"}, output="ok"),
    ]
    steps = _items_to_steps(items)
    tools = [t for s in steps for t in s.tool_calls]
    assert len(tools) == 1, f"dict raw_item not parsed correctly: {tools}"
    assert tools[0].name == "lookup_order"
    assert tools[0].arguments == {"order_id": "O-101"}
    assert tools[0].result == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIAgentsTracer public surface
# ─────────────────────────────────────────────────────────────────────────────

def test_tracer_capture_appends_steps_from_run_result():
    """capture(result) reads result.new_items and folds the parsed
    steps into the tracer's trace."""
    fake_result = SimpleNamespace(new_items=[
        _msg("Looking up."),
        _tool_call(call_id="c1", name="lookup", args_json="{}"),
        _tool_output(call_id="c1", output="ok"),
        _msg("Done."),
    ])
    tracer = OpenAIAgentsTracer()
    tracer.capture(fake_result)
    steps = tracer.get_trace()
    assert len(steps) == 2
    assert steps[0].tool_calls[0].name == "lookup"


def test_tracer_capture_handles_missing_new_items():
    """A RunResult-like object without new_items doesn't crash."""
    tracer = OpenAIAgentsTracer()
    tracer.capture(SimpleNamespace())          # no new_items at all
    tracer.capture(SimpleNamespace(new_items=None))
    assert tracer.get_trace() == []


def test_instrument_clears_state_before_each_call():
    """Codex D16 cycle 2 ISSUE 4: the suite calls reset() between
    cases via instrument()'s wrapper. Each case starts with a clean
    trace, even if the user forgets to call reset() themselves."""
    tracer = OpenAIAgentsTracer()
    tracer._steps.append(AgentStep(thought="stale"))
    wrapped = tracer.instrument(lambda _: "ok")
    wrapped("anything")
    # The pre-existing "stale" step was cleared by the instrument
    # wrapper's reset() call.
    assert tracer.get_trace() == []


def test_run_hooks_isolated_buffer(monkeypatch):
    """Codex D16 cycle 2 ISSUE 4 + cycle 4 ISSUE 8: live RunHooks
    must have a PRIVATE buffer so concurrent runs don't interleave.

    Behavioral test (no private-attribute assertion): create TWO hooks
    instances and run them as two parallel agent runs would. The
    second hook's events must NOT flow into the first hook's trace,
    and merge() must yield only the merged hook's data."""
    pytest.importorskip("agents")
    tracer = OpenAIAgentsTracer()

    hooks_a = tracer.run_hooks()
    hooks_b = tracer.run_hooks()

    # Simulate two concurrent runs writing to their hooks.
    hooks_a.steps.append(AgentStep(thought="from-A", tool_calls=[]))
    hooks_b.steps.append(AgentStep(thought="from-B", tool_calls=[]))

    # Before any merge, the tracer sees nothing.
    assert tracer.get_trace() == [], "hooks must not auto-write to tracer"

    # Merging A alone yields only A.
    tracer.merge(hooks_a)
    trace = tracer.get_trace()
    assert len(trace) == 1
    assert trace[0].thought == "from-A"

    # Merging B then appends only B.
    tracer.merge(hooks_b)
    trace = tracer.get_trace()
    assert len(trace) == 2
    assert {s.thought for s in trace} == {"from-A", "from-B"}


def test_merge_handles_objects_without_steps_attr():
    """merge() shouldn't crash if passed something that's not a hooks
    instance (e.g. a typo)."""
    tracer = OpenAIAgentsTracer()
    tracer.merge(object())   # no .steps attribute — must be a no-op
    tracer.merge(SimpleNamespace(steps=None))
    tracer.merge(SimpleNamespace(steps=[]))
    assert tracer.get_trace() == []
