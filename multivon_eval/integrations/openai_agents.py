"""
OpenAIAgentsTracer — captures traces from the OpenAI Agents SDK.

Two integration paths, both supported:

  1. **Post-hoc** (default, recommended): the user's ``model_fn`` calls
     ``Runner.run_sync(...)`` and the tracer reads ``RunResult.new_items``
     after the run completes. Simple, no global state, no thread-safety
     concerns. Codex D16 cycle 1 picked this over the global trace
     processor.

  2. **Live RunHooks**: an async hooks object passed as
     ``Runner.run(..., hooks=tracer.run_hooks())`` captures the trace
     as the run progresses. Each hooks instance has ITS OWN private
     step buffer (no shared mutable state across concurrent runs).
     Call ``tracer.merge(hooks)`` to fold the buffer into the tracer's
     trace after the run completes. Useful for streaming or when you
     want to intercept events live.

The semantic model (codex cycle 1 critique): **one AgentStep per
agent/LLM turn**. ``ToolCallItem`` and ``ToolCallOutputItem`` are
paired into ``ToolCall`` and attached to the step started by the
preceding ``MessageOutputItem`` / ``ReasoningItem``. Adjacent
message-like items belonging to the same turn append to the open
step rather than over-segmenting (codex D16 cycle 2 finding).

Requires: ``pip install 'multivon-eval[openai-agents]'``.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import AgentTracer
from ..case import AgentStep, ToolCall

__all__ = ["OpenAIAgentsTracer"]


class OpenAIAgentsTracer(AgentTracer):
    """
    Captures traces from the OpenAI Agents SDK.

    Default usage (post-hoc — recommended):

        from multivon_eval.integrations.openai_agents import OpenAIAgentsTracer
        from agents import Runner

        tracer = OpenAIAgentsTracer()

        def model_fn(input_text: str) -> str:
            result = Runner.run_sync(my_agent, input_text)
            tracer.capture(result)   # <-- MUST happen inside model_fn
            return result.final_output

        suite.run(model_fn, tracer=tracer)

    ``tracer.capture(result)`` parses ``RunResult.new_items`` and folds
    the derived ``AgentStep`` list into the tracer's trace. It must be
    called BEFORE ``model_fn`` returns the string output — the suite
    has no way to unwrap a ``RunResult`` itself.

    Live RunHooks variant (when you need event-time interception, e.g.
    streaming + cancel-on-guardrail):

        hooks = tracer.run_hooks()                      # PRIVATE buffer
        result = await Runner.run(my_agent, input_text, hooks=hooks)
        tracer.merge(hooks)                             # fold into trace

    Compatible with the openai-agents SDK as of late 2024 / 2025. Sync
    single-agent runs. Handoffs are captured (as ``HandoffCallItem``)
    but not expanded into sub-trace AgentSteps yet — file an issue if
    you need that.
    """

    def instrument(self, fn: Callable[[str], str]) -> Callable[[str], str]:
        """Wrap ``fn`` to clear the tracer state before each call.

        The SDK's ``RunResult`` is the source of truth, but users are
        responsible for calling :meth:`capture` (or returning a result
        the suite can unwrap). We DO clear ``self._steps`` here so
        case-to-case state doesn't leak.
        """
        tracer = self

        def wrapped(input_text: str, **kwargs: Any) -> str:
            tracer.reset()
            return fn(input_text, **kwargs)

        return wrapped

    def get_trace(self) -> list[AgentStep]:
        return list(self._steps)

    # ── post-hoc capture: parse RunResult.new_items ────────────────────

    def capture(self, run_result: Any) -> None:
        """Parse a ``RunResult`` and append the derived AgentSteps to
        the current trace.

        Idempotent on already-captured results (we only append; if you
        call it twice with the same result you'll get a duplicated
        trace — don't do that). Returns nothing because the trace
        lives on the tracer's ``_steps``.
        """
        items = getattr(run_result, "new_items", None) or []
        self._steps.extend(_items_to_steps(items))

    # ── live RunHooks variant ──────────────────────────────────────────

    def run_hooks(self) -> Any:
        """Return a ``RunHooks`` instance with its OWN private buffer.

        Pass to ``Runner.run(..., hooks=tracer.run_hooks())``. The hooks
        capture trace events as the run progresses, accumulating
        ``AgentStep`` objects in the hook's PRIVATE buffer — no shared
        mutable state with this tracer or other concurrent hooks. After
        the run completes, call ``tracer.merge(hooks)`` to fold the
        buffer into the tracer's trace.

        Why per-hook state: codex D16 cycle 2 caught that closing over
        ``tracer._steps`` would interleave traces across concurrent
        runs. This isolates each run.

        Step boundaries are driven by ``on_llm_start`` (each LLM call
        opens a new step), not ``on_agent_start`` (which fires once
        per agent activation and would collapse multi-turn loops into
        one step).
        """
        try:
            from agents.lifecycle import RunHooksBase
        except ImportError:
            raise ImportError(
                "OpenAIAgentsTracer.run_hooks() requires the openai-agents SDK: "
                "pip install 'multivon-eval[openai-agents]'"
            )

        class _Hooks(RunHooksBase):
            def __init__(self) -> None:
                super().__init__()
                # PRIVATE buffer — not the tracer's. The tracer merges
                # this in after the run completes.
                self.steps: list[AgentStep] = []
                # Tool calls in flight keyed by call_id when available,
                # else by Python id(tool). Defended against the SDK
                # change of args appearing in either on_tool_start or
                # on_tool_end.
                self._pending: dict[Any, ToolCall] = {}

            async def on_llm_start(  # type: ignore[override]
                self, context: Any, agent: Any, system_prompt: Any, input_items: Any,
            ) -> None:
                # New LLM turn = new step. Each Runner loop iteration
                # fires one on_llm_start.
                self.steps.append(AgentStep())

            async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:  # type: ignore[override]
                text = _extract_response_text(response)
                if not self.steps:
                    self.steps.append(AgentStep())
                last = self.steps[-1]
                if text:
                    last.thought = text
                    last.output = text

            async def on_tool_start(
                self, context: Any, agent: Any, tool: Any,
            ) -> None:  # type: ignore[override]
                # Codex D16 cycle 5 finding: prefer the SDK's
                # context.tool_call_id / tool_name / tool_arguments
                # over id(tool). Two parallel calls to the SAME tool
                # would share id() but have distinct tool_call_ids.
                call_id = (
                    _attr_or_key(context, "tool_call_id")
                    or getattr(tool, "call_id", None)
                )
                name = (
                    _attr_or_key(context, "tool_name")
                    or getattr(tool, "name", None)
                    or getattr(tool, "__name__", "tool")
                )
                args_raw = _attr_or_key(context, "tool_arguments")
                args = _coerce_args(args_raw) if args_raw else _extract_tool_args(tool, context)
                key = call_id or id(tool)
                self._pending[key] = ToolCall(name=str(name), arguments=args)

            async def on_tool_end(
                self, context: Any, agent: Any, tool: Any, result: Any,
            ) -> None:  # type: ignore[override]
                call_id = (
                    _attr_or_key(context, "tool_call_id")
                    or getattr(tool, "call_id", None)
                )
                key = call_id or id(tool)
                tool_call = self._pending.pop(key, None)
                if tool_call is None:
                    name = (
                        _attr_or_key(context, "tool_name")
                        or getattr(tool, "name", None)
                        or getattr(tool, "__name__", "tool")
                    )
                    args_raw = _attr_or_key(context, "tool_arguments")
                    tool_call = ToolCall(
                        name=str(name),
                        arguments=_coerce_args(args_raw) if args_raw else _extract_tool_args(tool, context),
                    )
                tool_call.result = _stringify(result)
                if not self.steps:
                    self.steps.append(AgentStep())
                self.steps[-1].tool_calls.append(tool_call)

            async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:  # type: ignore[override]
                if self.steps and not self.steps[-1].output:
                    self.steps[-1].output = _stringify(output)

        return _Hooks()

    def merge(self, hooks: Any) -> None:
        """Fold a live ``run_hooks()`` instance's private buffer into
        this tracer's trace, then clear the buffer.

        Idempotent: a second merge of the same hooks is a no-op
        because the first call clears ``hooks.steps``. Codex D16
        cycle 5 finding — without the clear, two ``merge`` calls
        duplicated the trace.
        """
        buffered = getattr(hooks, "steps", None)
        if not buffered:
            return
        self._steps.extend(buffered)
        # Clear so a second merge is a no-op. We assign to the
        # attribute rather than mutating in place — the caller may
        # still hold a reference to the OLD list and check it
        # separately.
        try:
            hooks.steps = []
        except (AttributeError, TypeError):
            # Read-only hook implementation — best effort.
            pass


# ── helpers (module-level for testability) ─────────────────────────────


_MESSAGE_LIKE = frozenset({"MessageOutputItem", "ReasoningItem"})

# Known SDK item types we don't fully model yet but want to PRESERVE
# as visible markers rather than silently drop. Codex D16 cycle 5
# finding: the SDK ships ToolSearchCallItem / ToolSearchOutputItem /
# ToolApprovalItem / MCPApprovalRequestItem / etc., and an eval that
# silently ignored them would mask real agent behavior.
_KNOWN_UNHANDLED = frozenset({
    "ToolSearchCallItem",
    "ToolSearchOutputItem",
    "ToolApprovalItem",
    "MCPApprovalRequestItem",
    "MCPApprovalResponseItem",
    "MCPListToolsItem",
    "ComputerCallItem",
    "ComputerCallOutputItem",
    "CodeInterpreterCallItem",
    "ImageGenerationCallItem",
    "LocalShellCallItem",
    # Conversation compaction (long agent run → SDK summarizes earlier
    # turns to fit context). Per SDK RunItem union docs.
    "CompactionItem",
})


def _items_to_steps(items: list[Any]) -> list[AgentStep]:
    """Convert a list of ``RunResult.new_items`` to AgentSteps.

    Walk items in order; group by agent turn. Rules (codex D16 cycle 2):

    - A NEW step opens on the FIRST message-like item after a
      ``ToolCallItem`` / ``ToolCallOutputItem`` (the model's response
      to tool output is a new turn).
    - Adjacent message-like items WITHIN the same turn (e.g.
      ReasoningItem followed by MessageOutputItem before any tool
      call) append to the open step instead of over-segmenting.
    - ``ToolCallItem`` attaches a partial ``ToolCall`` to the current
      step.
    - ``ToolCallOutputItem`` matches its call by ``call_id``; if the
      output arrives BEFORE its call (malformed ordering), buffer it
      and reconcile when the matching call item appears.
    - Handoffs append a marker to the current step's output.

    Class-name string matching is intentional: it lets the unit tests
    work without the openai-agents SDK installed. The names are
    stable in the SDK's public re-exports.
    """
    steps: list[AgentStep] = []
    pending_call: dict[str, ToolCall] = {}     # call_id -> partial ToolCall
    pending_output: dict[str, Any] = {}        # call_id -> output (orphan)
    last_kind: str | None = None

    def _current() -> AgentStep:
        if not steps:
            steps.append(AgentStep())
        return steps[-1]

    for item in items:
        cls = type(item).__name__
        if cls in _MESSAGE_LIKE:
            text = _extract_item_text(item)
            # Open a NEW step UNLESS the previous item was ALSO
            # message-like (reasoning+message in the same turn merges
            # into one step). Anything else — tool calls, handoffs,
            # known-unhandled items, truly unknown items — is a turn
            # boundary.
            if last_kind in _MESSAGE_LIKE and steps:
                step = steps[-1]
            else:
                step = AgentStep()
                steps.append(step)
            # Concat text fields when appending to an existing step.
            joined = (step.thought + " " + text).strip() if step.thought else text
            step.thought = joined
            step.output = joined
        elif cls == "ToolCallItem":
            tc_id, name, args = _extract_tool_call(item)
            tool = ToolCall(name=name, arguments=args, result=None)
            # Reconcile any orphan output that arrived before this call.
            if tc_id and tc_id in pending_output:
                tool.result = _stringify(pending_output.pop(tc_id))
            pending_call[tc_id] = tool
            _current().tool_calls.append(tool)
        elif cls == "ToolCallOutputItem":
            tc_id, output = _extract_tool_output(item)
            tool = pending_call.pop(tc_id, None) if tc_id else None
            if tool is not None:
                tool.result = _stringify(output)
            else:
                # Output arrived before its call — buffer and reconcile
                # on the matching ToolCallItem. Avoids the bogus
                # ``<unknown>`` synthetic from before.
                if tc_id:
                    pending_output[tc_id] = output
                else:
                    # No call_id at all — last resort attach to current.
                    _current().tool_calls.append(
                        ToolCall(name="<unknown>", arguments={},
                                 result=_stringify(output))
                    )
        elif cls in ("HandoffCallItem", "HandoffOutputItem"):
            step = _current()
            step.output = (step.output or "") + f"\n[handoff: {cls}]"
        elif cls in _KNOWN_UNHANDLED:
            # Preserve as a marker so a reviewer can see something
            # happened, instead of silently dropping the item.
            # Tracked as future v2 work.
            step = _current()
            step.output = (step.output or "") + f"\n[{cls}]"
        # Truly unknown classes (custom subclasses, future SDK types
        # we haven't catalogued) ARE silently skipped — the alternative
        # is noisy markers in every trace. Codex cycle 5 documented
        # the trade-off.
        last_kind = cls

    # Any orphan outputs that never matched a call become synthetic
    # entries on the final step — surfacing the issue beats silently
    # dropping the data.
    if pending_output:
        step = _current()
        for tc_id, output in pending_output.items():
            step.tool_calls.append(
                ToolCall(
                    name=f"<orphan output {tc_id[:8] if tc_id else ''}>",
                    arguments={},
                    result=_stringify(output),
                )
            )

    return steps


def _extract_item_text(item: Any) -> str:
    """Pull the assistant text out of a MessageOutputItem / ReasoningItem.

    Three real shapes (codex D16 cycle 4 finding):

    - ``MessageOutputItem.raw_item``: a Responses-API message with
      ``content: list[OutputContent]`` where each has a ``.text``.
    - ``ReasoningItem.raw_item`` (``ResponseReasoningItem``): has
      ``summary: list[Summary]`` (``Summary.text``) and optionally
      ``content: list[Content]``. We check both fields.
    - Some older / chat-completions paths: ``raw.message.content`` as
      a plain string.

    Falls back to empty string for unfamiliar shapes — better an empty
    thought than a crash from an SDK schema drift.
    """
    raw = getattr(item, "raw_item", None)
    if raw is None:
        return ""

    parts: list[str] = []

    # ReasoningItem path: pull from raw.summary first. Each Summary has
    # a .text field (verified against openai-agents 0.x SDK).
    summary = getattr(raw, "summary", None)
    if isinstance(summary, list):
        for s in summary:
            t = getattr(s, "text", None)
            if isinstance(t, str):
                parts.append(t)
            elif isinstance(s, dict):
                t = s.get("text")
                if isinstance(t, str):
                    parts.append(t)

    # MessageOutputItem path (and ReasoningItem's optional content
    # field): raw.content is a list of OutputContent / Content objects.
    content = getattr(raw, "content", None)
    if isinstance(content, list):
        for c in content:
            t = getattr(c, "text", None)
            if isinstance(t, str):
                parts.append(t)
            elif isinstance(c, dict):
                t = c.get("text")
                if isinstance(t, str):
                    parts.append(t)
    elif isinstance(content, str):
        parts.append(content)

    if parts:
        return " ".join(parts).strip()

    # Chat Completions style: raw.message.content (str)
    msg = getattr(raw, "message", None)
    if msg is not None:
        c = getattr(msg, "content", None)
        if isinstance(c, str):
            return c

    return ""


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from ``obj`` as either an attribute or a dict key.

    Defensive against SDK shape drift — some run-item snapshots
    surface fields as object attributes (dataclasses / pydantic
    models) and some as dict keys when they've been round-tripped
    through JSON. Codex D16 cycle 4 finding.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_tool_call(item: Any) -> tuple[str, str, dict[str, Any]]:
    """Return (call_id, tool_name, arguments_dict) for a ToolCallItem.

    Supports both object-shaped and dict-shaped ``raw_item`` — the
    SDK can emit either depending on whether the trace passed
    through a serializer."""
    import json
    raw = getattr(item, "raw_item", None)
    if raw is None:
        return ("", "<unknown>", {})
    call_id = _attr_or_key(raw, "call_id") or _attr_or_key(raw, "id") or ""
    name = _attr_or_key(raw, "name") or "<unknown>"
    # arguments is typically a JSON string on the Responses API
    args_raw = _attr_or_key(raw, "arguments")
    args: dict[str, Any] = {}
    if isinstance(args_raw, str):
        try:
            parsed = json.loads(args_raw)
            args = parsed if isinstance(parsed, dict) else {"input": args_raw}
        except (ValueError, TypeError):
            args = {"input": args_raw}
    elif isinstance(args_raw, dict):
        args = args_raw
    return (str(call_id), str(name), args)


def _extract_tool_output(item: Any) -> tuple[str, Any]:
    """Return (call_id, output) for a ToolCallOutputItem.

    Output may live on the item directly (``item.output``) or on the
    nested ``raw_item.output`` — both paths are checked. Same dict-
    or-attr defense as :func:`_extract_tool_call`."""
    raw = getattr(item, "raw_item", None)
    call_id = _attr_or_key(raw, "call_id") or ""
    output = getattr(item, "output", None)
    if output is None:
        output = _attr_or_key(raw, "output")
    return (str(call_id), output)


def _extract_response_text(response: Any) -> str:
    """Best-effort extraction of assistant text from a ModelResponse."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for o in output:
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
        if parts:
            return " ".join(parts).strip()
    return ""


def _coerce_args(args_raw: Any) -> dict[str, Any]:
    """Coerce a tool's argument blob (JSON string or dict) to a dict.

    The SDK's hook context typically surfaces ``tool_arguments`` as a
    JSON string; some shapes already pre-parse to a dict."""
    import json
    if args_raw is None:
        return {}
    if isinstance(args_raw, dict):
        return args_raw
    if isinstance(args_raw, str):
        s = args_raw.strip()
        if not s:
            return {}
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, TypeError):
                pass
        return {"input": args_raw}
    return {"input": str(args_raw)}


def _extract_tool_args(tool: Any, context: Any) -> dict[str, Any]:
    """Try to pull tool arguments from the live RunHooks context.

    Different SDK versions expose args in different places. Returns
    an empty dict rather than crashing if we can't find them — the
    user still gets the tool NAME and RESULT, just no args.
    """
    # Some SDK versions stash the in-flight call on the context.
    last_call = getattr(context, "last_tool_call", None)
    if last_call is not None:
        args = getattr(last_call, "arguments", None)
        if isinstance(args, dict):
            return args
    return {}


def _stringify(value: Any) -> str:
    """Render a value for trace storage. Dicts/lists → JSON, strings
    as-is, everything else via str()."""
    import json
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            pass
    return str(value)
