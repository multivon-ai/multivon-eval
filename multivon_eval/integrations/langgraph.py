"""
LangGraphTracer — graph-node-aware tracer for LangGraph agents.

LangGraph emits LangChain callback events with graph-specific metadata:
``metadata["langgraph_node"]`` names the executing node, tags include
``graph:step:N``, and ``langgraph_checkpoint_ns`` / ``langgraph_path``
disambiguate subgraphs. We use these to attribute LLM calls and tool
calls to the correct semantic agent turn.

The right granularity (codex D16 cycle 1 critique): **one AgentStep per
LLM/agent turn**, not per graph node. A ReAct graph has a separate
"tools" node, but semantically the tool calls belong to the preceding
LLM turn — that's the unit of agent decision-making evaluators want
to score. A "tools" node firing 3 tools in parallel yields ONE step
with 3 ToolCalls, not 3 separate steps.

Strategy:
  1. ``on_llm_start`` / ``on_chat_model_start`` opens a new step,
     flushing the previous one. Each LLM turn = one ``AgentStep``.
  2. ``on_llm_end`` writes the assistant text into ``thought``.
  3. ``on_tool_start`` / ``on_tool_end`` attach ``ToolCall`` instances
     to the currently-open step (LangGraph's tools node fires after
     the model decides to call them, so temporal proximity is right).
  4. ``on_chain_end`` for a LangGraph node either drops the step
     (empty / routing-only node) or flushes a tool-only step that has
     no following LLM turn to trigger the flush.
  5. ``get_trace()`` flushes any final in-flight step before returning,
     because the LAST LLM turn of a run never sees a follow-up
     ``on_llm_start``.

We track LangGraph-specific metadata by ``run_id`` because
``on_chain_end`` may not carry metadata in older LangGraph versions.

Requires: ``pip install 'multivon-eval[langgraph]'`` or
``pip install langgraph langchain-core``.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from .base import CallbackTracer
from ..case import AgentStep, ToolCall

__all__ = ["LangGraphTracer"]


class LangGraphTracer(CallbackTracer):
    """
    Captures agent traces from LangGraph compiled graphs.

    LangGraph builds on LangChain callbacks, so this tracer extends
    :class:`CallbackTracer` and uses the standard ``callbacks=[...]``
    config path. The user's ``model_fn`` invokes the compiled graph
    with the tracer's handler attached:

        from multivon_eval.integrations.langgraph import LangGraphTracer

        tracer = LangGraphTracer()

        def model_fn(input_text: str, **kwargs) -> str:
            result = graph.invoke(
                {"messages": [HumanMessage(content=input_text)]},
                config={"callbacks": kwargs.get("callbacks", [])},
            )
            return result["messages"][-1].content

        suite.run(model_fn, tracer=tracer)

    Compatible with: LangGraph >= 0.2. Verified on 0.5+ (latest at
    ship time). **Single-agent, single-branch sync graphs.** Known v1
    limitations (file an issue if you hit them):

      - Streaming (``graph.stream(...)``) and async (``graph.ainvoke``)
        emit the same callback events but the tracer hasn't been
        validated against them end-to-end.
      - Parallel branches via ``Send`` or parallel subgraphs use a
        single ``_current_step`` and may cross-attribute tool calls
        across branches. Use a separate tracer instance per branch
        until proper run_id-keyed step tracking lands.
      - Multi-agent handoffs are captured as adjacent steps but the
        handoff itself isn't a first-class trace event.
    """

    def __init__(self) -> None:
        super().__init__()
        # Live reference to the most recently built handler. Used by
        # ``get_trace()`` to flush any in-flight step the handler is
        # still holding — without this, the LAST LLM turn of a run is
        # lost (no following on_llm_start to trigger the flush).
        # Codex D16 cycle 2 finding.
        self._live_handler: Any = None

    def _build_handler(self) -> Any:
        try:
            from langchain_core.callbacks import BaseCallbackHandler
        except ImportError:
            raise ImportError(
                "LangGraphTracer requires langchain-core: "
                "pip install 'multivon-eval[langgraph]'"
            )

        tracer = self

        class _Handler(BaseCallbackHandler):
            def __init__(self) -> None:
                super().__init__()
                # Per-run_id metadata cache. on_chain_end may not carry
                # metadata in older LangGraph; we read it from on_chain_start.
                self._node_meta: dict[UUID, dict[str, Any]] = {}
                # The CURRENTLY OPEN step (one LLM turn). Tool calls are
                # attached here until the next on_llm_start opens a new step.
                self._current_step: AgentStep | None = None
                # Tool calls in flight, keyed by run_id so parallel tools
                # within a single node don't clobber each other.
                self._pending_tools: dict[UUID, ToolCall] = {}
                # The most recent assistant text — emitted on on_llm_end,
                # rolled into the step's `output` field on flush.
                self._last_llm_text: str = ""

            # ── helpers ────────────────────────────────────────────────

            def _flush_step(self) -> None:
                if self._current_step is None:
                    return
                # Set output from the most recent LLM text if not already set.
                if not self._current_step.output and self._last_llm_text:
                    self._current_step.output = self._last_llm_text
                tracer._steps.append(self._current_step)
                self._current_step = None
                self._last_llm_text = ""

            # ── chain (node) events ────────────────────────────────────

            def on_chain_start(
                self,
                serialized: dict[str, Any] | None,
                inputs: Any,
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                # Cache node identity by run_id so we can recognize the
                # corresponding on_chain_end even when its metadata is
                # None (observed in LangGraph 1.x).
                if metadata and metadata.get("langgraph_node"):
                    self._node_meta[run_id] = {
                        "node": metadata.get("langgraph_node"),
                        "checkpoint_ns": metadata.get("langgraph_checkpoint_ns"),
                        "step": metadata.get("langgraph_step"),
                    }

            def on_chain_end(
                self,
                outputs: Any,
                *,
                run_id: UUID,
                **kwargs: Any,
            ) -> None:
                # If this run_id was a LangGraph node, drop any empty
                # in-flight step (pure routing nodes shouldn't pollute
                # the trace) AND flush a populated tool-node step here
                # because no following on_llm_start will be there to
                # flush it. Codex D16 cycle 2 finding.
                if run_id in self._node_meta:
                    self._node_meta.pop(run_id, None)
                    if self._current_step is not None:
                        if not _step_has_content(self._current_step):
                            self._current_step = None
                        elif self._current_step.tool_calls and not self._current_step.thought:
                            # A pure-tool step closing without a new LLM
                            # turn after it — flush so it isn't lost.
                            self._flush_step()

            # ── LLM events — step boundaries ───────────────────────────

            def on_llm_start(
                self,
                serialized: dict[str, Any] | None,
                prompts: list[str],
                *,
                run_id: UUID,
                **kwargs: Any,
            ) -> None:
                # New LLM turn = new step. Flush whatever was open
                # (typically the previous turn whose tools have already
                # been bound to it).
                self._flush_step()
                self._current_step = AgentStep()

            def on_chat_model_start(
                self,
                serialized: dict[str, Any] | None,
                messages: Any,
                *,
                run_id: UUID,
                **kwargs: Any,
            ) -> None:
                # Chat models go through on_chat_model_start, not
                # on_llm_start. Same semantics.
                self.on_llm_start(serialized, [], run_id=run_id, **kwargs)

            def on_llm_end(
                self,
                response: Any,
                *,
                run_id: UUID,
                **kwargs: Any,
            ) -> None:
                # Pull assistant text from the LLM response. LangChain
                # responses expose .generations[0][0].text or .message.content.
                text = _extract_llm_text(response)
                if text:
                    self._last_llm_text = text
                    if self._current_step is not None:
                        # The thought is the assistant's textual reasoning
                        # BEFORE any tool calls. We populate `thought` here;
                        # `output` is set on flush.
                        self._current_step.thought = text

            # ── Tool events — attach to current step ───────────────────

            def on_tool_start(
                self,
                serialized: dict[str, Any] | None,
                input_str: str,
                *,
                run_id: UUID,
                **kwargs: Any,
            ) -> None:
                name = (serialized or {}).get("name", "unknown_tool")
                # Try to parse JSON args; fall back to {"input": raw_string}.
                args = _parse_tool_args(input_str)
                self._pending_tools[run_id] = ToolCall(name=name, arguments=args)
                # If a tool fires without a preceding LLM (e.g. a hard-coded
                # tools node), open a step so the call has somewhere to go.
                if self._current_step is None:
                    self._current_step = AgentStep()

            def on_tool_end(
                self,
                output: Any,
                *,
                run_id: UUID,
                **kwargs: Any,
            ) -> None:
                tool = self._pending_tools.pop(run_id, None)
                if tool is None:
                    return
                tool.result = _stringify_tool_result(output)
                if self._current_step is None:
                    self._current_step = AgentStep()
                self._current_step.tool_calls.append(tool)

            def on_tool_error(
                self,
                error: BaseException,
                *,
                run_id: UUID,
                **kwargs: Any,
            ) -> None:
                tool = self._pending_tools.pop(run_id, None)
                if tool is None:
                    return
                tool.result = f"[ERROR: {type(error).__name__}: {error}]"
                if self._current_step is None:
                    self._current_step = AgentStep()
                self._current_step.tool_calls.append(tool)

            # Flush expose — get_trace() calls this on the live
            # handler so the FINAL in-flight step (whose tools were
            # captured but never followed by another on_llm_start) is
            # not silently dropped.
            def flush_in_flight(self) -> None:
                if self._current_step is not None and _step_has_content(self._current_step):
                    self._flush_step()
                else:
                    self._current_step = None

        handler = _Handler()
        # Hold a reference so get_trace() can finalize the last step.
        tracer._live_handler = handler
        return handler

    def get_trace(self) -> list[AgentStep]:
        """Return the captured trace.

        Finalizes any in-flight step on the live handler before
        returning. The handler emits steps on each new ``on_llm_start``,
        so without an explicit flush the LAST step of a run is lost.
        Codex D16 cycle 2 finding.
        """
        if self._live_handler is not None:
            try:
                self._live_handler.flush_in_flight()
            except Exception:
                pass
        return list(self._steps)

    def reset(self) -> None:
        super().reset()
        # Drop the live handler reference so a stale handler from the
        # previous case can't accidentally flush into the next.
        self._live_handler = None


# ── Module-level helpers (kept out of the closure for testability) ─────


def _step_has_content(step: AgentStep) -> bool:
    """True if the step has anything worth keeping."""
    return bool(step.thought or step.tool_calls or step.output)


def _extract_llm_text(response: Any) -> str:
    """Best-effort extraction of the assistant message text from a
    LangChain LLMResult / ChatResult. Returns empty string if we can't
    find it — avoids crashing on unfamiliar response shapes from
    custom integrations."""
    try:
        generations = getattr(response, "generations", None)
        if generations:
            first = generations[0][0]
            # ChatGeneration carries the assistant message
            message = getattr(first, "message", None)
            if message is not None:
                content = getattr(message, "content", "")
                if isinstance(content, str):
                    return content
                # Some chat models return content as a list of blocks
                if isinstance(content, list):
                    return " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
            # LLMResult plain text
            return str(getattr(first, "text", "") or "")
    except (AttributeError, IndexError, KeyError):
        pass
    return ""


def _parse_tool_args(input_str: str) -> dict[str, Any]:
    """Parse a tool's ``input_str`` as JSON; fall back to wrapping the
    raw string as ``{"input": ...}``. LangGraph tools usually emit
    structured args as a JSON-serialized dict."""
    import json
    s = (input_str or "").strip()
    if not s:
        return {}
    if s.startswith("{") and s.endswith("}"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
    return {"input": input_str}


def _stringify_tool_result(output: Any) -> str:
    """Render a tool result for the trace. Keep structured types as
    JSON (audit-friendly), strings as-is, everything else via str()."""
    import json
    if isinstance(output, str):
        return output
    if isinstance(output, (dict, list, tuple)):
        try:
            return json.dumps(output, default=str)
        except (TypeError, ValueError):
            pass
    return str(output)
