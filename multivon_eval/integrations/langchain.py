"""
LangChainTracer — callback-based tracer for LangChain agents and LCEL chains.

Requires: pip install langchain-core
"""
from __future__ import annotations
from typing import Any, Callable

from .base import CallbackTracer
from ..case import AgentStep, ToolCall

__all__ = ["LangChainTracer"]


class LangChainTracer(CallbackTracer):
    """
    Captures agent traces from LangChain agents and LCEL chains.

    Hooks into LangChain's callback system — no changes to framework code needed.
    The user's model_fn must accept and forward **kwargs:

        def my_agent(input_text: str, **kwargs) -> str:
            return agent_executor.invoke(
                {"input": input_text},
                config={"callbacks": kwargs.get("callbacks", [])},
            )

    Usage:
        tracer = LangChainTracer()
        suite.run(my_agent, tracer=tracer)
    """

    def _build_handler(self) -> Any:
        try:
            from langchain_core.callbacks import BaseCallbackHandler
        except ImportError:
            raise ImportError(
                "LangChainTracer requires langchain-core: pip install langchain-core"
            )

        tracer = self

        class _Handler(BaseCallbackHandler):
            def __init__(self) -> None:
                super().__init__()
                self._current_step: AgentStep | None = None
                self._pending_tool: ToolCall | None = None

            def _flush_step(self) -> None:
                if self._current_step is not None:
                    tracer._steps.append(self._current_step)
                    self._current_step = None

            # Agent executor events
            def on_agent_action(self, action: Any, **kwargs: Any) -> None:
                self._flush_step()
                self._current_step = AgentStep(thought=str(getattr(action, "log", "") or ""))

            def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
                output = ""
                if hasattr(finish, "return_values") and isinstance(finish.return_values, dict):
                    output = str(finish.return_values.get("output", ""))
                if self._current_step is not None:
                    self._current_step.output = output
                    self._flush_step()
                else:
                    tracer._steps.append(AgentStep(output=output))

            # Tool events
            def on_tool_start(
                self, serialized: dict[str, Any], input_str: str, **kwargs: Any
            ) -> None:
                name = (serialized or {}).get("name", "unknown_tool")
                self._pending_tool = ToolCall(name=name, arguments={"input": input_str})
                if self._current_step is None:
                    self._current_step = AgentStep()

            def on_tool_end(self, output: Any, **kwargs: Any) -> None:
                if self._pending_tool is not None:
                    self._pending_tool.result = str(output)
                    if self._current_step is None:
                        self._current_step = AgentStep()
                    self._current_step.tool_calls.append(self._pending_tool)
                    self._pending_tool = None

            def on_tool_error(self, error: Any, **kwargs: Any) -> None:
                if self._pending_tool is not None:
                    self._pending_tool.result = f"[ERROR: {error}]"
                    if self._current_step is None:
                        self._current_step = AgentStep()
                    self._current_step.tool_calls.append(self._pending_tool)
                    self._pending_tool = None

            # Chain events — flush on chain end so LCEL chains without AgentExecutor
            # still produce steps
            def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
                if self._current_step is not None and self._current_step.tool_calls:
                    self._flush_step()

        return _Handler()
