"""
ManualTracer — explicit step recording for custom agents without a supported framework.
"""
from __future__ import annotations
from typing import Any, Callable

from .base import AgentTracer
from ..case import AgentStep, ToolCall

__all__ = ["ManualTracer"]


class _StepRecorder:
    """
    Context manager returned by ManualTracer.step().

    Records tool calls within a single agent step, then flushes to the
    tracer when the with-block exits.

    Usage:
        with tracer.step(thought="Searching for context") as s:
            result = search(query)
            s.record_tool_call("search", {"query": query}, result)
    """

    def __init__(self, tracer: "ManualTracer", thought: str = "") -> None:
        self._tracer = tracer
        self._step = AgentStep(thought=thought)

    def record_tool_call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
    ) -> None:
        """Record a tool call within this step."""
        self._step.tool_calls.append(
            ToolCall(name=name, arguments=arguments or {}, result=result)
        )

    def set_output(self, output: str) -> None:
        """Set the step's final output text."""
        self._step.output = output

    def __enter__(self) -> "_StepRecorder":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._tracer._steps.append(self._step)


class ManualTracer(AgentTracer):
    """
    Tracer for custom agents without a supported framework.

    model_fn is returned unwrapped. The user records steps manually
    by calling tracer.step() within their agent code.

    Usage:
        tracer = ManualTracer()

        def my_agent(input_text: str) -> str:
            with tracer.step(thought="Searching") as s:
                result = my_search_tool(input_text)
                s.record_tool_call("search", {"q": input_text}, result)
            answer = my_llm(result)
            tracer.record_output(answer)
            return answer

        suite.run(my_agent, tracer=tracer)
    """

    def instrument(self, fn: Callable[[str], str]) -> Callable[[str], str]:
        # No wrapping needed — user records manually.
        return fn

    def get_trace(self) -> list[AgentStep]:
        return list(self._steps)

    def step(self, thought: str = "") -> _StepRecorder:
        """Open a new step context. Record tool calls within the with-block."""
        return _StepRecorder(self, thought)

    def record_tool_call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
    ) -> None:
        """Record a tool call outside a step context (creates an implicit step)."""
        self._steps.append(
            AgentStep(tool_calls=[ToolCall(name=name, arguments=arguments or {}, result=result)])
        )

    def record_output(self, output: str) -> None:
        """Record a final output step (no tool calls)."""
        self._steps.append(AgentStep(output=output))
