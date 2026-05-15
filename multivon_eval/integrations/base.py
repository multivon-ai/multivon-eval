"""
Base abstractions for agent trace collection and case importing.

AgentTracer     — instrument any agent/pipeline to capture execution traces
CallbackTracer  — intermediate ABC for callback-style frameworks (LangChain, CrewAI)
CaseImporter    — pull pre-existing traces from external observability platforms
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable

from ..case import AgentStep, EvalCase

__all__ = ["AgentTracer", "CallbackTracer", "CaseImporter"]


class AgentTracer(ABC):
    """
    Base class for capturing agent execution traces.

    Implement this to integrate with any agent framework or custom agent.

    Contract:
        - instrument(fn) returns a wrapped fn with the same str→str signature.
        - The suite calls reset() before each case, then the instrumented fn,
          then get_trace() to retrieve the captured AgentStep list.
        - Supports use as a context manager for one-shot tracing outside a suite.

    Minimal implementation:
        class MyTracer(AgentTracer):
            def instrument(self, fn):
                def wrapped(input_text):
                    self._steps.clear()
                    output = fn(input_text)
                    # populate self._steps however your framework exposes them
                    return output
                return wrapped

            def get_trace(self):
                return list(self._steps)
    """

    def __init__(self) -> None:
        self._steps: list[AgentStep] = []

    @abstractmethod
    def instrument(self, fn: Callable[[str], str]) -> Callable[[str], str]:
        """
        Wrap model_fn to capture a trace on each call.

        The returned callable must have the same signature as fn (str → str).
        """
        ...

    @abstractmethod
    def get_trace(self) -> list[AgentStep]:
        """Return the trace captured by the most recent call to the instrumented fn."""
        ...

    def reset(self) -> None:
        """Clear captured trace. Called by EvalSuite before each case."""
        self._steps.clear()

    # Context manager support for one-shot use outside a suite
    def __enter__(self) -> "AgentTracer":
        self.reset()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @staticmethod
    def format_trace(steps: "list[AgentStep] | None") -> str:
        """Pretty-print a trace as plain-text. Returns the formatted
        string so callers can print, log, or attach it to a report.

        Use to debug what your agent actually did:

            tracer = HandRolledTracer()
            report = suite.run(my_agent, tracer=tracer)
            for cr in report.case_results:
                if not cr.passed:
                    print(f"--- {cr.case_input!r} ---")
                    print(AgentTracer.format_trace(cr.agent_trace))
        """
        if not steps:
            return "(no trace captured)"
        lines: list[str] = []
        for i, step in enumerate(steps, 1):
            lines.append(f"Step {i}:")
            if getattr(step, "thought", None):
                lines.append(f"  thought: {step.thought}")
            for tc in getattr(step, "tool_calls", []) or []:
                args = tc.arguments if isinstance(tc.arguments, dict) else {}
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                lines.append(f"  → {tc.name}({args_str})")
                if tc.result is not None:
                    lines.append(f"    = {tc.result!r}")
            if getattr(step, "output", None):
                lines.append(f"  output: {step.output}")
        return "\n".join(lines)

    def print_trace(self, steps: "list[AgentStep] | None" = None) -> None:
        """Print the trace to stdout. With no argument, prints the
        tracer's own captured steps (useful during interactive
        debugging in a notebook). Pass ``cr.agent_trace`` to print a
        specific case's trace from a report."""
        print(self.format_trace(steps if steps is not None else self.get_trace()))


class CallbackTracer(AgentTracer, ABC):
    """
    Intermediate ABC for callback-based frameworks (LangChain, CrewAI, etc.).

    Subclasses implement _build_handler() to return a framework-specific
    callback object. instrument() injects that handler as a kwarg.

    The user's model_fn must accept and forward **kwargs so the handler
    reaches the underlying framework call:

        def my_agent(input_text: str, **kwargs) -> str:
            return chain.invoke({"input": input_text}, config={"callbacks": kwargs.get("callbacks", [])})
    """

    @abstractmethod
    def _build_handler(self) -> Any:
        """Return a framework-specific callback handler instance."""
        ...

    def get_trace(self) -> list[AgentStep]:
        return list(self._steps)

    def instrument(self, fn: Callable[[str], str]) -> Callable[[str], str]:
        tracer = self

        def wrapped(input_text: str, **kwargs) -> str:
            handler = tracer._build_handler()
            existing = kwargs.pop("callbacks", []) or []
            return fn(input_text, callbacks=[*existing, handler], **kwargs)

        return wrapped


class CaseImporter(ABC):
    """
    Base class for pulling pre-existing traces from external platforms.

    Implement this to import runs from LangSmith, Datadog, Helicone,
    Braintrust, or any other observability store.

    Usage:
        importer = MyImporter(project="prod-agent")
        cases = importer.load(limit=200)
        suite.add_cases(cases)
        report = suite.run(importer.as_model_fn(cases))
    """

    @abstractmethod
    def load(self, **kwargs) -> list[EvalCase]:
        """
        Pull runs and return them as EvalCases.

        Each returned case should have agent_trace populated and
        metadata["_output"] set to the run's final output string
        so that as_model_fn() can replay it without re-running the agent.
        """
        ...

    def as_model_fn(self, cases: list[EvalCase]) -> Callable[[str], str]:
        """
        Return a passthrough model_fn that replays imported outputs in order.

        Uses a positional iterator so duplicate inputs are handled correctly.
        Requires suite.run() to call cases in the same order they were imported
        (always true with workers=1, which is required when using a tracer).

            cases = importer.load()
            suite.add_cases(cases)
            report = suite.run(importer.as_model_fn(cases))
        """
        outputs = [c.metadata.get("_output", "") for c in cases]
        it = iter(outputs)

        def _replay(_input_text: str) -> str:
            return next(it, "")

        return _replay
