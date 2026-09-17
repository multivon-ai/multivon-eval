"""
Base abstractions for agent trace collection and case importing.

AgentTracer     — instrument any agent/pipeline to capture execution traces
CallbackTracer  — intermediate ABC for callback-style frameworks (LangChain, CrewAI)
CaseImporter    — pull pre-existing traces from external observability platforms
"""
from __future__ import annotations

import warnings
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
        report = suite.run_on_cases([(case, case.metadata["_output"]) for case in cases])
    """

    @abstractmethod
    def load(self, **kwargs) -> list[EvalCase]:
        """
        Pull runs and return them as EvalCases.

        Each returned case should have agent_trace populated and
        metadata["_output"] set to the run's final output string
        so that run_on_cases() can grade it without re-running the agent.
        """
        ...

    def as_model_fn(self, cases: list[EvalCase]) -> Callable[[str], str]:
        """
        Return a deprecated, case-aware replay callable.

        Suites match immutable case identity, including for duplicate prompts,
        reordered execution and repeated grading. Direct string calls require
        an unambiguous output for that input. Missing outputs or unknown cases
        raise ValueError instead of silently fabricating an empty response.

        Prefer ``suite.run_on_cases([(c, c.metadata["_output"]) for c in cases])``:
        running a replay callable measures replay latency, not target latency,
        and repeating saved outputs cannot estimate target stochasticity.
        """
        warnings.warn(
            "as_model_fn is deprecated; use suite.run_on_cases with explicit (case, output) "
            "pairs to preserve saved-output provenance and unknown target latency",
            DeprecationWarning, stacklevel=2,
        )
        return _SavedReplay(cases)


class _SavedReplay:
    def __init__(self, cases: list[EvalCase]):
        self._by_case: dict[tuple[str, str], str] = {}
        self._by_input: dict[str, set[str]] = {}
        for case in cases:
            output = case.metadata.get("_output")
            if not isinstance(output, str):
                raise ValueError("Every imported case requires a string metadata['_output']")
            self._by_case[case.identity()] = output
            self._by_input.setdefault(case.input, set()).add(output)

    def __call__(self, input_text: str) -> str:
        outputs = self._by_input.get(input_text, set())
        if len(outputs) != 1:
            raise ValueError("Unknown or ambiguous replay input; use explicit (case, output) pairs")
        return next(iter(outputs))

    def _call_with_case(self, case: EvalCase) -> str:
        try:
            return self._by_case[case.identity()]
        except KeyError:
            raise ValueError("Replay case does not match the imported snapshot") from None

    async def _acall_with_case(self, case: EvalCase) -> str:
        return self._call_with_case(case)
