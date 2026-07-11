"""
LangSmithTracer  — live tracing that also logs runs to LangSmith.
LangSmithImporter — pull existing LangSmith runs as EvalCases.

Requires: pip install langsmith
"""
from __future__ import annotations
from typing import Any, Callable

from .base import CaseImporter
from .langchain import LangChainTracer
from ..case import AgentStep, EvalCase, ToolCall

__all__ = ["LangSmithTracer", "LangSmithImporter"]


class LangSmithTracer(LangChainTracer):
    """
    Extends LangChainTracer to also log runs to LangSmith.

    Builds the same AgentStep trace as LangChainTracer while simultaneously
    uploading to LangSmith for observability. Teams already using LangSmith
    for tracing get both without any extra work.

    Requires: pip install langsmith langchain-core

    Usage:
        tracer = LangSmithTracer(project_name="my-agent-evals")
        suite.run(my_agent, tracer=tracer)
    """

    def __init__(
        self,
        project_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        self._project = project_name
        self._api_key = api_key

    def _build_handler(self) -> list[Any]:
        our_handler = super()._build_handler()
        try:
            # LangSmith's callback handler lives in langchain_core (v0.7+)
            from langchain_core.tracers.langchain import LangChainTracer as _LSTracer
            ls_kwargs: dict[str, Any] = {}
            if self._project:
                ls_kwargs["project_name"] = self._project
            if self._api_key:
                from langsmith import Client
                ls_kwargs["client"] = Client(api_key=self._api_key)
            ls_handler = _LSTracer(**ls_kwargs)
            return [our_handler, ls_handler]
        except ImportError:
            # langchain_core or langsmith not installed — capture-only fallback
            return [our_handler]

    def instrument(self, fn: Callable[[str], str]) -> Callable[[str], str]:
        tracer = self

        def wrapped(input_text: str, **kwargs) -> str:
            handlers = tracer._build_handler()
            existing = kwargs.pop("callbacks", []) or []
            return fn(input_text, callbacks=[*existing, *handlers], **kwargs)

        return wrapped


class LangSmithImporter(CaseImporter):
    """
    Import existing LangSmith runs as EvalCases.

    Each imported case has agent_trace populated from the run's child runs
    and metadata["_output"] set to the run's final output so you can replay
    evaluation without re-running the agent.

    Requires: pip install langsmith

    Usage:
        importer = LangSmithImporter(project_name="my-agent")
        cases = importer.load(limit=100)

        suite.add_cases(cases)
        report = suite.run(importer.as_model_fn(cases))

    Filtering examples:
        # Only successful runs
        importer.load(filter='and(eq(error, ""), gt(latency, 0))')

        # Runs with a specific tag
        importer.load(filter='has(tags, "production")')
    """

    def __init__(
        self,
        project_name: str,
        api_key: str | None = None,
    ) -> None:
        self._project = project_name
        self._api_key = api_key

    def _client(self) -> Any:
        try:
            from langsmith import Client
        except ImportError:
            raise ImportError(
                "LangSmithImporter requires langsmith: pip install langsmith"
            )
        return Client(api_key=self._api_key)

    def load(
        self,
        *,
        limit: int = 100,
        filter: str | None = None,
        run_type: str = "chain",
        input_key: str | None = None,
        output_key: str | None = None,
    ) -> list[EvalCase]:
        """
        Pull runs from LangSmith and return as EvalCases.

        Args:
            limit:      Maximum number of runs to import (default 100).
            filter:     LangSmith filter string. See LangSmith docs for syntax.
            run_type:   Run type to pull ("chain", "llm", "tool"). Default "chain"
                        captures top-level agent/chain runs.
            input_key:  Key to use from run.inputs dict. Auto-detected if None.
            output_key: Key to use from run.outputs dict. Auto-detected if None.
        """
        client = self._client()
        # list_runs does not populate child_runs — fetch each full run separately
        run_stubs = client.list_runs(
            project_name=self._project,
            run_type=run_type,
            limit=limit,
            filter=filter,
        )
        cases = []
        for stub in run_stubs:
            # load_child_runs=True populates the nested tool/llm calls
            full_run = client.read_run(stub.id, load_child_runs=True)
            cases.append(self._run_to_case(full_run, input_key=input_key, output_key=output_key))
        return cases

    def _run_to_case(
        self,
        run: Any,
        *,
        input_key: str | None,
        output_key: str | None,
    ) -> EvalCase:
        input_text = self._extract_value(run.inputs, input_key)
        output_text = self._extract_value(run.outputs, output_key)
        trace = self._extract_trace(run)

        return EvalCase(
            input=input_text,
            agent_trace=trace or None,
            metadata={
                "_run_id": str(run.id),
                "_output": output_text,
                "_project": self._project,
                "_error": str(run.error) if getattr(run, "error", None) else None,
                "_run_name": getattr(run, "name", ""),
            },
        )

    @staticmethod
    def _extract_value(data: dict[str, Any] | None, key: str | None) -> str:
        if not data:
            return ""
        if key and key in data:
            return str(data[key])
        # Auto-detect: prefer common keys, fall back to first value
        for candidate in ("input", "output", "question", "answer", "text", "content"):
            if candidate in data:
                return str(data[candidate])
        first = next(iter(data.values()), "")
        return str(first) if first is not None else ""

    def _extract_trace(self, run: Any) -> list[AgentStep]:
        children = sorted(
            getattr(run, "child_runs", None) or [],
            key=lambda r: getattr(r, "start_time", None) or 0,
        )
        steps: list[AgentStep] = []
        current: AgentStep = AgentStep()

        for child in children:
            run_type = getattr(child, "run_type", "")
            if run_type == "tool":
                tool_output = self._extract_value(getattr(child, "outputs", None), None)
                if getattr(child, "error", None):
                    tool_output = f"[ERROR: {child.error}]"
                current.tool_calls.append(ToolCall(
                    name=getattr(child, "name", "unknown_tool") or "unknown_tool",
                    arguments=getattr(child, "inputs", {}) or {},
                    result=tool_output,
                ))
            elif run_type == "llm":
                # Flush accumulated tool calls as a step when we hit an LLM call
                if current.tool_calls:
                    steps.append(current)
                    current = AgentStep()

        # Flush remaining step
        if current.tool_calls:
            steps.append(current)

        # Attach final output to last step (or add a bare output step)
        output_text = self._extract_value(getattr(run, "outputs", None), None)
        if output_text:
            if steps:
                steps[-1].output = output_text
            else:
                steps.append(AgentStep(output=output_text))

        return steps
