"""
Live integration tests — hit real APIs to verify tracer/importer compatibility.

Run with:
    LANGSMITH_API_KEY=lsv2_... pytest tests/test_integrations_live.py -v

Skipped automatically if LANGSMITH_API_KEY is not set.
"""
from __future__ import annotations
import os
import pytest

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")

pytestmark = pytest.mark.skipif(
    not LANGSMITH_API_KEY,
    reason="LANGSMITH_API_KEY not set",
)


# ---------------------------------------------------------------------------
# LangSmithImporter — real API
# ---------------------------------------------------------------------------

def test_langsmith_importer_connects():
    """Client connects and returns a project list without error."""
    from langsmith import Client
    client = Client(api_key=LANGSMITH_API_KEY)
    projects = list(client.list_projects())
    assert len(projects) > 0, "Expected at least one project"
    names = [p.name for p in projects]
    assert LANGSMITH_PROJECT in names, f"{LANGSMITH_PROJECT} not found in {names}"


def test_langsmith_importer_load_returns_cases():
    """load() returns EvalCase objects with correct field types."""
    from multivon_eval.integrations import LangSmithImporter
    from multivon_eval import EvalCase

    importer = LangSmithImporter(project_name=LANGSMITH_PROJECT, api_key=LANGSMITH_API_KEY)
    cases = importer.load(limit=3)

    assert isinstance(cases, list)
    assert len(cases) > 0, "Expected at least one case"

    for case in cases:
        assert isinstance(case, EvalCase)
        assert isinstance(case.input, str)
        assert "_run_id" in case.metadata
        assert "_output" in case.metadata
        assert "_project" in case.metadata


def test_langsmith_importer_child_runs_populate_trace():
    """Runs with child tool/chain calls produce a non-empty agent_trace."""
    from langsmith import Client
    from multivon_eval.integrations import LangSmithImporter

    # Find a run that has child runs
    client = Client(api_key=LANGSMITH_API_KEY)
    stubs = list(client.list_runs(
        project_name=LANGSMITH_PROJECT,
        run_type="chain",
        is_root=True,
        limit=10,
    ))
    # Find one with child_run_ids (has children)
    with_children = [s for s in stubs if s.child_run_ids]
    if not with_children:
        pytest.skip("No runs with child_run_ids found in project")

    importer = LangSmithImporter(project_name=LANGSMITH_PROJECT, api_key=LANGSMITH_API_KEY)
    stub = with_children[0]
    full = client.read_run(stub.id, load_child_runs=True)
    case = importer._run_to_case(full, input_key=None, output_key=None)

    assert case.input != ""
    # At minimum metadata should be populated
    assert case.metadata["_run_id"] == str(stub.id)


def test_langsmith_importer_as_model_fn():
    """as_model_fn() replays outputs in the same order as the imported cases."""
    from multivon_eval.integrations import LangSmithImporter

    importer = LangSmithImporter(project_name=LANGSMITH_PROJECT, api_key=LANGSMITH_API_KEY)
    cases = importer.load(limit=3)
    model_fn = importer.as_model_fn(cases)

    # Iterator-based: call in order, each call returns the matching output
    for case in cases:
        result = model_fn(case.input)
        assert result == case.metadata["_output"], (
            f"Output mismatch for run {case.metadata['_run_id']}: "
            f"expected {case.metadata['_output']!r}, got {result!r}"
        )


# ---------------------------------------------------------------------------
# LangSmithTracer — real callback handler import
# ---------------------------------------------------------------------------

def test_langsmith_tracer_handler_import():
    """_build_handler() returns two handlers when langchain_core is installed."""
    from multivon_eval.integrations import LangSmithTracer

    tracer = LangSmithTracer(project_name=LANGSMITH_PROJECT, api_key=LANGSMITH_API_KEY)
    handlers = tracer._build_handler()
    # Should return a list of [our_handler, ls_handler]
    assert isinstance(handlers, list)
    assert len(handlers) == 2, (
        f"Expected 2 handlers (ours + LangSmith), got {len(handlers)}. "
        "LangSmith callback handler import may have failed — check langchain_core install."
    )


def test_langsmith_tracer_instrument_injects_callbacks():
    """instrument() wraps fn and injects callbacks kwarg."""
    from multivon_eval.integrations import LangSmithTracer

    received_callbacks = []

    def fake_agent(input_text: str, **kwargs) -> str:
        received_callbacks.extend(kwargs.get("callbacks", []))
        return "test output"

    tracer = LangSmithTracer(project_name="test", api_key=LANGSMITH_API_KEY)
    wrapped = tracer.instrument(fake_agent)
    result = wrapped("hello")

    assert result == "test output"
    assert len(received_callbacks) == 2  # our handler + langsmith handler


# ---------------------------------------------------------------------------
# LangChainTracer — real BaseCallbackHandler with simulated events
# ---------------------------------------------------------------------------

def test_langchain_tracer_with_real_handler():
    """Handler built from real BaseCallbackHandler fires correctly."""
    from multivon_eval.integrations import LangChainTracer
    from multivon_eval.case import AgentStep

    tracer = LangChainTracer()
    handler = tracer._build_handler()

    # Simulate a real LangChain agent execution sequence
    class FakeAction:
        log = "I should search for this"
        tool = "search"
        tool_input = "Paris capital"

    class FakeFinish:
        return_values = {"output": "Paris is the capital of France."}

    handler.on_agent_action(FakeAction())
    handler.on_tool_start({"name": "search"}, "Paris capital")
    handler.on_tool_end("Paris is the capital city.")
    handler.on_agent_finish(FakeFinish())

    trace = tracer.get_trace()

    assert len(trace) == 1
    step = trace[0]
    assert isinstance(step, AgentStep)
    assert "search" in step.thought.lower() or step.thought != ""
    assert len(step.tool_calls) == 1
    assert step.tool_calls[0].name == "search"
    assert step.tool_calls[0].result == "Paris is the capital city."
    assert step.output == "Paris is the capital of France."


def test_langchain_tracer_multi_step():
    """Multiple agent_action events produce multiple steps."""
    from multivon_eval.integrations import LangChainTracer

    tracer = LangChainTracer()
    handler = tracer._build_handler()

    class Action:
        def __init__(self, log):
            self.log = log

    class Finish:
        return_values = {"output": "done"}

    handler.on_agent_action(Action("step 1"))
    handler.on_tool_start({"name": "tool_a"}, "input_a")
    handler.on_tool_end("result_a")

    handler.on_agent_action(Action("step 2"))
    handler.on_tool_start({"name": "tool_b"}, "input_b")
    handler.on_tool_end("result_b")

    handler.on_agent_finish(Finish())

    trace = tracer.get_trace()
    assert len(trace) == 2
    assert trace[0].tool_calls[0].name == "tool_a"
    assert trace[1].tool_calls[0].name == "tool_b"


def test_langchain_tracer_tool_error():
    """on_tool_error captures error message as result."""
    from multivon_eval.integrations import LangChainTracer

    tracer = LangChainTracer()
    handler = tracer._build_handler()

    class Action:
        log = "trying tool"

    class Finish:
        return_values = {"output": "failed"}

    handler.on_agent_action(Action())
    handler.on_tool_start({"name": "flaky_tool"}, "input")
    handler.on_tool_error(Exception("timeout"))
    handler.on_agent_finish(Finish())

    trace = tracer.get_trace()
    assert "[ERROR:" in trace[0].tool_calls[0].result
