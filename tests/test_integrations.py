"""
Tests for multivon_eval.integrations — ManualTracer, LangChainTracer,
LangSmithImporter, CaseImporter.as_model_fn, and suite integration.
"""
from __future__ import annotations
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call

from multivon_eval.case import AgentStep, EvalCase, ToolCall
from multivon_eval.integrations.manual import ManualTracer
from multivon_eval.integrations.base import AgentTracer, CaseImporter
from multivon_eval.integrations.langsmith import LangSmithImporter
from multivon_eval.suite import EvalSuite


# ---------------------------------------------------------------------------
# 1. ManualTracer
# ---------------------------------------------------------------------------

class TestManualTracer(unittest.TestCase):

    def test_instrument_returns_fn_unchanged(self):
        """instrument() must return the same callable (no wrapping)."""
        tracer = ManualTracer()
        fn = lambda x: x + " response"
        instrumented = tracer.instrument(fn)
        self.assertIs(instrumented, fn)

    def test_step_context_manager_records_tool_calls(self):
        """step() context manager appends an AgentStep with tool calls on exit."""
        tracer = ManualTracer()
        with tracer.step(thought="Thinking") as s:
            s.record_tool_call("search", {"q": "hello"}, "result text")
            s.record_tool_call("lookup", {"id": 42}, {"value": 1})

        trace = tracer.get_trace()
        self.assertEqual(len(trace), 1)
        step = trace[0]
        self.assertEqual(step.thought, "Thinking")
        self.assertEqual(len(step.tool_calls), 2)
        self.assertEqual(step.tool_calls[0].name, "search")
        self.assertEqual(step.tool_calls[0].arguments, {"q": "hello"})
        self.assertEqual(step.tool_calls[0].result, "result text")
        self.assertEqual(step.tool_calls[1].name, "lookup")

    def test_step_set_output(self):
        """set_output() sets the step's output field."""
        tracer = ManualTracer()
        with tracer.step(thought="Planning") as s:
            s.set_output("final answer")

        step = tracer.get_trace()[0]
        self.assertEqual(step.output, "final answer")

    def test_get_trace_returns_copy(self):
        """get_trace() returns a fresh list each time (not the internal list)."""
        tracer = ManualTracer()
        with tracer.step():
            pass
        t1 = tracer.get_trace()
        t2 = tracer.get_trace()
        self.assertIsNot(t1, t2)
        self.assertEqual(len(t1), len(t2))

    def test_reset_clears_steps(self):
        """reset() empties the captured trace."""
        tracer = ManualTracer()
        with tracer.step(thought="step1"):
            pass
        self.assertEqual(len(tracer.get_trace()), 1)
        tracer.reset()
        self.assertEqual(len(tracer.get_trace()), 0)

    def test_record_tool_call_creates_implicit_step(self):
        """record_tool_call() outside a step context creates an implicit AgentStep."""
        tracer = ManualTracer()
        tracer.record_tool_call("fetch", {"url": "http://example.com"}, "page content")
        trace = tracer.get_trace()
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0].tool_calls[0].name, "fetch")

    def test_record_output_creates_bare_output_step(self):
        """record_output() appends a step with only the output field set."""
        tracer = ManualTracer()
        tracer.record_output("Here is the answer.")
        trace = tracer.get_trace()
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0].output, "Here is the answer.")
        self.assertEqual(trace[0].tool_calls, [])

    def test_multiple_steps_accumulated(self):
        """Multiple step() calls accumulate steps in order."""
        tracer = ManualTracer()
        with tracer.step(thought="step A") as s:
            s.record_tool_call("tool1", {}, "r1")
        with tracer.step(thought="step B") as s:
            s.record_tool_call("tool2", {}, "r2")
        trace = tracer.get_trace()
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0].thought, "step A")
        self.assertEqual(trace[1].thought, "step B")

    def test_context_manager_enter_exit(self):
        """AgentTracer context manager calls reset on enter."""
        tracer = ManualTracer()
        with tracer.step():
            pass
        self.assertEqual(len(tracer.get_trace()), 1)
        with tracer:  # __enter__ calls reset()
            pass
        self.assertEqual(len(tracer.get_trace()), 0)


# ---------------------------------------------------------------------------
# 2. LangChainTracer
# ---------------------------------------------------------------------------

class TestLangChainTracer(unittest.TestCase):

    def _make_tracer_with_mock_langchain(self):
        """
        Install a minimal fake langchain_core into sys.modules so that
        LangChainTracer._build_handler() doesn't require the real package.
        Returns the tracer and the fake BaseCallbackHandler class.
        """
        # Build a minimal fake module tree
        fake_lc_core = types.ModuleType("langchain_core")
        fake_callbacks = types.ModuleType("langchain_core.callbacks")

        class FakeBaseCallbackHandler:
            def __init__(self):
                pass

        fake_callbacks.BaseCallbackHandler = FakeBaseCallbackHandler
        fake_lc_core.callbacks = fake_callbacks

        sys.modules["langchain_core"] = fake_lc_core
        sys.modules["langchain_core.callbacks"] = fake_callbacks

        from multivon_eval.integrations.langchain import LangChainTracer
        tracer = LangChainTracer()
        return tracer, FakeBaseCallbackHandler

    def tearDown(self):
        # Clean up any fake modules we injected
        for key in list(sys.modules.keys()):
            if key.startswith("langchain_core"):
                del sys.modules[key]

    def test_build_handler_raises_when_langchain_core_missing(self):
        """_build_handler() raises ImportError when langchain-core is not installed."""
        # Ensure langchain_core is not importable
        sys.modules["langchain_core"] = None  # type: ignore
        sys.modules["langchain_core.callbacks"] = None  # type: ignore

        from multivon_eval.integrations.langchain import LangChainTracer
        tracer = LangChainTracer()
        with self.assertRaises(ImportError):
            tracer._build_handler()

    def test_instrument_injects_callbacks_kwarg(self):
        """instrument() wraps fn so that callbacks=[handler] is passed as kwarg."""
        tracer, _ = self._make_tracer_with_mock_langchain()

        received_kwargs = {}

        def fake_fn(input_text: str, **kwargs) -> str:
            received_kwargs.update(kwargs)
            return "output"

        instrumented = tracer.instrument(fake_fn)
        result = instrumented("hello")

        self.assertEqual(result, "output")
        self.assertIn("callbacks", received_kwargs)
        self.assertEqual(len(received_kwargs["callbacks"]), 1)

    def test_instrument_merges_existing_callbacks(self):
        """instrument() prepends existing callbacks rather than replacing them."""
        tracer, _ = self._make_tracer_with_mock_langchain()

        existing_cb = MagicMock()
        received_kwargs = {}

        def fake_fn(input_text: str, **kwargs) -> str:
            received_kwargs.update(kwargs)
            return "output"

        instrumented = tracer.instrument(fake_fn)
        instrumented("hello", callbacks=[existing_cb])

        callbacks = received_kwargs["callbacks"]
        self.assertEqual(len(callbacks), 2)
        self.assertIs(callbacks[0], existing_cb)

    def test_get_trace_returns_steps_populated_by_handler_events(self):
        """
        Simulate the callback sequence: on_agent_action → on_tool_start →
        on_tool_end → on_agent_finish. Verify get_trace() returns the
        correct AgentStep list.
        """
        tracer, _ = self._make_tracer_with_mock_langchain()
        handler = tracer._build_handler()

        # Simulate agent action
        action = MagicMock()
        action.log = "I will search for the answer."
        handler.on_agent_action(action)

        # Simulate tool start + end
        handler.on_tool_start({"name": "search_tool"}, "query string")
        handler.on_tool_end("search result")

        # Simulate agent finish
        finish = MagicMock()
        finish.return_values = {"output": "final answer"}
        handler.on_agent_finish(finish)

        trace = tracer.get_trace()
        self.assertEqual(len(trace), 1)
        step = trace[0]
        self.assertEqual(step.thought, "I will search for the answer.")
        self.assertEqual(len(step.tool_calls), 1)
        self.assertEqual(step.tool_calls[0].name, "search_tool")
        self.assertEqual(step.tool_calls[0].arguments, {"input": "query string"})
        self.assertEqual(step.tool_calls[0].result, "search result")
        self.assertEqual(step.output, "final answer")

    def test_on_tool_error_records_error_result(self):
        """on_tool_error() stores the error message on the pending tool call."""
        tracer, _ = self._make_tracer_with_mock_langchain()
        handler = tracer._build_handler()

        handler.on_tool_start({"name": "flaky_tool"}, "input")
        handler.on_tool_error(RuntimeError("network failure"))

        trace = tracer.get_trace()
        # on_agent_finish hasn't fired, so step is still pending — flush via chain_end
        handler.on_chain_end({})

        trace = tracer.get_trace()
        self.assertEqual(len(trace), 1)
        self.assertIn("ERROR", trace[0].tool_calls[0].result)

    def test_get_trace_returns_copy(self):
        """get_trace() returns a new list each call (CallbackTracer.get_trace)."""
        tracer, _ = self._make_tracer_with_mock_langchain()
        t1 = tracer.get_trace()
        t2 = tracer.get_trace()
        self.assertIsNot(t1, t2)


# ---------------------------------------------------------------------------
# 3. LangSmithImporter._run_to_case
# ---------------------------------------------------------------------------

class TestLangSmithImporterRunToCase(unittest.TestCase):

    def _make_run(self, *, input_val="hello", output_val="world", child_runs=None, run_id="abc-123", name="chain"):
        run = MagicMock()
        run.id = run_id
        run.name = name
        run.error = None
        run.inputs = {"input": input_val}
        run.outputs = {"output": output_val}
        run.child_runs = child_runs or []
        return run

    def _make_tool_child(self, name, input_val, output_val, start_time=0):
        child = MagicMock()
        child.run_type = "tool"
        child.name = name
        child.inputs = {"input": input_val}
        child.outputs = {"output": output_val}
        child.error = None
        child.start_time = start_time
        return child

    def _make_llm_child(self, start_time=1):
        child = MagicMock()
        child.run_type = "llm"
        child.name = "llm"
        child.start_time = start_time
        return child

    def setUp(self):
        self.importer = LangSmithImporter(project_name="test-project")

    def test_basic_run_to_case(self):
        """_run_to_case() returns EvalCase with correct input and metadata."""
        run = self._make_run(input_val="what is 2+2", output_val="4")
        case = self.importer._run_to_case(run, input_key=None, output_key=None)

        self.assertIsInstance(case, EvalCase)
        self.assertEqual(case.input, "what is 2+2")
        self.assertEqual(case.metadata["_output"], "4")
        self.assertEqual(case.metadata["_run_id"], "abc-123")
        self.assertEqual(case.metadata["_project"], "test-project")

    def test_run_with_tool_child_populates_agent_trace(self):
        """Tool child runs should appear as ToolCalls in agent_trace."""
        tool_child = self._make_tool_child("search", "query", "result", start_time=1)
        run = self._make_run(child_runs=[tool_child], output_val="final answer")

        case = self.importer._run_to_case(run, input_key=None, output_key=None)

        self.assertIsNotNone(case.agent_trace)
        assert case.agent_trace is not None
        # Should have one step (the tool), with output attached
        found_tool = any(
            tc.name == "search"
            for step in case.agent_trace
            for tc in step.tool_calls
        )
        self.assertTrue(found_tool, "Expected 'search' tool call in trace")

    def test_tool_with_error_sets_error_output(self):
        """A tool child with error should produce [ERROR: ...] result."""
        child = self._make_tool_child("bad_tool", "input", "", start_time=1)
        child.error = "timeout"
        run = self._make_run(child_runs=[child], output_val="fallback")

        case = self.importer._run_to_case(run, input_key=None, output_key=None)
        assert case.agent_trace is not None
        tool_results = [
            tc.result
            for step in case.agent_trace
            for tc in step.tool_calls
        ]
        self.assertTrue(any("ERROR" in str(r) for r in tool_results))

    def test_llm_child_flushes_step(self):
        """An LLM child run after tool calls should flush a step."""
        tool_child = self._make_tool_child("fetch", "url", "page", start_time=1)
        llm_child = self._make_llm_child(start_time=2)
        tool_child2 = self._make_tool_child("parse", "html", "data", start_time=3)

        run = self._make_run(child_runs=[tool_child, llm_child, tool_child2], output_val="done")
        case = self.importer._run_to_case(run, input_key=None, output_key=None)

        assert case.agent_trace is not None
        # First flush (before LLM) should have "fetch", second group has "parse"
        all_tool_names = [
            tc.name
            for step in case.agent_trace
            for tc in step.tool_calls
        ]
        self.assertIn("fetch", all_tool_names)
        self.assertIn("parse", all_tool_names)

    def test_no_children_bare_output_step(self):
        """Run with no child runs should produce a single bare output step."""
        run = self._make_run(output_val="just an answer", child_runs=[])
        case = self.importer._run_to_case(run, input_key=None, output_key=None)

        assert case.agent_trace is not None
        self.assertEqual(len(case.agent_trace), 1)
        self.assertEqual(case.agent_trace[0].output, "just an answer")
        self.assertEqual(case.agent_trace[0].tool_calls, [])

    def test_metadata_error_field_set_when_run_has_error(self):
        """metadata['_error'] is set when the run has an error."""
        run = self._make_run()
        run.error = "some error occurred"
        case = self.importer._run_to_case(run, input_key=None, output_key=None)
        self.assertEqual(case.metadata["_error"], "some error occurred")

    def test_metadata_error_field_none_on_success(self):
        """metadata['_error'] is None when run succeeded."""
        run = self._make_run()
        run.error = None
        case = self.importer._run_to_case(run, input_key=None, output_key=None)
        self.assertIsNone(case.metadata["_error"])


# ---------------------------------------------------------------------------
# 4. LangSmithImporter.as_model_fn (also covers CaseImporter.as_model_fn)
# ---------------------------------------------------------------------------

class TestLangSmithImporterAsModelFn(unittest.TestCase):

    def setUp(self):
        self.importer = LangSmithImporter(project_name="test")

    def test_as_model_fn_replays_in_order(self):
        """as_model_fn() replays outputs positionally — order matches cases."""
        cases = [
            EvalCase(input="What is 2+2?", metadata={"_output": "4"}),
            EvalCase(input="Capital of France?", metadata={"_output": "Paris"}),
        ]
        fn = self.importer.as_model_fn(cases)
        # Input arg is ignored — outputs come out in case order
        self.assertEqual(fn("What is 2+2?"), "4")
        self.assertEqual(fn("Capital of France?"), "Paris")

    def test_as_model_fn_exhausted_returns_empty(self):
        """as_model_fn() returns empty string after all outputs consumed."""
        cases = [EvalCase(input="q", metadata={"_output": "a"})]
        fn = self.importer.as_model_fn(cases)
        self.assertEqual(fn("q"), "a")
        self.assertEqual(fn("q"), "")  # exhausted

    def test_as_model_fn_missing_output_key_returns_empty(self):
        """as_model_fn() handles cases where _output key is absent."""
        cases = [EvalCase(input="query", metadata={})]
        fn = self.importer.as_model_fn(cases)
        self.assertEqual(fn("query"), "")


# ---------------------------------------------------------------------------
# 5. Suite integration — tracer wiring
# ---------------------------------------------------------------------------

class _MockEvaluator:
    """Minimal evaluator that always passes, records calls for inspection."""
    name = "mock_eval"

    def __init__(self):
        self.calls: list[tuple[EvalCase, str]] = []

    def evaluate(self, case: EvalCase, output: str, **kwargs):
        self.calls.append((case, output))
        from multivon_eval.result import EvalResult
        return EvalResult(evaluator=self.name, score=1.0, passed=True, reason="ok")


class TestSuiteTracerIntegration(unittest.TestCase):

    def _make_suite(self) -> tuple[EvalSuite, _MockEvaluator]:
        suite = EvalSuite("test")
        ev = _MockEvaluator()
        suite.add_evaluator(ev)  # type: ignore[arg-type]
        return suite, ev

    def test_tracer_reset_called_before_each_case(self):
        """suite.run() must call tracer.reset() once per case."""
        tracer = MagicMock(spec=ManualTracer)
        tracer.instrument.return_value = lambda x: "output"
        tracer.get_trace.return_value = []

        suite, _ = self._make_suite()
        suite.add_cases([
            EvalCase(input="q1"),
            EvalCase(input="q2"),
        ])
        suite.run(lambda x: "output", tracer=tracer, verbose=False)

        self.assertEqual(tracer.reset.call_count, 2)

    def test_tracer_get_trace_called_after_each_case(self):
        """suite.run() must call tracer.get_trace() once per case."""
        tracer = MagicMock(spec=ManualTracer)
        tracer.instrument.return_value = lambda x: "output"
        tracer.get_trace.return_value = []

        suite, _ = self._make_suite()
        suite.add_cases([EvalCase(input="q1"), EvalCase(input="q2")])
        suite.run(lambda x: "output", tracer=tracer, verbose=False)

        self.assertEqual(tracer.get_trace.call_count, 2)

    def test_trace_attached_to_case_for_evaluator(self):
        """When tracer returns steps, evaluator sees case with agent_trace set."""
        step = AgentStep(thought="test", tool_calls=[], output="done")
        tracer = MagicMock(spec=ManualTracer)
        tracer.instrument.return_value = lambda x: "output"
        tracer.get_trace.return_value = [step]

        suite, ev = self._make_suite()
        suite.add_case(EvalCase(input="question"))
        suite.run(lambda x: "output", tracer=tracer, verbose=False)

        self.assertEqual(len(ev.calls), 1)
        called_case, _ = ev.calls[0]
        self.assertIsNotNone(called_case.agent_trace)
        self.assertEqual(called_case.agent_trace, [step])

    def test_empty_trace_does_not_overwrite_existing(self):
        """If tracer.get_trace() returns [], case.agent_trace is not overwritten."""
        existing_step = AgentStep(thought="pre-existing")
        tracer = MagicMock(spec=ManualTracer)
        tracer.instrument.return_value = lambda x: "output"
        tracer.get_trace.return_value = []  # empty trace

        suite, ev = self._make_suite()
        suite.add_case(EvalCase(input="q", agent_trace=[existing_step]))
        suite.run(lambda x: "output", tracer=tracer, verbose=False)

        called_case, _ = ev.calls[0]
        # Empty trace → no replace → original agent_trace preserved
        self.assertEqual(called_case.agent_trace, [existing_step])

    def test_tracer_with_workers_gt_1_raises(self):
        """suite.run() with tracer + workers > 1 must raise ValueError."""
        tracer = MagicMock(spec=ManualTracer)
        tracer.instrument.return_value = lambda x: "output"

        suite, _ = self._make_suite()
        suite.add_case(EvalCase(input="q"))
        with self.assertRaises(ValueError):
            suite.run(lambda x: "output", tracer=tracer, workers=2, verbose=False)

    def test_instrument_called_once_before_loop(self):
        """tracer.instrument() is called exactly once, not per-case."""
        tracer = MagicMock(spec=ManualTracer)
        tracer.instrument.return_value = lambda x: "output"
        tracer.get_trace.return_value = []

        suite, _ = self._make_suite()
        suite.add_cases([EvalCase(input=f"q{i}") for i in range(5)])
        suite.run(lambda x: "output", tracer=tracer, verbose=False)

        tracer.instrument.assert_called_once()

    def test_run_without_tracer_works_unchanged(self):
        """suite.run() without tracer still works normally."""
        suite, ev = self._make_suite()
        suite.add_case(EvalCase(input="hello"))
        report = suite.run(lambda x: "hi", verbose=False)
        self.assertEqual(len(report.case_results), 1)


# ---------------------------------------------------------------------------
# 6. run_on_cases
# ---------------------------------------------------------------------------

class TestRunOnCases(unittest.TestCase):

    def test_run_on_cases_evaluates_provided_outputs(self):
        """run_on_cases() evaluates the given (case, output) pairs."""
        ev = _MockEvaluator()
        suite = EvalSuite("test")
        suite.add_evaluator(ev)  # type: ignore[arg-type]

        case1 = EvalCase(input="q1")
        case2 = EvalCase(input="q2")
        pairs = [(case1, "answer1"), (case2, "answer2")]

        report = suite.run_on_cases(pairs, verbose=False)

        self.assertEqual(len(report.case_results), 2)
        self.assertEqual(ev.calls[0][1], "answer1")
        self.assertEqual(ev.calls[1][1], "answer2")

    def test_run_on_cases_respects_fail_threshold(self):
        """run_on_cases() raises SystemExit when pass_rate < fail_threshold."""
        from multivon_eval.result import EvalResult
        from multivon_eval.evaluators.base import Evaluator

        class _FailEv(Evaluator):
            name = "always_fail"
            def evaluate(self, case, output, **kw):
                return EvalResult(evaluator=self.name, score=0.0, passed=False, reason="no")

        suite = EvalSuite("test")
        suite.add_evaluator(_FailEv())
        pairs = [(EvalCase(input="q"), "answer")]

        with self.assertRaises(SystemExit):
            suite.run_on_cases(pairs, verbose=False, fail_threshold=1.0)


if __name__ == "__main__":
    unittest.main()
