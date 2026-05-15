from unittest.mock import patch

from multivon_eval import AgentMemoryEval, AgentStep, EvalCase, ToolCall
from multivon_eval.evaluators.agent import (
    PlanQuality,
    StepFaithfulness,
    TaskCompletion,
    ToolArgumentAccuracy,
    ToolCallAccuracy,
    ToolCallNecessity,
    TrajectoryEfficiency,
)


def make_case(
    *,
    input_text="Find the weather and summarize it",
    steps=None,
    expected_tool_calls=None,
    context=None,
    expected_output=None,
):
    return EvalCase(
        input=input_text,
        agent_trace=steps,
        expected_tool_calls=expected_tool_calls,
        context=context,
        expected_output=expected_output,
    )


def make_steps():
    return [
        AgentStep(
            thought="Need the current weather first",
            tool_calls=[ToolCall(name="search_weather", arguments={"city": "Paris"}, result="18C and sunny")],
            output="Fetched the weather",
        ),
        AgentStep(
            thought="Now summarize the result",
            tool_calls=[ToolCall(name="summarize", arguments={"style": "brief"}, result="Paris is 18C and sunny")],
            output="Prepared the summary",
        ),
    ]


class TestToolCallAccuracy:
    def test_passing_unordered_case(self):
        case = make_case(steps=make_steps(), expected_tool_calls=["search_weather", "summarize"])
        result = ToolCallAccuracy().evaluate(case, "Paris is 18C and sunny.")
        assert result.passed
        assert result.score == 1.0

    def test_failing_missing_and_unexpected_tools(self):
        case = make_case(
            steps=[AgentStep(tool_calls=[ToolCall(name="search_weather"), ToolCall(name="translate")])],
            expected_tool_calls=["search_weather", "summarize"],
        )
        result = ToolCallAccuracy().evaluate(case, "done")
        assert not result.passed
        assert result.score == 0.5
        assert "Missing tools" in result.reason
        assert "Unexpected tools" in result.reason

    def test_empty_steps_fail(self):
        result = ToolCallAccuracy().evaluate(make_case(steps=[] , expected_tool_calls=["search_weather"]), "done")
        assert not result.passed
        assert "No agent_trace provided" in result.reason

    def test_no_tool_calls_still_records_failure_when_expected(self):
        case = make_case(steps=[AgentStep(thought="I can answer directly")], expected_tool_calls=["search_weather"])
        result = ToolCallAccuracy().evaluate(case, "done")
        assert not result.passed
        assert result.score == 0.0

    def test_ordered_case_fails_on_wrong_order(self):
        steps = [
            AgentStep(tool_calls=[ToolCall(name="summarize")]),
            AgentStep(tool_calls=[ToolCall(name="search_weather")]),
        ]
        case = make_case(steps=steps, expected_tool_calls=["search_weather", "summarize"])
        result = ToolCallAccuracy(require_order=True).evaluate(case, "done")
        assert not result.passed
        assert result.score == 0.0

    # ── Codex D16 cycle 5 ISSUE 1: penalize_unexpected ─────────────

    def test_default_mode_does_not_penalize_extra_tools(self):
        """Default behavior (back-compat): if the expected tool is
        called AND extras are called too, score is still 1.0 because
        the case asserted what MUST be called, not what mustn't."""
        steps = [AgentStep(tool_calls=[
            ToolCall(name="lookup_order"),
            ToolCall(name="refund_order"),
        ])]
        case = make_case(steps=steps, expected_tool_calls=["lookup_order"])
        result = ToolCallAccuracy().evaluate(case, "done")
        assert result.score == 1.0
        # Unexpected is REPORTED, just not penalized.
        assert "Unexpected tools" in result.reason
        assert "refund_order" in result.reason

    def test_strict_mode_penalizes_unexpected_tools(self):
        """penalize_unexpected=True drops the score when the agent
        called extras. This is what the framework templates'
        negative cases ("don't refund a processing order") need."""
        steps = [AgentStep(tool_calls=[
            ToolCall(name="lookup_order"),
            ToolCall(name="refund_order"),
        ])]
        case = make_case(steps=steps, expected_tool_calls=["lookup_order"])
        result = ToolCallAccuracy(penalize_unexpected=True).evaluate(case, "done")
        # matched=1 (lookup), denom=2 (lookup ∪ refund) → 0.5
        assert result.score == 0.5
        assert not result.passed
        assert "strict mode" in result.reason.lower()

    def test_strict_mode_unchanged_when_no_extras(self):
        """If only the expected tools fired, strict mode == default."""
        steps = [AgentStep(tool_calls=[ToolCall(name="lookup_order")])]
        case = make_case(steps=steps, expected_tool_calls=["lookup_order"])
        result = ToolCallAccuracy(penalize_unexpected=True).evaluate(case, "done")
        assert result.score == 1.0
        assert result.passed


class TestToolArgumentAccuracy:
    @patch("multivon_eval.evaluators.agent._judge_call", side_effect=["Yes", "Yes"])
    def test_passing_arguments(self, judge_call):
        result = ToolArgumentAccuracy().evaluate(make_case(steps=make_steps()), "done")
        assert result.passed
        assert result.score == 1.0
        assert judge_call.call_count == 2

    @patch("multivon_eval.evaluators.agent._judge_call", side_effect=["No", "Yes"])
    def test_failing_arguments(self, _judge_call):
        result = ToolArgumentAccuracy().evaluate(make_case(steps=make_steps()), "done")
        assert not result.passed
        assert result.score == 0.5

    def test_empty_steps_fail(self):
        result = ToolArgumentAccuracy().evaluate(make_case(steps=[]), "done")
        assert not result.passed
        assert "No agent_trace provided" in result.reason

    def test_no_tool_calls_passes(self):
        result = ToolArgumentAccuracy().evaluate(make_case(steps=[AgentStep(thought="No tools needed")]), "done")
        assert result.passed
        assert result.score == 1.0
        assert "No tool calls in trace" in result.reason


class TestPlanQuality:
    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(1.0, ["ok"]))
    def test_passing_case(self, qag_eval):
        result = PlanQuality().evaluate(make_case(steps=make_steps()), "done")
        assert result.passed
        assert result.score == 1.0
        assert qag_eval.called

    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(0.4, ["bad plan"]))
    def test_failing_case(self, _qag_eval):
        result = PlanQuality().evaluate(make_case(steps=make_steps()), "done")
        assert not result.passed
        assert result.score == 0.4

    def test_empty_steps_fail(self):
        result = PlanQuality().evaluate(make_case(steps=[]), "done")
        assert not result.passed

    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(0.8, ["single step ok"]))
    def test_no_tool_calls_still_evaluates_trace(self, _qag_eval):
        result = PlanQuality().evaluate(make_case(steps=[AgentStep(thought="Answer directly")]), "done")
        assert result.passed
        assert result.score == 0.8


class TestTaskCompletion:
    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(1.0, ["completed"]))
    def test_passing_case(self, _qag_eval):
        result = TaskCompletion().evaluate(make_case(steps=make_steps()), "Paris is 18C and sunny.")
        assert result.passed

    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(0.25, ["incomplete"]))
    def test_failing_case(self, _qag_eval):
        result = TaskCompletion().evaluate(make_case(steps=make_steps()), "I do not know.")
        assert not result.passed
        assert result.score == 0.25

    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(0.75, ["answered directly"]))
    def test_empty_steps_are_allowed(self, _qag_eval):
        result = TaskCompletion().evaluate(make_case(steps=[]), "done")
        assert result.passed

    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(0.75, ["no tools needed"]))
    def test_no_tool_calls_are_allowed(self, _qag_eval):
        result = TaskCompletion().evaluate(make_case(steps=[AgentStep(output="Answered directly")]), "done")
        assert result.passed


class TestStepFaithfulness:
    @patch("multivon_eval.evaluators.agent._judge_call", side_effect=["Yes", "Yes"])
    def test_passing_case(self, judge_call):
        result = StepFaithfulness().evaluate(make_case(steps=make_steps()), "done")
        assert result.passed
        assert result.score == 1.0
        assert judge_call.call_count == 2

    @patch("multivon_eval.evaluators.agent._judge_call", side_effect=["Yes", "No"])
    def test_failing_case(self, _judge_call):
        result = StepFaithfulness().evaluate(make_case(steps=make_steps()), "done")
        assert not result.passed
        assert result.score == 0.5

    def test_empty_steps_fail(self):
        result = StepFaithfulness().evaluate(make_case(steps=[]), "done")
        assert not result.passed

    @patch("multivon_eval.evaluators.agent._judge_call", side_effect=["Yes"])
    def test_no_tool_calls_still_scores_steps(self, _judge_call):
        result = StepFaithfulness().evaluate(make_case(steps=[AgentStep(thought="Reason directly")]), "done")
        assert result.passed
        assert result.score == 1.0


class TestToolCallNecessity:
    @patch("multivon_eval.evaluators.agent._judge_call", side_effect=["Yes", "Yes"])
    def test_passing_case(self, judge_call):
        result = ToolCallNecessity().evaluate(make_case(steps=make_steps()), "done")
        assert result.passed
        assert result.score == 1.0
        assert judge_call.call_count == 2

    @patch("multivon_eval.evaluators.agent._judge_call", side_effect=["Yes", "No"])
    def test_failing_case(self, _judge_call):
        result = ToolCallNecessity().evaluate(make_case(steps=make_steps()), "done")
        assert not result.passed
        assert result.score == 0.5

    def test_empty_steps_fail(self):
        result = ToolCallNecessity().evaluate(make_case(steps=[]), "done")
        assert not result.passed

    def test_no_tool_calls_passes(self):
        result = ToolCallNecessity().evaluate(make_case(steps=[AgentStep(thought="No tools needed")]), "done")
        assert result.passed
        assert result.score == 1.0


class TestTrajectoryEfficiency:
    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(1.0, ["efficient"]))
    def test_passing_case(self, _qag_eval):
        result = TrajectoryEfficiency().evaluate(make_case(steps=make_steps()), "done")
        assert result.passed
        assert result.score == 1.0

    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(0.4, ["inefficient"]))
    def test_failing_case(self, _qag_eval):
        result = TrajectoryEfficiency().evaluate(make_case(steps=make_steps()), "done")
        assert not result.passed
        assert result.score == 0.4

    def test_empty_steps_fail(self):
        result = TrajectoryEfficiency().evaluate(make_case(steps=[]), "done")
        assert not result.passed

    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(0.9, ["direct answer"]))
    def test_no_tool_calls_still_evaluates(self, _qag_eval):
        result = TrajectoryEfficiency().evaluate(make_case(steps=[AgentStep(thought="Answer directly")]), "done")
        assert result.passed
        assert result.score == 0.9

    # Recovery scoring now uses `_judge_call_with` (per-evaluator judge) instead
    # of the global `_judge_call`, so the patch target has changed accordingly.
    @patch("multivon_eval.evaluators.agent._judge_call_with", return_value="No")
    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(0.8, ["reasonable"]))
    def test_failed_tool_recovery_penalizes_score(self, _qag_eval, _judge_call_with):
        steps = [
            AgentStep(
                thought="Try the tool",
                tool_calls=[ToolCall(name="search_weather", arguments={"city": "Paris"}, result="error: timeout")],
            )
        ]
        result = TrajectoryEfficiency().evaluate(make_case(steps=steps), "done")
        assert not result.passed
        assert result.score == 0.6000000000000001


class TestAgentMemoryEval:
    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(1.0, ["used memory"]))
    def test_passing_case(self, _qag_eval):
        case = make_case(
            steps=make_steps(),
            context="User prefers metric units and lives in Paris.",
            expected_output="metric units",
        )
        result = AgentMemoryEval().evaluate(case, "You prefer metric units.")
        assert result.passed

    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(0.5, ["hallucinated memory"]))
    def test_failing_case(self, _qag_eval):
        case = make_case(context="User prefers metric units.", expected_output="metric units")
        result = AgentMemoryEval().evaluate(case, "You prefer imperial units.")
        assert not result.passed
        assert result.score == 0.5

    def test_missing_context_fails(self):
        result = AgentMemoryEval().evaluate(make_case(context=None), "done")
        assert not result.passed
        assert "No context provided" in result.reason

    @patch("multivon_eval.evaluators.agent._qag_eval", return_value=(0.75, ["nothing to remember but okay"]))
    def test_no_tool_calls_are_allowed(self, _qag_eval):
        case = make_case(context="The user likes concise answers.", steps=[AgentStep(output="Answer directly")])
        result = AgentMemoryEval().evaluate(case, "Concise answer.")
        assert result.passed
