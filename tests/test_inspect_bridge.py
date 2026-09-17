"""Exercise the real Inspect runner and native log format without API calls."""
import json

import pytest

inspect_ai = pytest.importorskip("inspect_ai")

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput
from inspect_ai.solver import solver

from multivon_eval import (
    AcceptancePolicy,
    CaseManifest,
    CheckRequirement,
    EvalCase,
    EvalReport,
    ExactMatch,
    NotEmpty,
)
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.integrations.inspect import (
    as_inspect_scorer,
    bind_inspect_task,
    from_inspect_log,
    to_inspect_dataset,
)


@solver
def fixture_output():
    async def solve(state, generate):
        from inspect_ai.model import ChatMessageAssistant
        answer = "wrong" if "bad" in str(state.sample_id) else "yes"
        state.output = ModelOutput.from_content("fixture", answer)
        state.messages.append(ChatMessageAssistant(content=answer))
        return state
    return solve


def execute(tmp_path, cases, graders, epochs=1, bound=False):
    manifest = CaseManifest("inspect bridge", cases)
    task = Task(dataset=to_inspect_dataset(manifest), solver=fixture_output(),
                scorer=[as_inspect_scorer(ev) for ev in graders], epochs=epochs)
    if bound:
        bind_inspect_task(task, version='bridge-fixture/v1', dependencies={})
    logs = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none",
                        log_format="eval", log_model_api=True)
    return logs[0]


def test_native_runner_log_roundtrip_and_acceptance(tmp_path):
    log = execute(tmp_path, [EvalCase("x", "yes", case_id="good", source_id="one"),
                             EvalCase("y", "yes", case_id="bad", source_id="two")],
                  [ExactMatch(), NotEmpty()], epochs=2)
    assert log.status == "success", log.error
    loaded = read_eval_log(log.location)
    report = from_inspect_log(loaded)
    assert report.total == 2 and report.passed == 1
    assert all(len(r.trials) == 2 for r in report.case_results)
    evidence = report.case_results[0].trials[0].data
    assert evidence["upstream"]["log_location"] == log.location
    assert len(evidence["upstream"]["sample_digest"]) == 64
    assert evidence["latency_ms"] is None
    copied = EvalReport.from_dict(json.loads(report.to_json()))
    assert copied.case_results[0].trials == report.case_results[0].trials
    policy = AcceptancePolicy((CheckRequirement("exact_match"),), min_cases=2, min_source_groups=2)
    assert policy.evaluate(report).decision == "reject"


def test_native_skip_is_null_and_policy_indeterminate(tmp_path):
    log = execute(tmp_path, [EvalCase("x", case_id="unknown")], [ExactMatch()])
    assert log.status == "success"
    score = next(iter(log.samples[0].scores.values()))
    assert score.value == {"score": None, "passed": None}
    report = from_inspect_log(log)
    assert report.skipped == 1
    assert AcceptancePolicy((CheckRequirement("exact_match"),)).evaluate(report).decision == "indeterminate"


def test_native_scoring_exception_stays_an_error(tmp_path):
    class Crash(Evaluator):
        name = "crash"
        def evaluate(self, case, output):
            raise RuntimeError("grader crashed")
    log = execute(tmp_path, [EvalCase("x", case_id="error")], [Crash()])
    report = from_inspect_log(log)
    assert report.errors == 1 and report.passed == 0
    assert report.evidence_issues
    result = AcceptancePolicy((CheckRequirement("crash", critical=True),)).evaluate(report)
    assert result.decision == "indeterminate"


def test_context_is_delivered_to_native_model_and_case_id_checked():
    from multivon_eval.integrations.inspect import _case_from_metadata
    samples = to_inspect_dataset(CaseManifest("context", [EvalCase("x", context="source", case_id="id")]))
    sample = samples[0]
    assert sample.input[0].role == "system" and "source" in sample.input[0].text
    with pytest.raises(ValueError, match="ID differs"):
        _case_from_metadata(sample.metadata, "other")


def test_native_tool_transcript_converts_without_reusing_static_trace():
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
    from inspect_ai.tool import ToolCall as InspectToolCall

    from multivon_eval import AgentStep, ToolCall
    from multivon_eval.integrations.inspect import _execution_case
    case = EvalCase("x", agent_trace=[AgentStep(tool_calls=[ToolCall("fake")])])
    messages = [ChatMessageAssistant(content="", tool_calls=[InspectToolCall(
        id="call-1", function="lookup", arguments={"id": 1})]),
        ChatMessageTool(content="found", tool_call_id="call-1")]
    converted = _execution_case(case, messages)
    assert converted.agent_trace[0].tool_calls == [ToolCall("lookup", {"id": 1}, "found")]
    assert case.agent_trace[0].tool_calls[0].name == "fake"


def test_documented_inspect_example(tmp_path):
    import re
    import subprocess
    import sys
    from pathlib import Path
    source = (Path(__file__).parents[1] / "docs/guides/inspect-integration.mdx").read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
    script = tmp_path / "inspect_example.py"
    script.write_text("\n".join(blocks))
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_partial_epoch_retry_history_preserves_executions_without_double_counting(tmp_path):
    from inspect_ai.log import EvalError

    from multivon_eval.trials import trial_integrity_issues
    log = execute(tmp_path, [EvalCase("x", "yes", case_id="good")], [ExactMatch()], epochs=2, bound=True)
    prior = log.model_copy(deep=True)
    prior.status = "error"
    interrupted = next(sample for sample in prior.samples if sample.epoch == 2)
    interrupted.uuid = "interrupted-original-execution"
    interrupted.error = EvalError(message="interrupted", traceback="", traceback_ansi="")
    interrupted.scores = None
    combined = from_inspect_log(log, previous_logs=[prior])
    row = combined.case_results[0]
    assert row.runs == 2 and row.retry_attempts == 1 and len(row.trials) == 3
    assert [(t.data["attempt"], t.data["run_index"]) for t in row.trials] == [(1, 1), (1, 2), (2, 2)]
    assert not trial_integrity_issues(row)
    requirement = (CheckRequirement("exact_match"),)
    assert AcceptancePolicy(requirement).evaluate(combined).decision == "indeterminate"
    assert AcceptancePolicy(requirement, trial_scope="final_attempt").evaluate(combined).decision == "accept"
    altered = prior.model_copy(deep=True)
    altered.eval.task_id = "unrelated"
    with pytest.raises(ValueError, match="same task ID"):
        from_inspect_log(log, previous_logs=[altered])
    prior.status = "started"
    with pytest.raises(ValueError, match="Recover started"):
        from_inspect_log(log, previous_logs=[prior])


def test_native_tool_errors_survive_projection_and_scoring(tmp_path):
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
    from inspect_ai.tool import ToolCall as InspectToolCall
    from inspect_ai.tool import ToolCallError

    from multivon_eval.integrations.inspect import _execution_case

    messages = [ChatMessageAssistant(content="", tool_calls=[InspectToolCall(
        id="bad-call", function="post_entry", arguments={"amount": "1.00"})]),
        ChatMessageTool(content="currency is required", tool_call_id="bad-call",
                        error=ToolCallError(type="parsing", message="currency is required"))]
    case = EvalCase("Post", case_id="bad-call-case")
    result = _execution_case(case, messages)
    assert result.agent_trace[0].tool_calls[0].result == {
        "error": {"type": "parsing", "message": "currency is required"}, "text": "currency is required"}

    @solver
    def failed_tool():
        async def solve(state, generate):
            state.messages.extend(messages)
            state.output = ModelOutput.from_content("fixture", "")
            return state
        return solve

    task = Task(dataset=to_inspect_dataset(CaseManifest("tool errors", [case])),
                solver=failed_tool(), scorer=as_inspect_scorer(NotEmpty()))
    log = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success", log.error
    report = from_inspect_log(read_eval_log(log.location))
    assert report.errors == 0 and report.failed == 1
    assert report.case_results[0].agent_trace[0].tool_calls[0].result["error"]["type"] == "parsing"
