"""Exercise the real Inspect runner and native log format without API calls."""
import json

import pytest

inspect_ai = pytest.importorskip("inspect_ai")

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput
from inspect_ai.solver import solver
from multivon_eval import (
    AcceptancePolicy, CaseManifest, CheckRequirement, EvalCase, EvalReport,
    EvalResult, ExactMatch, NotEmpty,
)
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.integrations.inspect import as_inspect_scorer, from_inspect_log, to_inspect_dataset


@solver
def fixture_output():
    async def solve(state, generate):
        from inspect_ai.model import ChatMessageAssistant
        answer = "wrong" if "bad" in str(state.sample_id) else "yes"
        state.output = ModelOutput.from_content("fixture", answer)
        state.messages.append(ChatMessageAssistant(content=answer))
        return state
    return solve


def execute(tmp_path, cases, graders, epochs=1):
    manifest = CaseManifest("inspect bridge", cases)
    task = Task(dataset=to_inspect_dataset(manifest), solver=fixture_output(),
                scorer=[as_inspect_scorer(ev) for ev in graders], epochs=epochs)
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
    from multivon_eval.integrations.inspect import _execution_case
    from multivon_eval import AgentStep, ToolCall
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
    blocks = re.findall(r"```python\n(.*?)```", source, re.S)
    script = tmp_path / "inspect_example.py"
    script.write_text("\n".join(blocks))
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
