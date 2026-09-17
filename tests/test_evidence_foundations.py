"""Regression tests for identity, split leakage and lost execution evidence."""
import asyncio
import dataclasses
import json

import pytest

from multivon_eval import (
    AgentStep, CaseManifest, EvalCase, EvalReport, EvalResult, EvalSuite, ExactMatch,
    JudgeRetry, ToolCall, TrialRecord, compare_reports, load_jsonl, regrade, save_jsonl,
)
from multivon_eval.compare import _cli
from multivon_eval.case_manifest import case_from_dict, case_to_dict
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.exceptions import JudgeUnavailable
from multivon_eval.result import CaseResult


def test_jsonl_preserves_all_case_fields(tmp_path):
    original = EvalCase(
        "read", "done", ["source"], [{"role": "user", "content": "read"}],
        [AgentStep("check", [ToolCall("lookup", {"id": 1}, {"found": True})], "done")],
        ["lookup"], {"nested": {"label": "café"}}, ["critical"], "reference",
        "invoice-1", "2", "document-1",
    )
    path = tmp_path / "cases.jsonl"
    save_jsonl([original], str(path))
    assert load_jsonl(str(path)) == [original]
    data = case_to_dict(original)
    data["metadata"]["nested"]["label"] = "changed"
    assert original.metadata["nested"]["label"] == "café"


@pytest.mark.parametrize("data", [
    {"input": "x", "expectd_output": "typo"}, {"input": 2},
    {"input": "x", "tags": "critical"}, {"input": "x", "case_id": ""},
    {"input": "x", "metadata": {1: "lossy"}},
    {"input": "x", "context": [1]},
    {"input": "x", "agent_trace": [{"tool_calls": [{"name": 4}]}]},
    {"input": "x", "metadata": {"score": float("nan")}},
])
def test_portable_case_rejects_silent_corruption(data):
    with pytest.raises((ValueError, TypeError)):
        case_from_dict(data)


def test_callable_reference_is_not_execution_identity_and_cannot_silently_export(tmp_path):
    original = EvalCase("x")
    reference = dataclasses.replace(original, reference_output=lambda case: "answer")
    assert original.identity() == reference.identity()
    path = tmp_path / "cases.jsonl"
    path.write_text("existing")
    with pytest.raises(ValueError, match="Callable"):
        save_jsonl([original, reference], str(path))
    assert path.read_text() == "existing"


def test_dataset_detaches_inputs_and_is_order_independent(tmp_path):
    cases = [EvalCase("a", case_id="a", source_id="source-a"),
             EvalCase("b", case_id="b", source_id="source-b")]
    dataset = CaseManifest("invoices", cases, splits={"dev": ["a"], "test": ["b"]})
    assert dataset.digest == CaseManifest("invoices", cases[::-1],
                                     splits={"test": ["b"], "dev": ["a"]}).digest
    cases[0].metadata["changed"] = True
    dataset.cases[0].input = "mutated"
    dataset.manifest["cases"].clear()
    assert dataset.cases[0].metadata == {}
    assert dataset.split("test")[0].input == "b"
    path = tmp_path / "dataset.json"
    dataset.save(path)
    assert CaseManifest.load(path).manifest == dataset.manifest
    corrupt = dataset.manifest
    corrupt["cases"][0]["case"]["expected_output"] = "tampered"
    with pytest.raises(ValueError, match="digest"):
        CaseManifest.from_dict(corrupt)


@pytest.mark.parametrize("splits,match", [
    ({"dev": ["a"], "test": ["b"]}, "Source leakage"),
    ({"dev": ["a"]}, "every case"),
    ({"dev": ["a", "b", "a"]}, "more than once"),
    ({"dev": ["a", "b", "missing"]}, "Unknown case"),
])
def test_dataset_rejects_leakage_and_bad_assignments(splits, match):
    cases = [EvalCase("a", case_id="a", source_id="same"),
             EvalCase("b", case_id="b", source_id="same")]
    with pytest.raises(ValueError, match=match):
        CaseManifest("x", cases, splits=splits)


def test_duplicate_auto_identity_requires_explicit_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        CaseManifest("x", [EvalCase("a"), EvalCase("a")])


def _suite(cases):
    return EvalSuite("evidence").add_cases(cases).add_evaluator(ExactMatch())


@pytest.mark.parametrize("field,value", [
    ("context", "changed source"), ("expected_output", "new answer"),
    ("expected_tool_calls", ["delete"]), ("metadata", {"amount": 12}),
    ("revision", "2"), ("tags", ["critical"]),
])
def test_changed_definition_cannot_produce_paired_improvement(field, value):
    case = EvalCase("same prompt", "yes", case_id="case-1")
    baseline = _suite([case]).run(lambda _: "no", verbose=False)
    proposal = _suite([dataclasses.replace(case, **{field: value})]).run(lambda _: "yes", verbose=False)
    diff = compare_reports(baseline, proposal)
    assert diff.identity_issues
    assert not diff.paired and not diff.improvements
    assert diff.mcnemar_p is None


def test_reordered_duplicate_prompts_pair_by_definition():
    a = EvalCase("same", "A", context="A")
    b = EvalCase("same", "B", context="B")
    class Model:
        def _call_with_case(self, case):
            return case.expected_output
    baseline = _suite([a, b]).run(Model(), verbose=False)
    proposal = _suite([b, a]).run(Model(), verbose=False)
    diff = compare_reports(baseline, proposal)
    assert len(diff.paired) == 2 and not diff.identity_issues
    assert not diff.regressions and not diff.added and not diff.removed
    assert diff.paired[0].case_id == b.identity()[0]


def test_duplicate_explicit_ids_are_never_paired():
    report = _suite([EvalCase("a", case_id="same"), EvalCase("b", case_id="same")]).run(
        lambda _: "ok", verbose=False)
    diff = compare_reports(report, report)
    assert "Duplicate" in " ".join(diff.identity_issues)
    assert not diff.paired and diff.mcnemar_p is None


def test_legacy_gate_requires_explicit_override(tmp_path):
    report = EvalReport("old", [CaseResult("x", "yes", [EvalResult("e", 1, True)])])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    report.save_json(str(a))
    report.save_json(str(b))
    assert compare_reports(report, report).mcnemar_p is None
    assert _cli([str(a), str(b), "--fail-on-regression"]) == 2
    assert _cli([str(a), str(b), "--fail-on-regression", "--allow-legacy-identity"]) == 0


@pytest.mark.parametrize("mode", ["sync", "parallel", "async"])
def test_every_run_and_retry_survives_json_and_regrade(mode):
    class Transient(Evaluator):
        name = "transient"
        def evaluate(self, case, output):
            if output == "attempt-1":
                raise JudgeUnavailable("temporary")
            return EvalResult(self.name, 1, True, f"saw {output}", {"raw": [output]})
    calls = 0
    def target(_):
        nonlocal calls
        calls += 1
        return f"attempt-{calls}"
    async def async_target(prompt):
        return target(prompt)
    case = EvalCase("x", agent_trace=[AgentStep(tool_calls=[ToolCall("read", {}, "safe")])])
    suite = _suite([case]).add_evaluator(Transient())
    kwargs = dict(runs=2, verbose=False, judge_retry=JudgeRetry(max_attempts=2, base_backoff=0))
    if mode == "async":
        report = asyncio.run(suite.run_async(async_target, **kwargs))
    else:
        report = suite.run(target, workers=2 if mode == "parallel" else 1, **kwargs)
    assert calls == 4
    result = report.case_results[0]
    assert len(result.trials) == 4 and result.retry_attempts == 1
    assert [(t.data["attempt"], t.data["run_index"]) for t in result.trials] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    for trial in result.trials:
        evidence = trial.data["provider_evidence"]
        assert evidence["state"] == "closed"
        for event in evidence["events"]:
            assert event["labels"]["attempt"] == trial.data["attempt"]
            assert event["labels"]["run_index"] == trial.data["run_index"]
            assert event["labels"]["case_id"] == result.case_id
    assert result.trials[0].data["status"] == "judge_error"
    assert result.trials[1].data["evaluators"][1]["reason"] == "saw attempt-2"
    copy = EvalReport.from_dict(json.loads(report.to_json()))
    assert copy.case_results[0].trials == result.trials
    assert copy.case_results[0].agent_trace == result.agent_trace
    reviewed = regrade(copy, _suite([]))
    assert calls == 4 and reviewed.total == 4
    assert reviewed.case_results[1].trials[0].data["parent_trial"] == result.trials[1].digest
    case.agent_trace[0].tool_calls[0].result = "mutated"
    assert result.trials[0].data["agent_trace"][0]["tool_calls"][0]["result"] == "safe"
    detached = result.trials[0].data
    detached["output"] = "forged"
    with pytest.raises(ValueError, match="digest"):
        TrialRecord.from_dict(detached)


def test_nonportable_metadata_keeps_result_but_marks_missing_evidence():
    report = _suite([EvalCase("x", "x", metadata={"opaque": object()})]).run(
        lambda _: "x", verbose=False)
    result = report.case_results[0]
    assert result.passed and result.evidence_error and not result.trials
    assert compare_reports(report, report).identity_issues
    with pytest.raises(ValueError, match="complete saved trials"):
        regrade(report, _suite([]))


def test_identity_is_captured_before_target_mutation():
    case = EvalCase("x", "ok", metadata={"state": "before"})
    original = case.identity()
    class Model:
        def _call_with_case(self, case):
            case.metadata["state"] = "after"
            return "ok"
    result = _suite([case]).run(Model(), verbose=False).case_results[0]
    assert (result.case_id, result.case_digest) == original
    assert result.trials[0].data["case"]["metadata"]["state"] == "before"


def test_legacy_report_does_not_invent_trials_or_individual_scores():
    report = EvalReport.from_dict({"cases": [{"input": "x", "output": "last",
        "runs": 3, "score": 0.5, "score_std": 0.2}]})
    assert report.case_results[0].all_scores == []
    assert report.case_results[0].trials == ()


def test_model_error_regrade_remains_error():
    def target(_):
        raise RuntimeError("target failed")
    report = _suite([EvalCase("x")]).run(target, verbose=False)
    reviewed = regrade(report, _suite([]))
    assert reviewed.case_results[0].model_error == "target failed"
    assert not reviewed.case_results[0].results


def test_regrade_uses_evaluation_time_metadata():
    class AssertState(Evaluator):
        name = "state"
        def evaluate(self, case, output):
            ok = case.metadata["state"] == "after"
            return EvalResult(self.name, float(ok), ok)
    class Model:
        def _call_with_case(self, case):
            case.metadata["state"] = "after"
            return "ok"
    suite = EvalSuite("state").add_case(EvalCase("x", metadata={"state": "before"}))
    suite.add_evaluator(AssertState())
    report = suite.run(Model(), verbose=False)
    assert report.passed == 1
    reviewed = regrade(report, suite)
    assert reviewed.passed == 1
    assert reviewed.case_results[0].case_digest == report.case_results[0].case_digest


def test_imported_latency_is_unknown_unless_supplied():
    from multivon_eval import MaxLatency
    suite = EvalSuite("latency").add_evaluator(MaxLatency(10))
    pairs = [(EvalCase("x"), "answer")]
    missing = suite.run_on_cases(pairs, verbose=False)
    assert missing.skipped == 1
    assert missing.case_results[0].trials[0].data["latency_ms"] is None
    assert regrade(missing, suite).skipped == 1
    slow = suite.run_on_cases(pairs, verbose=False, latencies_ms=[100])
    assert slow.failed == 1 and regrade(slow, suite).failed == 1


def test_report_preserves_costs_and_suite_lock():
    from multivon_eval import Costs, ProviderUsage
    report = _suite([EvalCase("x", "x")]).run(lambda _: "x", verbose=False)
    report.costs = Costs([ProviderUsage("test", "unknown", 20, 5, 1, None)])
    copy = EvalReport.from_dict(json.loads(report.to_json()))
    assert copy.costs.to_dict() == report.costs.to_dict()
    assert copy.suite_lock.to_dict() == report.suite_lock.to_dict()


def test_documented_evidence_workflow_runs(tmp_path, monkeypatch):
    import re
    from pathlib import Path
    source = (Path(__file__).parents[1] / "docs/guides/versioned-evidence.mdx").read_text()
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", source, re.S):
        if "from datasets import" in block:
            continue  # exercised with the real optional dependency in interoperability tests
        exec(compile(block, "versioned-evidence.mdx", "exec"), namespace)
    assert namespace["reviewed"].total == 3
