"""Release gates distinguish rejection, missing evidence and acceptance."""
import dataclasses
import json

import pytest

from multivon_eval import (
    AcceptancePolicy, CheckRequirement, EvalCase, EvalGateFailure, EvalReport,
    EvalResult, EvalSuite, ExactMatch, JudgeRetry, NotEmpty, SliceRequirement,
)
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.exceptions import JudgeUnavailable
from multivon_eval.result import CaseResult


def policy(**kwargs):
    return AcceptancePolicy((CheckRequirement("exact_match"),), **kwargs)


def run(cases, outputs):
    return EvalSuite("test").add_cases(cases).add_evaluator(ExactMatch()).run(
        outputs.__getitem__, verbose=False)


def test_complete_cases_pass_and_policy_digest_changes_with_contract():
    report = run([EvalCase("x", "yes", source_id="document")], {"x": "yes"})
    contract = policy(min_source_groups=1)
    result = contract.evaluate(report)
    assert result.decision == "accept" and result.exit_code == 0
    assert result.findings == ()
    result.assert_accepted()
    assert json.loads(json.dumps(result.to_dict()))["policy_digest"] == contract.digest
    assert dataclasses.replace(contract, min_cases=2).digest != contract.digest


def test_missing_required_check_cannot_pass_from_other_checks():
    report = EvalSuite("missing").add_case(EvalCase("x")).add_evaluator(NotEmpty()).run(
        lambda _: "ok", verbose=False)
    assert report.pass_rate == 1
    result = policy().evaluate(report)
    assert result.decision == "indeterminate" and result.exit_code == 2
    with pytest.raises(EvalGateFailure) as caught:
        result.assert_accepted()
    assert caught.value.code == 2 and caught.value.acceptance is result


def test_partial_skip_cannot_pass_required_coverage():
    class Skip(Evaluator):
        name = "required"
        def evaluate(self, case, output):
            return EvalResult(self.name, 0, False, "missing input", {"skipped": True})
    report = EvalSuite("partial").add_case(EvalCase("x")).add_evaluators(NotEmpty(), Skip()).run(
        lambda _: "ok", verbose=False)
    assert report.passed == 1
    contract = AcceptancePolicy((CheckRequirement("not_empty"), CheckRequirement("required")))
    result = contract.evaluate(report)
    assert result.decision == "indeterminate"
    assert any(f.code == "check_coverage" for f in result.findings)


def test_known_quality_failure_rejects_even_with_missing_evidence():
    report = run([EvalCase("x", "yes")], {"x": "no"})
    result = policy(min_cases=20).evaluate(report)
    assert result.decision == "reject" and result.exit_code == 1
    assert {f.kind for f in result.findings} == {"quality", "evidence"}


def test_source_group_count_does_not_count_variants_as_independent():
    report = run([EvalCase("a", "yes", source_id="same"),
                  EvalCase("b", "yes", source_id="same")], {"a": "yes", "b": "yes"})
    result = policy(min_source_groups=2).evaluate(report)
    assert result.decision == "indeterminate"
    assert any(f.code == "insufficient_source_groups" for f in result.findings)


def test_missing_source_provenance_is_indeterminate():
    report = run([EvalCase("a", "yes")], {"a": "yes"})
    assert policy(min_source_groups=1).evaluate(report).decision == "indeterminate"


def test_missing_slice_and_small_slice_are_indeterminate():
    report = run([EvalCase("a", "yes", tags=["edge"])], {"a": "yes"})
    for requirement in (SliceRequirement("absent"), SliceRequirement("edge", min_cases=2)):
        assert policy(slices=(requirement,)).evaluate(report).decision == "indeterminate"


def test_slice_can_reject_despite_global_threshold_passing():
    report = run([EvalCase("a", "yes"), EvalCase("b", "yes", tags=["edge"])],
                 {"a": "yes", "b": "no"})
    contract = AcceptancePolicy((CheckRequirement("exact_match", min_pass_rate=0.5),),
                                 slices=(SliceRequirement("edge", min_pass_rate=1),))
    result = contract.evaluate(report)
    assert result.decision == "reject"
    assert all(f.scope == "tag:edge" for f in result.findings)


def test_repeated_trials_do_not_inflate_sample_size():
    report = EvalSuite("repeats").add_case(EvalCase("x", "yes")).add_evaluator(ExactMatch()).run(
        lambda _: "yes", runs=20, verbose=False)
    assert policy(min_cases=2).evaluate(report).decision == "indeterminate"
    assert policy().evaluate(report).measurements[1]["measured_cases"] == 1


def test_critical_failure_survives_aggregate_majority_vote():
    outputs = iter(["no", "yes", "yes"])
    report = EvalSuite("unsafe").add_case(EvalCase("x", "yes")).add_evaluator(ExactMatch()).run(
        lambda _: next(outputs), runs=3, verbose=False)
    assert report.case_results[0].results[0].passed
    contract = AcceptancePolicy((CheckRequirement("exact_match", min_pass_rate=0, critical=True),))
    result = contract.evaluate(report)
    assert result.decision == "reject"
    assert result.findings[0].code == "critical_failure"


def test_retry_policy_explicitly_controls_which_attempts_count():
    class FailFirst(Evaluator):
        name = "available"
        calls = 0
        def evaluate(self, case, output):
            self.calls += 1
            if self.calls == 1:
                raise JudgeUnavailable("outage")
            return EvalResult(self.name, 1, True)
    suite = EvalSuite("retries").add_case(EvalCase("x", "yes")).add_evaluators(ExactMatch(), FailFirst())
    report = suite.run(lambda _: "yes", verbose=False, judge_retry=JudgeRetry(base_backoff=0))
    assert report.passed == 1
    assert policy().evaluate(report).decision == "indeterminate"
    assert policy(trial_scope="final_attempt").evaluate(report).decision == "accept"


def test_imported_judge_error_does_not_become_critical_quality_rejection():
    class Outage(Evaluator):
        name = "critical"
        def evaluate(self, case, output):
            raise JudgeUnavailable("outage")
    report = EvalSuite("outage").add_evaluator(Outage()).run_on_cases([(EvalCase("x"), "yes")], verbose=False)
    result = AcceptancePolicy((CheckRequirement("critical", critical=True),)).evaluate(report)
    assert result.decision == "indeterminate"
    assert not any(f.kind == "quality" for f in result.findings)


def test_empty_and_legacy_reports_are_indeterminate():
    assert policy().evaluate(EvalReport("empty", [])).decision == "indeterminate"
    legacy = EvalReport("legacy", [CaseResult("x", "yes", [EvalResult("exact_match", 1, True)])])
    assert policy().evaluate(legacy).decision == "indeterminate"
    assert policy(require_identity=False, require_trials=False).evaluate(legacy).decision == "accept"


@pytest.mark.parametrize("kwargs", [
    {"min_cases": 0}, {"min_cases": True}, {"min_source_groups": -1},
    {"max_error_rate": float("nan")}, {"max_error_rate": 1.1},
    {"trial_scope": "random"}, {"require_trials": "yes"},
])
def test_invalid_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        policy(**kwargs)


@pytest.mark.parametrize("requirement", [
    lambda: CheckRequirement(""), lambda: CheckRequirement("x", min_pass_rate=-1),
    lambda: CheckRequirement("x", min_coverage=float("inf")),
    lambda: SliceRequirement(""), lambda: SliceRequirement("edge", min_cases=0),
    lambda: AcceptancePolicy(()),
])
def test_invalid_requirements_are_rejected(requirement):
    with pytest.raises(ValueError):
        requirement()


def test_actual_cli_saves_decision_before_nonzero_exit(tmp_path):
    import subprocess
    import sys
    report = run([EvalCase("x", "yes")], {"x": "no"})
    report_path, policy_path, decision_path = [tmp_path / name for name in ("report.json", "policy.json", "decision.json")]
    report.save_json(str(report_path))
    policy_path.write_text(json.dumps(policy().to_dict()))
    completed = subprocess.run([sys.executable, "-c", "from multivon_eval.cli import main; main()",
        "gate", str(report_path), "--policy", str(policy_path), "--output", str(decision_path)],
        capture_output=True, text=True)
    assert completed.returncode == 1, completed.stderr
    assert json.loads(decision_path.read_text())["decision"] == "reject"
    assert AcceptancePolicy.from_dict(json.loads(policy_path.read_text())).digest == policy().digest


def test_actual_cli_legacy_override_reaches_comparison(tmp_path):
    import subprocess
    import sys
    report = EvalReport("old", [CaseResult("x", "ok", [EvalResult("required", 1, True)])])
    path = tmp_path / "old.json"
    report.save_json(str(path))
    completed = subprocess.run([sys.executable, "-c", "from multivon_eval.cli import main; main()",
        "compare", str(path), str(path), "--fail-on-regression", "--allow-legacy-identity"],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_upstream_incomplete_log_cannot_pass_policy_or_comparison():
    report = run([EvalCase("x", "yes")], {"x": "yes"})
    report.evidence_issues.append("Upstream run was interrupted")
    assert policy().evaluate(report).decision == "indeterminate"
    from multivon_eval import compare_reports
    assert compare_reports(report, report).identity_issues
    assert compare_reports(report, report).mcnemar_p is None


def test_documented_policy_example(tmp_path, monkeypatch):
    import re
    from pathlib import Path
    source = (Path(__file__).parents[1] / "docs/guides/acceptance-policies.mdx").read_text()
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", source, re.S):
        exec(compile(block, "acceptance-policies.mdx", "exec"), namespace)
    assert namespace["decision"].decision == "accept"


@pytest.mark.parametrize("mutation", ["id", "output", "tags", "lost_run", "duplicate_run", "all_trials"])
def test_detached_headers_or_lost_trials_cannot_pass(mutation):
    report = EvalSuite("integrity").add_case(EvalCase("x", "yes")).add_evaluator(ExactMatch()).run(
        lambda _: "yes", runs=2, verbose=False)
    result = report.case_results[0]
    if mutation == "id":
        result.case_id = "different"
    elif mutation == "output":
        result.actual_output = "edited"
    elif mutation == "tags":
        result.tags = ["different"]
    elif mutation == "all_trials":
        result.trials = ()
    elif mutation == "lost_run":
        result.trials = result.trials[:1]
    else:
        result.trials = (result.trials[0], result.trials[0])
    assert policy().evaluate(report).decision == "indeterminate"
    from multivon_eval import compare_reports
    assert compare_reports(report, report).mcnemar_p is None


def test_missing_trace_cannot_establish_no_forbidden_tools():
    from multivon_eval import ToolCallAccuracy
    case = EvalCase("Do not call tools", expected_tool_calls=[])
    missing = ToolCallAccuracy().evaluate(case, "done")
    assert missing.metadata["skipped"] and not missing.passed
    case.agent_trace = []
    observed_empty = ToolCallAccuracy().evaluate(case, "done")
    assert observed_empty.passed and not observed_empty.metadata.get("skipped")


def test_recorded_grader_and_repeat_configuration_must_match():
    from multivon_eval import compare_reports
    cases = [EvalCase("x", "yes")]
    base = EvalSuite("one").add_cases(cases).add_evaluator(ExactMatch()).run(lambda _: "yes", verbose=False)
    repeats = EvalSuite("two").add_cases(cases).add_evaluator(ExactMatch()).run(lambda _: "yes", runs=2, verbose=False)
    assert compare_reports(base, repeats).mcnemar_p is None
    changed = EvalSuite("three").add_cases(cases).add_evaluator(ExactMatch(threshold=0.7)).run(lambda _: "yes", verbose=False)
    result = compare_reports(base, changed)
    assert result.mcnemar_p is None
    assert any("evaluator configuration" in issue for issue in result.identity_issues)
