"""Migration fixtures and malformed-input behavior for downstream report readers."""
import copy
import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from multivon_eval import EvalReport
from multivon_eval.report_schema import report_schema, validate_report

FIXTURES = Path(__file__).parent / "fixtures" / "reports"


def published():
    return json.loads((FIXTURES / "published-0.18.0.json").read_text())


def test_schema_is_standard_detached_and_offline():
    schema = report_schema()
    Draft202012Validator.check_schema(schema)
    schema.clear()
    assert report_schema()["$schema"].endswith("2020-12/schema")


def test_published_report_migrates_without_losing_raw_trials():
    data = published()
    validate_report(data)
    report = EvalReport.from_dict(data)
    assert [r.status.value for r in report.case_results] == ["passed", "failed_quality", "skipped"]
    saved = json.loads(report.to_json())
    validate_report(saved)
    assert [c["trials"] for c in saved["cases"]] == [c["trials"] for c in data["cases"]]
    assert saved["suite_lock"] == data["suite_lock"]


@pytest.mark.parametrize("explicit", [False, True])
def test_v1_migrates_without_inventing_evidence(explicit):
    data = json.loads((FIXTURES / "legacy-v1.json").read_text())
    if explicit:
        data["schema"] = "multivon.report/v1"
    report = EvalReport.from_dict(data)
    assert [r.status.value for r in report.case_results] == ["passed", "model_error", "skipped"]
    assert all(not r.trials and r.case_id is None for r in report.case_results)
    validate_report(json.loads(report.to_json()))


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema="multivon.report/v999"),
    lambda d: d.pop("cases"),
    lambda d: d.update(cases={}),
    lambda d: d["cases"][0].update(runs=0),
    lambda d: d["cases"][0].update(runs=True),
    lambda d: d["cases"][0].update(output=None),
    lambda d: d["cases"][0].update(latency_ms=-1),
    lambda d: d["cases"][0].update(status="future_status"),
    lambda d: d["cases"][0].update(status="judge_error"),
    lambda d: d["cases"][0].update(passed=False),
    lambda d: d["cases"][0]["evaluators"][0].update(passed="false"),
    lambda d: d["cases"][0]["evaluators"][0].update(score=float("nan")),
    lambda d: d["cases"][0]["evaluators"][0].update(score=1.1),
])
def test_invalid_envelope_rejected_without_echoing_private_values(mutate):
    data = published()
    data["cases"][0]["input"] = "private report content"
    mutate(data)
    with pytest.raises(ValueError) as caught:
        EvalReport.from_dict(data)
    assert "private report content" not in str(caught.value)


def test_new_additive_fields_are_accepted_but_not_promised_lossless():
    data = published()
    data["future_annotation"] = {"value": 1}
    data["cases"][0]["future_annotation"] = True
    assert EvalReport.from_dict(data).passed == 1


def test_nested_trial_integrity_remains_required():
    data = copy.deepcopy(published())
    data["cases"][0]["trials"][0]["output"] = "tampered"
    validate_report(data)  # Envelope shape alone does not authenticate evidence.
    with pytest.raises(ValueError, match="digest"):
        EvalReport.from_dict(data)


def test_extension_guide_python_examples_execute():
    guide = Path(__file__).resolve().parents[1] / "docs/guides/extensions-and-compatibility.mdx"
    blocks = re.findall(r"```python\n(.*?)```", guide.read_text(), re.DOTALL)
    assert len(blocks) == 2
    namespace = {}
    for block in blocks:
        exec(compile(block, str(guide), "exec"), namespace)  # noqa: S102 - repository-owned examples
