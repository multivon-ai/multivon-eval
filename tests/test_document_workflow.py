"""Outcome checks must reject a model that says success without correct state."""

import pytest

pytest.importorskip("inspect_ai")

from benchmarks.industrial.document_task import LedgerOutcome, initialize, post
from multivon_eval import EvalCase


def test_ledger_oracle_checks_real_state_and_idempotency(tmp_path):
    database = tmp_path / "ledger.sqlite"
    initialize(database)
    case = EvalCase("post", case_id="one", metadata={
        "expected_amount": "10.00", "expected_currency": "USD"})
    grader = LedgerOutcome(database)
    assert not grader.evaluate(case, "Successfully posted 10.00 USD").passed
    post(database, "one", "10.00", "UNSPECIFIED")
    assert not grader.evaluate(case, "posted").passed
    with pytest.raises(ValueError, match="Idempotency conflict"):
        post(database, "one", "10.00", "USD")
    second = EvalCase("post", case_id="two", metadata=case.metadata)
    post(database, "two", "10.00", "USD")
    post(database, "two", "10.00", "USD")
    result = grader.evaluate(second, "")
    assert result.passed
    assert result.metadata["observed_rows"] == [["10.00", "USD"]]


@pytest.mark.parametrize("amount,currency", [
    (True, "USD"), (10, "USD"), ("10.0", "USD"), ("1,000.00", "USD"),
    ("10.00", "EUR"), ("", "UNSPECIFIED"),
])
def test_tool_rejects_invalid_values_before_writing(tmp_path, amount, currency):
    database = tmp_path / "ledger.sqlite"
    initialize(database)
    with pytest.raises(ValueError):
        post(database, "one", amount, currency)


def test_inspect_tool_execution_is_checked_against_persisted_state(tmp_path):
    import hashlib

    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput
    from inspect_ai.tool import ToolCall

    from benchmarks.industrial.document_task import document_ledger
    from multivon_eval import CaseManifest
    from multivon_eval.integrations.inspect import from_inspect_log

    source = tmp_path / "receipt.txt"
    source.write_text("Total 10.00")
    case = EvalCase("Post the total", case_id="fixture", source_id="receipt", metadata={
        "expected_amount": "10.00", "expected_currency": "UNSPECIFIED",
        "asset": {"path": source.name, "media_type": "text/plain",
                  "bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}})
    CaseManifest("fixture", [case]).save(tmp_path / "manifest.json")
    task = document_ledger(str(tmp_path), str(tmp_path / "ledger.sqlite"))
    task.cost_limit = None
    output = ModelOutput(model="mockllm", choices=[ChatCompletionChoice(message=ChatMessageAssistant(
        content="Posting the total", tool_calls=[ToolCall(id="call-1", function="post_entry",
            arguments={"amount": "10.00", "currency": "UNSPECIFIED"})]))])
    log = inspect_eval(task, model="mockllm/model", model_args={"custom_outputs": [output]},
                       log_dir=str(tmp_path / "logs"), display="none", log_model_api=True)[0]
    assert log.status == "success", log.error
    report = from_inspect_log(log)
    assert report.passed == 1, report.to_json()
    assert report.case_results[0].results[-1].metadata["observed_rows"] == [["10.00", "UNSPECIFIED"]]
    assert len(report.case_results[0].agent_trace[0].tool_calls) == 1
