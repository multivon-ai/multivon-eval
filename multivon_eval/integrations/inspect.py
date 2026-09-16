"""Use Multivon graders and acceptance policies with the Inspect runtime.

Inspect owns model execution, logs, provider settings, sandboxes and resume.
The bridge keeps its native log as the authoritative execution artifact.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from ..case import AgentStep, EvalCase, ToolCall
from ..case_manifest import CaseManifest, case_from_dict, case_to_dict, digest
from ..evaluators.base import Evaluator
from ..evaluators.deterministic import Latency, MaxLatency
from ..result import CaseResult, EvalReport, EvalResult
from ..trials import TrialRecord, attach_trial, capture_case

_NAMESPACE = "multivon_case_v1"
_RESULT = "multivon_result_v1"
_EVALUATION_CASE = "multivon_evaluation_case_v1"


def _require_inspect() -> None:
    try:
        import inspect_ai  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install multivon-eval[inspect] for the Inspect integration") from exc


def to_inspect_dataset(manifest: CaseManifest, *, split: str | None = None):
    """Create native Inspect samples with stable IDs and full case metadata.

    Text context becomes a system message. Conversation messages precede the
    current input. Multimodal attachments and agent tool events need explicit
    task/solver configuration; metadata is preserved without guessing wiring.
    """
    _require_inspect()
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
    samples = []
    for case in manifest.split(split) if split else manifest.cases:
        case_id, case_digest = case.identity()
        messages = []
        if case.context is not None:
            messages.append(ChatMessageSystem(content="Use the provided context to answer.\n\n" + case.context_str()))
        roles = {"user": ChatMessageUser, "assistant": ChatMessageAssistant, "system": ChatMessageSystem}
        for message in case.conversation or []:
            if message["role"] not in roles:
                raise ValueError("Tool conversation messages require an explicit Inspect task mapper")
            messages.append(roles[message["role"]](content=message["content"]))
        messages.append(ChatMessageUser(content=case.input))
        samples.append(Sample(id=case_id, input=messages, target=case.expected_output or "",
                              metadata={_NAMESPACE: {"case": case_to_dict(case),
                                        "case_id": case_id, "case_digest": case_digest,
                                        "manifest_digest": manifest.digest}}))
    return MemoryDataset(samples, name=manifest.manifest["name"])


def _case_from_metadata(metadata: dict, sample_id: str | int) -> EvalCase:
    envelope = metadata.get(_NAMESPACE)
    if not isinstance(envelope, dict):
        raise TypeError("Inspect sample lacks Multivon case metadata; use to_inspect_dataset")
    case = case_from_dict(envelope["case"])
    if case.identity() != (envelope["case_id"], envelope["case_digest"]):
        raise ValueError("Inspect sample case digest mismatch")
    if str(sample_id) != envelope["case_id"]:
        raise ValueError("Inspect sample ID differs from case identity")
    return case


def _execution_case(case: EvalCase, messages: list) -> EvalCase:
    responses = {}
    for message in messages:
        if message.role == "tool" and message.tool_call_id:
            if message.tool_call_id in responses:
                raise ValueError("Duplicate Inspect tool response ID")
            responses[message.tool_call_id] = (
                {"error": asdict(message.error), "text": message.text}
                if message.error else message.text)
    trace, conversation = [], []
    for message in messages:
        if message.role in {"user", "assistant"}:
            conversation.append({"role": message.role, "content": message.text})
        if message.role == "assistant":
            trace.append(AgentStep(output=message.text, tool_calls=[
                ToolCall(call.function, call.arguments, responses.get(call.id))
                for call in message.tool_calls or []]))
    return replace(case, agent_trace=trace or None, conversation=conversation or None)


def as_inspect_scorer(evaluator: Evaluator, *, name: str | None = None):
    """Wrap a Multivon evaluator as a native Inspect scorer.

    Exceptions remain Inspect scoring errors; skipped checks have null values.
    No average metric is registered because missing values require coverage
    policy. Recreate this scorer in a registered Inspect @task for eval-retry.
    """
    _require_inspect()
    from inspect_ai.scorer import Score, scorer
    @scorer(metrics=[], name=name or f"multivon_{evaluator.name}")
    def factory():
        if hasattr(evaluator, "prepare"):
            evaluator.prepare()
        async def score(state, target):
            case = _execution_case(_case_from_metadata(state.metadata, state.sample_id), state.messages)
            if isinstance(evaluator, (Latency, MaxLatency)):
                result = EvalResult(evaluator.name, 0.0, False,
                                    "Inspect task duration is not target request latency", {"skipped": True})
            else:
                result = await evaluator.aevaluate(case, state.output.completion)
            payload = {"evaluator": result.evaluator, "score": result.score,
                       "passed": result.passed, "reason": result.reason, "metadata": result.metadata}
            missing = result.metadata.get("skipped") or result.metadata.get("error_kind")
            return Score(value={"score": None if missing else result.score,
                                "passed": None if missing else result.passed},
                         answer=state.output.completion, explanation=result.reason,
                         metadata={_RESULT: payload, _EVALUATION_CASE: case_to_dict(case)})
        return score
    return factory()


def from_inspect_log(log: Any, *, previous_logs: list[Any] | None = None) -> EvalReport:
    """Import a log, optionally including earlier retry logs oldest first.

    Native retry logs can preserve completed samples while omitting failed
    attempts. Supply those earlier logs to retain the complete provided history.
    The bridge cannot infer that an omitted log exists. Keep native logs together.
    """
    current = _from_inspect_log(log)
    if not previous_logs:
        return current
    if any(previous.status == "started" for previous in previous_logs):
        raise ValueError("Recover started Inspect logs before importing retry history")
    if any(previous.eval.task_id != log.eval.task_id or previous.eval.model != log.eval.model
           for previous in previous_logs):
        raise ValueError("Inspect retry history must use the same task ID and model")
    from .inspect_history import merge_history
    return merge_history(current, [_from_inspect_log(previous) for previous in previous_logs])


def _from_inspect_log(log: Any) -> EvalReport:
    """Import a native EvalLog produced with this bridge, grouping epochs.

    Keeps an upstream sample content digest, log location, and model usage on
    each imported trial. This does not flatten arbitrary tool/model events into
    a lossy AgentStep transcript. Inspect's native log retains those events.
    Native scorers need an explicit mapping and are rejected by this adapter.
    """
    _require_inspect()
    from ..suite import _aggregate_runs
    groups: dict[str, list[tuple[int, EvalCase, CaseResult]]] = {}
    evidence_issues = []
    if log.status != "success":
        evidence_issues.append(f"Inspect evaluation status is {log.status}; expected samples may be missing")
    for sample in log.samples or []:
        case = _case_from_metadata(sample.metadata, sample.id)
        results = []
        evaluation_case = _execution_case(case, sample.messages)
        for score_name, score in (sample.scores or {}).items():
            payload = (score.metadata or {}).get(_RESULT)
            if payload is None:
                raise ValueError(f"Inspect scorer {score_name!r} has no Multivon result mapping")
            missing = payload.get("metadata", {}).get("skipped") or payload.get("metadata", {}).get("error_kind")
            expected_value = {"score": None if missing else payload["score"],
                              "passed": None if missing else payload["passed"]}
            if score.value != expected_value:
                raise ValueError("Inspect score was edited without updating its Multivon result mapping")
            results.append(EvalResult(**payload))
            if (_EVALUATION_CASE in score.metadata
                    and digest(score.metadata[_EVALUATION_CASE]) != digest(case_to_dict(evaluation_case))):
                raise ValueError("Inspect scorer case differs from recorded messages")
        error = sample.error.message if sample.error else None
        cr = CaseResult(case.input, sample.output.completion, results, tags=case.tags,
                        evaluator_error=f"Inspect sample error: {error}" if error else None,
                        agent_trace=evaluation_case.agent_trace)
        attach_trial(cr, capture_case(case), origin="inspect", latency_known=False,
                     evaluation_snapshot=capture_case(evaluation_case))
        if cr.trials:
            data = cr.trials[0].data
            data.pop("digest")
            data["upstream"] = {
                "format": "inspect", "schema_version": log.version,
                "log_location": getattr(log, "location", None), "sample_id": sample.id,
                "epoch": sample.epoch, "sample_uuid": sample.uuid,
                "sample_digest": digest(sample.model_dump(mode="json")),
                "model_usage": {k: v.model_dump(mode="json") for k, v in sample.model_usage.items()},
            }
            data["evidence_gaps"] = [
                "Execution events and provider requests remain in the referenced native Inspect log",
                "Multivon judge calls are not automatically instrumented as Inspect model calls",
                "Inspect sample errors are not classified as model versus grader failures by this bridge",
                "AgentStep is a text/tool projection; multimodal and timing detail remain upstream",
                "Only provided logs are imported; omitted retry logs cannot be inferred",
            ]
            cr.trials = (TrialRecord.from_dict({**data, "digest": digest(data)}),)
        groups.setdefault(case.identity()[0], []).append((sample.epoch, case, cr))
    case_results = []
    for rows in groups.values():
        rows.sort(key=lambda row: row[0])
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError("Duplicate Inspect sample ID/epoch")
        if len({row[1].identity()[1] for row in rows}) != 1:
            raise ValueError("Inspect sample definition changed between epochs")
        case_results.append(rows[0][2] if len(rows) == 1 else
                            _aggregate_runs(rows[0][1], [row[2] for row in rows]))
    return EvalReport(suite_name=log.eval.task, case_results=case_results,
                      model_id=str(log.eval.model), evidence_issues=evidence_issues)
