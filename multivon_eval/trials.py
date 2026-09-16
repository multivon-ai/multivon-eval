"""Immutable per-execution evidence; aggregates never replace raw trials."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .case import EvalCase
from .case_manifest import (
    canonical_json,
    case_from_dict,
    case_identity,
    case_to_dict,
    digest,
    trace_from_data,
    trace_to_data,
)

if TYPE_CHECKING:
    from .result import CaseResult, EvalReport
    from .suite import EvalSuite

TRIAL_SCHEMA = "multivon.trial/v1"


@dataclass(frozen=True)
class TrialRecord:
    """A detached, content-addressed trial. Accessors return fresh copies.

    A digest detects accidental changes, not a trusted signature. Provider
    request evidence is separately instrumented; its absence is explicit.
    """
    _json: str

    def __post_init__(self) -> None:
        data = json.loads(self._json)
        claimed = data.pop("digest", None)
        if data.get("schema") != TRIAL_SCHEMA or digest(data) != claimed:
            raise ValueError("Invalid trial schema or digest")
        if not isinstance(data.get("attempt"), int) or data["attempt"] < 1:
            raise ValueError("Trial attempt must be positive")
        if not isinstance(data.get("run_index"), int) or data["run_index"] < 1:
            raise ValueError("Trial run_index must be positive")
        case = case_from_dict(data["case"])
        if case_identity(case) != (data["case_id"], data["case_digest"]):
            raise ValueError("Trial case identity does not match snapshot")

    @classmethod
    def from_dict(cls, data: dict) -> TrialRecord:
        return cls(canonical_json(data))

    @property
    def data(self) -> dict:
        return json.loads(self._json)

    @property
    def digest(self) -> str:
        return self.data["digest"]

    def with_position(self, *, attempt: int, run_index: int) -> TrialRecord:
        data = self.data
        data.pop("digest")
        data.update(attempt=attempt, run_index=run_index)
        return TrialRecord.from_dict({**data, "digest": digest(data)})


@dataclass(frozen=True)
class CaseSnapshot:
    case_id: str | None
    case_digest: str | None
    payload: str | None
    error: str | None = None


def capture_case(case: EvalCase) -> CaseSnapshot:
    try:
        case_id, case_digest = case_identity(case)
        return CaseSnapshot(case_id, case_digest,
                            canonical_json(case_to_dict(case, include_reference=False)))
    except (TypeError, ValueError, AttributeError) as exc:
        # Existing custom metadata can be nonportable. Keep the evaluation,
        # but explicitly block treating it as reproducible evidence.
        return CaseSnapshot(None, None, None, f"{type(exc).__name__}: {exc}")


def attach_trial(result: CaseResult, snapshot: CaseSnapshot, *,
                 origin: str = "execution", evaluation_snapshot: CaseSnapshot | None = None,
                 latency_known: bool = True) -> CaseResult:
    result.case_id = snapshot.case_id
    result.case_digest = snapshot.case_digest
    result.evidence_error = snapshot.error
    if snapshot.payload is None:
        return result
    result.case_input = json.loads(snapshot.payload)["input"]
    result.tags = json.loads(snapshot.payload)["tags"]
    evaluation_snapshot = evaluation_snapshot or snapshot
    if evaluation_snapshot.error:
        result.evidence_error = evaluation_snapshot.error
        return result
    try:
        data = {
            "schema": TRIAL_SCHEMA, "case_id": snapshot.case_id,
            "case_digest": snapshot.case_digest, "case": json.loads(snapshot.payload),
            "evaluation_case": json.loads(evaluation_snapshot.payload),
            "attempt": 1, "run_index": 1, "origin": origin,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "output": result.actual_output, "status": result.status.value,
            "model_error": result.model_error, "judge_error": result.judge_error,
            "evaluator_error": result.evaluator_error,
            "latency_ms": result.latency_ms if latency_known else None,
            "agent_trace": trace_to_data(result.agent_trace),
            "evaluators": [{"name": r.evaluator, "score": r.score, "passed": r.passed,
                            "reason": r.reason, "metadata": r.metadata} for r in result.results],
            "provider_requests": None,
            "evidence_gaps": ["Provider requests and per-trial usage are not captured"],
        }
        result.trials = (TrialRecord.from_dict({**data, "digest": digest(data)}),)
    except (TypeError, ValueError, AttributeError) as exc:
        result.evidence_error = f"{type(exc).__name__}: {exc}"
    return result


def trial_integrity_issues(result: CaseResult) -> list[str]:
    """Detect detached headers or missing/duplicate recorded execution slots."""
    if not result.trials:
        return []
    trials = [trial.data for trial in result.trials]
    issues = []
    if any((t["case_id"], t["case_digest"]) != (result.case_id, result.case_digest)
           or t["case"]["input"] != result.case_input
           or t["case"]["tags"] != result.tags for t in trials):
        issues.append("Case headers do not match retained trial identities")
    if result.actual_output != trials[-1]["output"]:
        issues.append("Case output does not match its final retained trial")
    if all(t["origin"] == "regrade" for t in trials):
        return issues
    if all(t["origin"] == "inspect" for t in trials):
        slots: dict[int, list[int]] = {}
        for trial in trials:
            slots.setdefault(trial["run_index"], []).append(trial["attempt"])
        if sorted(slots) != list(range(1, result.runs + 1)):
            issues.append("Retained Inspect epoch slots do not match runs")
        if any(sorted(attempts) != list(range(1, len(attempts) + 1)) for attempts in slots.values()):
            issues.append("Retained Inspect epoch attempts are missing or duplicated")
        if max(t["attempt"] for t in trials) != result.retry_attempts + 1:
            issues.append("Retained Inspect attempts do not match retry history")
        return issues
    attempts: dict[int, list[int]] = {}
    for trial in trials:
        attempts.setdefault(trial["attempt"], []).append(trial["run_index"])
    if sorted(attempts) != list(range(1, result.retry_attempts + 2)):
        issues.append("Retained trial attempts do not match retry history")
    if any(sorted(indices) != list(range(1, len(indices) + 1)) for indices in attempts.values()):
        issues.append("Retained trial positions are missing or duplicated")
    if len(attempts[max(attempts)]) != result.runs:
        issues.append("Retained final-attempt trial count does not match runs")
    return issues


def regrade(report: EvalReport, suite: EvalSuite) -> EvalReport:
    """Grade every saved trial, including failed retry attempts, without a target.

    Model-error trials retain their infrastructure status and are not graded as
    text answers. Returns one case result per original trial, with parent digest
    provenance. This function may call judges configured on the supplied suite.
    """
    from .result import CaseResult, EvalReport
    if any(not result.trials or result.evidence_error for result in report.case_results):
        raise ValueError("Regrading requires complete saved trials for every case")
    results = []
    for result in report.case_results:
        for trial in result.trials:
            data = trial.data
            snapshot = capture_case(case_from_dict(data["case"]))
            case = case_from_dict(data["evaluation_case"])
            evaluation_snapshot = capture_case(case)
            case.agent_trace = trace_from_data(data["agent_trace"])
            if data["model_error"] is not None:
                graded = CaseResult(case.input, data["output"], [],
                                    model_error=data["model_error"], agent_trace=case.agent_trace)
            else:
                graded = suite.run_on_cases([(case, data["output"])], verbose=False,
                                            latencies_ms=[data["latency_ms"]]).case_results[0]
            attach_trial(graded, snapshot, origin="regrade", evaluation_snapshot=evaluation_snapshot,
                         latency_known=data["latency_ms"] is not None)
            if graded.trials:
                child = graded.trials[0].data
                child.pop("digest")
                child.update(parent_trial=trial.digest, attempt=data["attempt"],
                             run_index=data["run_index"])
                graded.trials = (TrialRecord.from_dict({**child, "digest": digest(child)}),)
            results.append(graded)
    return EvalReport(suite.name, results, model_id=report.model_id, purpose=report.purpose)
