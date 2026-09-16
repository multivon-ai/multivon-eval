"""Combine native Inspect retry logs without replaying preserved samples."""
from __future__ import annotations

from dataclasses import replace

from ..result import EvalReport


def merge_history(current: EvalReport, earlier: list[EvalReport]) -> EvalReport:
    previous_by_id = {}
    for report in earlier:
        for row in report.case_results:
            previous_by_id.setdefault(row.case_id, []).append(row)
    final_ids = {row.case_id for row in current.case_results}
    if set(previous_by_id) - final_ids:
        current.evidence_issues.append("Some cases from earlier Inspect logs are absent from the current log")
    merged = []
    for result in current.case_results:
        candidates = previous_by_id.get(result.case_id, []) + [result]
        epochs = {}
        seen = {}
        for candidate in candidates:
            if candidate.evidence_error or not candidate.trials:
                raise ValueError("Inspect history contains incomplete trial evidence")
            if candidate.case_digest != result.case_digest:
                raise ValueError("Inspect case definition changed across retry logs")
            for trial in candidate.trials:
                data = trial.data
                source = data["upstream"]
                uuid = source["sample_uuid"]
                if not uuid:
                    raise ValueError("Inspect history requires sample UUIDs to identify preserved executions")
                if uuid in seen:
                    if seen[uuid] != source["sample_digest"]:
                        raise ValueError("Preserved Inspect sample changed across logs")
                    continue
                seen[uuid] = source["sample_digest"]
                epochs.setdefault(source["epoch"], []).append(trial)
        trials = []
        for run_index, epoch in enumerate(sorted(epochs), 1):
            trials.extend(trial.with_position(attempt=attempt, run_index=run_index)
                          for attempt, trial in enumerate(epochs[epoch], 1))
        # Keep the current log's aggregate quality verdict. Retain all older
        # executions separately so the acceptance policy can choose its scope.
        retry_count = max((len(records) - 1 for records in epochs.values()), default=0)
        retry_errors = [f"Inspect retry round {i + 1}; see retained per-epoch trials"
                        for i in range(retry_count)]
        merged.append(replace(result, trials=tuple(trials), retry_attempts=retry_count,
                              retry_errors=retry_errors))
    return replace(current, case_results=merged)
