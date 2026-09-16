"""Repair the known tool-error projection crash, re-score, then use Inspect retry.

This narrowly scoped recovery preserves the original artifact and never repeats
inference for the completed response that triggered the grader bug. Interrupted
samples are replayed by Inspect under the task's idempotent local-write policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inspect_ai import eval_retry, score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelCost, set_model_cost

from benchmarks.industrial.document_task import LedgerOutcome, ValidPostingCall
from benchmarks.industrial.run_documents import MODELS, summarize
from multivon_eval import CaseManifest
from multivon_eval.integrations.inspect import as_inspect_scorer, from_inspect_log

ERROR = "AttributeError(\"'ToolCallError' object has no attribute 'model_dump'\")"


def repair(original_path: Path, database: Path, output: Path):
    original = read_eval_log(original_path)
    from_inspect_log(original).save_json(str(output / "original-report.json"))
    repaired = original.model_copy(deep=True)
    repaired_ids = []
    for sample in repaired.samples or []:
        if sample.error and sample.error.message == ERROR:
            events = [e for e in sample.events if e.event == "model"]
            if len(events) != 1 or not events[0].completed or events[0].error or not events[0].call.response:
                raise ValueError("Cannot establish that inference finished before scoring crashed")
            sample.metadata["multivon_scoring_repair"] = {
                "original_error": sample.error.model_dump(mode="json"),
                "original_log_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
                "kind": "offline regrade, no new model generation",
            }
            sample.error = None
            repaired_ids.append(str(sample.id))
    if repaired_ids != ["cord-test-14-oracle-text"]:
        raise ValueError(f"Recovery is scoped to the observed grader crash, got {repaired_ids}")
    # Public Inspect scoring API; the active grader model is a mock and both
    # graders are deterministic. Cancelled samples retain their original error.
    repaired = score(repaired, [as_inspect_scorer(ValidPostingCall()),
                               as_inspect_scorer(LedgerOutcome(database))],
                     model="mockllm/model", action="overwrite", display="none")
    target = output / "regraded.eval"
    write_eval_log(repaired, target)
    (output / "repair.json").write_text(json.dumps({
        "original_log": str(original_path), "original_log_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "regraded_log": str(target), "regraded_sample_ids": repaired_ids,
        "model_generations_for_regrading": 0,
        "note": "Original grading errors remain in original-report.json; do not overwrite the original log"}, indent=2))
    return read_eval_log(target)


def resume(original: Path, root: Path, database: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    regraded = repair(original, database, output)
    completed = {str(s.id): s.uuid for s in regraded.samples or [] if s.error is None}
    model = regraded.eval.model
    price_in, price_out = MODELS[model]
    set_model_cost(model, ModelCost(input=price_in, output=price_out,
                                   input_cache_write=price_in * 1.25, input_cache_read=price_in * 0.1))
    result = eval_retry(str(output / "regraded.eval"), log_dir=str(output / "logs"),
                        max_samples=2, max_connections=2, max_retries=0, timeout=90,
                        display="none", log_model_api=True, log_images=True, log_buffer=1)[0]
    preserved = {str(s.id): s.uuid for s in result.samples or [] if str(s.id) in completed}
    if preserved != completed:
        raise AssertionError("Completed generations were not preserved across native retry")
    summary = summarize(result, CaseManifest.load(root / "manifest.json"), output, model,
                        previous_logs=[regraded], evidence_note=(
                            "One original grader crash was corrected by offline regrading. "
                            "Original grading history is retained separately in original-report.json and repair.json."))
    summary["preserved_generation_count"] = len(completed)
    summary["original_usage"] = {k: v.model_dump(mode="json") for k, v in read_eval_log(original).stats.model_usage.items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("original", "root", "database", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    result = resume(args.original, args.root, args.database, args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "success" else 2)
