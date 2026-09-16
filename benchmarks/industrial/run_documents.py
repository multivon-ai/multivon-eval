"""Run one frozen document treatment set; credentials come from the environment."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelCost

from multivon_eval import AcceptancePolicy, CaseManifest, CheckRequirement
from multivon_eval.integrations.inspect import from_inspect_log

MODELS = {
    "anthropic/claude-haiku-4-5-20251001": (1.0, 5.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
    "mockllm/model": (0.0, 0.0),
}


def run(root: Path, output: Path, model: str) -> dict:
    manifest = CaseManifest.load(root / "manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    price_in, price_out = MODELS[model]
    logs = inspect_eval(
        f"{Path(__file__).with_name('document_task.py').resolve()}@document_ledger",
        task_args={"root": str(root.resolve()), "database": str((output / "ledger.sqlite").resolve())},
        model=model, max_samples=2, max_connections=2, log_buffer=1, display="none",
        log_dir=str(output / "logs"), log_model_api=True, log_images=True,
        model_cost_config={model: ModelCost(input=price_in, output=price_out,
                                           input_cache_write=price_in * 1.25, input_cache_read=price_in * 0.1)},
    )
    log = logs[0]
    report = from_inspect_log(log)
    report.save_json(str(output / "report.json"))
    policy = AcceptancePolicy(checks=(CheckRequirement("ledger_outcome", critical=True),
                                     CheckRequirement("valid_posting_call")),
                              min_cases=len(manifest.cases),
                              min_source_groups=len({c.source_id for c in manifest.cases}))
    decision = policy.evaluate(report)
    (output / "decision.json").write_text(json.dumps(decision.to_dict(), indent=2))
    (output / "policy.json").write_text(json.dumps(policy.to_dict(), indent=2))
    usage = {name: value.model_dump(mode="json") for name, value in log.stats.model_usage.items()}
    slices = defaultdict(lambda: {"n": 0, "passed": 0, "errors": 0, "response_only_false_accepts": 0})
    for result in report.case_results:
        case = next(c for c in manifest.cases if c.case_id == result.case_id)
        row = slices[f"{case.metadata['family']}/{case.metadata['modality']}"]
        row["n"] += 1
        outcome = next((r for r in result.results if r.evaluator == "ledger_outcome"), None)
        row["passed"] += int(outcome is not None and outcome.passed)
        row["errors"] += int(bool(result.model_error or result.judge_error or result.evaluator_error))
        row["response_only_false_accepts"] += int(bool(result.actual_output.strip()) and outcome is not None and not outcome.passed)
    estimated_cost = sum((u.get("input_tokens", 0) * price_in + u.get("output_tokens", 0) * price_out
                         + (u.get("input_tokens_cache_read") or 0) * price_in * 0.1
                         + (u.get("input_tokens_cache_write") or 0) * price_in * 1.25) / 1_000_000
                         for u in usage.values())
    summary = {"status": log.status, "model": model, "manifest_digest": manifest.digest,
               "decision": decision.decision, "slices": dict(slices), "usage": usage,
               "estimated_usd": estimated_cost, "cost_is_invoice": False,
               "pricing_date": "2026-09-17", "versions": {name: version(name) for name in (
                   "inspect-ai", "multivon-eval", "anthropic")}, "log": log.location}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output, args.model), indent=2))
