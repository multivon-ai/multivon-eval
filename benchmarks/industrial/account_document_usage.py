"""Deduplicate preserved Inspect model events across retries and regrading."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from benchmarks.industrial.run_documents import MODELS


def account(paths: list[Path]) -> dict:
    seen, artifacts = {}, []
    for path in paths:
        log = read_eval_log(path)
        artifacts.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "status": log.status, "samples": len(log.samples or [])})
        for sample in log.samples or []:
            for event in sample.events:
                if event.event != "model":
                    continue
                usage = event.output.usage.model_dump(mode="json") if event.output and event.output.usage else None
                record = {"event_id": event.uuid, "model": event.model, "usage": usage,
                          "case_id": sample.id, "error": event.error}
                if event.uuid in seen and seen[event.uuid] != record:
                    raise ValueError("A preserved model event changed accounting data")
                seen[event.uuid] = record
    estimated = 0.0
    missing = []
    for event in seen.values():
        if event["usage"] is None:
            missing.append({k: v for k, v in event.items() if k != "usage"})
            continue
        usage = event["usage"]
        price_in, price_out = MODELS[event["model"]]
        estimated += (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out
                      + (usage.get("input_tokens_cache_read") or 0) * price_in * 0.1
                      + (usage.get("input_tokens_cache_write") or 0) * price_in * 1.25) / 1_000_000
    return {"model_events": len(seen), "events_with_usage": len(seen) - len(missing),
            "known_usage_estimate_usd": round(estimated, 9), "unknown_usage_events": missing,
            "billable_total_known": not missing,
            "note": "API list price estimate, not an invoice. Interrupted requests may be billable. Repeated native events counted once.",
            "artifacts": artifacts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(account(args.logs), indent=2))
