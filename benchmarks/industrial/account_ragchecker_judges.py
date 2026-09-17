"""Reconcile a completed judge run using existing provider accounting and pricing.

For --price, install LiteLLM and set LITELLM_LOCAL_MODEL_COST_MAP=True beforehand.
No model requests are made. The catalog is an estimate, not a provider invoice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from reproduce_ragchecker_meta import write_json

from multivon_eval import ProviderJournal
from multivon_eval.provider_accounting import account_provider_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--price", action="store_true")
    args = parser.parse_args()
    if json.loads((args.run / "execution.json").read_text())["jobs_completed"] != 1120:
        raise ValueError("Run incomplete")
    pricer = None
    if args.price:
        import litellm

        from multivon_eval.integrations.litellm_pricing import LiteLLMPricer
        litellm.telemetry = False
        pricer = LiteLLMPricer(litellm)
    with ProviderJournal(args.run / "events.sqlite") as journal:
        events = journal.events()
    summary = {}
    for method in ("answer_accuracy", "direct_rating"):
        selected = [e for e in events if e["labels"]["method"] == method]
        account = account_provider_events(
            selected, price_estimator=pricer,
            coverage_declaration="Frozen runner invokes only the instrumented native Anthropic judge for this method",
        ).to_dict()
        responses = [e["response"] for e in selected if e["kind"] == "http_response"]
        provenance = next((r["pricing"] for r in account["evidence"]["requests"] if r["pricing"]), None)
        summary[method] = {
            **{k: v for k, v in account.items() if k != "evidence"},
            "http_status_counts": dict(Counter(r["status_code"] for r in responses)),
            "response_models": dict(Counter((r["body"].get("value") or {}).get("model", "unknown")
                                            for r in responses if r.get("body"))),
            "requests_missing_price": sum(r["cost_usd"] is None for r in account["evidence"]["requests"]),
            "pricing_errors": dict(Counter(r.get("pricing_error") for r in account["evidence"]["requests"]
                                           if r.get("pricing_error"))),
            "pricing_provenance": provenance,
            "requests_digest": hashlib.sha256(json.dumps(account["evidence"]["requests"], sort_keys=True).encode()).hexdigest(),
        }
    write_json(args.out, {"methods": summary,
                         "journal_sha256": hashlib.sha256((args.run / "events.sqlite").read_bytes()).hexdigest(),
                         "limits": "Recorded API usage; any dollar values are catalog estimates, not invoices"})
    print(args.out)


if __name__ == "__main__":
    main()
