"""
EU AI Act high-risk eval — auditor-ready in three steps.

This example runs fully offline (no API keys) so you can see the compliance
workflow end-to-end:

  1. Build a high-risk suite via the factory.
  2. Print a coverage report — which Articles does the suite exercise?
  3. Record an audit run, verify the hash chain, demonstrate tamper detection.

To run a real eval, wire `suite.run(your_model_fn, runs=5)` to a model
function and pass that report to `reporter.record(...)` instead of the
synthetic report below.
"""
from __future__ import annotations

import json
from pathlib import Path

from multivon_eval import (
    CaseResult, ComplianceReporter, EvalReport, EvalResult, EvalSuite,
)


def main() -> int:
    # ── 1. Factory ──────────────────────────────────────────────────────────
    suite = EvalSuite.eu_ai_act_high_risk(jurisdiction="gdpr")

    # ── 2. Coverage analysis (no model needed) ──────────────────────────────
    reporter = ComplianceReporter(output_dir="./audit-logs", framework="eu-ai-act")
    print(reporter.coverage(suite))
    print()

    # In a real eval you would do::
    #
    #   suite.add_cases(load_jsonl("cases.jsonl"))
    #   report = suite.run(your_model_fn, runs=5, fail_threshold=0.85)
    #
    # For this offline demo we synthesize a tiny report.
    synthetic = EvalReport(
        suite_name=suite.name,
        model_id="demo-model-v1",
        case_results=[
            CaseResult(
                case_input="Summarize the refund policy.",
                actual_output="Refunds are processed within 14 days of receipt.",
                results=[
                    EvalResult(evaluator="faithfulness", score=0.92, passed=True),
                    EvalResult(evaluator="hallucination", score=0.96, passed=True),
                    EvalResult(evaluator="relevance", score=0.88, passed=True),
                    EvalResult(evaluator="toxicity", score=1.0, passed=True),
                    EvalResult(evaluator="bias", score=0.95, passed=True),
                    EvalResult(evaluator="pii_detection", score=1.0, passed=True),
                    EvalResult(evaluator="not_empty", score=1.0, passed=True),
                ],
            ),
        ],
    )

    # ── 3. Record, verify, tamper-detect ────────────────────────────────────
    print("Recording audit trail:")
    reporter.record(synthetic, tags={"system": "support-bot", "version": "1.0"})
    reporter.record(synthetic, tags={"system": "support-bot", "version": "1.1"})

    print("\nVerifying audit chain (intact):")
    reporter.verify(suite.name)

    # Demonstrate that editing the log is detected.
    log_path = Path("./audit-logs") / f"{suite.name.replace(' ', '_')}.audit.ndjson"
    lines = log_path.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["summary"]["pass_rate"] = 0.0
    lines[0] = json.dumps(tampered, separators=(",", ":"))
    log_path.write_text("\n".join(lines) + "\n")

    print("\nVerifying audit chain (after tampering with record 1):")
    try:
        reporter.verify(suite.name)
        print("  WARNING: verifier did not detect the tamper — bug!")
        return 1
    except Exception as exc:
        # Expected — the verifier raises ComplianceError on a broken chain.
        print(f"  ✓ Tamper detected: {type(exc).__name__}: {exc}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
