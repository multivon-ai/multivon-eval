"""RAG faithfulness over a small insurance knowledge base.

Purpose:       Show how Faithfulness + Relevance catch one deliberately ungrounded answer.
Runtime:       ~30s. Cost: <$0.05 (Anthropic claude-haiku-4-5 judge).
Output shape:  Per-case PASS/FAIL table, aggregate scores, saved JSON with full
               judge reasons. Exits 1 if any case falls below the faithfulness threshold.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from multivon_eval import (
    EvalCase,
    EvalSuite,
    Faithfulness,
    JudgeConfig,
    Relevance,
    configure,
)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic insurance knowledge base (5 short documents)
# ──────────────────────────────────────────────────────────────────────────────

KB = {
    "policy_basics": (
        "Acme Auto policies cover collision, comprehensive, and liability. "
        "Standard collision deductible is $500. Comprehensive deductible is $250. "
        "Liability limits default to $100,000/$300,000 (per person / per accident). "
        "Roadside assistance is included on all policies at no extra cost."
    ),
    "claims_process": (
        "To file a claim, call 1-800-555-0123 within 72 hours of the incident. "
        "Photos of damage and a police report (for accidents involving another party) are required. "
        "Claims are typically processed within 10 business days. Rental car reimbursement "
        "is available up to $30/day for a maximum of 14 days while your vehicle is being repaired."
    ),
    "discounts": (
        "Acme offers a 12% multi-policy discount when you bundle auto and home insurance. "
        "Safe-driver discounts of up to 15% are available after 3 years of claim-free driving. "
        "Students with a GPA above 3.0 qualify for a 7% good-student discount. "
        "Anti-theft device installation reduces comprehensive premiums by 5%."
    ),
    "cancellation": (
        "You may cancel your policy at any time by submitting a written request. "
        "Refunds for unused premium are pro-rated and issued within 30 days. "
        "A short-rate cancellation fee of $25 applies if cancelled within the first 60 days. "
        "Cancellation does not retroactively void coverage for prior incidents."
    ),
    "exclusions": (
        "Standard auto policies exclude damage from racing, intentional acts, "
        "and use of the vehicle for commercial delivery (Uber, DoorDash, etc. require a rider). "
        "Wear and tear, mechanical breakdown, and freezing damage are not covered. "
        "Personal items stolen from inside the vehicle are not covered under auto — "
        "they fall under homeowner's or renter's insurance."
    ),
}

FULL_CONTEXT = "\n\n".join(f"[{k}]\n{v}" for k, v in KB.items())


# ──────────────────────────────────────────────────────────────────────────────
# 5 Q&A pairs. The "answers" dict is what a (real or simulated) RAG system
# might return. Case 5 is deliberately ungrounded — it cites a $750 rental
# reimbursement number that isn't in the KB. The eval should catch this.
# ──────────────────────────────────────────────────────────────────────────────

CASES = [
    {
        "question": "What is the standard deductible for collision coverage?",
        "answer": (
            "The standard collision deductible on an Acme Auto policy is $500. "
            "Comprehensive coverage has a separate $250 deductible."
        ),
    },
    {
        "question": "How do I file a claim and how long does it take?",
        "answer": (
            "To file a claim, call 1-800-555-0123 within 72 hours. You'll need photos "
            "and a police report if another party was involved. Most claims are processed "
            "within 10 business days."
        ),
    },
    {
        "question": "What discounts are available for bundling and safe driving?",
        "answer": (
            "Bundling auto with home gets you 12% off. Safe drivers with 3+ claim-free years "
            "save up to 15%. Students with a GPA above 3.0 get an additional 7% discount."
        ),
    },
    {
        "question": "Does my auto policy cover items stolen from inside my car?",
        "answer": (
            "No — personal items stolen from inside the vehicle are not covered under your "
            "auto policy. They are typically covered by homeowner's or renter's insurance."
        ),
    },
    # Deliberately ungrounded — should fail Faithfulness.
    {
        "question": "What is the rental car reimbursement limit on a standard policy?",
        "answer": (
            "Acme reimburses rental cars at up to $75 per day for a maximum of 30 days "
            "while your vehicle is in the shop. Premium policyholders receive unlimited "
            "rental reimbursement."
        ),
    },
]


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Export it before running.")
        return 2

    # Use Anthropic claude-haiku-4-5 as the judge (cheap + calibrated).
    configure(JudgeConfig(provider="anthropic", model="claude-haiku-4-5"))

    eval_cases = [
        EvalCase(input=c["question"], context=FULL_CONTEXT, metadata={"answer": c["answer"]})
        for c in CASES
    ]

    suite = EvalSuite("RAG Faithfulness — Insurance KB", model_id="precomputed-answers")
    suite.add_cases(eval_cases)
    suite.add_evaluators(Faithfulness(), Relevance())

    # Model fn returns the pre-computed answer attached to each case.
    def model_fn(question: str) -> str:
        for c in CASES:
            if c["question"] == question:
                return c["answer"]
        return ""

    report = suite.run(model_fn)

    here = Path(__file__).parent
    out = here / "01_rag_insurance_faithfulness_output.json"
    report.save_json(str(out))
    print(f"\nSaved full results -> {out.name}")

    # Surface which cases failed Faithfulness, with one-line judge reason.
    print("\n=== Faithfulness verdict per case ===")
    faithfulness_fails = 0
    for cr in report.case_results:
        f_results = [er for er in cr.results if er.evaluator == "faithfulness"]
        if not f_results:
            continue
        er = f_results[0]
        verdict = "PASS" if er.passed else "FAIL"
        if not er.passed:
            faithfulness_fails += 1
        snippet = cr.case_input[:60] + ("..." if len(cr.case_input) > 60 else "")
        print(f"  [{verdict}] faithfulness={er.score:.2f}  Q: {snippet}")

    if faithfulness_fails:
        print(f"\nResult: FAIL — {faithfulness_fails}/{len(report.case_results)} case(s) ungrounded.")
        return 1
    print("\nResult: PASS — every claim grounded in context.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
