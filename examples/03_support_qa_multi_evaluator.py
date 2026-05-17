"""Customer support QA — Faithfulness + Relevance + a plain-English CheckEvaluator.

Purpose:       10 hand-crafted support tickets (input + retrieved context + bot output).
               Some bot outputs are clean, some are ungrounded, some are vague deferrals.
               Three evaluators run against each. The plain-English CheckEvaluator turns
               an English criterion into yes/no questions automatically.
Runtime:       ~60-90s. Cost: <$0.15 (Anthropic claude-haiku-4-5 judge).
Output shape:  Per-case table, evaluator summary, JSON dump. Exits 1 if pass rate < 70%.
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
from multivon_eval.evaluators.llm_judge import CheckEvaluator


KB = {
    "shipping": (
        "Standard shipping is 5-7 business days. Express shipping is 2-3 business days. "
        "Overnight shipping is available for $24.99. We ship to the US, Canada, and the EU. "
        "Orders placed before 2pm EST ship the same day."
    ),
    "returns": (
        "Items can be returned within 30 days of delivery for a full refund. "
        "Items must be unused and in original packaging. Sale items are final — "
        "no returns or exchanges. Refunds post to the original payment method within "
        "5-7 business days after we receive the return."
    ),
    "account": (
        "To reset your password, go to /forgot-password and enter the email on file. "
        "We'll send a reset link valid for 30 minutes. To change the email on your "
        "account, sign in and visit Account Settings → Email. We do not delete accounts "
        "automatically — contact support to close one."
    ),
    "billing": (
        "We accept Visa, Mastercard, American Express, and PayPal. Subscriptions auto-renew "
        "monthly on the date of original purchase. To cancel, go to Account Settings → "
        "Subscription → Cancel. You retain access until the end of the current billing period. "
        "Refunds for partial periods are not issued."
    ),
    "outage": (
        "If the app shows a white screen on launch, force-quit and reopen. If that fails, "
        "uninstall and reinstall — your data is stored on our servers and will sync back. "
        "Status page: status.example.com. For ongoing issues, attach a screenshot when "
        "contacting support so we can debug."
    ),
}


# 10 support tickets. Each has the retrieved KB context and the bot's actual reply.
# Mix: 6 good answers, 4 problematic (ungrounded, vague, or deferring).
TICKETS = [
    {
        "id": "T1",
        "input": "How long does standard shipping take?",
        "context": KB["shipping"],
        "answer": "Standard shipping takes 5-7 business days. If you need it faster, express is 2-3 business days.",
    },
    {
        "id": "T2",
        "input": "Can I return a sale item?",
        "context": KB["returns"],
        "answer": "Unfortunately no — sale items are final and cannot be returned or exchanged.",
    },
    {
        "id": "T3",
        "input": "I forgot my password — how do I reset it?",
        "context": KB["account"],
        "answer": "Visit /forgot-password and enter the email on file. We'll send a reset link that's valid for 30 minutes.",
    },
    {
        "id": "T4",
        "input": "Do you accept Apple Pay?",
        "context": KB["billing"],
        # Ungrounded: KB lists Visa/MC/Amex/PayPal — does NOT mention Apple Pay.
        "answer": "Yes, we accept Apple Pay along with Visa, Mastercard, Amex, and PayPal — just select it at checkout.",
    },
    {
        "id": "T5",
        "input": "How do I cancel my subscription?",
        "context": KB["billing"],
        "answer": "Go to Account Settings → Subscription → Cancel. You'll keep access until the end of the current billing period.",
    },
    {
        "id": "T6",
        "input": "The app is showing a white screen and won't load — what should I do?",
        "context": KB["outage"],
        # Vague deferral — doesn't name the problem or give the documented next step.
        "answer": "We're sorry you're having trouble! Our team will look into this and get back to you.",
    },
    {
        "id": "T7",
        "input": "How long do refunds take after I send the item back?",
        "context": KB["returns"],
        "answer": "Refunds post to your original payment method within 5-7 business days after we receive the returned item.",
    },
    {
        "id": "T8",
        "input": "I want to delete my account.",
        "context": KB["account"],
        "answer": "We don't delete accounts automatically — please contact our support team and we'll close it for you.",
    },
    {
        "id": "T9",
        "input": "When will my order arrive if I pick overnight shipping?",
        "context": KB["shipping"],
        # Ungrounded: KB doesn't promise a specific delivery time, just that overnight is $24.99.
        "answer": "With overnight shipping ($24.99) your order will arrive by 10am the next business day, guaranteed.",
    },
    {
        "id": "T10",
        "input": "Refund for order #4839 — it never arrived.",
        "context": KB["shipping"] + "\n\n" + KB["returns"],
        # Vague deferral again — doesn't name the specific problem or next step.
        "answer": "Please contact our support team for assistance.",
    },
]


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Export it before running.")
        return 2

    configure(JudgeConfig(provider="anthropic", model="claude-haiku-4-5"))

    cases = [
        EvalCase(input=t["input"], context=t["context"], metadata={"ticket_id": t["id"], "answer": t["answer"]})
        for t in TICKETS
    ]

    suite = EvalSuite("Customer Support QA", model_id="precomputed-answers")
    suite.add_cases(cases)
    suite.add_evaluators(
        Faithfulness(),
        Relevance(),
        CheckEvaluator(
            criterion="Response should name the customer's specific problem and provide a concrete next step.",
            # Pin questions so this is reproducible in CI without re-generating.
            questions=[
                "Does the response name or restate the customer's specific problem?",
                "Does the response provide a concrete next step or action the customer can take?",
                "Does the response avoid vague deferrals like 'we will look into this'?",
            ],
            name="actionability",
        ),
    )

    def model_fn(question: str) -> str:
        for t in TICKETS:
            if t["input"] == question:
                return t["answer"]
        return ""

    report = suite.run(model_fn)

    here = Path(__file__).parent
    out = here / "03_support_qa_multi_evaluator_output.json"
    report.save_json(str(out))
    print(f"\nSaved full results -> {out.name}")

    # Per-evaluator summary.
    print("\n=== Per-case breakdown ===")
    for cr in report.case_results:
        tid = next((t["id"] for t in TICKETS if t["input"] == cr.case_input), "?")
        per_eval = {er.evaluator: er for er in cr.results}
        f_ = per_eval.get("faithfulness")
        r_ = per_eval.get("relevance")
        a_ = per_eval.get("actionability")
        verdict = "PASS" if cr.passed else "FAIL"
        print(
            f"  [{verdict}] {tid:>3}  "
            f"faith={f_.score:.2f}{'✓' if f_.passed else '✗'}  "
            f"rel={r_.score:.2f}{'✓' if r_.passed else '✗'}  "
            f"act={a_.score:.2f}{'✓' if a_.passed else '✗'}"
        )

    print(f"\nOverall pass rate: {report.pass_rate:.0%}  ({report.passed}/{report.total})")
    if report.pass_rate < 0.7:
        print("Result: FAIL — pass rate below 70% gate.")
        return 1
    print("Result: PASS — pass rate meets gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
