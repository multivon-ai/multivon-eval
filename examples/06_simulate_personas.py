"""Persona simulation: adaptive multi-turn eval against a toy support bot.

Purpose:       Show `simulate()` driving a conversation a static script can't —
               the persona LLM adapts each turn to what the bot actually said,
               then the transcript is scored by the conversation evaluators
               plus a goal-completion judge.
Runtime:       ~30s. Cost: ~$0.01-0.05 (claude-haiku-4-5 drives the persona and
               judges the goal; hard ceiling $0.25 enforced by budget_usd).
Output shape:  Per-persona stop reason + goal verdict + conversation scores,
               and the standing disclaimer every simulate output carries:
               simulated personas measure behavior under synthetic users, not
               real traffic. Exits 1 if any persona's goal judge says "no".
Requires:      ANTHROPIC_API_KEY (or configure another judge below).
"""
from __future__ import annotations

import os
import sys

from multivon_eval import JudgeConfig, Persona, score_simulations, simulate


# ── The system under test: a deliberately imperfect support bot ─────────────
# It answers refund questions well but stonewalls cancellation questions —
# exactly the kind of asymmetry a goal-directed persona surfaces and a
# hand-written three-turn script usually misses.

def support_bot(rendered_conversation: str) -> str:
    # model_fn receives the FULL rendered conversation ("USER: ...\n
    # ASSISTANT: ..."). Route on the LAST user message — keyword-matching
    # the whole transcript means matching your own earlier replies. (The
    # simulator catches that bug loudly: flat-zero relevance scores.)
    text = rendered_conversation.rsplit("USER:", 1)[-1].lower()
    if "refund" in text:
        return ("You can return any item within 30 days for a full refund. "
                "Want me to start the process for your latest order?")
    if "cancel" in text:
        return "Thanks for reaching out! Is there anything else I can help with?"
    return "Hi! I can help with orders, refunds, and shipping."


PERSONAS = [
    Persona(
        name="rushed_customer",
        profile="A customer in a hurry who wants a refund for order #1234.",
        goal="Find out how to get a refund and confirm the refund window.",
        success_criteria="The assistant stated the 30-day refund window. "
                         "(Offering to start the process is a bonus, not required.)",
        traits=["terse", "impatient"],
    ),
    Persona(
        name="cancellation_seeker",
        profile="A polite customer who wants to cancel their subscription today.",
        goal="Get clear instructions for cancelling the subscription.",
        success_criteria="The assistant gave concrete cancellation steps "
                         "(not a deflection).",
        traits=["persistent", "polite"],
    ),
]


def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this example "
              "(persona + judge calls cost ~$0.01-0.05).", file=sys.stderr)
        return 2

    judge = JudgeConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    results = simulate(
        support_bot,
        PERSONAS,
        max_turns=5,
        judge=judge,
        budget_usd=0.25,  # hard ceiling — partial transcripts survive a cutoff
    )
    summary = score_simulations(results, judge=judge)

    failures = 0
    for r in results:
        verdict = {True: "goal reached", False: "GOAL NOT REACHED", None: "unjudged"}
        print(f"  {r.persona.name:22s} turns={r.turns} stop={r.stop_reason:18s} "
              f"{verdict[r.goal_achieved]}")
        if r.goal_achieved is False:
            failures += 1
            last = r.transcript[-1]["content"] if r.transcript else "(no transcript)"
            print(f"    last bot reply: {last[:90]!r}")

    gc = summary["goal_completion"]
    print(f"\n  goal completion: {gc['achieved']}/{gc['judged']} judged")
    print(f"  judge cost: ${summary['total_cost_usd']:.4f}")
    print(f"  {summary['disclaimer']}")
    # The disclaimer is part of the product, not this script: every simulate
    # output is labeled so synthetic traffic never blends into real metrics.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
