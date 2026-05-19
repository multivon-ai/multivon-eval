"""Validate the intelligent-eval prototype end-to-end with a real LLM.

Runs:
  1. generate_adversarial_cases (Claude Haiku) for "ungrounded_claim"
  2. validate_adversarial_cases with N_SHOTS=3 against TWO baselines:
       - WEAK: always confabulates (should fail most cases)
       - STRONG: always refuses ("I don't have that info") — should pass most
  3. Reports mean failure RATE per baseline + per-case shot detail so
     judge noise can be quantified and separated from real signal.

Goal: answer the "are these synthetic noise or real adversarial cases?"
question with data, not vibes. With single-shot validation the result is
1 bit per case and judge noise dominates. With N-shot aggregation the
failure_rate per case has real granularity (0/3, 1/3, 2/3, 3/3) and the
mean across cases gives a noise-corrected discrimination score between
baselines. If WEAK mean_failure_rate >> STRONG mean_failure_rate, the
filter is doing useful work above the noise floor.

Usage:  .venv/bin/python scripts/validate_auto_prototype.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env so ANTHROPIC_API_KEY is available
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ANTHROPIC_API_KEY not set; aborting.", file=sys.stderr)
    sys.exit(1)

from multivon_eval import JudgeConfig, configure  # noqa: E402
from multivon_eval.auto import (  # noqa: E402
    generate_adversarial_cases,
    validate_adversarial_cases,
)

# Cheap, fast judge
configure(JudgeConfig(provider="anthropic", model="claude-haiku-4-5"))

SEED = (
    "AcmeCorp ships products in 3-5 business days. Refunds are available within "
    "30 days of purchase. Returns require the original receipt. Customer support "
    "is available Mon-Fri 9am-5pm Pacific Time."
)


def weak_baseline(prompt: str) -> str:
    """Always confabulates — invents plausible-sounding info."""
    return (
        "Based on our records, the answer is yes. We offer that service with a "
        "standard 60-day window and free expedited shipping included."
    )


def strong_baseline(prompt: str) -> str:
    """Always refuses cleanly when info isn't available."""
    return "I don't have that information in the provided context."


N_SHOTS = 3
HARDNESS_BAND = (0.5, 1.0)  # keep cases the baseline fails ≥ half the shots


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _print_reports(reports: list, label: str) -> tuple[float, int]:
    """Print per-case shot detail; return (mean_failure_rate, kept_count)."""
    kept = sum(1 for r in reports if r.in_hardness_band)
    mean_fr = _mean([r.failure_rate for r in reports])
    print(f"  kept: {kept} / {len(reports)}  (band={HARDNESS_BAND}, n_shots={N_SHOTS})")
    print(f"  mean failure_rate across cases: {mean_fr:.2f}")
    for i, r in enumerate(reports, 1):
        marker = "✓" if r.in_hardness_band else "·"
        scores_str = " ".join(f"{s:.2f}" for s in r.scores)
        print(
            f"  {marker} case {i} [{r.evaluator_name}] "
            f"failure_rate={r.failure_rate:.2f}  shot_scores=[{scores_str}]"
        )
    return mean_fr, kept


def main() -> int:
    print("=" * 72)
    print(f"STEP 1 — generate_adversarial_cases(mode='ungrounded_claim', n=5)")
    print("=" * 72)
    cases = generate_adversarial_cases(SEED, "ungrounded_claim", n=5)
    print(f"Generated {len(cases)} cases.\n")

    for i, c in enumerate(cases, 1):
        print(f"--- case {i} ---")
        print(f"  input:    {c.input[:120]}")
        print(f"  context:  {(c.context or '')[:120]}")
        print(f"  expected: {(c.expected_output or '')[:120]}")
        print(f"  tags:     {c.tags}")
        print(f"  stress:   {c.metadata.get('stress_tests')}")
        print(f"  prompt_v: {c.metadata.get('prompt_version')}")
        print()

    if not cases:
        print("No cases generated — aborting validation step.", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"STEP 2 — validate against WEAK baseline (always-confabulate), n_shots={N_SHOTS}")
    print("=" * 72)
    _, reports_weak = validate_adversarial_cases(
        cases, weak_baseline,
        n_shots=N_SHOTS, hardness_band=HARDNESS_BAND,
    )
    weak_mean, weak_kept = _print_reports(reports_weak, "WEAK")
    print()

    print("=" * 72)
    print(f"STEP 3 — validate against STRONG baseline (always-refuse), n_shots={N_SHOTS}")
    print("=" * 72)
    _, reports_strong = validate_adversarial_cases(
        cases, strong_baseline,
        n_shots=N_SHOTS, hardness_band=HARDNESS_BAND,
    )
    strong_mean, strong_kept = _print_reports(reports_strong, "STRONG")
    print()

    print("=" * 72)
    print("STEP 4 — verdict")
    print("=" * 72)
    n = len(cases)
    print(f"  WEAK   keep-rate: {weak_kept}/{n} = {weak_kept/n:.0%}  "
          f"mean_failure_rate={weak_mean:.2f}")
    print(f"  STRONG keep-rate: {strong_kept}/{n} = {strong_kept/n:.0%}  "
          f"mean_failure_rate={strong_mean:.2f}")
    print(f"  signal (Δ mean failure_rate): {weak_mean - strong_mean:+.2f}")
    print()
    if weak_mean - strong_mean >= 0.3:
        print("  ✓ Strong signal: cases discriminate weak from strong baseline.")
    elif weak_mean > strong_mean:
        print("  ~ Weak signal: filter discriminates but separation < 0.3 — judge")
        print("    noise may be eating into it. Consider raising n_shots.")
    elif weak_mean == strong_mean:
        print("  ~ No separation: filter isn't distinguishing weak from strong.")
    else:
        print("  ✗ Inverted: strong baseline fails MORE than weak. Filter is broken.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
