"""Contract analysis trap: footnote overrides the body, GPT-4o misses it.

Purpose:       Generate a real adversarial contract PDF where a 6pt footnote
               overrides the body clause, then ask gpt-4o (vision) for the
               liability cap and score with pdfhell's code-based scorer.
Runtime:       ~30s. Cost: <$0.30 (one gpt-4o vision call on a small PDF).
Output shape:  Prints the PDF question, the model's free-text answer, and
               pdfhell's correctness verdict. Exits 1 if the model fell for the trap.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

from openai import OpenAI
from pdfhell.generators.footnote_override import generate as generate_footnote_override
from pdfhell.scorer import score_case


# Seed 2 of the footnote_override generator deterministically produces the
# "liability_cap" variant: a Master Services Agreement that confidently caps
# liability at N months in the body, then a 6pt footnote carves out specific
# Sections as uncapped. We pin this seed so the case is reproducible and the
# expected answer is fixed across runs.
SEED = 2


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Export it before running.")
        return 2

    here = Path(__file__).parent

    # 1. Generate the adversarial contract PDF from code (procedural ground truth).
    pdf_bytes, case = generate_footnote_override(seed=SEED)
    pdf_path = here / "02_contract.pdf"
    pdf_path.write_bytes(pdf_bytes)

    print("=" * 78)
    print("Contract analysis trap — footnote_override (pdfhell)")
    print("=" * 78)
    print(f"PDF:              {pdf_path.name}  ({len(pdf_bytes):,} bytes)")
    print(f"Trap family:      {case.trap_family}")
    print(f"Question to model: {case.question}")
    print(f"Expected answer:  {case.expected_answer}")
    print(f"Expected tokens:  {case.expected_tokens}")
    print(f"Forbidden answer: {case.forbidden_answers[0]}")
    print()

    # 2. Ask GPT-4o (vision) to answer the question by looking at the PDF.
    # The OpenAI API accepts PDFs via the file input or as base64 data URLs;
    # the simplest portable route is base64 inline.
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    client = OpenAI()
    print("Calling gpt-4o ...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": "contract.pdf",
                            "file_data": f"data:application/pdf;base64,{pdf_b64}",
                        },
                    },
                    {"type": "text", "text": case.question},
                ],
            }
        ],
        max_tokens=400,
    )
    model_output = response.choices[0].message.content or ""
    print()
    print("=== Model output ===")
    print(model_output.strip())
    print()

    # 3. Score with pdfhell's code-based scorer (no LLM judge).
    score = score_case(case, model_output)
    print("=== pdfhell score ===")
    print(f"  correct:           {score.correct}")
    print(f"  matched_expected:  {score.matched_expected}")
    print(f"  fell_for_trap:     {score.fell_for_trap}")
    print(f"  matched_forbidden: {score.matched_forbidden}")
    print(f"  refused:           {score.refused}")
    if score.failure_mode:
        print(f"  failure_mode:      {score.failure_mode}")

    out = here / "02_contract_pdfhell_trap_output.json"
    payload = {
        "trap_family": case.trap_family,
        "seed": SEED,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "expected_tokens": case.expected_tokens,
        "forbidden_answers": case.forbidden_answers,
        "model": "gpt-4o",
        "model_output": model_output,
        "score": score.to_dict(),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved full results -> {out.name}")

    if score.fell_for_trap:
        print("\nResult: FAIL — model fell for the footnote_override trap (answered body-only).")
        return 1
    if not score.correct:
        print("\nResult: FAIL — model did not produce the expected answer.")
        return 1
    print("\nResult: PASS — model captured the footnote carve-out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
