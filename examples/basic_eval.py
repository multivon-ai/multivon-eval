"""
Basic eval — deterministic + lexical similarity evaluators.

No LLM judge call required (a real Anthropic API call still happens
for the model under test). Run as a sanity check on any model.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

import anthropic
from multivon_eval import (
    EvalSuite, EvalCase,
    ExactMatch, NotEmpty, WordCount, Levenshtein,
)


_MODEL_ID = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
client = anthropic.Anthropic()


def my_model(prompt: str) -> str:
    response = client.messages.create(
        model=_MODEL_ID,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def build_suite() -> EvalSuite:
    cases = [
        EvalCase(
            input="What is 2 + 2? Reply with just the number.",
            expected_output="4",
            tags=["math"],
        ),
        EvalCase(
            input="Name three primary colors. Reply with a comma-separated list.",
            tags=["knowledge"],
        ),
        EvalCase(
            input="Write a one-sentence description of photosynthesis.",
            tags=["science"],
        ),
        EvalCase(
            input="What is the capital of France? One word only.",
            expected_output="Paris",
            tags=["geography"],
        ),
    ]

    suite = EvalSuite("Basic Deterministic Eval", model_id=_MODEL_ID)
    suite.add_cases(cases)
    # ExactMatch and Levenshtein both no-op (return score 0.0 with a clear
    # reason) when expected_output is absent — no per-case branching needed.
    suite.add_evaluators(
        NotEmpty(),
        WordCount(min=1, max=100),
        ExactMatch(),
        Levenshtein(threshold=0.8),
    )
    return suite


def main() -> int:
    suite = build_suite()
    report = suite.run(my_model)
    report.save_json("eval_results.json")
    print("\nSaved to eval_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
