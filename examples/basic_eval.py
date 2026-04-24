"""
Basic eval — deterministic evaluators, no LLM judge needed.
"""
from dotenv import load_dotenv
load_dotenv()

import anthropic
from llm_evals import EvalSuite, EvalCase, ExactMatch, Contains, NotEmpty, WordCount

client = anthropic.Anthropic()

def my_model(prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


cases = [
    EvalCase(
        input="What is 2 + 2?",
        expected_output="4",
        tags=["math"],
    ),
    EvalCase(
        input="Name three primary colors.",
        tags=["knowledge"],
    ),
    EvalCase(
        input="Write a one-sentence description of photosynthesis.",
        tags=["science"],
    ),
]

suite = EvalSuite("Basic Deterministic Eval", model_id="claude-haiku")
suite.add_cases(cases)
suite.add_evaluators(
    NotEmpty(),
    WordCount(min_words=1, max_words=200),
    Contains(substrings=["4"], threshold=1.0),   # only applies to first case conceptually
)

report = suite.run(my_model)
report.save_json("eval_results.json")
print("\nSaved to eval_results.json")
