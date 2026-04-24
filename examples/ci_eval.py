"""
CI/CD eval — exits with code 1 if pass rate drops below threshold.
Add to your GitHub Actions workflow to catch regressions before they ship.

Example GitHub Actions step:
  - name: Run LLM evals
    run: python examples/ci_eval.py
    env:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
"""
from dotenv import load_dotenv
load_dotenv()

import anthropic
from llm_evals import EvalSuite, load, NotEmpty, Relevance, WordCount

client = anthropic.Anthropic()

def model(prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# Load test cases from a dataset file
cases = load("examples/datasets/qa_sample.jsonl")

suite = EvalSuite("CI Regression Eval", model_id="claude-haiku")
suite.add_cases(cases)
suite.add_evaluators(
    NotEmpty(),
    WordCount(min_words=2, max_words=500),
    Relevance(threshold=0.6),
)

# fail_threshold=0.8 means: exit(1) if fewer than 80% of cases pass
report = suite.run(model, fail_threshold=0.8)
report.save_json("ci_eval_results.json")
