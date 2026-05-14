"""
CI/CD eval — exits non-zero on quality regression or infrastructure error.

Add to your GitHub Actions workflow to catch regressions before they ship.
The JUnit XML output renders natively in the PR test panel.

GitHub Actions step::

    - name: Run LLM evals
      run: python examples/ci_eval.py
      env:
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    - name: Publish results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        path: ci_eval_results.*
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv
load_dotenv()

import anthropic
from multivon_eval import EvalSuite, load, NotEmpty, Relevance, WordCount


client = anthropic.Anthropic()


def model(prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main() -> int:
    # Load pinned test cases — a snapshot file gives reproducible CI runs.
    # Commit qa_sample.jsonl to your repo so the case set is stable across runs.
    cases = load("examples/datasets/qa_sample.jsonl")

    suite = EvalSuite("CI Regression Eval", model_id="claude-haiku")
    suite.add_cases(cases)
    suite.add_evaluators(
        NotEmpty(),
        WordCount(min=2, max=500),
        Relevance(threshold=0.6),
    )
    suite.add_check("Response should directly answer the question asked")
    suite.add_check("Response should not contain placeholder or error text")

    # fail_threshold=0.8 raises EvalGateFailure (subclass of SystemExit)
    # inside suite.run() when pass_rate < 80%. The rest of this function
    # runs only on the happy path.
    report = suite.run(model, fail_threshold=0.8)

    # Persist artifacts: JSON for `multivon-eval report/view`, JUnit XML
    # for GitHub Actions / GitLab PR rendering.
    report.save_json("ci_eval_results.json")
    report.save_junit_xml("ci_eval_results.junit.xml")

    # Distinct exit code (2) for infrastructure problems (judge outage,
    # model crash) so on-call can route differently from quality
    # regressions (which exit 1 via fail_threshold).
    if report.errors > 0:
        print(
            f"\nCI WARNING — {report.errors} case(s) errored "
            f"({report.errors_by_kind}). Quality result is for "
            f"{report.evaluated}/{report.total} cases.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
