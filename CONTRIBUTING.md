# Contributing to multivon-eval

Thanks for considering it. This document is the shortest path from clone to merged PR.

## Dev setup

```bash
git clone https://github.com/multivon-ai/multivon-eval
cd multivon-eval
python3 -m venv .venv && source .venv/bin/activate   # or use uv
pip install -e ".[dev]"                               # or: uv pip install -e ".[dev]"
```

Python 3.10 is the floor. If you use syntax added after 3.10 (e.g. nested
f-string quotes from PEP 701), it has to be gated or avoided — CI runs the
whole matrix from 3.10 up.

## Running tests

```bash
python3 -m pytest tests/ --ignore=tests/test_integrations_live.py -q -p no:cacheprovider
```

Notes:

- Always invoke via `python3 -m pytest`, never bare `pytest` — it guarantees
  the interpreter you installed into is the one collecting tests.
- `tests/test_integrations_live.py` hits real model APIs and needs keys; CI
  skips it and so should you unless you're changing a live integration.
- If your working tree has local files that aren't in git, restrict the run
  to tracked tests: `python3 -m pytest $(git ls-files 'tests/test_*.py' ':!:tests/test_integrations_live.py') -q -p no:cacheprovider`.
  Pass newly added test paths explicitly until they are staged.
- Lint changed Python files with `ruff check path/to/file.py`; disclose existing
  repository-wide lint failures rather than claiming a clean lint baseline.

## Extension and format contracts

Start with [extensions and compatibility](docs/guides/extensions-and-compatibility.mdx).
Use existing evaluator/importer interfaces, native Inspect tasks and Gymnasium
environments. Reuse upstream dataset and execution tools instead of adding
parallel implementations. Custom graders must distinguish missing evidence,
execution errors and measured failure, and be safe under the selected concurrency.

`tests/test_extension_contracts.py` exercises a small custom grader across sync,
async and saved-output paths, plus imported-output pairing. Historical report
fixtures in `tests/fixtures/reports/` test migration from saved artifacts. Keep
those artifacts frozen; add a fixture and migration or rejection test when
changing a format. JSON Schema validates the envelope; trial/lock validators
remain responsible for nested evidence. Keep heavy dependencies optional and
exercise the affected integration with its real upstream package installed.

## Before release

1. Run the tracked core suite with Python 3.10 and a current supported Python.
   Check the configured 3.10–3.14 CI jobs; local tests are not proof of remote CI.
2. Install the affected extras and run their integration tests. A skipped test
   does not verify compatibility. Preserve the package versions used.
3. Build both distributions with `python -m build`, then run
   `python -m twine check dist/*`. Build the wheel from the sdist as well so
   missing packaged resources cannot hide behind the working tree.
4. Install that wheel into a fresh environment with no dev extras. From a
   directory outside the checkout, run:

   ```bash
   /path/to/clean-venv/bin/python /path/to/multivon-eval/scripts/check_installed.py --bare
   /path/to/clean-venv/bin/python -m pip check
   ```

   The script rejects source shadowing, blocks outbound socket connections and
   checks public imports, packaged schemas, saved grading and migration fixtures.
   Also run `multivon-eval --help` from outside the checkout.
5. Execute changed documentation examples, validate MDX/navigation, and smoke
   test affected companion consumers against the wheel. Record limitations in
   the changelog before publishing and verify the uploaded version afterward.

The historical `all` extra covers older model/browser/agent integrations.
Install development extras such as `datasets`, `inspect`, `gymnasium`, `review`,
`otel`, `media` and `pricing` explicitly. Core CI does not exercise all extras.

## PR expectations

- Tests for every behavior change. A bug fix comes with the test that would
  have caught it.
- Docs updated in the same PR — README, ROADMAP.md, or docstrings, whichever
  the change touches. CHANGELOG.md gets an entry under `[Unreleased]`.
- Keep files under 500 lines. Comments explain non-obvious *why*, not what.
- Small, focused PRs review faster than large ones.

## Honesty rules (non-negotiable)

This project's brand is measurement you can trust, so the repo holds itself
to the same standard:

- No unverifiable claims — in code comments, docs, or benchmark copy. If you
  can't point at the data or the test, don't write the sentence.
- Every reported number carries a confidence interval or an explicit n.
- Prefer an honest `UNKNOWN` over a confident wrong answer, in evaluator
  output and in documentation alike.
- If a measurement catches a mistake of ours, we publish the correction, not
  just the fix.

## Questions

Open an issue, or email <hello@multivon.ai>.
