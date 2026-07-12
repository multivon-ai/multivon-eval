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
  to tracked tests: `python3 -m pytest $(git ls-files 'tests/test_*.py') -q -p no:cacheprovider`.
- Lint with `ruff check .` before pushing.

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
