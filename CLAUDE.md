# multivon-eval — Claude Code instructions

Apache-2.0 Python eval framework. The package is `multivon_eval/` at the repo
root; tests live in `tests/`; docs guides in `docs/`; benchmark evidence in
`benchmarks/`. There is exactly one package — anything else at root claiming
to be one is local clutter, not code.

## Running tests

Never run bare `pytest` — the working tree accumulates untracked local test
files that must not be collected. Run tracked tests only:

```bash
python3 -m pytest $(git ls-files 'tests/test_*.py') -q -p no:cacheprovider
```

CI equivalent: `python3 -m pytest tests/ --ignore=tests/test_integrations_live.py`
(the live file needs real API keys; skip it locally too).

## Rules

- Python 3.10 is the floor. The dev machine runs 3.14 — before release, check
  any new syntax against 3.10/3.11 (PEP 701 f-strings have bitten us before).
- Keep files under 500 lines.
- Comments only for non-obvious WHY, never restating the code.
- Honesty rules apply to code, docs, and commit messages alike: no
  unverifiable claims; every reported number carries a CI or an explicit n;
  prefer an honest UNKNOWN over a confident wrong answer.
- Commit identity: `Multivon <hello@multivon.ai>`. No `Co-Authored-By`
  trailers.
- Update CHANGELOG.md under `[Unreleased]` for behavior changes.
