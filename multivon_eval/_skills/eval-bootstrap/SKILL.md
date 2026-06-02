---
name: eval-bootstrap
description: |
  Auto-generate an evaluation suite for an LLM-touching codebase using
  multivon-eval. Reads the project description + sample traces, picks
  the right evaluators, calibrates thresholds, emits a runnable
  eval_suite.py + 30 adversarial seed cases + EVALS.md.

  Invoke when the user says "add evals to this project", "set up
  evaluation", "eval this codebase", or when you detect an LLM-touching
  project (imports from anthropic/openai/google/litellm) with no
  existing eval/ or tests/eval/ directory.
trigger_phrases:
  - "add evals"
  - "set up evaluation"
  - "eval this codebase"
  - "evaluate this project"
  - "what evaluators should I run"
provides:
  - multivon-eval bootstrap workflow
  - EVALS.md scaffold
  - eval_suite.py with stub_model replaced by detected provider
requires:
  - multivon-eval >= 0.9.4 (pip install multivon-eval)
  - either ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY in env,
    OR a running Ollama instance for fully local bootstrap
---

# eval-bootstrap

Turns a one-paragraph product description + a handful of sample traces
into a runnable eval suite. The point: a fresh project can go from "no
eval scaffolding" to "first eval running" in under three minutes.

## When to invoke

Auto-invoke if:
- The user explicitly asks for evals ("add evals", "set up evaluation",
  "eval this codebase", "evaluate this project", "what evaluators
  should I run").
- The current repo imports from `anthropic`, `openai`, `google.genai`,
  `litellm`, `langchain`, or `llama_index` AND has no `eval/`,
  `evals/`, `tests/eval/`, or `evaluation/` directory.
- The user is reviewing a PR that touches prompts/system-messages/
  model parameters AND there's no eval gate on the repo (use
  `/eval-audit` instead if evals already exist — see that skill).

Do NOT auto-invoke if:
- The repo already has an `eval/` directory with a working suite
  (suggest `/eval-audit` instead).
- The user is in the middle of a different task and you'd be
  context-switching them.

## What the skill does

1. **Scope check** — read `pyproject.toml` / `package.json` to identify
   the LLM provider in use. If multiple, ask the user which to target.
2. **Trace collection** — find sample traces. Check, in order:
   - `traces/*.jsonl` or `data/traces/*.jsonl` in the repo
   - `notebooks/*/traces.jsonl`
   - Ask the user to paste 5-20 sample (input, output) pairs into a
     temp file
   If the project uses LangSmith / LangFuse / Phoenix, prompt for a
   dump command (the user typically runs this themselves — these are
   their secrets).
3. **Product description** — if the repo has `PRODUCT.md`,
   `OVERVIEW.md`, or a top-level README, use that as the product
   description. Otherwise ask the user for 2-3 sentences describing
   what their LLM does.
4. **Run bootstrap** — execute (in a fresh terminal the user can
   inspect):
   ```bash
   multivon-eval bootstrap \
       --product PRODUCT.md \
       --traces sample_traces.jsonl \
       --output ./eval-bootstrap \
       --judge-provider <detected_provider> \
       --judge-model <sensible_default> \
       --pii-policy redact
   ```
   The bootstrap CLI emits four files: `eval_suite.py` (runnable),
   `seed_cases.jsonl` (30 adversarial cases), `thresholds.yaml`
   (calibrated from traces), `DISCOVERY_REPORT.md` (rationale for
   each evaluator).
5. **Rewrite `stub_model`** — `eval_suite.py` ships with a placeholder
   `stub_model()` function. Replace it with a real call into the
   project's model — read 1-2 of the project's existing LLM call sites
   to copy the pattern (don't reinvent client setup; reuse what's
   there).
6. **Write `EVALS.md`** — a short doc the next Claude Code session
   reads first. Include:
   - What evaluators were picked and why (1 sentence each)
   - The CLI command to re-run the suite
   - The CI wiring TODO (link to `/eval-audit` skill for PR gating)
7. **Sanity-check** — run `python eval_suite.py --runs 1` once. If it
   fails on the first 1-2 cases, surface the error to the user with a
   clear "this is a config issue at line N, not a model issue" framing.

## Local-judge path (no API key)

If no cloud API key is in env, check for a running Ollama instance
(`curl -s http://localhost:11434/api/tags`). If present, use:

```bash
multivon-eval bootstrap \
    --product PRODUCT.md \
    --traces sample_traces.jsonl \
    --judge-provider ollama \
    --judge-model qwen2.5:14b  # or whichever model the user has pulled
```

Bootstrap with a local judge is slower (~5× wall-clock vs Haiku) but
runs entirely offline. The calibrated thresholds in the shipped
`_calibration_data/v2.json` are for cloud judges — flag this and
suggest re-running calibration locally if the user cares about
threshold accuracy:

```bash
python -m multivon_eval.benchmarks.run_calibration_v2 \
    --judges "ollama:qwen2.5:72b-instruct"
```

## What it doesn't do

- Doesn't promise the generated suite covers everything the project
  should evaluate. It covers the shape it can infer from traces and
  the product description. Treat it as a starting point.
- Doesn't replace `pytest` or the project's existing test suite.
- Doesn't ship a hosted dashboard — output is plain JSON + markdown
  + runnable Python.
- Doesn't gate PRs by itself — pair with `eval-action` (GitHub
  Action) for CI gating.

## Costs

Default bootstrap (claude-haiku judge): ~$0.12 per run, hard ceiling
$0.15. Local judge (Ollama): free, ~5× wall-clock.

## Verification

Confirm the bootstrap worked by:
```bash
ls eval-bootstrap/
# expect: eval_suite.py  seed_cases.jsonl  thresholds.yaml  DISCOVERY_REPORT.md
python eval-bootstrap/eval_suite.py --runs 1
```

If the user wants this wired into CI gating, suggest installing
[eval-action](https://github.com/marketplace/actions/multivon-eval).
