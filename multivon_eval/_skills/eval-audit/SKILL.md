---
name: eval-audit
description: |
  Pre-flight eval check on a PR diff. Reads the proposed code change,
  identifies which existing eval cases stress the changed surface, runs
  only those, reports score delta vs main, and blocks merging if any
  case regresses past threshold.

  Invoke between /review and /ship when the PR diff touches prompts,
  model calls, system prompts, tool definitions, or anything in an
  evaluator-recommended code path. Also invoke when the user says
  "audit this prompt change" or "will this regress evals".
trigger_phrases:
  - "audit this prompt"
  - "will this regress evals"
  - "check this against evals"
  - "eval impact of this PR"
  - "regression check"
provides:
  - Targeted eval run on changed surfaces
  - Per-evaluator delta vs main
  - Wilson CI / paired McNemar on each regression
  - Pre-ship block on safety-class regressions
requires:
  - multivon-eval >= 0.9.4
  - Repo has an existing eval suite (eval_suite.py from /eval-bootstrap,
    or a hand-written multivon_eval.EvalSuite)
  - git baseline reachable (default: origin/main)
---

# eval-audit

Eval as a pre-flight check, not a nightly batch. Runs only the cases
that stress what the PR actually changed.

## When to invoke

- Auto-invoke after `/review` succeeds AND the diff touches:
  - any file with LLM call sites (anthropic/openai/google/litellm
    imports + `.create()` / `.completion()` calls)
  - system prompts or instruction-tuned templates
  - tool definitions (`tools=` arg, function-call schemas)
  - retrieval pipeline code (chunkers, embedders, rerankers)
  - evaluator threshold YAML or `_calibration_data/`
- Auto-invoke on user phrases: "audit this prompt change", "will this
  regress evals", "regression check before I ship", "eval impact".
- DON'T auto-invoke for diff that touches only:
  - tests, docs, type stubs, comments
  - infra (CI YAML, Dockerfile) unless eval pipeline itself is touched

## What the skill does

1. **Scope the diff** — run `git diff --name-only origin/main...HEAD`
   to enumerate changed files. Cross-reference against
   `multivon_eval.attribution.scan` (added in 0.9.4) to find which
   prompt fingerprints changed.
2. **Identify stressing cases** — read the existing eval suite. For
   each evaluator, mark which seed cases exercise the changed surface
   (e.g., a system-prompt edit affects all cases; a single tool
   definition change affects cases whose `expected_tool_calls`
   reference that tool).
3. **Targeted run** — execute only the marked cases. Don't run the
   full suite — that's the point. Aim for <60s wall-clock on a
   typical PR. Use multi-run (--runs 3) on flaky-sensitive evaluators
   to surface real signal vs noise.
4. **Compare against baseline** — load `baseline_report.json` if
   present (committed by previous `/ship` runs), or re-run against
   `origin/main` if not. Compute per-evaluator delta + Wilson CI +
   paired-McNemar p-value. (multivon-eval's `compare_reports` does
   all of this; don't reimplement.)
5. **Block on safety-class regression** — any evaluator with
   `safety`/`toxicity`/`bias`/`pii`/`hallucination` in its name that
   regresses at p<0.05: report as BLOCK. Non-safety regression: report
   as WARN with delta + CI. No regression: report as PASS.
6. **Print a short summary** — 5-10 lines max. The summary tells the
   user (a) what changed, (b) what regressed and by how much, (c) the
   one-line CI calculation that justifies (b), (d) what to do next
   (fix-and-retry, accept-and-document-baseline, or ship-as-is).

## Output format

Print one of:

```
✓ eval-audit PASS — 4/4 stressed cases held at baseline (n=12 reruns,
  no eval regressed at p<0.05). Safe to ship.
```

```
⚠ eval-audit WARN — Faithfulness dropped 4pp (0.78 → 0.74) on 6/6
  stressed cases. Wilson CI overlap [0.61–0.85] vs baseline
  [0.65–0.89], paired McNemar p=0.14. Within noise but worth noting
  in the PR description.
```

```
✗ eval-audit BLOCK — PII evaluator regressed 12pp (0.95 → 0.83) on
  the 8 cases that exercise the changed input-sanitization path.
  Paired McNemar p=0.003, CIs do not overlap. SAFETY-CLASS — do not
  ship. See benchmarks/results/eval-audit/<sha>.json for the failing
  cases.
```

## What it doesn't do

- Doesn't replace the full nightly suite. This is a targeted
  pre-ship check; the comprehensive suite runs on a schedule
  (eval-action handles that).
- Doesn't auto-fix regressions. It surfaces them — fixes are still
  human judgment.
- Doesn't add new eval cases. If a regression points at an
  unexercised surface, suggest adding a case but don't do it
  inline (the user picks the right framing).

## Verification

After the audit completes:
```bash
cat eval-audit-<sha>.json | jq '.summary'
```

Should show: `verdict`, `cases_run`, `evaluators_assessed`, `regressions`,
`baseline_sha`, `head_sha`.

## Pairs with

- `/eval-bootstrap` — to create the eval suite this skill audits against.
- `eval-action` (GitHub Action) — for the nightly + post-merge runs
  this skill complements.
