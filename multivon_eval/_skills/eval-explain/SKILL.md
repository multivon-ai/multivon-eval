---
name: eval-explain
description: |
  Explain why multivon-eval recommended a particular evaluator,
  threshold, or methodology. Reads the rationale from the bootstrap
  output (DISCOVERY_REPORT.md), the evaluator's docstring, and one or
  two example cases, then answers in 3 sentences.

  Invoke when the user asks "why did multivon recommend Faithfulness
  here", "what does this evaluator measure", "is this the right eval
  for my use case", "explain this threshold", "why this evaluator",
  or right after /eval-bootstrap completes.

  Provides: a 3-sentence rationale per evaluator, 1-2 example cases
  that exercise the evaluator, and links to the methodology +
  benchmark page.

  Requires: multivon-eval >= 0.9.8, and either a DISCOVERY_REPORT.md
  (from /eval-bootstrap) in the project or a known evaluator name.
allowed-tools: Read, Grep, WebFetch
---

# eval-explain

Closes the "black box recommender" DX gap. Bootstrap picks evaluators
based on inferred shape; this skill explains why.

## When to invoke

- Auto-invoke after `/eval-bootstrap` completes — surface the
  rationale for the top 1-2 evaluators that were picked, so the user
  understands what they just got before scrolling DISCOVERY_REPORT.md.
- User asks "why did multivon recommend X" / "what does X evaluator
  do" / "is X the right eval for my use case" / "explain this
  threshold".

## What the skill does

1. Locate the source of truth:
   - First, check for `DISCOVERY_REPORT.md` next to `eval_suite.py`.
     If present, read the rationale block for the named evaluator.
   - If not, read the evaluator's docstring directly:
     `python -c "from multivon_eval import Faithfulness;
     print(Faithfulness.__doc__)"`
2. Find 1-2 example cases that exercise the evaluator (from
   `seed_cases.jsonl` if present, or by generating one with
   `generate_hallucination_pairs` for hallucination/faithfulness).
3. Answer in **exactly 3 sentences**:
   - Sentence 1: what the evaluator measures (paraphrase the docstring,
     don't quote it).
   - Sentence 2: why bootstrap picked it for *this* project (cite the
     trace pattern or product-shape signal that drove the pick).
   - Sentence 3: what alternatives exist and when you'd use them
     instead.
4. Optionally append 1 example case formatted as `input: ...` /
   `expected: ...` so the user sees what the evaluator actually does.

## Example

User: "Why did multivon recommend Faithfulness here?"

Skill output:
> Faithfulness measures whether your agent's answer is grounded in the
> retrieved context — it generates yes/no questions about claims in
> the answer and scores by the fraction the context supports.
> Bootstrap picked it because your traces contain a `context` field on
> every row (RAG shape) and your product description mentions
> "answers from our internal docs." If your context were short (1-2
> sentences), Hallucination would be the better pick; if it were
> long-form generated text without retrieval, Coherence + AnswerAccuracy
> would be the right pair instead.
>
> Example case:
> ```
> input: "what is the company's return policy?"
> context: "Returns accepted within 30 days with receipt..."
> output: "We accept returns within 60 days with no receipt."
> expected: faithfulness LOW — output contradicts context on both
>           the time window and the receipt requirement
> ```

## What it doesn't do

- Doesn't argue with the user's choice. If they want to override
  bootstrap's recommendation, surface why bootstrap picked X, then
  step back. The user has context bootstrap doesn't.
- Doesn't lecture about statistical methodology. Keep the explanation
  to the chosen evaluator's behavior, not eval theory.
- Doesn't generate cases the user didn't ask for. The example case
  is illustrative, not material the user has to use.

## Pairs with

- `/eval-bootstrap` — runs first, this skill explains its output.
- `/eval-audit` — when a particular evaluator flags a regression and
  the user asks "wait, what does this even measure?"
