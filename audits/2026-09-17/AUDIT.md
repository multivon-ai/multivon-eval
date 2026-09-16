# Multivon evaluation audit — 17 September 2026

**Verdict:** the project has a useful direction and substantial working functionality, but the evidence does not establish state-of-the-art evaluation accuracy. Its strongest opportunity is trustworthy release decisions. Before this audit, several paths contradicted that promise by converting missing measurements into successes or statistically significant improvements.

The audit started at library commit `b5027d5` and website commit `36c1d52`, both clean. Scope: runner, result aggregation, judges, comparison, grader validation, calibration and benchmark methodology, representative agent evaluators, public website content, and `multivon-web` source. This is a targeted engineering and product audit, not a certification of every evaluator or a new empirical accuracy study.

## The goal to optimize

**Given a specific AI task and a proposed system change, determine whether task success improved without unacceptable regressions—and show when the evidence is insufficient.**

For each evaluation, identify:

1. The user's intended outcome, including permitted and forbidden actions.
2. The evidence that establishes success: exact answer, final environment state, source-supported claims, expert rubric, or a combination.
3. The population and important slices the cases represent.
4. The mistakes that matter most, and their acceptable rates.
5. The decision: ship, reject, or gather more evidence.

This is broader than collecting evaluator scores. `auto_evaluators` infers case shape; bootstrap proposes a suite. Neither establishes the business goal or validates that a proxy metric measures it. Make this distinction explicit in both the product and docs.

A defensible positioning is **a local evaluation framework for release decisions, with grader validation and inspectable evidence**. A defensible SoTA claim would be narrower: best demonstrated detection of a defined class of errors on frozen, independently labeled data at a declared cost and coverage level.

## What is already good

- Offline first run, a compact Python API, practical framework integrations, and portable reports.
- Explicit infrastructure error types and the ability to inspect failure reasons.
- Known-good reference validation plus known-bad contrasts: the right direction for testing graders.
- Repeated trials, uncertainty reporting, explicit sample-size limitations, and calibration provenance.
- Published corrections rather than silently rewriting failures.
- Before changes, the tracked offline suite passed **1,421 tests**, with **4 skips**, on Python 3.12. Passing tests did not establish the measurement guarantees: some explicitly required the problematic behavior.

## Confirmed defects addressed in 0.17.0

The pre-change reproduction output is in `reproductions-before.json`. Regression checks are in `tests/test_audit_measurement_integrity.py`.

| Priority | Finding and concrete consequence | Correction |
|---|---|---|
| P0 | A Faithfulness-only case without context produced score 1.0, pass rate 100%, and passed a threshold-1 gate although nothing was graded. | Skips are unmeasured; all-skipped cases are SKIPPED. Measured aggregates exclude them. Metadata survives repeated runs and JSON round trips. |
| P0 | “Sorry, the capital of France is Berlin” against context saying Paris bypassed both Faithfulness and Hallucination and scored 1.0. | Factuality evaluators no longer use the refusal-prefix shortcut. Claimless responses can still pass after actual evaluation. |
| P0 | Recovery from 20 judge errors became 20 model improvements, with McNemar p ≈ 0.0000215. | Only completed quality pairs enter quality directions and McNemar. Infrastructure counts remain separate. Incomplete comparison gates exit 2. |
| P0 | A mostly errored run could pass a quality gate using only the surviving cases; errors merely warned unless an error budget was set. | Active quality gates default to zero errors and reject unmeasured cases. An explicit error budget remains available. |
| P1 | “I cannot say yes” parsed YES; “There is no way to determine this” parsed NO. | Verdict mentions in explanatory text are UNKNOWN; leading verdicts and complete explicit answer phrases are accepted. |
| P1 | OpenAI and Anthropic SDK spies showed configured timeout=3 was absent from actual requests/client construction. | Timeouts reach sync/async provider paths. This is per-request control, not an end-to-end execution deadline. |
| P1 | Reliability checks could replay cached judge responses and manufacture perfect agreement. | Reliability reruns bypass cache reads and writes. |
| P1 | Ordered strict tool scoring returned **1.5** for three expected calls of the same name plus an unexpected call. | Denominator counts expected occurrences consistently, yielding 0.75 in that case. |
| P1 | Per-evaluator/tag summaries and percentiles included skips/errors while the headline excluded errors. | Quality aggregates use completed measurements consistently. |

The release also preserves infrastructure status for uncaught parallel-runner errors, excludes skipped scores from trace-threshold suggestions, and labels skips in terminal/HTML/JUnit exports.

**Migration:** skipped individual results now have `passed=False` and a placeholder score, and previously permissive CI gates may stop passing. Read `docs/guides/migration-0-17.mdx`. Historical benchmark accuracy is not automatically applicable after a parser or grading-path change.

## Remaining engineering work, ordered by risk

These are not represented as fixed by 0.17.0.

**P1 — Reproducibility must record effective requests.** In `judge.py`, native OpenAI/Anthropic requests omit configured temperature; locks/cache keys can nevertheless include it. Some reasoning models reject sampling controls, so blanket forwarding is also wrong. Build provider/model-aware request construction, expose requested versus effective settings, test it with SDK contract fixtures, then remeasure calibration. Temperature zero is not a guarantee of deterministic API output. Parameter restrictions are model dependent; see [official OpenAI guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2).

**P1 — Stable task identity and comparison compatibility.** `compare.py:_pair_by_input` pairs by prompt text and occurrence order. Reordering two tasks with identical prompts but different contexts can pair the wrong tasks. Add immutable case IDs, task/context/reference hashes, and comparison checks for grader/model/rubric changes. The current migration notes instruct callers to hold those constant; the implementation does not enforce the full contract.

**P1 — Preserve trials as records.** `_aggregate_runs` keeps the final output, score list, pass count, and aggregated evaluator rows; it drops individual reasons/metadata and does not preserve every captured trial trace. An error in one trial excludes the whole case. Store each trial's status, output, trace, judge settings, attempts and costs, then derive case summaries. This enables independently auditable reliability metrics and more honest partial-coverage reporting.

**P1 — Separate baseline normalization from calibration.** `discover.py:calibrate_thresholds` uses the 25th percentile of unlabeled trace scores. That can normalize poor behavior into a passing baseline; it does not measure agreement with humans. Rename this in public explanations as a threshold suggestion. Human calibration needs labeled successes/failures, a declared loss function, separate fitting and evaluation sets, and out-of-domain warnings. The website now makes this distinction.

**P1 — Verify judge resistance to adversarial evaluated text.** Judge prompts directly interpolate model output and retrieved text. A real provider-backed attack benchmark is needed: instruction injection, answer-position/length effects, polite false claims, numerical contradictions, multilingual cases, and abstention. This audit did not measure live prompt-injection success. Delimiters or a system message alone should not be treated as proof of resistance.

**P1 — Explicit measurement coverage.** Faithfulness checks only the first ten extracted claims. Partial UNKNOWN votes are excluded once sufficient parseable votes remain. The reason string discloses this, but a score can describe only a subset of an answer. Expose structured attempted/verified/unknown/truncated counts and a configurable coverage gate; use sampling or complete verification appropriate to the decision.

**P2 — Statistical assumptions.** Fixed-sample combinatorial pass@k/pass^k estimates should not be called unbiased under arbitrary adaptive stopping. Current early stopping changes the recorded sample; reporting UNKNOWN only when k exceeds recorded n does not address selection bias. Degenerate bootstrap/Wilson fallback behavior also needs simulation-based coverage validation. Publish the estimand and assumptions, not just an interval label. “Not significant” does not mean equivalent or safe to ship.

**P2 — Agent outcomes.** Tool names and plausible arguments are useful diagnostics. Task success should also be tested against actual state: correct record changed, correct amount, no duplicate action, and required constraints satisfied. Add executable outcome checks, environment reset/isolation, and tool argument schemas. Avoid requiring a single trajectory when several are valid. Anthropic's [agent-evaluation guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) distinguishes outcomes, transcripts, graders, and repeated trials.

**P2 — Maintenance and operations.** At audit start, `suite.py` was 1,836 lines, `result.py` 1,085, and `llm_judge.py` 918, despite the repository's 500-line guideline. Split execution, measurements, exports, and provider transport along tested boundaries. Repository-wide Ruff reports substantial existing debt; do not confuse clean tests with clean lint. Add cancellation/deadlines, resumable trials, bounded concurrency/rate limits, provider contract tests and wheel-install checks. Test fault injection, not only successful mocked replies.

## Why the existing benchmark evidence is not sufficient for SoTA

- The documented HaluEval-QA threshold is tuned on the same QA subset used for the headline score. The Faithfulness/Sum result is likewise a calibration result. These disclosures are good; those rows still cannot establish generalization.
- `run_truly_held_out.py` tests **30 source documents / 60 paired answers** from the start of HaluEval-Sum. This supports a limited cross-task result within that benchmark, not broad external superiority.
- `_add_cis.py` reconstructs independent rows from confusion counts. Paired answers share a source, so document clustering is lost. Store predictions with source IDs and bootstrap the sampling units. Comparisons should estimate paired differences directly.
- Some baseline wrappers turn exceptions into score 0.5. Those failures can enter accuracy counts as real predictions. Apply the same error and coverage rules to every framework.
- The older DeepEval comparison changes the underlying judge model. It compares configurations, not the isolated benefit of Multivon's algorithm.
- The website's newer same-judge RAGTruth pilot is useful additional evidence, but explicitly lacks a public standalone harness. A summary JSON cannot independently reproduce the result.
- The website formerly displayed F1=0 for a baseline with 100/100 errors. Its quality is unmeasured, not zero. The local website change now shows that correctly.
- Label provenance needs precision. HaluEval contains both generated task examples and human-annotated material; do not describe every task split as independently human labeled without checking the exact split. See the [HaluEval paper](https://aclanthology.org/2023.emnlp-main.397/).

## A practical path to stronger evidence

Start with one target: **grounded-answer regression detection for RAG applications**. It matches the existing implementation and data. Treat agent completion and multimodal grading as separate evaluation tracks.

1. Freeze a task contract and loss function. For example, prioritize detecting unsupported material claims while bounding false alarms on correct answers. Set acceptable error and abstention rates before scoring the test set.
2. Create separate development, calibration and test partitions grouped by source. Deduplicate documents and task families. Include a later temporal holdout and a small representative production sample labeled by independent reviewers with adjudication.
3. Compare QAG, a simple structured single-call judge, DeepEval, RAGAS, and a relevant specialist judge using the same underlying models and a disclosed budget. Report both defaults and development-tuned configurations. Never tune on the final test set.
4. Report false-negative and false-positive rates, precision/recall/F1, abstention/coverage, judge failures, p50/p95 latency and actual cost. Include clustered uncertainty and paired differences, plus slices for long answers, numbers, negation, distractors and language.
5. Repeat judge runs with cache bypass, publish every item-level verdict and versioned request, and test whether the method changes real release decisions correctly.
6. Publish the harness, frozen manifests, dependency lock, run command, model snapshots, raw outputs and known limitations. Have another person reproduce it before making comparative claims.

Useful reference tracks include [FaithBench](https://github.com/vectara/FaithBench), [FaithJudge's study](https://arxiv.org/abs/2505.04847), [JudgeBench](https://arxiv.org/abs/2410.12784), and [RewardBench 2 tooling](https://github.com/allenai/reward-bench). JudgeBench/RewardBench test broader judging abilities; they are not interchangeable with a RAG grounding test. The recent [AgentJudgeBench preprint](https://arxiv.org/abs/2608.26623) is a candidate for a later tool-calling judge track, not proof about this library.

Success criteria should be falsifiable: no green gate from missing measurements; no judge outage interpreted as quality change; held-out improvement at a predeclared operating point; bounded cost; and reproducibility by a second operator. Avoid expanding the evaluator catalog until these contracts are reliable.

## Website assessment and changes

The current headline explains the benefit well. The source has a consistent restrained visual system, semantic sections, responsive layouts, a skip link, and reduced-motion handling. A redesign of the brand is unnecessary.

The reading problem is sequencing and claim precision:

- The product page led with paid bootstrap, low-level auto APIs and prompt-drift details. A newcomer needs one local run, a familiar use case, and a concrete explanation of success/failure first.
- The homepage put a 20.9% static-scanner failure story in the hero before the reader understood the product. Keep it on the existing track-record section; transparency is useful once its relevance is clear.
- Technical body copy repeatedly used 14px type, with captions at 10–12px. Key explanatory paragraphs now use 16px and a narrower prose measure. Preserve compact type for actual reference tables.
- “Live output” described hardcoded example text; now labeled example output.
- “CI on every number” and “every aggregate” overstate interval coverage; now limited to headline scores.
- The claimed bootstrap $0.15 hard ceiling conflicts with the current $2 estimated seed-generation budget, which does not cap the total provider bill. Corrected the explanation and removed the old fixed-price promise.
- The benchmark page now frames its table as a recorded configuration comparison with an unavailable harness and no general ranking claim.

Local edits: homepage, `/eval`, `/benchmark`, and the benchmark summary JSON. The product page now leads with an offline three-step start and the task-goal explanation; use cases precede advanced APIs. No website deployment is implied by these edits.

Recommended next content artifact: one worked before/after evaluation with the task goal, ten understandable cases, one grader failure, one model failure, an uncertain result, and the final release decision. Show the report artifact beside the code. Keep advanced details in the docs and link from each relevant result.

## Validation and limits

Validation results and release artifact hashes are recorded separately in this audit directory. Paid judge benchmarks were not rerun. Browser tooling returned “No browser is available”; website review used source, public text retrieval, lint, TypeScript and production build checks. Responsive visual inspection and a real assistive-technology audit remain outstanding. The 0.17.0 release is a correctness improvement, not a certification that all remaining risks are resolved.

Release outcome: [multivon-eval 0.17.0](https://pypi.org/project/multivon-eval/0.17.0/) was published successfully. Both PyPI artifacts match the locally tested SHA-256 hashes. Final suites passed 1,450 tests with 4 skips on each of Python 3.10 and 3.12; the fresh wheel passed all 29 new regression checks and the offline demo. Website changes remain local and are not deployed.
