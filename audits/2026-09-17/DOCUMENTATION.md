# Documentation and README audit

September 17, 2026. Audited against multivon-eval 0.17.0.

The documentation should help a user define task success, obtain a first real
measurement, and understand when a release decision lacks evidence. The README
now follows that order and is 204 lines, with one complete offline first example.

## Corrections

| Finding | Correction |
|---|---|
| Old CI instructions allowed errors without an explicit budget. | Document zero-error default, exit codes, empty/all-skipped coverage, and the need for separate required-check coverage. |
| Report reference used `case.input`, a nonexistent `evaluators` alias, and reversed baseline/proposal order. | Match actual fields; use `prev.compare(report)` and verify comparison direction by executing the example. |
| Multimodal examples called nonexistent `suite.run_case`. | Use each evaluator's actual `evaluate(case, output)` method. |
| Gated experiment examples recorded runs only after the gate could raise. | Save inside `run()` and record the artifact in `finally`. |
| Bootstrap claimed validated calibration and a fixed total budget. | Describe p25 suggestions, estimated seed-generation budget scope, variable call costs, and review requirements. Correct the bundled skill text as well. |
| A calibration command referenced a nonexistent installed module. | Remove that recipe; describe development/test label review and the actual purpose of `suite.calibrate()`. |
| Judge docs described old verdict parsing and promised effective sampling settings. | Describe the stricter parser, historical threshold limitations, and text-provider temperature-forwarding gap. |
| Statistical docs conflated repeatability, correctness, and significance. | Explain paired versus summary tests, missing denominators, clustering, estimator assumptions, and limitations of interval overlap. Remove universal “safe for CI” agreement thresholds. |
| Benchmark docs called generated HaluEval task examples human-adjudicated and implied independent superiority. | Correct provenance, distinguish threshold-selection data, disclose judge confounding and unclustered intervals, and label historical results as not rerun for 0.17.0. |
| Introduction/selection guide overstated confidence and product readiness. | Lead with the decision being measured and distinguish current capability from unproven accuracy. |
| Examples page used heading syntax that failed standard MDX compilation. | Use normal headings with automatically generated anchors. |
| No concise guide connected user goals to verifiable outcomes. | Add `guides/task-success`, including final state, side effects, grader validation, and release policy. |

HaluEval provenance was checked against its upstream data-generation description:
https://github.com/RUCAIBox/HaluEval#data-generation. QA derives from HotpotQA;
summarization derives from CNN/DailyMail. The generated task-specific subsets
are distinct from the human-annotated general-query subset.

## Validation

- 57 MDX pages compiled with `@mdx-js/mdx` 3.1.1 after removing YAML frontmatter.
  This checks MDX syntax, not a hosted Mintlify visual rendering.
- 215 Python documentation blocks parsed successfully. Public library imports,
  recognized receiver methods, and explicitly named call keywords were checked
  against the installed source API. Fragments requiring application objects
  were not falsely treated as standalone programs.
- Documentation navigation targets and internal page links resolve.
- Report-reference fields checked against actual dataclasses and properties.
- README, quickstart, task-success, and baseline comparison examples executed
  offline; the README writes its report artifacts in a temporary directory.
- Full tracked offline suite: **1,455 passed, 4 skipped, 5 warnings**, Python 3.12.
  Command: `.venv/bin/python -m pytest $(git ls-files 'tests/test_*.py' ':!:tests/test_integrations_live.py') -q -p no:cacheprovider`
- `ruff check tests/test_documentation_contract.py` and `git diff --check` pass.

No paid evaluator benchmarks or live provider integration tests were run for
this documentation update. Corrected prose does not repair the remaining
implementation limitations identified in `AUDIT.md`: stable identity, complete
trial evidence, effective provider configuration, required-check coverage,
and independent task-specific calibration remain important work.

The bundled PDF Hell and MCP pages describe their companion packages; their
external services and all commercial/legal assertions were not independently
validated by these static checks. This is a technical documentation audit,
not a certification of legal compliance or a visual QA of the hosted site.

## Release handling

The measurement fixes were published as 0.17.0. Documentation and bundled skill
wording corrections are recorded under Unreleased and pushed to GitHub; they do
not replace the already-published PyPI artifacts. No new package behavior was
introduced in this documentation commit.
