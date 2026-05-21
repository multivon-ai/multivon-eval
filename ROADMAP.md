# multivon-eval roadmap

Track of what's shipped + what's coming. Updated when a feature lands or moves between sections. See [CHANGELOG.md](CHANGELOG.md) for the dated release history.

## Shipped

### Core SDK

- [x] Deterministic evaluators (BLEU, ROUGE, regex, JSON schema, latency, exact match, contains, word count, Levenshtein, ChrfScore, BERTScore)
- [x] LLM-as-judge with QAG scoring (Faithfulness, Hallucination, Relevance, Coherence, Toxicity, Bias, AnswerAccuracy, ContextPrecision, ContextRecall, Summarization, CustomRubric, GEval, CheckEvaluator)
- [x] Agent trace evaluators — ToolCallAccuracy, ToolArgumentAccuracy, ToolCallNecessity, TrajectoryEfficiency, PlanQuality, TaskCompletion, StepFaithfulness
- [x] Multi-session `AgentMemoryEval`
- [x] Conversation evaluators — ConversationRelevance, KnowledgeRetention, ConversationCompleteness, TurnConsistency
- [x] Multimodal evaluators — VQAFaithfulness, DocumentGrounding (experimental, 0.7.3+)
- [x] SelfConsistency for non-deterministic outputs
- [x] Plain-English checks (`suite.add_check()`) with auto-generated QAG questions
- [x] Built-in model adapters (`run_with_openai`, `run_with_anthropic`, `run_with_litellm`)
- [x] Parallel runner (`workers=N`) + async runner (`run_async`)

### Bootstrap + auto module (new in 0.8.0)

- [x] `multivon-eval bootstrap` cold-start CLI — product description + traces → tuned EvalSuite
- [x] `multivon_eval.auto.auto_evaluators(case)` — pure-heuristic evaluator recommender
- [x] `multivon_eval.auto.generate_adversarial_cases` — LLM-generated cases across 10 failure modes
- [x] `multivon_eval.auto.validate_adversarial_cases` — N-shot judge-noise filter
- [x] Local PII redaction before any LLM call in the bootstrap pipeline

### Compliance & privacy

- [x] `PIIEvaluator` — local regex, zero API calls, multi-jurisdiction (GDPR, CCPA, HIPAA, DPDP India, PIPEDA)
- [x] `SchemaEvaluator` — Pydantic + JSON Schema validation with per-field failures
- [x] `ComplianceReporter` — hash-chained NDJSON audit trails, EU AI Act + NIST AI RMF Article mappings, tamper-evident verification
- [x] `EvalSuite.eu_ai_act_high_risk()` / `for_regulated(jurisdiction=...)` factories
- [x] Audit-pack generation (`audit_package.build_audit_package`)

### Statistical rigor

- [x] Wilson 95% CI on pass rate + bootstrap 95% CI on avg score (shown by default)
- [x] Score percentiles (p10 / p50 / p90) to expose bimodal distributions
- [x] Power warning when dataset is too small
- [x] `runs_needed(delta=)` + `min_detectable_effect(n=)` sizing helpers
- [x] Benjamini-Hochberg multiple-comparison correction in `exp.compare()`
- [x] Cohen's h effect size in experiment comparison
- [x] Judge calibration vs human labels (`suite.calibrate()`)
- [x] Judge reliability check (`JudgeConfig(reliability_check=True)`)

### Operations & integrations

- [x] CLI — `multivon-eval init / run / report / view / generate / experiments / compare / bootstrap / discover`
- [x] HTML report export (self-contained, shareable)
- [x] JUnit XML output for native CI rendering
- [x] Cost + latency budgets (`report.assert_budget(...)`)
- [x] Suite lockfile + fingerprinting (`build_suite_lock`, `verify_suite_against_lock`)
- [x] Cache (`JudgeCache`, `set_cache()`) — 2,271× speedup on cache hits
- [x] Framework integrations — LangChain, LangSmith, LangGraph, OpenAI Agents SDK, ManualTracer
- [x] Production targets — `DeployedAPITarget`, `BrowserTarget`, `MultiTurnAPITarget`, `BearerAuth`, `APIKeyAuth`

## In flight (not yet shipped)

- [ ] LlamaIndex / CrewAI tracer integrations
- [ ] Pytest plugin (`@eval_case` decorator)
- [ ] LiteLLM adapter (covers Azure, Bedrock, Vertex AI, 100+ providers via one interface)
- [ ] Tiered eval cost optimizer — start with heuristic, escalate to local model, escalate to frontier only when needed
- [ ] Agent simulation / adversarial user testing
- [ ] Multi-LLM consultation as v0.9 upgrade to the bootstrap CLI (Claude + GPT + Gemini for metric proposal)

## How to influence the roadmap

- File an issue at <https://github.com/multivon-ai/multivon-eval/issues> with the use case
- Open a PR if you've already built it
- Email <hello@multivon.ai> for enterprise / regulated-industry priorities

If a feature is on this list, it's either shipped (`[x]`) or actively considered (`[ ]`). If it's not on the list, it's not necessarily off the table — but file an issue so we can track demand.
