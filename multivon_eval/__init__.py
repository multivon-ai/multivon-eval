"""
multivon-eval — AI evaluation for teams that ship models to production.

Evaluate language models, agents, and pipelines across text, RAG,
agentic workflows, and multi-turn conversations.

Quick start:
    from multivon_eval import EvalSuite, EvalCase, Relevance, Faithfulness, NotEmpty

    suite = EvalSuite("My Eval")
    suite.add_cases([EvalCase(input="What is 2+2?", expected_output="4")])
    suite.add_evaluators(NotEmpty(), Relevance())
    report = suite.run(my_model_fn)

Synthetic dataset generation (no labeled data needed):
    from multivon_eval import generate_from_file
    cases = generate_from_file("docs/faq.md", n=20)

Experiment tracking (compare runs across versions):
    from multivon_eval import Experiment
    exp = Experiment("my-pipeline")
    run_id = exp.record(report, tags={"model": "gpt-4o", "prompt_v": "3"})
    exp.compare(old_run_id, run_id)
"""

__version__ = "0.15.0"

from .suite import EvalSuite
from .case import EvalCase, AgentStep, ToolCall
from .judge import JudgeConfig, configure
from .adapters import ModelAdapter, OpenAIAdapter, AnthropicAdapter, LiteLLMAdapter
from .vision import call_vision  # vision-call dispatch for image/PDF inputs (0.9.1)
from .integrations import (
    AgentTracer, CaseImporter,
    ManualTracer, LangChainTracer,
    LangSmithTracer, LangSmithImporter,
    LangGraphTracer, OpenAIAgentsTracer,
)
from .dataset import load, load_jsonl, load_csv
from .generate import (
    generate_from_text, generate_from_file, generate_hallucination_pairs,
    generate_contrast_pairs,
)
# Deterministic generation toolkit (0.13.0): mutators + template grids ($0)
from .mutate import MUTATIONS, mutate_cases, cases_from_template
from .case_gates import GenerationReport
from .input_gate import assess_input, InputQualityReport, SignalFinding
from .experiments import (
    Experiment, list_experiments, wilson_interval, bootstrap_interval,
    runs_needed, min_detectable_effect, cohens_h, benjamini_hochberg,
    mcnemar_test, bayesian_interval,
)
from .compliance import ComplianceReporter, github_actions_anchor
from .reporters.html_compliance import ComplianceHtmlReporter, render_compliance_html
from .exceptions import (
    MultivonError, JudgeUnavailable, CalibrationMissing,
    EvaluatorPrereqMissing, CacheError, SecretsError, ComplianceError,
)
from .lockfile import (
    SuiteLock, EvaluatorFingerprint, LockMismatch,
    build_suite_lock, fingerprint_evaluator, verify_suite_against_lock,
)
from .costs import Costs, CostTracker, ProviderUsage, ModelPricing, register_pricing
from .audit_package import build_audit_package
from .retry import JudgeRetry
from .compare import CaseDiff, ReportDiff, compare_reports

# Bootstrap pipeline (0.8.0) — cold-start eval suite generator
from .discover import (
    bootstrap, BootstrapResult, RecommendedEvaluator, TraceSummary,
    infer_product_shape, summarize_traces, load_traces,
)
# Auto / intelligent-eval prototype (0.8.0)
from .auto import (
    auto_evaluators, EvaluatorRecommendation, AmbiguousCaseShape,
    generate_adversarial_cases, generate_unicode_obfuscation_cases,
    validate_adversarial_cases, HardnessReport,
)
# Persona simulator — adaptive multi-turn eval against the live system.
# Outputs are SIMULATED: synthetic users, not real traffic.
from .simulate import (
    Persona, SimulationResult, simulate,
    personas_from_jsonl, propose_personas, score_simulations,
    results_to_cases,
)
# Prompt-drift staleness + case provenance (0.9.x)
from .staleness import (
    StalenessReport, SiteVerdict, CaseVerdict, BaselineError,
    build_staleness_report, write_baseline, load_baseline,
)
from .provenance import (
    stamp_jsonl, stamp as stamp_provenance,
    read_provenance, StampResult,
    ProvenanceError, AmbiguousSiteError,
)
# Runtime prompt recorder (0.11.0) — opt-in capture of rendered prompts;
# importing it performs NO patching (zero overhead when off).
from .recorder import (
    PromptRecorder, record_prompts, recording_active,
    set_active_case, reset_active_case,
    load_recordings, merge_recordings_into_baseline,
)

# Pytest plugin: pytest is an optional dependency. Guard the import so a
# user who installs multivon-eval without pytest can still `import multivon_eval`.
# Narrow the catch to ModuleNotFoundError for pytest specifically so any
# OTHER ImportError raised by the plugin module (real regressions) bubbles up
# instead of being silently downgraded to "install pytest".
_PYTEST_MISSING_MSG = (
    "multivon_eval.assert_evaluators requires pytest. "
    "Install with: pip install 'multivon-eval[pytest]'"
)
try:
    from .pytest_plugin import assert_evaluators, EvaluatorFailure
except ModuleNotFoundError as _exc:
    if _exc.name != "pytest":
        raise

    def assert_evaluators(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError(_PYTEST_MISSING_MSG)

    class EvaluatorFailure(Exception):  # type: ignore[no-redef]
        """Placeholder for the real pytest-plugin exception when pytest is missing."""

from .secrets import (
    SecretsResolver, EnvResolver, ChainedResolver, StaticResolver,
    get_secret, set_resolver, get_resolver, reset_resolver,
)
from .cache import JudgeCache, get_cache, set_cache
from .targets import (
    BearerAuth, APIKeyAuth,
    DeployedAPITarget, MultiTurnAPITarget, BrowserTarget,
    simulate_users,
)
from .result import (
    CalibrationResult, CaseResult, EvalGateFailure, EvalReport, EvalResult,
    EvalStatus, EVALUATION_STATUSES, ERROR_STATUSES,
    PairwiseReport, PairwiseResult,
)
from .calibration import (
    calibrated_threshold, threshold_table,
    calibration_provenance, load_calibration, CalibrationEntry,
    calibration_versions, set_calibration_fallback_policy,
)
from .evaluators import (
    # Deterministic
    NotEmpty, ExactMatch, Contains, RegexMatch,
    JSONSchemaEval, WordCount, Latency, MaxLatency,
    BLEU, ROUGE, StartsWith, BERTScore,
    Levenshtein, ChrfScore,
    # LLM-as-judge (QAG)
    Faithfulness, Hallucination, Relevance, Coherence,
    Toxicity, Bias, Summarization, AnswerAccuracy,
    ContextPrecision, ContextRecall, CustomRubric, GEval,
    CheckEvaluator,
    # Agent
    ToolCallAccuracy, ToolArgumentAccuracy,
    PlanQuality, TaskCompletion, StepFaithfulness,
    ToolCallNecessity, TrajectoryEfficiency, AgentMemoryEval,
    # Compliance
    PIIEvaluator, SchemaEvaluator,
    # Conversation
    ConversationRelevance, KnowledgeRetention,
    ConversationCompleteness, TurnConsistency,
    # Consistency
    SelfConsistency,
    # Multimodal (experimental, 0.7.3)
    VQAFaithfulness, DocumentGrounding,
)

__all__ = [
    "__version__",
    "EvalSuite", "EvalCase", "AgentStep", "ToolCall",
    "JudgeConfig", "configure",
    "JudgeRetry",
    # Compare two runs
    "CaseDiff", "ReportDiff", "compare_reports",
    "ModelAdapter", "OpenAIAdapter", "AnthropicAdapter", "LiteLLMAdapter",
    # Integrations
    "AgentTracer", "CaseImporter",
    "ManualTracer", "LangChainTracer",
    "LangSmithTracer", "LangSmithImporter",
    "LangGraphTracer", "OpenAIAgentsTracer",
    "load", "load_jsonl", "load_csv",
    # Generation
    "generate_from_text", "generate_from_file", "generate_hallucination_pairs",
    # Generation toolkit (0.13.0): mutators, template grids, contrast pairs
    "generate_contrast_pairs", "mutate_cases", "cases_from_template",
    "MUTATIONS", "GenerationReport",
    "assess_input", "InputQualityReport", "SignalFinding",
    # Experiments + statistics
    "Experiment", "list_experiments", "wilson_interval", "bootstrap_interval",
    "runs_needed", "min_detectable_effect", "cohens_h", "benjamini_hochberg",
    "mcnemar_test", "bayesian_interval",
    # Exceptions
    "MultivonError", "JudgeUnavailable", "CalibrationMissing",
    "EvaluatorPrereqMissing", "CacheError", "SecretsError", "ComplianceError",
    "EvalGateFailure",
    # Calibration
    "calibrated_threshold", "threshold_table",
    "calibration_provenance", "load_calibration", "calibration_versions",
    "CalibrationEntry", "set_calibration_fallback_policy",
    # Secrets
    "SecretsResolver", "EnvResolver", "ChainedResolver", "StaticResolver",
    "get_secret", "set_resolver", "get_resolver", "reset_resolver",
    # Cache
    "JudgeCache", "get_cache", "set_cache",
    # Calibration
    "CalibrationResult",
    # Pairwise
    "PairwiseReport", "PairwiseResult",
    # Compliance
    "ComplianceReporter", "ComplianceHtmlReporter", "render_compliance_html",
    "github_actions_anchor",
    # Production targets
    "BearerAuth", "APIKeyAuth",
    "DeployedAPITarget", "MultiTurnAPITarget", "BrowserTarget",
    "simulate_users",
    # Result types (CaseResult.status enum is the headline 0.7.0 feature)
    "CaseResult", "EvalResult", "EvalReport",
    "EvalStatus", "EVALUATION_STATUSES", "ERROR_STATUSES",
    # Cost tracking
    "Costs", "CostTracker", "ProviderUsage", "ModelPricing", "register_pricing",
    # Suite locking / fingerprinting
    "SuiteLock", "EvaluatorFingerprint", "LockMismatch",
    "build_suite_lock", "fingerprint_evaluator", "verify_suite_against_lock",
    # Audit packaging
    "build_audit_package",
    # Pytest plugin (no-op when pytest is not installed)
    "assert_evaluators", "EvaluatorFailure",
    # Deterministic
    "NotEmpty", "ExactMatch", "Contains", "RegexMatch",
    "JSONSchemaEval", "WordCount", "Latency", "MaxLatency",
    "BLEU", "ROUGE", "StartsWith", "BERTScore",
    "Levenshtein", "ChrfScore",
    # LLM-as-judge
    "Faithfulness", "Hallucination", "Relevance", "Coherence",
    "Toxicity", "Bias", "Summarization", "AnswerAccuracy",
    "ContextPrecision", "ContextRecall", "CustomRubric", "GEval", "CheckEvaluator",
    # Agent
    "ToolCallAccuracy", "ToolArgumentAccuracy",
    "PlanQuality", "TaskCompletion", "StepFaithfulness",
    "ToolCallNecessity", "TrajectoryEfficiency", "AgentMemoryEval",
    # Compliance evaluators
    "PIIEvaluator", "SchemaEvaluator",
    # Conversation
    "ConversationRelevance", "KnowledgeRetention",
    "ConversationCompleteness", "TurnConsistency",
    # Multimodal (experimental, 0.7.3)
    "VQAFaithfulness", "DocumentGrounding",
    # Consistency
    "SelfConsistency",
    # Bootstrap pipeline (0.8.0)
    "bootstrap", "BootstrapResult", "RecommendedEvaluator", "TraceSummary",
    "infer_product_shape", "summarize_traces", "load_traces",
    # Persona simulator (simulated personas — synthetic users, not real traffic)
    "Persona", "SimulationResult", "simulate",
    "personas_from_jsonl", "propose_personas", "score_simulations",
    "results_to_cases",
    # Intelligent-eval (auto) prototype (0.8.0)
    "auto_evaluators", "EvaluatorRecommendation", "AmbiguousCaseShape",
    "generate_adversarial_cases", "generate_unicode_obfuscation_cases",
    "validate_adversarial_cases", "HardnessReport",
    # Prompt-drift staleness + case provenance (0.9.x)
    "StalenessReport", "SiteVerdict", "CaseVerdict", "BaselineError",
    "build_staleness_report", "write_baseline", "load_baseline",
    "stamp_jsonl", "stamp_provenance", "read_provenance", "StampResult",
    # Runtime prompt recorder (0.11.0)
    "PromptRecorder", "record_prompts", "recording_active",
    "set_active_case", "reset_active_case",
    "load_recordings", "merge_recordings_into_baseline",
    "ProvenanceError", "AmbiguousSiteError",
]
