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

__version__ = "0.7.0"

from .suite import EvalSuite
from .case import EvalCase, AgentStep, ToolCall
from .judge import JudgeConfig, configure
from .adapters import ModelAdapter, OpenAIAdapter, AnthropicAdapter, LiteLLMAdapter
from .integrations import (
    AgentTracer, CaseImporter,
    ManualTracer, LangChainTracer,
    LangSmithTracer, LangSmithImporter,
)
from .dataset import load, load_jsonl, load_csv
from .generate import generate_from_text, generate_from_file, generate_hallucination_pairs
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
)
from .evaluators import (
    # Deterministic
    NotEmpty, ExactMatch, Contains, RegexMatch,
    JSONSchemaEval, WordCount, Latency, MaxLatency,
    BLEU, ROUGE, StartsWith, BERTScore,
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
)

__all__ = [
    "__version__",
    "EvalSuite", "EvalCase", "AgentStep", "ToolCall",
    "JudgeConfig", "configure",
    "ModelAdapter", "OpenAIAdapter", "AnthropicAdapter", "LiteLLMAdapter",
    # Integrations
    "AgentTracer", "CaseImporter",
    "ManualTracer", "LangChainTracer",
    "LangSmithTracer", "LangSmithImporter",
    "load", "load_jsonl", "load_csv",
    # Generation
    "generate_from_text", "generate_from_file", "generate_hallucination_pairs",
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
    "calibration_provenance", "load_calibration", "CalibrationEntry",
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
    # Deterministic
    "NotEmpty", "ExactMatch", "Contains", "RegexMatch",
    "JSONSchemaEval", "WordCount", "Latency", "MaxLatency",
    "BLEU", "ROUGE", "StartsWith", "BERTScore",
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
    # Consistency
    "SelfConsistency",
]
