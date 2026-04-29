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

__version__ = "0.3.0"

from .suite import EvalSuite
from .case import EvalCase, AgentStep, ToolCall
from .judge import JudgeConfig, configure
from .integrations import (
    AgentTracer, CaseImporter,
    ManualTracer, LangChainTracer,
    LangSmithTracer, LangSmithImporter,
)
from .dataset import load, load_jsonl, load_csv
from .generate import generate_from_text, generate_from_file, generate_hallucination_pairs
from .experiments import (
    Experiment, list_experiments, wilson_interval, bootstrap_interval,
    runs_needed, min_detectable_effect, cohens_h,
)
from .compliance import ComplianceReporter
from .evaluators import (
    # Deterministic
    NotEmpty, ExactMatch, Contains, RegexMatch,
    JSONSchemaEval, WordCount, Latency, MaxLatency,
    BLEU, ROUGE, StartsWith,
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
    # Integrations
    "AgentTracer", "CaseImporter",
    "ManualTracer", "LangChainTracer",
    "LangSmithTracer", "LangSmithImporter",
    "load", "load_jsonl", "load_csv",
    # Generation
    "generate_from_text", "generate_from_file", "generate_hallucination_pairs",
    # Experiments
    "Experiment", "list_experiments", "wilson_interval", "bootstrap_interval",
    "runs_needed", "min_detectable_effect", "cohens_h",
    # Compliance
    "ComplianceReporter",
    # Deterministic
    "NotEmpty", "ExactMatch", "Contains", "RegexMatch",
    "JSONSchemaEval", "WordCount", "Latency", "MaxLatency",
    "BLEU", "ROUGE", "StartsWith",
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
