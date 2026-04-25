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
"""

__version__ = "0.1.0"

from .suite import EvalSuite
from .case import EvalCase, AgentStep, ToolCall
from .dataset import load, load_jsonl, load_csv
from .evaluators import (
    # Deterministic
    NotEmpty, ExactMatch, Contains, RegexMatch,
    JSONSchemaEval, WordCount, Latency, MaxLatency,
    BLEU, ROUGE, StartsWith,
    # LLM-as-judge (QAG)
    Faithfulness, Hallucination, Relevance, Coherence,
    Toxicity, Bias, Summarization, AnswerAccuracy,
    ContextPrecision, ContextRecall, CustomRubric, GEval,
    # Agent
    ToolCallAccuracy, ToolArgumentAccuracy,
    PlanQuality, TaskCompletion, StepFaithfulness,
    # Conversation
    ConversationRelevance, KnowledgeRetention,
    ConversationCompleteness, TurnConsistency,
)

__all__ = [
    "__version__",
    "EvalSuite", "EvalCase", "AgentStep", "ToolCall",
    "load", "load_jsonl", "load_csv",
    # Deterministic
    "NotEmpty", "ExactMatch", "Contains", "RegexMatch",
    "JSONSchemaEval", "WordCount", "Latency", "MaxLatency",
    "BLEU", "ROUGE", "StartsWith",
    # LLM-as-judge
    "Faithfulness", "Hallucination", "Relevance", "Coherence",
    "Toxicity", "Bias", "Summarization", "AnswerAccuracy",
    "ContextPrecision", "ContextRecall", "CustomRubric", "GEval",
    # Agent
    "ToolCallAccuracy", "ToolArgumentAccuracy",
    "PlanQuality", "TaskCompletion", "StepFaithfulness",
    # Conversation
    "ConversationRelevance", "KnowledgeRetention",
    "ConversationCompleteness", "TurnConsistency",
]
