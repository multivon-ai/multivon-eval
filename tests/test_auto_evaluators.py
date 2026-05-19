"""Validation tests for multivon_eval.auto.auto_evaluators.

Pure heuristic — no LLM calls. Covers every case-shape path declared by
the prototype plus the strict_mode + ambiguity behavior.
"""
from __future__ import annotations

import pytest

from multivon_eval import EvalCase
from multivon_eval.auto import (
    AmbiguousCaseShape,
    EvaluatorRecommendation,
    auto_evaluators,
)


def _names(recs: list[EvaluatorRecommendation]) -> list[str]:
    return [r.evaluator.__name__ for r in recs]


def _tier_names(recs: list[EvaluatorRecommendation], tier: str) -> list[str]:
    return [r.evaluator.__name__ for r in recs if r.tier == tier]


# ─── RAG shape ─────────────────────────────────────────────────────────────

def test_rag_case_recommends_faithfulness_and_hallucination_as_primary():
    case = EvalCase(
        input="What is the refund window?",
        context="Refunds within 30 days of purchase.",
    )
    recs = auto_evaluators(case)
    primaries = _tier_names(recs, "primary")
    assert "Faithfulness" in primaries
    assert "Hallucination" in primaries
    secondaries = _tier_names(recs, "secondary")
    assert "ContextPrecision" in secondaries
    assert "Relevance" in secondaries
    # ContextRecall only when expected_output is also present
    assert "ContextRecall" not in _names(recs)


def test_rag_with_expected_output_adds_context_recall():
    case = EvalCase(
        input="What is the refund window?",
        context="Refunds within 30 days of purchase.",
        expected_output="30 days",
    )
    recs = auto_evaluators(case)
    assert "ContextRecall" in _names(recs)


# ─── QA shape (input + expected, no context) ──────────────────────────────

def test_qa_case_recommends_answer_accuracy():
    case = EvalCase(input="2+2?", expected_output="4")
    recs = auto_evaluators(case)
    assert "AnswerAccuracy" in _tier_names(recs, "primary")
    assert "Relevance" in _tier_names(recs, "secondary")


def test_bare_input_falls_into_degraded_qa_with_low_confidence():
    case = EvalCase(input="hello")
    recs = auto_evaluators(case)
    primaries = [r for r in recs if r.tier == "primary"]
    assert len(primaries) == 1
    assert primaries[0].evaluator.__name__ == "Relevance"
    assert primaries[0].confidence == "low"


# ─── Agent paths ───────────────────────────────────────────────────────────

def test_expected_tool_calls_recommends_tool_call_accuracy():
    case = EvalCase(input="book a meeting", expected_tool_calls=["create_event"])
    recs = auto_evaluators(case)
    assert "ToolCallAccuracy" in _tier_names(recs, "primary")
    assert "ToolCallNecessity" in _tier_names(recs, "secondary")


def test_agent_trace_recommends_plan_quality_and_step_faithfulness():
    from multivon_eval import AgentStep
    trace = [AgentStep(thought="t", output="o")]
    case = EvalCase(input="x", agent_trace=trace)
    recs = auto_evaluators(case)
    primaries = _tier_names(recs, "primary")
    assert "PlanQuality" in primaries
    assert "StepFaithfulness" in primaries
    assert "TaskCompletion" in _tier_names(recs, "secondary")


# ─── Conversation ──────────────────────────────────────────────────────────

def test_conversation_recommends_relevance_retention_consistency():
    case = EvalCase(
        input="latest reply",
        conversation=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    recs = auto_evaluators(case)
    primaries = _tier_names(recs, "primary")
    assert "ConversationRelevance" in primaries
    assert "KnowledgeRetention" in primaries
    assert "TurnConsistency" in _tier_names(recs, "secondary")


# ─── Multimodal + Schema (metadata-driven add-ons) ─────────────────────────

def test_image_metadata_appends_vqa_faithfulness():
    case = EvalCase(
        input="what is in the image?",
        metadata={"image_url": "https://example.com/x.png"},
    )
    recs = auto_evaluators(case)
    assert "VQAFaithfulness" in _names(recs)


def test_images_metadata_appends_document_grounding():
    case = EvalCase(
        input="summarize this document",
        metadata={"images": ["p1.png", "p2.png"]},
    )
    recs = auto_evaluators(case)
    assert "DocumentGrounding" in _names(recs)


def test_schema_metadata_appends_schema_evaluator():
    case = EvalCase(
        input="extract fields",
        metadata={"schema": {"type": "object"}},
    )
    recs = auto_evaluators(case)
    assert "SchemaEvaluator" in _names(recs)


# ─── Always-on guardrails + opt-ins ────────────────────────────────────────

def test_not_empty_is_always_recommended_as_guardrail():
    case = EvalCase(input="x", expected_output="y")
    recs = auto_evaluators(case)
    assert "NotEmpty" in _tier_names(recs, "guardrail")


def test_pii_opt_in_appends_pii_evaluator_with_rationale():
    case = EvalCase(input="x", expected_output="y")
    recs = auto_evaluators(case, include_pii=True, pii_jurisdiction="dpdp")
    pii_recs = [r for r in recs if r.evaluator.__name__ == "PIIEvaluator"]
    assert len(pii_recs) == 1
    assert pii_recs[0].tier == "guardrail"
    assert "dpdp" in pii_recs[0].rationale


def test_safety_opt_in_appends_toxicity_and_bias():
    case = EvalCase(input="x", expected_output="y")
    recs = auto_evaluators(case, include_safety=True)
    guardrails = _tier_names(recs, "guardrail")
    assert "Toxicity" in guardrails
    assert "Bias" in guardrails


# ─── task_type override pins the path ──────────────────────────────────────

def test_task_type_override_pins_path():
    # A case whose shape would auto-detect as RAG (input + context)
    case = EvalCase(input="2+2?", context="Math facts.", expected_output="4")
    # Force QA path explicitly
    recs = auto_evaluators(case, task_type="qa")
    assert "AnswerAccuracy" in _tier_names(recs, "primary")
    # Faithfulness should NOT be recommended because we pinned QA
    assert "Faithfulness" not in _names(recs)


# ─── Ambiguity scoring + strict_mode ───────────────────────────────────────

def test_ambiguous_case_lowers_confidence():
    # Multiple primary signals: input+context (RAG) + expected_tool_calls (agent)
    # + conversation. That's 3 primary signals → confidence drops to low.
    case = EvalCase(
        input="x",
        context="some context",
        expected_tool_calls=["t"],
        conversation=[{"role": "user", "content": "hi"}],
    )
    recs = auto_evaluators(case)
    # Whatever path is picked, its primaries should be low confidence
    primaries = [r for r in recs if r.tier == "primary"]
    assert primaries  # something was picked
    assert all(r.confidence == "low" for r in primaries)


def test_strict_mode_raises_on_low_confidence_primary():
    case = EvalCase(
        input="x",
        context="some context",
        expected_tool_calls=["t"],
        conversation=[{"role": "user", "content": "hi"}],
    )
    with pytest.raises(AmbiguousCaseShape):
        auto_evaluators(case, strict_mode=True)


def test_strict_mode_raises_on_completely_bare_case():
    case = EvalCase(input="")  # nothing usable
    with pytest.raises(AmbiguousCaseShape):
        auto_evaluators(case, strict_mode=True)


def test_strict_mode_passes_on_clean_rag_case():
    case = EvalCase(input="q?", context="ctx")
    # Should NOT raise — single clean signal
    recs = auto_evaluators(case, strict_mode=True)
    assert "Faithfulness" in _names(recs)


# ─── Output ordering: primary → secondary → guardrail ──────────────────────

def test_recommendations_ordered_primary_secondary_guardrail():
    case = EvalCase(input="q?", context="ctx", expected_output="a")
    recs = auto_evaluators(case, include_pii=True, include_safety=True)
    # The guardrail tier must come AFTER all primary/secondary recs
    tiers = [r.tier for r in recs]
    last_primary = max((i for i, t in enumerate(tiers) if t == "primary"), default=-1)
    first_guardrail = min((i for i, t in enumerate(tiers) if t == "guardrail"), default=len(tiers))
    assert last_primary < first_guardrail
