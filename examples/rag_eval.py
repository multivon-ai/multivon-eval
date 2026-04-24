"""
RAG eval — tests faithfulness and hallucination for a retrieval-augmented system.
This is the most common production use case for llm-evals.
"""
from dotenv import load_dotenv
load_dotenv()

import anthropic
from llm_evals import EvalSuite, EvalCase, Faithfulness, Hallucination, Relevance, NotEmpty

client = anthropic.Anthropic()

CONTEXT = """
OmniTensorLabs is an AI services company founded in 2026. The company offers four core services:
custom AI development, AI integration, intelligent automation, and data analytics.
The company works with both startups and enterprises. Their mission is to democratize access
to world-class AI. They are headquartered in San Francisco and have a team of 12 people.
"""

def rag_model(question: str) -> str:
    """Simulate a RAG system that answers based on retrieved context."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=f"Answer the question using ONLY the following context. Do not use outside knowledge.\n\nContext:\n{CONTEXT}",
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


cases = [
    EvalCase(
        input="What services does OmniTensorLabs offer?",
        context=CONTEXT,
        tags=["factual"],
    ),
    EvalCase(
        input="When was OmniTensorLabs founded?",
        context=CONTEXT,
        tags=["factual"],
    ),
    EvalCase(
        input="How many employees does OmniTensorLabs have?",
        context=CONTEXT,
        tags=["factual"],
    ),
    EvalCase(
        input="What is OmniTensorLabs' pricing?",  # Not in context — should trigger hallucination
        context=CONTEXT,
        tags=["out-of-context"],
    ),
]

suite = EvalSuite("RAG Faithfulness Eval", model_id="claude-haiku + RAG")
suite.add_cases(cases)
suite.add_evaluators(
    NotEmpty(),
    Relevance(threshold=0.6),
    Faithfulness(threshold=0.7),
    Hallucination(threshold=0.7),
)

report = suite.run(my_model := rag_model)
report.save_json("rag_eval_results.json")
report.save_csv("rag_eval_results.csv")
print("\nSaved to rag_eval_results.json and rag_eval_results.csv")
