"""
RAG eval — faithfulness and hallucination for a retrieval-augmented system.
The most common production use case: ensure answers stay grounded in retrieved context.
"""
from dotenv import load_dotenv
load_dotenv()

import anthropic
from multivon_eval import EvalSuite, EvalCase, Faithfulness, Hallucination, Relevance, NotEmpty

client = anthropic.Anthropic()

CONTEXT = """
Acme Corp return policy: Items can be returned within 30 days of purchase for a full refund.
Items must be unused and in original packaging. Electronics have a 14-day return window.
Sale items are final — no returns or exchanges. Refunds are processed within 5–7 business days
to the original payment method. To start a return, visit acme.com/returns or call 1-800-555-0100.
"""


def rag_model(question: str) -> str:
    """Simulate a RAG system that answers based on retrieved context."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=(
            "Answer the question using ONLY the following context. "
            "If the answer is not in the context, say so explicitly.\n\n"
            f"Context:\n{CONTEXT}"
        ),
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


cases = [
    EvalCase(
        input="What is the standard return window?",
        context=CONTEXT,
        tags=["factual"],
    ),
    EvalCase(
        input="Can I return electronics after 20 days?",
        context=CONTEXT,
        tags=["factual", "edge-case"],
    ),
    EvalCase(
        input="How long do refunds take to process?",
        context=CONTEXT,
        tags=["factual"],
    ),
    EvalCase(
        input="What is the return policy for sale items?",
        context=CONTEXT,
        tags=["factual"],
    ),
    EvalCase(
        input="What is Acme Corp's revenue last quarter?",  # not in context — should admit it
        context=CONTEXT,
        tags=["out-of-context"],
    ),
]

suite = EvalSuite("RAG Faithfulness Eval", model_id="claude-haiku + RAG")
suite.add_cases(cases)
suite.add_evaluators(
    NotEmpty(),
    Relevance(),
    Faithfulness(),
    Hallucination(),
)

if __name__ == "__main__":
    report = suite.run(rag_model)
    report.save_json("rag_eval_results.json")
    report.save_csv("rag_eval_results.csv")
    print("\nSaved to rag_eval_results.json and rag_eval_results.csv")
