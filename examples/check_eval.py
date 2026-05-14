"""
Plain-English checks — write criteria in English, not evaluator classes.

add_check() generates specific yes/no questions from your criterion,
runs QAG scoring, and fails cases that don't meet the bar.

Best for: product engineers who want eval coverage without learning
the full evaluator API. Start here, graduate to CustomRubric when
you need fine-grained control over the questions.
"""
from dotenv import load_dotenv
load_dotenv()

import logging
import anthropic
from multivon_eval import EvalSuite, EvalCase

# Show generated questions in the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

client = anthropic.Anthropic()

SYSTEM = """You are a customer support agent for Acme Corp.
Be empathetic, specific, and always provide a clear next step.
Keep responses under 100 words."""


def support_bot(question: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


cases = [
    EvalCase(input="My order hasn't arrived after 2 weeks.", tags=["shipping"]),
    EvalCase(input="I was charged twice for the same order.", tags=["billing"]),
    EvalCase(input="The product I received is damaged.", tags=["returns"]),
    EvalCase(input="How do I reset my account password?", tags=["account"]),
    EvalCase(input="I want to cancel my subscription.", tags=["billing"]),
]

suite = EvalSuite("Support Bot Check Eval", model_id="claude-haiku")
suite.add_cases(cases)

# Plain-English criteria — no evaluator knowledge required.
# Questions are generated once before the eval loop starts.
suite.add_check("Response should acknowledge the customer's specific problem")
suite.add_check("Response should provide a concrete next step or resolution")
suite.add_check("Response should be empathetic and avoid defensive language")
suite.add_check("Response should be concise (under 100 words)", threshold=0.8)

# Pin questions for reproducible CI — skip LLM generation entirely:
# suite.add_check(
#     "Response should provide a concrete next step",
#     questions=[
#         "Does the response name a specific action the customer should take?",
#         "Does the response include a link, phone number, or contact method?",
#         "Does the response avoid vague phrases like 'we'll look into it'?",
#     ],
# )

if __name__ == "__main__":
    report = suite.run(support_bot)
    report.save_json("check_eval_results.json")
    print("\nSaved to check_eval_results.json")
