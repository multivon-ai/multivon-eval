"""
Agent eval — tool call accuracy for a customer support agent with tools.

Uses ManualTracer to record each tool call, then evaluates:
  - ToolCallAccuracy: did the agent call the right tools?
  - TaskCompletion: did the agent resolve the customer's request?
  - add_check: plain-English quality criteria

Run with runs=3 to surface flaky cases (non-deterministic tool selection).
"""
from dotenv import load_dotenv
load_dotenv()

import anthropic
from multivon_eval import (
    EvalSuite, EvalCase, AgentStep, ToolCall,
    ManualTracer, ToolCallAccuracy, TaskCompletion, NotEmpty,
)

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "get_order_status",
        "description": "Look up the current status of an order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund for an order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "reason"],
        },
    },
    {
        "name": "send_followup_email",
        "description": "Send a follow-up email to the customer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

TOOL_RESULTS = {
    "get_order_status": {"status": "delivered", "delivered_at": "2025-04-28"},
    "issue_refund":     {"refund_id": "REF-001", "status": "approved"},
    "send_followup_email": {"message_id": "MSG-001", "status": "sent"},
}

tracer = ManualTracer()


def agent(prompt: str) -> str:
    """One-turn agent: calls tools, records trace, returns final response."""
    messages = [
        {
            "role": "user",
            "content": (
                "You are a customer support agent. Use the available tools.\n\n"
                f"Customer: {prompt}"
            ),
        }
    ]

    # Agentic loop: keep calling until no more tool calls
    with tracer.step(thought="Processing customer request") as step:
        while True:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                tools=TOOLS,
                messages=messages,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                # Final text response
                text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                step.set_output(text)
                return text

            # Execute tool calls (simulated results here; swap in real calls)
            tool_results = []
            for tu in tool_uses:
                result = TOOL_RESULTS.get(tu.name, {"error": "unknown tool"})
                step.record_tool_call(tu.name, dict(tu.input), result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": str(result),
                })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})


cases = [
    EvalCase(
        input="My order #A123 hasn't arrived. Please check and issue a refund.",
        expected_tool_calls=["get_order_status", "issue_refund"],
        tags=["refund"],
    ),
    EvalCase(
        input="What is the status of order #B456?",
        expected_tool_calls=["get_order_status"],
        tags=["status"],
    ),
    EvalCase(
        input="I want a refund for order #C789. The item was broken.",
        expected_tool_calls=["get_order_status", "issue_refund"],
        tags=["refund"],
    ),
]

suite = EvalSuite("Customer Support Agent Eval", model_id="claude-haiku + tools")
suite.add_cases(cases)
suite.add_evaluators(
    NotEmpty(),
    ToolCallAccuracy(),
    TaskCompletion(),
)
suite.add_check("Response should acknowledge the customer's specific issue")
suite.add_check("Response should confirm what action was taken")

if __name__ == "__main__":
    # runs=3 surfaces non-deterministic tool selection (flaky cases)
    report = suite.run(agent, tracer=tracer, runs=3)
    report.save_json("agent_eval_results.json")
    print("\nSaved to agent_eval_results.json")
