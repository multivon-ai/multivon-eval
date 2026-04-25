from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool/function call made by an agent."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass
class AgentStep:
    """One step in an agent's execution trace."""
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    output: str = ""


@dataclass
class EvalCase:
    """
    A single test case for evaluation.

    Attributes:
        input:                The prompt, question, or user message.
        expected_output:      Ideal response (used by ExactMatch, AnswerAccuracy).
        context:              Retrieved documents or context (used by Faithfulness, Hallucination).
        conversation:         Multi-turn message history for conversation evaluators.
                              Format: [{"role": "user"/"assistant", "content": "..."}]
        agent_trace:          Sequence of agent steps for agent evaluators.
        expected_tool_calls:  Ordered list of tool names the agent should call.
        metadata:             Arbitrary key-value data (e.g. source_id, difficulty).
        tags:                 Labels for filtering reports.
    """
    input: str
    expected_output: str | None = None
    context: str | list[str] | None = None
    conversation: list[dict[str, str]] | None = None
    agent_trace: list[AgentStep] | None = None
    expected_tool_calls: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def context_str(self) -> str:
        if self.context is None:
            return ""
        if isinstance(self.context, list):
            return "\n\n".join(self.context)
        return self.context

    def conversation_str(self) -> str:
        if not self.conversation:
            return ""
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.conversation)
