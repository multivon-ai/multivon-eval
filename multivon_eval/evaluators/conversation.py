"""
Conversation evaluators — evaluate multi-turn dialogue quality.

These evaluators assess the quality of conversational AI systems
over the full dialogue, not just a single response.
"""
from __future__ import annotations

from .base import Evaluator
from .llm_judge import _judge_call, _parse_yes_no, _qag_eval
from ..case import EvalCase
from ..judge import JudgeConfig, resolve_judge
from ..result import EvalResult


class ConversationRelevance(Evaluator):
    """
    Evaluates whether each assistant turn is relevant to the ongoing conversation.
    Detects responses that ignore prior context or change subject unexpectedly.
    Requires case.conversation.
    """
    name = "conversation_relevance"

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.conversation:
            return self._skipped("Requires case.conversation — add a multi-turn dialog to enable this evaluator.")

        ctx = f"Conversation history:\n{case.conversation_str()}\n\nLatest response: {output}"
        questions = [
            ("Is the latest response relevant to the conversation history?", True),
            ("Does the response address what the user was asking or discussing?", True),
            ("Does the response ignore important context from earlier in the conversation?", False),
            ("Does the response follow naturally from the preceding turns?", True),
        ]
        judge = resolve_judge(self._judge_cfg)
        score, reasons = _qag_eval(questions, ctx, judge)
        return self._result(score, "\n".join(reasons))


class KnowledgeRetention(Evaluator):
    """
    Evaluates whether the model retains and correctly uses facts established
    earlier in the conversation.

    Requires case.conversation — checks that the final response (output)
    is consistent with facts introduced in earlier turns.
    """
    name = "knowledge_retention"

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.conversation:
            return self._skipped("Requires case.conversation — add a multi-turn dialog to enable this evaluator.")

        # Extract facts from user turns earlier in conversation
        user_turns = [m["content"] for m in case.conversation if m["role"] == "user"]
        if not user_turns:
            return self._result(1.0, "No user turns to retain facts from")

        ctx = (
            f"Conversation history:\n{case.conversation_str()}\n\n"
            f"Latest response:\n{output}"
        )
        questions = [
            ("Does the response correctly recall facts mentioned by the user earlier in the conversation?", True),
            ("Does the response contradict information the user provided in a prior turn?", False),
            ("Does the response show awareness of the user's preferences or context established earlier?", True),
        ]
        judge = resolve_judge(self._judge_cfg)
        score, reasons = _qag_eval(questions, ctx, judge)
        return self._result(score, "\n".join(reasons))


class ConversationCompleteness(Evaluator):
    """
    Evaluates whether the assistant fully resolved the user's goals
    over the course of the conversation.

    Assesses the final output as the culmination of a multi-turn dialogue.
    Requires case.conversation.
    """
    name = "conversation_completeness"

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.conversation:
            return self._skipped("Requires case.conversation — add a multi-turn dialog to enable this evaluator.")

        # Infer the user's original goal from first user turn
        first_user = next(
            (m["content"] for m in case.conversation if m["role"] == "user"), case.input
        )
        ctx = (
            f"Original user goal: {first_user}\n\n"
            f"Full conversation:\n{case.conversation_str()}\n\n"
            f"Final response:\n{output}"
        )
        questions = [
            ("By the end of the conversation, has the user's original goal been fully addressed?", True),
            ("Has the assistant left important questions from the user unanswered?", False),
            ("Does the final response bring the conversation to a satisfying resolution?", True),
            ("Would the user need to ask follow-up questions to get what they originally wanted?", False),
        ]
        judge = resolve_judge(self._judge_cfg)
        score, reasons = _qag_eval(questions, ctx, judge)
        return self._result(score, "\n".join(reasons))


class TurnConsistency(Evaluator):
    """
    Checks that the assistant doesn't contradict itself across turns.
    Requires case.conversation.
    """
    name = "turn_consistency"

    def __init__(self, threshold: float = 0.8, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.conversation:
            return self._skipped("Requires case.conversation — add a multi-turn dialog to enable this evaluator.")

        ctx = (
            f"Conversation:\n{case.conversation_str()}\n\n"
            f"Latest response:\n{output}"
        )
        questions = [
            ("Is the latest response consistent with all prior assistant responses in the conversation?", True),
            ("Does the assistant contradict something it said in a previous turn?", False),
            ("Does the assistant maintain a consistent persona and tone throughout?", True),
        ]
        judge = resolve_judge(self._judge_cfg)
        score, reasons = _qag_eval(questions, ctx, judge)
        return self._result(score, "\n".join(reasons))
