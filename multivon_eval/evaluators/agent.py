"""
Agent evaluators — evaluate multi-step AI agent execution traces.

These evaluators operate on AgentStep traces attached to EvalCase,
alongside the final output string. They assess supplied trace evidence;
external task outcomes require separate measurements.
"""
from __future__ import annotations

import json

from ..case import AgentStep, EvalCase
from ..judge import JudgeConfig, resolve_judge
from ..result import EvalResult
from .agent_judgments import PROTOCOL, capture_judgments, judge_binary, strict_qag
from .base import Evaluator
from .llm_judge import _call as _judge_call_with


def _judge_call(prompt, max_tokens=100, judge=None):
    return _judge_call_with(prompt, resolve_judge(judge), max_tokens=max_tokens)


def _qag_eval(questions, context, judge):
    return strict_qag(questions, context,
                      lambda prompt: _judge_call_with(prompt, judge, max_tokens=100))


def _check_limit(evaluator, count):
    if count > evaluator.max_items:
        result = evaluator._skipped(
            f"Trace has {count} items, exceeding max_items={evaluator.max_items}; no prefix was graded.")
        result.metadata.update(requested_items=count, measured_items=0, max_items=evaluator.max_items)
        return result
    return None


def _trace_str(trace: list[AgentStep]) -> str:
    """Render an agent trace as readable text for the judge."""
    lines = []
    for i, step in enumerate(trace, 1):
        lines.append(f"Step {i}:")
        if step.thought:
            lines.append(f"  Thought: {step.thought}")
        for tc in step.tool_calls:
            args = json.dumps(tc.arguments, indent=2, allow_nan=False) if tc.arguments else "{}"
            result_str = json.dumps(tc.result, ensure_ascii=False, allow_nan=False) if tc.result is not None else "(no result)"
            lines.append(f"  Tool call: {tc.name}({args})")
            lines.append(f"  Result: {result_str}")
        if step.output:
            lines.append(f"  Output: {step.output}")
    return "\n".join(lines)


class ToolCallAccuracy(Evaluator):
    """
    Evaluates whether the agent called the right tools.

    Checks:
    - Were all expected tools called?
    - Were they called in the correct order (if ``require_order``)? Order is
      checked as a subsequence: extra calls interleaved among the expected
      ones do not break the match, a reordering does.
    - Were any unexpected tools called?

    Scoring:

      - ``penalize_unexpected=False`` (default, backward-compat): score
        is the fraction of expected tools called; unexpected tools are
        reported but don't drop the score. This lets a case assert
        "the agent must call lookup_order" without forbidding other
        tools.
      - ``penalize_unexpected=True``: score = matched / (expected ∪ unexpected).
        Every unexpected tool drags the score down. Use this for
        negative cases like "the agent must NOT call refund_order
        on an already-refunded order."

    Requires case.agent_trace and case.expected_tool_calls.
    """
    name = "tool_call_accuracy"

    def __init__(
        self,
        require_order: bool = False,
        threshold: float = 0.7,
        *,
        penalize_unexpected: bool = False,
    ):
        super().__init__(threshold)
        self.require_order = require_order
        self.penalize_unexpected = penalize_unexpected

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        # Distinguish the three cases sharply:
        #   1. expected_tool_calls=None    → user didn't assert anything; skip.
        #   2. expected_tool_calls=[]      → user explicitly says "no tools";
        #                                    if trace has no tool calls too,
        #                                    that's a PASS, not a skip.
        #   3. expected_tool_calls=[...]   → real expectation; need a trace.
        if case.expected_tool_calls is None:
            return self._skipped(
                "Requires case.expected_tool_calls — set it (or [] to assert no tools) to enable ToolCallAccuracy.",
            )

        if case.agent_trace is None:
            return self._skipped("Requires a measured agent_trace; use [] for an observed empty trace.")

        actual_calls = [
            tc.name
            for step in (case.agent_trace or [])
            for tc in step.tool_calls
        ]
        expected = case.expected_tool_calls

        # Explicit "no tools should be called" assertion.
        if expected == []:
            if not actual_calls:
                return self._result(1.0, "Correctly called no tools (expected: [])")
            # Tools were called when none were expected — that's a real failure
            # the user wants to see, even when penalize_unexpected=False.
            return self._result(
                0.0,
                f"Unexpected tool calls — expected none, got: {actual_calls}",
            )

        if self.require_order:
            # Ordered match is a subsequence test, not a positional one: the
            # agent is allowed to take extra steps around the expected ones.
            # Positional zip would score a correct-order run with one extra
            # leading call the same 0.0 as a completely reversed one.
            remaining = list(expected)
            consumed = []
            for a in actual_calls:
                if remaining and a == remaining[0]:
                    remaining.pop(0)
                    consumed.append(a)
            matches = len(consumed)
            score = matches / len(expected)
            missing = remaining
            unexpected = [a for a in actual_calls if a not in expected]
        else:
            # Unordered: fraction of expected tools that were called
            called_set = set(actual_calls)
            expected_set = set(expected)
            matched = called_set & expected_set
            score = len(matched) / len(expected_set)
            missing = list(expected_set - called_set)
            unexpected = list(called_set - expected_set)

        # Strict mode: recompute score = matched / (expected ∪ unexpected).
        if self.penalize_unexpected and unexpected:
            denom = len(set(expected) | set(actual_calls))
            # Count matches the same way each branch does above.
            if self.require_order:
                score = matches / (len(expected) + len(unexpected))
            else:
                score = len(matched) / denom

        reasons = [f"Called: {actual_calls}", f"Expected: {expected}"]
        if missing:
            reasons.append(f"Missing tools: {missing}")
        if unexpected:
            reasons.append(f"Unexpected tools: {unexpected}")
        if self.penalize_unexpected:
            reasons.append("(strict mode: unexpected tools penalized)")

        return self._result(score, "\n".join(reasons))


class ToolArgumentAccuracy(Evaluator):
    """
    Evaluates whether tool arguments were correct and well-formed.
    Uses LLM judge to assess argument quality since exact matching is too rigid.
    Requires case.agent_trace.
    """
    name = "tool_argument_accuracy"
    uses_llm_judge = True

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None, *, max_items: int = 8):
        super().__init__(threshold)
        if type(max_items) is not int or max_items < 1:
            raise ValueError("max_items must be a positive integer")
        self.max_items = max_items
        self._judge_cfg = judge
        self.protocol = PROTOCOL

    @capture_judgments
    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.agent_trace:
            return self._skipped(
                "Requires case.agent_trace — no tool calls to inspect.",
            )

        all_tool_calls = [tc for step in case.agent_trace for tc in step.tool_calls]
        if not all_tool_calls:
            return self._skipped("No tool calls in trace; argument quality is undefined.")

        results, reasons = [], []
        limited = _check_limit(self, len(all_tool_calls))
        if limited is not None:
            return limited
        for tc in all_tool_calls:
            args_str = json.dumps(tc.arguments, indent=2, allow_nan=False) if tc.arguments else "{}"
            prompt = (
                f"Task: {case.input}\n\n"
                f"Tool called: {tc.name}\n"
                f"Arguments provided:\n{args_str}\n\n"
                f"Are these arguments appropriate and well-formed for the tool '{tc.name}' given the task?"
                f"\nAnswer \"Yes\" or \"No\"."
            )
            good = judge_binary(prompt, lambda p: _judge_call(p, max_tokens=100, judge=self._judge_cfg))
            results.append(good)
            reasons.append(f"{'✓' if good else '✗'} {tc.name}({args_str[:60]})")

        score = sum(results) / len(results)
        return self._result(score, "\n".join(reasons))


class PlanQuality(Evaluator):
    """
    Evaluates whether the agent's plan is logical, complete, and efficient.
    Looks at the sequence of steps and tool calls as a whole.
    Requires case.agent_trace.
    """
    name = "plan_quality"
    uses_llm_judge = True

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge
        self.protocol = PROTOCOL

    @capture_judgments
    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.agent_trace:
            return self._skipped("Requires case.agent_trace — no execution trace to score.")

        trace = _trace_str(case.agent_trace)
        ctx = f"Task: {case.input}\n\nAgent execution trace:\n{trace}\n\nFinal output: {output}"
        questions = [
            ("Does the agent's plan address all aspects of the task?", True),
            ("Are the steps in the agent's plan in a logical order?", True),
            ("Does the agent avoid redundant or unnecessary steps?", True),
            ("Does each step in the plan follow logically from the previous one?", True),
            ("Would an expert consider this plan efficient for the task?", True),
        ]
        judge = resolve_judge(self._judge_cfg)
        score, reasons = _qag_eval(questions, ctx, judge)
        return self._result(score, "\n".join(reasons))


class TaskCompletion(Evaluator):
    """
    Evaluates whether the agent successfully completed the given task.
    Assesses the final output against the task goal — not just the process.
    """
    name = "task_completion"
    uses_llm_judge = True

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge
        self.protocol = PROTOCOL

    @capture_judgments
    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        trace_str = ""
        if case.agent_trace:
            trace_str = f"\n\nAgent trace summary:\n{_trace_str(case.agent_trace)}"

        ctx = f"Task: {case.input}{trace_str}\n\nFinal output: {output}"
        questions = [
            ("Does the final output successfully complete the task?", True),
            ("Does the final output address all requirements of the task?", True),
            ("Is the final output a complete response (not partial or cut off)?", True),
            ("Did the agent fail to complete the task or produce an error?", False),
        ]
        judge = resolve_judge(self._judge_cfg)
        score, reasons = _qag_eval(questions, ctx, judge)
        return self._result(score, "\n".join(reasons))


class ToolCallNecessity(Evaluator):
    """
    Evaluates whether each tool call was actually needed given the task and context.

    Detects redundant or spurious tool use — agents that call tools "just in case"
    or repeat calls they already made. Low scores here mean inefficient, noisy agents.
    Requires case.agent_trace.
    """
    name = "tool_call_necessity"
    uses_llm_judge = True

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None, *, max_items: int = 8):
        super().__init__(threshold)
        if type(max_items) is not int or max_items < 1:
            raise ValueError("max_items must be a positive integer")
        self.max_items = max_items
        self._judge_cfg = judge
        self.protocol = PROTOCOL

    @capture_judgments
    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if case.agent_trace is None:
            return self._skipped("Requires a measured agent_trace; use [] for an observed empty trace.")

        all_tool_calls = [tc for step in case.agent_trace for tc in step.tool_calls]
        if not all_tool_calls:
            return self._skipped("No tool calls in the observed trace; necessity is undefined.")

        results, reasons = [], []
        prior_calls = []

        limited = _check_limit(self, len(all_tool_calls))
        if limited is not None:
            return limited
        for tc in all_tool_calls:
            args_str = json.dumps(tc.arguments, indent=2, allow_nan=False) if tc.arguments else "{}"
            prior_str = _trace_str([AgentStep(tool_calls=prior_calls)]) if prior_calls else "(none yet)"
            prompt = (
                f"Task: {case.input}\n\n"
                f"Prior tool calls already made:\n{prior_str}\n\n"
                f"Current tool call: {tc.name}({args_str})\n\n"
                f"Is this tool call strictly necessary to complete the task, "
                f"or is it redundant/unnecessary given what was already done?"
                f"\nAnswer \"Yes\" (necessary) or \"No\" (redundant/unnecessary)."
            )
            needed = judge_binary(prompt, lambda p: _judge_call(p, max_tokens=100, judge=self._judge_cfg))
            results.append(needed)
            reasons.append(f"{'✓' if needed else '✗'} {tc.name}({args_str[:50]})")
            prior_calls.append(tc)

        score = sum(results) / len(results)
        return self._result(score, "\n".join(reasons))


class TrajectoryEfficiency(Evaluator):
    """
    Judge heuristic for redundant steps and recovery; does not measure an optimum.
    Recovery screening looks for "error" in tool results, which can misidentify failures.
    Requires case.agent_trace.
    """
    name = "trajectory_efficiency"
    uses_llm_judge = True

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge
        self.protocol = PROTOCOL

    @capture_judgments
    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.agent_trace:
            return self._skipped(
                "Requires case.agent_trace — no execution trace to score for efficiency.",
            )

        trace = _trace_str(case.agent_trace)
        step_count = len(case.agent_trace)
        tool_count = sum(len(s.tool_calls) for s in case.agent_trace)
        failed_tools = [
            tc for step in case.agent_trace for tc in step.tool_calls
            if tc.result is not None and "error" in str(tc.result).lower()
        ]

        ctx = (
            f"Task: {case.input}\n\n"
            f"Agent execution trace ({step_count} steps, {tool_count} tool calls):\n{trace}\n\n"
            f"Final output: {output}"
        )
        questions = [
            ("Did the agent complete the task without unnecessary detours or repeated steps?", True),
            ("Is the number of steps taken proportionate to the task complexity?", True),
            ("Did the agent avoid making the same tool call more than once with identical arguments?", True),
        ]

        judge = resolve_judge(self._judge_cfg)
        score, reasons = _qag_eval(questions, ctx, judge)

        # This text heuristic is not a structured tool-error observation.
        if failed_tools:
            recovery_prompt = (
                f"Task: {case.input}\n\nTrace:\n{trace}\n\n"
                f"{len(failed_tools)} tool result(s) contain the word error; these are suspected failures. "
                f"Did the agent appropriately detect and recover from these failures "
                f"(e.g., retried, used an alternative, or reported the failure clearly)?"
                f"\nAnswer \"Yes\" or \"No\"."
            )
            recovered = judge_binary(
                recovery_prompt, lambda p: _judge_call_with(p, judge, max_tokens=100))
            if not recovered:
                score = max(0.0, score - 0.2)
                reasons.append(f"✗ Did not recover well from {len(failed_tools)} suspected tool failure(s)")
            else:
                reasons.append(f"✓ Recovered from {len(failed_tools)} suspected tool failure(s)")

        return self._result(score, "\n".join(reasons))


class AgentMemoryEval(Evaluator):
    """
    Evaluates memory quality in multi-session agents.

    Tests whether the agent correctly uses context from a prior session
    (provided via case.context) — including accurate retrieval, avoiding
    stale information, and not hallucinating non-existent prior context.

    A context-use heuristic, not an implementation of a memory benchmark.

    Requires:
      - case.context: summary or log of prior session(s)
      - case.input: current session query that requires memory
      - case.expected_output (optional): expected recalled information
    """
    name = "agent_memory"
    uses_llm_judge = True

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge
        self.protocol = PROTOCOL

    @capture_judgments
    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.context:
            return self._skipped("Requires case.context — supply prior session context to enable AgentMemoryEval.")

        ctx = (
            f"Prior session context:\n{case.context_str}\n\n"
            f"Current query: {case.input}\n\n"
            f"Agent response: {output}"
        )
        questions = [
            ("Does the agent's response correctly use information from the prior context?", True),
            ("Does the agent avoid hallucinating facts not present in the prior context?", True),
            ("Does the agent correctly ignore stale or superseded information from prior context?", True),
        ]
        if case.expected_output:
            questions.append(
                (f"Does the response include: \"{case.expected_output}\"?", True)
            )

        judge = resolve_judge(self._judge_cfg)
        score, reasons = _qag_eval(questions, ctx, judge)
        return self._result(score, "\n".join(reasons))


class StepFaithfulness(Evaluator):
    """
    Evaluates whether each agent step faithfully follows from the task and prior steps.
    Detects hallucinated reasoning or steps that contradict the task.
    Requires case.agent_trace.
    """
    name = "step_faithfulness"
    uses_llm_judge = True

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None, *, max_items: int = 8):
        super().__init__(threshold)
        if type(max_items) is not int or max_items < 1:
            raise ValueError("max_items must be a positive integer")
        self.max_items = max_items
        self._judge_cfg = judge
        self.protocol = PROTOCOL

    @capture_judgments
    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.agent_trace:
            return self._skipped("Requires case.agent_trace — no execution trace to score.")

        results, reasons = [], []
        limited = _check_limit(self, len(case.agent_trace))
        if limited is not None:
            return limited
        for i, step in enumerate(case.agent_trace, 1):
            prior = _trace_str(case.agent_trace[:i-1]) if i > 1 else "(no prior steps)"
            step_str = _trace_str([step])
            prompt = (
                f"Task: {case.input}\n\n"
                f"Prior steps:\n{prior}\n\n"
                f"Current step {i}:\n{step_str}\n\n"
                f"Does this step follow logically from the task and prior steps, "
                f"without introducing contradictions or hallucinated information?"
                f"\nAnswer \"Yes\" or \"No\"."
            )
            faithful = judge_binary(prompt, lambda p: _judge_call(p, max_tokens=100, judge=self._judge_cfg))
            results.append(faithful)
            reasons.append(f"{'✓' if faithful else '✗'} Step {i}")

        score = sum(results) / len(results)
        return self._result(score, f"{sum(results)}/{len(results)} steps faithful\n" + "\n".join(reasons))
