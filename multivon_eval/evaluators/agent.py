"""
Agent evaluators — evaluate multi-step AI agent execution traces.

These evaluators operate on AgentStep traces attached to EvalCase,
not just the final output string. This is the key differentiator:
framework-agnostic evaluation of tool use, planning, and task completion.
"""
from __future__ import annotations
import json
import re

from .base import Evaluator
from .llm_judge import _judge_call, _call as _judge_call_with, _parse_yes_no, _qag_eval
from ..case import EvalCase, AgentStep
from ..exceptions import JudgeUnavailable
from ..judge import JudgeConfig, resolve_judge
from ..result import EvalResult


def _trace_str(trace: list[AgentStep]) -> str:
    """Render an agent trace as readable text for the judge."""
    lines = []
    for i, step in enumerate(trace, 1):
        lines.append(f"Step {i}:")
        if step.thought:
            lines.append(f"  Thought: {step.thought}")
        for tc in step.tool_calls:
            args = json.dumps(tc.arguments, indent=2) if tc.arguments else "{}"
            result_str = str(tc.result)[:200] if tc.result is not None else "(no result)"
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
    - Were they called in the correct order (if ``require_order``)?
    - Were any unexpected tools called?

    Scoring (codex D16 cycle 5 finding):

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

        # Real expectation but no trace — we can't compute anything meaningful.
        if not case.agent_trace:
            return self._skipped(
                "Requires case.agent_trace to evaluate expected_tool_calls.",
            )

        if self.require_order:
            # Ordered match
            matches = sum(1 for a, e in zip(actual_calls, expected) if a == e)
            score = matches / len(expected)
            missing = [e for e in expected if e not in actual_calls]
            unexpected = [a for a in actual_calls if a not in expected]
        else:
            # Unordered: fraction of expected tools that were called
            called_set = set(actual_calls)
            expected_set = set(expected)
            matched = called_set & expected_set
            score = len(matched) / len(expected_set)
            missing = list(expected_set - called_set)
            unexpected = list(called_set - expected_set)

        # Strict mode: penalize every unexpected tool by recomputing
        # score = matched / (expected ∪ unexpected). Lets negative
        # cases like "agent must NOT call refund_order" actually fail.
        if self.penalize_unexpected and unexpected:
            denom = len(set(expected) | set(actual_calls))
            if denom > 0:
                # Count matches the same way each branch does above.
                if self.require_order:
                    score = matches / denom
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

    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold)

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.agent_trace:
            return self._skipped(
                "Requires case.agent_trace — no tool calls to inspect.",
            )

        all_tool_calls = [tc for step in case.agent_trace for tc in step.tool_calls]
        if not all_tool_calls:
            return self._result(1.0, "No tool calls in trace")

        results, reasons = [], []
        for tc in all_tool_calls[:8]:
            args_str = json.dumps(tc.arguments, indent=2) if tc.arguments else "{}"
            prompt = (
                f"Task: {case.input}\n\n"
                f"Tool called: {tc.name}\n"
                f"Arguments provided:\n{args_str}\n\n"
                f"Are these arguments appropriate and well-formed for the tool '{tc.name}' given the task?"
                f"\nAnswer \"Yes\" or \"No\"."
            )
            try:
                answer = _judge_call(prompt, max_tokens=10)
                good = _parse_yes_no(answer)
                results.append(good)
                reasons.append(f"{'✓' if good else '✗'} {tc.name}({args_str[:60]})")
            except JudgeUnavailable:
                raise
            except Exception as e:
                results.append(False)
                reasons.append(f"✗ {tc.name} (error: {e})")

        score = sum(results) / len(results) if results else 0.0
        return self._result(score, "\n".join(reasons))


class PlanQuality(Evaluator):
    """
    Evaluates whether the agent's plan is logical, complete, and efficient.
    Looks at the sequence of steps and tool calls as a whole.
    Requires case.agent_trace.
    """
    name = "plan_quality"

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge

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

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge

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

    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold)

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.agent_trace:
            # No trace + no claimed tool calls = "agent didn't call any tools",
            # which is the correct, non-redundant outcome for many cases
            # (e.g. trivial questions). Treat as PASS, not as a missing-data
            # failure. Distinguishes from "user didn't supply trace at all" by
            # the absence of expected_tool_calls — covered by accuracy evaluator.
            return self._result(1.0, "No agent trace and no tool calls — nothing to flag as redundant.")

        all_tool_calls = [tc for step in case.agent_trace for tc in step.tool_calls]
        if not all_tool_calls:
            return self._result(1.0, "No tool calls — nothing to evaluate")

        results, reasons = [], []
        prior_calls = []

        for tc in all_tool_calls[:8]:
            args_str = json.dumps(tc.arguments, indent=2) if tc.arguments else "{}"
            prior_str = "\n".join(
                f"- {p.name}({json.dumps(p.arguments) if p.arguments else '{}'})"
                for p in prior_calls
            ) or "(none yet)"
            prompt = (
                f"Task: {case.input}\n\n"
                f"Prior tool calls already made:\n{prior_str}\n\n"
                f"Current tool call: {tc.name}({args_str})\n\n"
                f"Is this tool call strictly necessary to complete the task, "
                f"or is it redundant/unnecessary given what was already done?"
                f"\nAnswer \"Yes\" (necessary) or \"No\" (redundant/unnecessary)."
            )
            try:
                answer = _judge_call(prompt, max_tokens=10)
                needed = _parse_yes_no(answer)
                results.append(needed)
                reasons.append(f"{'✓' if needed else '✗ redundant'} {tc.name}({args_str[:50]})")
            except JudgeUnavailable:
                raise
            except Exception as e:
                results.append(True)
                reasons.append(f"? {tc.name} (error: {e})")
            prior_calls.append(tc)

        score = sum(results) / len(results) if results else 1.0
        return self._result(score, "\n".join(reasons))


class TrajectoryEfficiency(Evaluator):
    """
    Evaluates how efficiently the agent completed the task — did it take the
    optimal number of steps, or did it meander, repeat, or over-engineer?

    Compares actual step count against an LLM-estimated optimal.
    Also scores error recovery: if a tool failed, did the agent respond correctly?
    Requires case.agent_trace.
    """
    name = "trajectory_efficiency"

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge

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

        # Bonus: error recovery — if there were failed tool calls, did the agent handle them?
        if failed_tools:
            recovery_prompt = (
                f"Task: {case.input}\n\nTrace:\n{trace}\n\n"
                f"{len(failed_tools)} tool call(s) returned errors. "
                f"Did the agent appropriately detect and recover from these failures "
                f"(e.g., retried, used an alternative, or reported the failure clearly)?"
                f"\nAnswer \"Yes\" or \"No\"."
            )
            try:
                # Use the same resolved judge as the QAG above so a caller's
                # `judge=` is honored consistently across both scoring paths.
                answer = _judge_call_with(recovery_prompt, judge, max_tokens=10)
                recovered = _parse_yes_no(answer)
                if not recovered:
                    score = max(0.0, score - 0.2)
                    reasons.append(f"✗ Did not recover well from {len(failed_tools)} tool failure(s)")
                else:
                    reasons.append(f"✓ Recovered from {len(failed_tools)} tool failure(s)")
            except JudgeUnavailable:
                raise
            except Exception:
                pass

        return self._result(score, "\n".join(reasons))


class AgentMemoryEval(Evaluator):
    """
    Evaluates memory quality in multi-session agents.

    Tests whether the agent correctly uses context from a prior session
    (provided via case.context) — including accurate retrieval, avoiding
    stale information, and not hallucinating non-existent prior context.

    Aligned with AMA-Bench (2025): tests retrieval accuracy, test-time learning,
    long-range understanding, and selective forgetting.

    Requires:
      - case.context: summary or log of prior session(s)
      - case.input: current session query that requires memory
      - case.expected_output (optional): expected recalled information
    """
    name = "agent_memory"

    def __init__(self, threshold: float = 0.7, judge: JudgeConfig | None = None):
        super().__init__(threshold)
        self._judge_cfg = judge

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.context:
            return self._skipped("Requires case.context — supply prior session context to enable AgentMemoryEval.")

        ctx = (
            f"Prior session context:\n{case.context}\n\n"
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
                (f"Does the response include: \"{case.expected_output[:200]}\"?", True)
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

    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold)

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        if not case.agent_trace:
            return self._skipped("Requires case.agent_trace — no execution trace to score.")

        results, reasons = [], []
        for i, step in enumerate(case.agent_trace[:8], 1):
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
            try:
                answer = _judge_call(prompt, max_tokens=10)
                faithful = _parse_yes_no(answer)
                results.append(faithful)
                thought_preview = step.thought[:60] if step.thought else "(no thought)"
                reasons.append(f"{'✓' if faithful else '✗'} Step {i}: {thought_preview}")
            except JudgeUnavailable:
                raise
            except Exception as e:
                results.append(False)
                reasons.append(f"✗ Step {i} (error: {e})")

        score = sum(results) / len(results) if results else 0.0
        return self._result(score, f"{sum(results)}/{len(results)} steps faithful\n" + "\n".join(reasons))
