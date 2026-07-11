"""``multivon-eval simulate`` — persona-driven adaptive multi-turn simulation.

Static multi-turn cases assume a fixed conversation path; the moment the
model responds differently, a scripted dialogue is meaningless. This module
*drives* a conversation instead: a persona-LLM (the same judge plumbing
:mod:`multivon_eval.discover` uses) generates the next user turn conditioned
on the persona + goal + transcript so far, the system under test replies,
and the loop stops on goal-reached, refusal, ``max_turns``, a driver error,
or the hard budget ceiling.

Honesty rules (non-negotiable, test-pinned):
  - Every result's case metadata carries ``"simulated": True`` and reports
    print: "simulated personas — measures behavior under synthetic users,
    not real traffic".
  - A spend estimate is printed up front; ``budget_usd`` is a HARD ceiling
    across all personas — when exceeded, partial results are returned with
    ``stop_reason="budget_exceeded"``, never an exception that loses
    completed transcripts.
  - Runs are STOCHASTIC. Persona proposal is seeded (the seed is folded
    into the prompt and recorded in metadata), but LLM turns vary between
    runs — nothing here claims run-to-run determinism. Result metadata
    records the judge model + configured temperature (provider adapters
    may clamp temperature; see ``discover._call_judge``).
  - A ``driver_error`` on one persona never kills the run — it is recorded
    and the simulator continues to the next persona.

Recorder synergy: each persona's conversation runs with its case identity
bound via :func:`multivon_eval.recorder.bind_case` (zero overhead when
recording is off), so ``--record-prompts`` during a simulation yields
observed case→site bindings for free — simulation *with provenance*.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .case import EvalCase
# Reuse the exact LLM-call + cost-estimation mechanism bootstrap uses.
from .discover import _call_judge, _estimate_cost, _estimate_cost_from_tokens
from .evaluators.llm_judge import _is_refusal
from .judge import JudgeConfig, resolve_judge
from .provenance import git_info, stamp_metadata_inplace
from .recorder import bind_case, unbind_case
from ._simulate_prompts import (
    _PROPOSE_SYSTEM, _VERDICT_SYSTEM,
    _parse_persona_turn, _parse_verdict,
    _persona_system_prompt, _persona_user_prompt,
    _render_conversation, _strip_fences,
    _verdict_user_prompt,
)

SIMULATED_DISCLAIMER = (
    "simulated personas — measures behavior under synthetic users, "
    "not real traffic"
)

#: Every stop_reason a SimulationResult may carry.
STOP_REASONS = (
    "goal_reached", "max_turns", "assistant_refused",
    "driver_error", "budget_exceeded",
)


# ─── Public types ─────────────────────────────────────────────────────────


@dataclass
class Persona:
    """A synthetic user the simulator role-plays against the system."""
    name: str
    profile: str            # who they are, 1-3 sentences
    goal: str               # what they're trying to accomplish
    success_criteria: str   # how a judge decides the goal was met
    traits: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """One persona's simulated conversation + verdict.

    ``stop_reason`` is one of :data:`STOP_REASONS` — ``budget_exceeded``
    marks personas cut short (or never started) by the hard budget ceiling,
    so partial runs stay explainable instead of raising.
    """
    persona: Persona
    transcript: list[dict[str, str]]   # [{"role": "user"|"assistant", "content": ...}]
    turns: int                         # number of assistant replies
    stop_reason: str
    goal_achieved: bool | None         # judge verdict; None if judging skipped/failed
    cost_usd: float
    case: EvalCase                     # conversation-shaped case for scoring


# ─── Budget tracking ──────────────────────────────────────────────────────


class _Spend:
    """Accumulates estimated judge spend against the hard ceiling."""

    def __init__(self, budget_usd: float):
        self.budget_usd = budget_usd
        self.total = 0.0

    def add(self, cost: float) -> None:
        self.total += cost

    def exceeded(self) -> bool:
        return self.total >= self.budget_usd


# ─── Driver loop ──────────────────────────────────────────────────────────


def _persona_turn(
    resolved: JudgeConfig, persona: Persona,
    transcript: list[dict[str, str]], turns: int, max_turns: int,
    seed: int, spend: _Spend,
) -> tuple[str, bool] | None:
    """One persona-LLM call (retried once on malformed output).

    Returns ``(message, goal_reached)`` or None after two malformed
    responses. Provider exceptions propagate (caller maps to driver_error).
    """
    system = _persona_system_prompt(persona, seed)
    user = _persona_user_prompt(persona, transcript, turns, max_turns)
    for _attempt in range(2):
        text = _call_judge(resolved, system, user)
        spend.add(_estimate_cost(resolved, system + user, text))
        parsed = _parse_persona_turn(text)
        if parsed is not None:
            return parsed
    return None


def _judge_goal(
    resolved: JudgeConfig, persona: Persona,
    transcript: list[dict[str, str]], spend: _Spend,
) -> bool | None:
    """Final goal-completion verdict. None when judging fails or the
    budget is already exhausted (skipped, never a crash)."""
    if not transcript or spend.exceeded():
        return None
    user = _verdict_user_prompt(persona, transcript)
    try:
        text = _call_judge(resolved, _VERDICT_SYSTEM, user)
    except Exception:
        return None
    spend.add(_estimate_cost(resolved, _VERDICT_SYSTEM + user, text))
    return _parse_verdict(text)


def _simulate_one(
    model_fn: Callable[[str], str],
    persona: Persona,
    *,
    resolved: JudgeConfig,
    max_turns: int,
    seed: int,
    spend: _Spend,
    git: dict[str, Any],
) -> SimulationResult:
    """Drive one persona's conversation. Never raises — every failure mode
    becomes a stop_reason on the returned result."""
    metadata: dict[str, Any] = {
        "simulated": True,
        "persona": persona.name,
        "persona_traits": list(persona.traits),
        "success_criteria": persona.success_criteria,
        "judge_provider": resolved.provider,
        "judge_model": resolved.model,
        "judge_temperature": resolved.temperature,
        "seed": seed,
        "stop_reason": None,  # filled after the loop
    }
    # Stamp BEFORE binding so bind_case picks up the case_uid; recordings
    # made by model_fn during this loop carry this case's identity.
    stamp_metadata_inplace(metadata, authored_by="simulator", git=git, targets=[])

    transcript: list[dict[str, str]] = []
    turns = 0
    stop_reason = "max_turns"
    cost_before = spend.total

    token = bind_case(metadata)  # None-check inside — zero overhead when off
    try:
        while turns < max_turns:
            if spend.exceeded():
                stop_reason = "budget_exceeded"
                break
            try:
                parsed = _persona_turn(
                    resolved, persona, transcript, turns, max_turns, seed, spend,
                )
            except Exception as exc:
                stop_reason = "driver_error"
                metadata["driver_error"] = f"{type(exc).__name__}: {exc}"
                break
            if parsed is None:
                stop_reason = "driver_error"
                metadata["driver_error"] = (
                    "persona LLM returned malformed JSON twice"
                )
                break
            message, goal_reached = parsed
            if goal_reached:
                stop_reason = "goal_reached"
                break
            transcript.append({"role": "user", "content": message})
            try:
                reply = model_fn(_render_conversation(transcript))
            except Exception as exc:
                stop_reason = "driver_error"
                metadata["driver_error"] = (
                    f"model_fn raised {type(exc).__name__}: {exc}"
                )
                break
            transcript.append({"role": "assistant", "content": str(reply)})
            turns += 1
            if _is_refusal(str(reply)):
                # Adversarial personas exist to probe PAST refusals — a
                # one-turn "I can't do that" ending the probe would
                # systematically undertest the system. They run to
                # goal/max_turns; refusals are recorded, not terminal.
                if any("adversarial" in t.lower() for t in persona.traits):
                    metadata.setdefault("refusals_observed", 0)
                    metadata["refusals_observed"] += 1
                else:
                    stop_reason = "assistant_refused"
                    break
    finally:
        unbind_case(token)

    metadata["stop_reason"] = stop_reason
    goal_achieved: bool | None = None
    if stop_reason != "driver_error":
        goal_achieved = _judge_goal(resolved, persona, transcript, spend)
    metadata["goal_achieved"] = goal_achieved

    case = EvalCase(input=persona.goal, conversation=transcript, metadata=metadata)
    return SimulationResult(
        persona=persona,
        transcript=transcript,
        turns=turns,
        stop_reason=stop_reason,
        goal_achieved=goal_achieved,
        cost_usd=round(spend.total - cost_before, 6),
        case=case,
    )


def simulate(
    model_fn: Callable[[str], str],
    personas: list[Persona],
    *,
    max_turns: int = 8,
    judge: JudgeConfig | None = None,
    seed: int = 0,
    budget_usd: float = 1.00,
    verbose: bool = True,
) -> list[SimulationResult]:
    """Run persona-driven simulations against ``model_fn``.

    ``model_fn`` has the same contract as ``EvalSuite.run`` — a callable
    taking one rendered-prompt string (the full conversation, rendered the
    way ``EvalCase.conversation_str()`` renders it) and returning the
    assistant's reply.

    Returns one :class:`SimulationResult` per persona, always — personas
    cut off (or never started) by the ``budget_usd`` hard ceiling carry
    ``stop_reason="budget_exceeded"`` with whatever transcript completed.

    Stochasticity: LLM turns vary between runs. ``seed`` is folded into
    the persona prompt and recorded in metadata, but this function makes
    NO determinism claim.
    """
    resolved = resolve_judge(judge)
    spend = _Spend(budget_usd)
    git = git_info(".")

    if verbose:
        est_calls = len(personas) * max_turns * 2  # persona turn + verdict headroom
        est_cost = _estimate_cost_from_tokens(
            resolved, input_tokens=1200, output_tokens=150) * est_calls
        cost_note = (f"≈${est_cost:.4f}" if est_cost
                     else "unknown (no cost model for this judge)")
        print(f"[simulate] {SIMULATED_DISCLAIMER}", file=sys.stderr)
        print(
            f"[simulate] spend estimate: {len(personas)} persona(s) × "
            f"{max_turns} turns × ~2 LLM calls = ~{est_calls} judge calls "
            f"{cost_note} · hard ceiling ${budget_usd:.2f}",
            file=sys.stderr,
        )

    results: list[SimulationResult] = []
    for persona in personas:
        result = _simulate_one(
            model_fn, persona, resolved=resolved, max_turns=max_turns,
            seed=seed, spend=spend, git=git,
        )
        results.append(result)
        if verbose:
            print(
                f"[simulate] {persona.name}: {result.turns} turn(s), "
                f"stop={result.stop_reason}, goal_achieved={result.goal_achieved}",
                file=sys.stderr,
            )
    return results


# ─── Persona sources ──────────────────────────────────────────────────────


_REQUIRED_PERSONA_FIELDS = ("name", "profile", "goal", "success_criteria")


def _persona_from_obj(obj: Any) -> Persona | None:
    """Build a Persona from a parsed dict; None when malformed."""
    if not isinstance(obj, dict):
        return None
    values: dict[str, str] = {}
    for key in _REQUIRED_PERSONA_FIELDS:
        v = obj.get(key)
        if not isinstance(v, str) or not v.strip():
            return None
        values[key] = v.strip()
    traits = obj.get("traits") or []
    if not isinstance(traits, list):
        traits = []
    traits = [str(t) for t in traits if isinstance(t, str) and t.strip()]
    return Persona(traits=traits, **values)


def personas_from_jsonl(path: Path | str) -> list[Persona]:
    """Load user-authored personas from a JSONL file.

    Each line needs ``name``, ``profile``, ``goal``, ``success_criteria``
    (strings) and optionally ``traits`` (list of strings). A malformed
    line raises ``ValueError`` with file:line — user-authored input fails
    loudly, never a silent skip.
    """
    p = Path(path)
    personas: list[Persona] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{i}: malformed JSONL ({exc})") from exc
        persona = _persona_from_obj(obj)
        if persona is None:
            raise ValueError(
                f"{p}:{i}: persona needs string fields "
                f"{_REQUIRED_PERSONA_FIELDS} (traits optional)"
            )
        personas.append(persona)
    return personas


def propose_personas(
    description: str,
    n: int = 5,
    judge: JudgeConfig | None = None,
    seed: int = 0,
) -> list[Persona]:
    """ONE LLM call proposing ``n`` diverse personas from a product description.

    Same shape as bootstrap's ``propose_evaluators_via_llm``: constrained
    JSON output, retry-once on parse failure, malformed entries rejected
    rather than crashing. Returns ``[]`` when both attempts fail.

    ``seed`` is folded into the prompt for variation control — proposal is
    seeded, not deterministic (the LLM still samples).
    """
    resolved = resolve_judge(judge)
    user = (
        f"PRODUCT DESCRIPTION:\n{description.strip()}\n\n"
        f"Propose exactly {n} personas (variation seed: {seed}). "
        f"At least one must have the \"adversarial\" trait. Return JSON only."
    )
    for _attempt in range(2):
        text = _call_judge(resolved, _PROPOSE_SYSTEM, user)
        try:
            obj = json.loads(_strip_fences(text))
        except json.JSONDecodeError:
            continue
        raw = obj.get("personas") if isinstance(obj, dict) else None
        if not isinstance(raw, list):
            continue
        personas = [p for p in (_persona_from_obj(r) for r in raw) if p is not None]
        if personas:
            return personas[:n]
    return []


# ─── Scoring ──────────────────────────────────────────────────────────────


def score_simulations(
    results: list[SimulationResult],
    evaluators: list[Any] | None = None,
    judge: JudgeConfig | None = None,
) -> dict[str, Any]:
    """Score simulated conversations with the conversation evaluators.

    Runs each evaluator's ``evaluate(case, output)`` over every result's
    conversation-shaped case (output = the last assistant reply), the same
    call shape ``EvalSuite`` uses — but deliberately NOT entangled with
    EvalSuite internals. Adds the goal_achieved verdicts and returns a
    plain summary dict. Evaluator crashes are recorded per-evaluator,
    never raised.

    Honesty rule (absence of evidence ≠ evidence): a persona with an EMPTY
    transcript (e.g. ``driver_error`` before the first exchange), or an
    evaluator that *skipped* the case (``EvalResult.metadata["skipped"]``),
    records the score as ``None`` with a ``"skipped: <reason>"`` marker —
    NEVER the evaluator's pass-through 1.0, and never an invented 0.0.
    """
    if evaluators is None:
        from .evaluators.conversation import (
            ConversationRelevance, KnowledgeRetention, TurnConsistency,
        )
        evaluators = [
            ConversationRelevance(judge=judge),
            KnowledgeRetention(judge=judge),
            TurnConsistency(judge=judge),
        ]

    per_persona: dict[str, Any] = {}
    for r in results:
        output = next(
            (m["content"] for m in reversed(r.transcript)
             if m.get("role") == "assistant"),
            "",
        )
        scores: dict[str, float | None] = {}
        reasons: dict[str, str] = {}
        for ev in evaluators:
            name = getattr(ev, "name", type(ev).__name__)
            if not r.transcript:
                # Nothing was ever said — there is no conversation to score.
                # A None-with-reason beats a vacuous 1.00 or an invented 0.
                scores[name] = None
                reasons[name] = (
                    f"skipped: empty transcript "
                    f"(stop_reason={r.stop_reason}) — no conversation to score"
                )
                continue
            try:
                res = ev.evaluate(r.case, output)
                if res.metadata.get("skipped"):
                    # Evaluator skipped the case (shape mismatch) — its
                    # pass-through score must not masquerade as a real 1.00.
                    reason = res.reason.removeprefix("[skipped] ")
                    scores[name] = None
                    reasons[name] = f"skipped: {reason}"
                else:
                    scores[name] = float(res.score)
                    reasons[name] = res.reason
            except Exception as exc:  # evaluator crash is data, not fatal
                scores[name] = None
                reasons[name] = f"[evaluator error: {type(exc).__name__}: {exc}]"
        per_persona[r.persona.name] = {
            "scores": scores,
            "reasons": reasons,
            "turns": r.turns,
            "stop_reason": r.stop_reason,
            "goal_achieved": r.goal_achieved,
        }

    judged = [r for r in results if r.goal_achieved is not None]
    achieved = sum(1 for r in judged if r.goal_achieved)
    return {
        "simulated": True,
        "disclaimer": SIMULATED_DISCLAIMER,
        "n_personas": len(results),
        "per_persona": per_persona,
        "goal_completion": {
            "achieved": achieved,
            "judged": len(judged),
            "rate": (achieved / len(judged)) if judged else None,
        },
        "total_cost_usd": round(sum(r.cost_usd for r in results), 6),
    }


# ─── Transcript → dataset export ──────────────────────────────────────────


def results_to_cases(
    results: list[SimulationResult],
) -> tuple[list[EvalCase], "GenerationReport"]:
    """Convert simulation transcripts into conversation EvalCases.

    Each result becomes an EvalCase with ``conversation=transcript`` and
    ``input=persona.goal``; metadata carries the persona name/traits/
    stop_reason/``simulated=True`` plus the provenance stamp the
    simulator already wrote (``authored_by="simulator"`` — not restamped
    here). The persona's ``success_criteria`` is recorded as
    ``expected_behavior`` so the well-formed gate (and downstream judges)
    know what success looks like.

    Empty transcripts are skipped and counted (``dropped_malformed`` —
    there is no conversation to evaluate). Accepted cases pass
    ``gate_well_formed`` + ``gate_duplicate`` vs this batch.

    Returns ``(cases, GenerationReport)``.
    """
    from .case_gates import GenerationReport, gate_duplicate, gate_well_formed

    report = GenerationReport(requested=len(results), kind="simulate_export")
    accepted: list[EvalCase] = []
    for r in results:
        report.generated += 1
        if not r.transcript:
            # Nothing was ever said (e.g. driver_error / budget_exceeded
            # before turn 1) — no conversation to export. Counted, not kept.
            report.dropped_malformed += 1
            continue
        metadata = dict(r.case.metadata)
        metadata.setdefault("expected_behavior", r.persona.success_criteria)
        case = EvalCase(
            input=r.persona.goal,
            conversation=[dict(m) for m in r.transcript],
            metadata=metadata,
            tags=list(r.case.tags),
        )
        if not gate_well_formed(case).passed:
            report.dropped_malformed += 1
            continue
        if not gate_duplicate(case, accepted).passed:
            report.dropped_duplicate += 1
            continue
        accepted.append(case)
    report.accepted = len(accepted)
    return accepted, report


__all__ = [
    "SIMULATED_DISCLAIMER",
    "STOP_REASONS",
    "Persona",
    "SimulationResult",
    "simulate",
    "personas_from_jsonl",
    "propose_personas",
    "score_simulations",
    "results_to_cases",
]
