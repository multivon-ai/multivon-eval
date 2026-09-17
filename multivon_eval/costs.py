"""
Recorded usage subtotals and coverage-aware provider accounting.

Every call to :func:`multivon_eval.judge.make_judge_call` (and its async
sibling) reports its token usage to this module. The active
:class:`CostTracker` (set per evaluation run by :class:`EvalSuite`)
accumulates a subset of successful text-judge counts. It cannot establish the
full run's cost. Reconcile native events with ``account_provider_events`` and
an explicit coverage declaration for run provider budget gates.

Cost accounting is *advisory*: a missing tracker, a provider that does
not report usage, or an unknown model never breaks an evaluation. They
leave the run cost unknown. ``recorded_cost_usd`` is a scoped subtotal estimate.

Architecture:

* :class:`CostTracker` — accumulator. Cheap, thread-safe via a Lock.
* :func:`record_call` — module-level hook called from
  ``judge._record_usage``. Forwards to the active tracker.
* :func:`active_tracker` / :func:`set_active_tracker` — used by
  :class:`EvalSuite.run` to push a tracker for the duration of a run.
* :class:`Costs` — dataclass returned by ``EvalReport.costs``.

Pricing comes from :mod:`multivon_eval._cost_models`. Override per
model with :func:`register_pricing`.
"""
from __future__ import annotations

import contextvars
import threading
import math
from dataclasses import dataclass, field

from ._cost_models import ModelPricing, estimate_cost_usd, register_pricing  # noqa: F401


@dataclass
class ProviderUsage:
    """Per-(provider, model) accumulated usage."""
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float | None = 0.0  # None if pricing unknown for this model

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in (self.input_tokens, self.output_tokens, self.calls)):
            raise ValueError("Usage counts must be nonnegative integers")
        if self.cost_usd is not None and (
            isinstance(self.cost_usd, bool) or not isinstance(self.cost_usd, (int, float))
            or not math.isfinite(self.cost_usd) or self.cost_usd < 0
        ):
            raise ValueError("Cost must be unknown or a finite nonnegative number")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "cost_usd": self.cost_usd,
        }


@dataclass
class Costs:
    """Aggregated cost report for an evaluation run.

    Returned by :attr:`multivon_eval.result.EvalReport.costs`.
    """
    by_model: list[ProviderUsage] = field(default_factory=list)
    """One entry per (provider, model) pair seen during the run."""
    scope: str = "recorded_judge_responses"
    coverage_declaration: str | None = None
    evidence_gaps: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.coverage_declaration is not None and (
            not isinstance(self.coverage_declaration, str) or not self.coverage_declaration.strip()
        ):
            raise ValueError("coverage_declaration must be a nonempty explanation of complete provider coverage")

    @property
    def complete(self) -> bool:
        """Caller-declared run provider coverage, with no detected evidence gaps."""
        return (self.scope == "run_provider_usage" and bool(self.coverage_declaration)
                and not self.evidence_gaps)

    @classmethod
    def from_dict(cls, data: dict) -> Costs:
        """Restore recorded usage; derived totals are recomputed from entries."""
        return cls([ProviderUsage(
            provider=row["provider"], model=row["model"],
            input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
            calls=row["calls"], cost_usd=row["cost_usd"],
        ) for row in data["by_model"]], scope=data.get("scope", "legacy_recorded_usage"),
            coverage_declaration=data.get("coverage_declaration"),
            evidence_gaps=list(data.get("evidence_gaps", [])), evidence=data.get("evidence", {}))

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.by_model)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.by_model)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_calls(self) -> int:
        return sum(u.calls for u in self.by_model)

    @property
    def total_cost_usd(self) -> float | None:
        """Run provider estimate, unknown unless coverage and pricing are complete."""
        return self.recorded_cost_usd if self.complete else None

    @property
    def recorded_cost_usd(self) -> float | None:
        """Estimate for recorded entries only; never establishes full run cost."""
        if any(u.cost_usd is None for u in self.by_model):
            return None
        return math.fsum(u.cost_usd or 0.0 for u in self.by_model)

    def to_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "total_cost_usd": self.total_cost_usd,
            "recorded_cost_usd": self.recorded_cost_usd,
            "scope": self.scope,
            "coverage_declaration": self.coverage_declaration,
            "complete": self.complete,
            "evidence_gaps": list(self.evidence_gaps),
            "evidence": self.evidence,
            "by_model": [u.to_dict() for u in self.by_model],
        }

    def __str__(self) -> str:
        if not self.by_model:
            return "Costs: no judge calls recorded; run provider cost unknown" if not self.complete else "Costs: declared complete zero provider usage"
        lines = [
            f"Recorded usage ({self.scope}): {self.total_calls} calls, "
            f"{self.total_input_tokens:,} → {self.total_output_tokens:,} tokens"
        ]
        for u in self.by_model:
            cost = f"${u.cost_usd:.4f}" if u.cost_usd is not None else "$?.??"
            lines.append(
                f"  {u.provider}/{u.model}: "
                f"{u.calls} calls, {u.input_tokens:,} in / {u.output_tokens:,} out, {cost}"
            )
        total = self.total_cost_usd
        if total is not None:
            lines.append(f"  total: ${total:.4f}")
        else:
            lines.append("  run provider total: unknown (coverage or pricing incomplete)")
        return "\n".join(lines)


class CostTracker:
    """Thread-safe accumulator. Created per EvalSuite.run() invocation."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], ProviderUsage] = {}
        self._lock = threading.Lock()

    def record(self, *, provider: str, model: str,
               input_tokens: int, output_tokens: int) -> None:
        key = (provider, model)
        with self._lock:
            slot = self._by_key.get(key)
            if slot is None:
                slot = ProviderUsage(provider=provider, model=model)
                self._by_key[key] = slot
            slot.input_tokens += input_tokens
            slot.output_tokens += output_tokens
            slot.calls += 1
            # Cost estimate (None if model isn't priced)
            inc = estimate_cost_usd(
                model, input_tokens=input_tokens, output_tokens=output_tokens,
            )
            if inc is None:
                slot.cost_usd = None
            elif slot.cost_usd is not None:
                slot.cost_usd += inc

    def snapshot(self) -> Costs:
        with self._lock:
            entries = sorted(
                self._by_key.values(),
                key=lambda u: (u.provider, u.model),
            )
            # Return copies so a later record() doesn't mutate the snapshot.
            return Costs(by_model=[
                ProviderUsage(
                    provider=u.provider, model=u.model,
                    input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                    calls=u.calls, cost_usd=u.cost_usd,
                )
                for u in entries
            ])

    def reset(self) -> None:
        with self._lock:
            self._by_key.clear()


# ContextVar so set_active_tracker is correct under asyncio.
_ACTIVE: contextvars.ContextVar[CostTracker | None] = contextvars.ContextVar(
    "multivon_active_cost_tracker", default=None,
)


def active_tracker() -> CostTracker | None:
    return _ACTIVE.get()


def set_active_tracker(tracker: CostTracker | None) -> contextvars.Token:
    """Set the active tracker. Returns a token usable with :func:`reset_token`."""
    return _ACTIVE.set(tracker)


def reset_token(token: contextvars.Token) -> None:
    _ACTIVE.reset(token)


def record_call(*, provider: str, model: str,
                input_tokens: int, output_tokens: int) -> None:
    """Hook called by :func:`judge._record_usage`.

    Looks up the active CostTracker via contextvar. If there's no tracker
    (eval running outside a suite, or user explicitly disabled), it's a no-op.
    """
    tracker = active_tracker()
    if tracker is None:
        return
    tracker.record(
        provider=provider, model=model,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )


__all__ = [
    "Costs",
    "CostTracker",
    "ProviderUsage",
    "active_tracker",
    "set_active_tracker",
    "reset_token",
    "record_call",
    "register_pricing",
    "ModelPricing",
]
