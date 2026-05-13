"""
Model price catalog for cost estimation.

USD per million tokens. Treat these as estimates, not invoices. Provider
prices change; users can override at runtime via ``register_pricing()``.
The catalog is shipped with the library so the basic "$0.05 spent on
this suite run" estimate works out of the box for the major judges, but
production cost accounting should use the provider's billed usage where
available (the cost-tracking pipeline prefers provider-reported counts).

Last updated: 2026-05.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens for one model."""
    input_per_million: float
    output_per_million: float


# ─── Anthropic ─────────────────────────────────────────────────────────────
# https://www.anthropic.com/pricing  (anthropic published pricing as of 2026-05)
_ANTHROPIC = {
    "claude-opus-4-7": ModelPricing(15.0, 75.0),
    "claude-opus-4-7-20251101": ModelPricing(15.0, 75.0),
    "claude-opus-4-6": ModelPricing(15.0, 75.0),
    "claude-opus-4-5": ModelPricing(15.0, 75.0),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0),
    "claude-sonnet-4-5": ModelPricing(3.0, 15.0),
    "claude-haiku-4-5": ModelPricing(0.80, 4.0),
    "claude-haiku-4-5-20251001": ModelPricing(0.80, 4.0),
    "claude-3-5-sonnet-20241022": ModelPricing(3.0, 15.0),
}

# ─── OpenAI ────────────────────────────────────────────────────────────────
_OPENAI = {
    "gpt-5.5": ModelPricing(2.50, 10.0),
    "gpt-5.4": ModelPricing(2.50, 10.0),
    "gpt-5.3": ModelPricing(2.50, 10.0),
    "gpt-5": ModelPricing(2.50, 10.0),
    "gpt-4o": ModelPricing(2.50, 10.0),
    "gpt-4o-2024-11-20": ModelPricing(2.50, 10.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4.1": ModelPricing(2.50, 10.0),
}

# ─── Self-hosted / on-prem ─────────────────────────────────────────────────
# Local models incur compute cost, not per-token API cost. We report $0 here
# so cost accounting is correct on-prem; users with explicit GPU bills can
# register their own pricing.
_ONPREM = {
    "llama-3.3-70b-instruct": ModelPricing(0.0, 0.0),
    "qwen-2.5-72b-instruct": ModelPricing(0.0, 0.0),
    "mistral-large": ModelPricing(0.0, 0.0),
}


_CATALOG: dict[str, ModelPricing] = {}
_CATALOG.update(_ANTHROPIC)
_CATALOG.update(_OPENAI)
_CATALOG.update(_ONPREM)


def get_pricing(model: str) -> ModelPricing | None:
    """Return per-million pricing for a model, or None if unknown."""
    return _CATALOG.get(model)


def register_pricing(model: str, pricing: ModelPricing) -> None:
    """Add or override pricing for ``model``. Lasts for the process."""
    _CATALOG[model] = pricing


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Estimate USD cost for one call. Return None if the model is unknown."""
    p = _CATALOG.get(model)
    if p is None:
        return None
    return (
        p.input_per_million * input_tokens / 1_000_000.0
        + p.output_per_million * output_tokens / 1_000_000.0
    )


def known_models() -> list[str]:
    """Sorted list of every model with shipped pricing."""
    return sorted(_CATALOG.keys())
