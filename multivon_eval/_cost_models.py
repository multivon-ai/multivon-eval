"""Legacy basic-text price estimates, not complete provider accounting.

This small compatibility catalog excludes cache, tier, tool and regional pricing.
Use native evidence plus an upstream estimator for those dimensions. Self-hosted
models are not assumed free. Caller overrides last for the current process.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens for one model."""
    input_per_million: float
    output_per_million: float

    def __post_init__(self):
        for rate in (self.input_per_million, self.output_per_million):
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
                raise ValueError("Token prices must be finite nonnegative numbers")


# Compatibility estimates for basic uncached text only. Verified 2026-09-17.
# Native reconciliation uses an explicit upstream/caller estimator instead.
# https://platform.claude.com/docs/en/about-claude/pricing
# https://developers.openai.com/api/docs/models/gpt-4o-mini
# https://developers.openai.com/api/docs/models/gpt-4o
# https://developers.openai.com/api/docs/models/gpt-4.1
_CATALOG: dict[str, ModelPricing] = {
    "claude-opus-4-7": ModelPricing(5.0, 25.0),
    "claude-opus-4-6": ModelPricing(5.0, 25.0),
    "claude-opus-4-5": ModelPricing(5.0, 25.0),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0),
    "claude-sonnet-4-5": ModelPricing(3.0, 15.0),
    "claude-haiku-4-5": ModelPricing(1.0, 5.0),
    "claude-haiku-4-5-20251001": ModelPricing(1.0, 5.0),
    "gpt-4o": ModelPricing(2.50, 10.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4.1": ModelPricing(2.0, 8.0),
}
# Removed speculative model IDs/rates, context/modality-dependent Gemini rates
# and zero-cost self-hosted assumptions. Unknown is preferable to a false price.


def get_pricing(model: str) -> ModelPricing | None:
    """Return per-million pricing for a model, or None if unknown."""
    return _CATALOG.get(model)


def register_pricing(model: str, pricing: ModelPricing) -> None:
    """Add or override pricing for ``model``. Lasts for the process."""
    if not isinstance(model, str) or not model.strip() or not isinstance(pricing, ModelPricing):
        raise ValueError("A nonempty model and ModelPricing are required")
    _CATALOG[model] = pricing


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Estimate USD cost for one call. Return None if the model is unknown."""
    if any(type(count) is not int or count < 0 for count in (input_tokens, output_tokens)):
        raise ValueError("Token counts must be nonnegative integers")
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
