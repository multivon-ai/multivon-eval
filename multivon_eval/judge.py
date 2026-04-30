"""
Judge configuration for LLM-as-judge evaluators.

Decouples the judge model from the metric so teams can use any
supported backend without changing evaluator code.

Precedence (highest to lowest):
  1. Per-evaluator JudgeConfig passed to the evaluator constructor
  2. Global config set via configure()
  3. Environment variables (JUDGE_PROVIDER, JUDGE_MODEL)
  4. Built-in defaults (anthropic / claude-haiku-4-5)

Usage:
    from multivon_eval import configure, JudgeConfig

    # Set globally once at startup
    configure(JudgeConfig(provider="openai", model="gpt-4o-mini"))

    # Override for a specific evaluator
    Faithfulness(judge=JudgeConfig(provider="anthropic", model="claude-opus-4-7"))
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = ["JudgeConfig", "configure", "get_global_judge"]

_SUPPORTED_PROVIDERS = {"anthropic", "openai"}

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
}


@dataclass
class JudgeConfig:
    """
    Configuration for the LLM judge used by evaluators.

    Args:
        provider:    "anthropic" or "openai". Defaults to JUDGE_PROVIDER env
                     var, then "anthropic".
        model:       Model name. Defaults to JUDGE_MODEL env var, then the
                     provider's default (claude-haiku-4-5 / gpt-4o-mini).
        temperature: Sampling temperature for the judge (default 0.0 for
                     determinism).
        max_tokens:  Token budget for judge responses (default 1024).
        timeout:     Request timeout in seconds (default 30).
    """
    provider: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 30
    reliability_check: bool = False
    reliability_sample: int = 5
    # Reserved for future backends (prometheus, minichek, local hf model)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider and self.provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{self.provider}'. "
                f"Choose from: {sorted(_SUPPORTED_PROVIDERS)}"
            )

    def resolve(self) -> "JudgeConfig":
        """Return a fully resolved config, filling blanks from env + defaults."""
        provider = self.provider or os.getenv("JUDGE_PROVIDER", "anthropic").lower()
        model = self.model or os.getenv("JUDGE_MODEL", _DEFAULT_MODELS.get(provider, ""))
        return JudgeConfig(
            provider=provider,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            reliability_check=self.reliability_check,
            reliability_sample=self.reliability_sample,
            extra=self.extra,
        )


_GLOBAL_JUDGE: JudgeConfig = JudgeConfig()


def configure(config: JudgeConfig) -> None:
    """Set the global judge config used by all evaluators that don't override it."""
    global _GLOBAL_JUDGE
    _GLOBAL_JUDGE = config


def get_global_judge() -> JudgeConfig:
    """Return the current global judge config."""
    return _GLOBAL_JUDGE


def resolve_judge(per_evaluator: JudgeConfig | None) -> JudgeConfig:
    """
    Merge per-evaluator config over global config, then resolve env vars.

    Explicit fields in per_evaluator take precedence over global config,
    which takes precedence over env vars, which take precedence over defaults.
    """
    base = _GLOBAL_JUDGE
    override = per_evaluator or JudgeConfig()

    merged = JudgeConfig(
        provider=override.provider or base.provider,
        model=override.model or base.model,
        temperature=override.temperature if override.temperature != 0.0 else base.temperature,
        max_tokens=override.max_tokens if override.max_tokens != 1024 else base.max_tokens,
        timeout=override.timeout if override.timeout != 30 else base.timeout,
        reliability_check=override.reliability_check or base.reliability_check,
        reliability_sample=override.reliability_sample if override.reliability_sample != 5 else base.reliability_sample,
        extra={**base.extra, **override.extra},
    )
    return merged.resolve()


def make_judge_call(prompt: str, config: JudgeConfig) -> str:
    """Execute a single judge call with the given resolved config."""
    if config.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    elif config.provider == "openai":
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=config.model,
            max_completion_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    else:
        raise ValueError(f"Unsupported provider: '{config.provider}'")
