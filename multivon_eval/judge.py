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

    # Route any LiteLLM-supported provider (Bedrock, Vertex, Azure, Ollama)
    configure(JudgeConfig(provider="litellm", model="bedrock/anthropic.claude-3-sonnet-…"))

    # Self-hosted OpenAI-compatible endpoint (vLLM, TGI, LM Studio, Ollama)
    configure(JudgeConfig(
        provider="openai",
        model="llama-3.3-70b-instruct",
        base_url="https://vllm.internal/v1",
    ))
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .exceptions import JudgeUnavailable

__all__ = ["JudgeConfig", "configure", "get_global_judge"]

_SUPPORTED_PROVIDERS = {"anthropic", "openai", "litellm"}

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "litellm": "",  # User must specify; LiteLLM model strings vary by provider.
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
        base_url:    Custom API endpoint for OpenAI-compatible local servers.
                     Examples: "http://localhost:11434/v1" (Ollama),
                     "http://localhost:1234/v1" (LM Studio). Ignored for
                     Anthropic. Also read from OPENAI_BASE_URL env var.
        temperature: Sampling temperature for the judge (default 0.0 for
                     determinism).
        max_tokens:  Token budget for judge responses (default 1024).
        timeout:     Request timeout in seconds (default 30).
    """
    provider: str = ""
    model: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 30
    reliability_check: bool = False
    reliability_sample: int = 5
    cache: bool = False
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
        base_url = self.base_url or (os.getenv("OPENAI_BASE_URL", "") if provider == "openai" else "")
        cache = self.cache or os.getenv("MULTIVON_JUDGE_CACHE", "").lower() in ("1", "true", "yes")
        return JudgeConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            reliability_check=self.reliability_check,
            reliability_sample=self.reliability_sample,
            cache=cache,
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
        base_url=override.base_url or base.base_url,
        temperature=override.temperature if override.temperature != 0.0 else base.temperature,
        max_tokens=override.max_tokens if override.max_tokens != 1024 else base.max_tokens,
        timeout=override.timeout if override.timeout != 30 else base.timeout,
        reliability_check=override.reliability_check or base.reliability_check,
        reliability_sample=override.reliability_sample if override.reliability_sample != 5 else base.reliability_sample,
        cache=override.cache or base.cache,
        extra={**base.extra, **override.extra},
    )
    return merged.resolve()


_RETRYABLE_HTTP = {429, 500, 502, 503, 504}
_RETRYABLE_EXC_NAMES = (
    "RateLimitError", "APIStatusError", "InternalServerError",
    "ServiceUnavailableError", "APIConnectionError", "Timeout",
    "APITimeoutError",
)
_MAX_ATTEMPTS = 3


def _is_retryable(exc: Exception) -> bool:
    exc_str = str(exc)
    if any(str(code) in exc_str for code in _RETRYABLE_HTTP):
        return True
    return type(exc).__name__ in _RETRYABLE_EXC_NAMES


def _wrap_provider_error(provider: str, model: str, exc: Exception) -> JudgeUnavailable:
    """Re-raise a provider SDK exception as JudgeUnavailable with __cause__ preserved."""
    err = JudgeUnavailable(
        f"Judge call failed: {type(exc).__name__}: {exc}",
        provider=provider,
        model=model,
    )
    err.__cause__ = exc
    return err


def _sync_anthropic_call(prompt: str, config: JudgeConfig) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _sync_openai_call(prompt: str, config: JudgeConfig) -> str:
    import openai
    client_kwargs: dict = {}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    client = openai.OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=config.model,
        max_completion_tokens=config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _sync_litellm_call(prompt: str, config: JudgeConfig) -> str:
    try:
        import litellm
    except ImportError as exc:
        raise JudgeUnavailable(
            "provider='litellm' requires the litellm package: pip install 'multivon-eval[litellm]'",
            provider="litellm",
            model=config.model,
        ) from exc
    extra = dict(config.extra)
    if config.base_url and "api_base" not in extra:
        extra["api_base"] = config.base_url
    response = litellm.completion(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
        **extra,
    )
    return response.choices[0].message.content or ""


def _make_judge_call_uncached(prompt: str, config: JudgeConfig) -> str:
    """Single judge call, retried with backoff; provider exceptions wrapped."""
    import time as _time

    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            if config.provider == "anthropic":
                return _sync_anthropic_call(prompt, config)
            if config.provider == "openai":
                return _sync_openai_call(prompt, config)
            if config.provider == "litellm":
                return _sync_litellm_call(prompt, config)
            raise JudgeUnavailable(
                f"Unsupported provider: {config.provider!r}",
                provider=config.provider,
                model=config.model,
            )
        except JudgeUnavailable:
            raise
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _MAX_ATTEMPTS - 1:
                _time.sleep(2 ** attempt)
                continue
            raise _wrap_provider_error(config.provider, config.model, exc) from exc

    # Unreachable; guard for type-checkers.
    raise _wrap_provider_error(config.provider, config.model, last_exc or RuntimeError("exhausted"))


def make_judge_call(prompt: str, config: JudgeConfig) -> str:
    """Execute a single judge call with the given resolved config.

    Retries up to 3 times on rate-limit (429) and transient server errors (5xx)
    with exponential backoff (1s, 2s, 4s). Any provider-side error that is not
    retried is re-raised as :class:`JudgeUnavailable`.

    If ``config.cache`` is true, results are read/written through the on-disk
    judge cache (see :mod:`multivon_eval.cache`).
    """
    if config.cache:
        # Imported lazily so installs that never enable the cache pay nothing.
        from .cache import get_cache
        cache = get_cache()
        cached = cache.get(prompt, config)
        if cached is not None:
            return cached
        result = _make_judge_call_uncached(prompt, config)
        cache.put(prompt, config, result)
        return result
    return _make_judge_call_uncached(prompt, config)


# ── Async siblings ──────────────────────────────────────────────────────────


async def _async_anthropic_call(prompt: str, config: JudgeConfig) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def _async_openai_call(prompt: str, config: JudgeConfig) -> str:
    import openai
    client_kwargs: dict = {}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    client = openai.AsyncOpenAI(**client_kwargs)
    response = await client.chat.completions.create(
        model=config.model,
        max_completion_tokens=config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


async def _async_litellm_call(prompt: str, config: JudgeConfig) -> str:
    try:
        import litellm
    except ImportError as exc:
        raise JudgeUnavailable(
            "provider='litellm' requires the litellm package: pip install 'multivon-eval[litellm]'",
            provider="litellm",
            model=config.model,
        ) from exc
    extra = dict(config.extra)
    if config.base_url and "api_base" not in extra:
        extra["api_base"] = config.base_url
    response = await litellm.acompletion(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
        **extra,
    )
    return response.choices[0].message.content or ""


async def _make_judge_call_async_uncached(prompt: str, config: JudgeConfig) -> str:
    import asyncio

    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            if config.provider == "anthropic":
                return await _async_anthropic_call(prompt, config)
            if config.provider == "openai":
                return await _async_openai_call(prompt, config)
            if config.provider == "litellm":
                return await _async_litellm_call(prompt, config)
            raise JudgeUnavailable(
                f"Unsupported provider: {config.provider!r}",
                provider=config.provider,
                model=config.model,
            )
        except JudgeUnavailable:
            raise
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise _wrap_provider_error(config.provider, config.model, exc) from exc

    raise _wrap_provider_error(config.provider, config.model, last_exc or RuntimeError("exhausted"))


async def make_judge_call_async(prompt: str, config: JudgeConfig) -> str:
    """Async sibling of :func:`make_judge_call`. Same retry, same wrapping,
    same cache integration — uses provider async SDKs (AsyncAnthropic /
    AsyncOpenAI / litellm.acompletion)."""
    if config.cache:
        from .cache import get_cache
        cache = get_cache()
        cached = cache.get(prompt, config)
        if cached is not None:
            return cached
        result = await _make_judge_call_async_uncached(prompt, config)
        cache.put(prompt, config, result)
        return result
    return await _make_judge_call_async_uncached(prompt, config)
