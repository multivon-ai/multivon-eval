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


# Conservative whitelist of provider-SDK exception NAMES that almost
# always mean "auth missing / wrong key / can't reach the server."
# Generic names like ``BadRequestError`` or ``APIError`` are deliberately
# excluded — they routinely fire for prompt-too-long, invalid model
# id, or unsupported params, and drowning those real bugs in setup
# advice is worse than no hint at all. Codex D15 round-1 finding.
_AUTH_HINT_EXC_NAMES = {
    "AuthenticationError",     # OpenAI / Anthropic — clear auth fail
    "PermissionDeniedError",   # Anthropic — clear permission fail
    "APIConnectionError",      # local server down / DNS / unreachable
    "ConnectError",            # httpx — connection-specific
}


def _looks_like_auth_or_connection_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in _AUTH_HINT_EXC_NAMES:
        return True
    msg = str(exc).lower()
    # Message-content sigs are targeted: "api_key" / "unauthorized" /
    # "401" / "could not connect" / "name or service not known" are
    # auth-and-connectivity-specific. We do NOT include "missing" or
    # "not found" — those false-positive on legitimate prompt and
    # model lookup errors.
    return any(
        sig in msg for sig in
        ("api_key", "api key", "unauthorized", "401",
         "could not connect", "connection refused",
         "name or service not known", "no such host")
    )


def _setup_hint(provider: str) -> str:
    """Plain-language setup hint for the most common beginner mistake:
    no API key set, or local LLM server not running. Returned only
    when the underlying exception looks auth- or connection-shaped, so
    real bugs aren't drowned in 'have you set your key?' noise."""
    lines = [
        "",
        "  To fix, pick ONE:",
    ]
    if provider == "anthropic":
        lines += [
            "    export ANTHROPIC_API_KEY=sk-ant-...",
            "    (get a key at https://console.anthropic.com)",
        ]
    elif provider == "openai":
        lines += [
            "    export OPENAI_API_KEY=sk-...",
            "    (get a key at https://platform.openai.com)",
        ]
    else:
        lines += [
            "    export ANTHROPIC_API_KEY=sk-ant-...    # or",
            "    export OPENAI_API_KEY=sk-...",
        ]
    lines += [
        "    OR run a local LLM (no key needed):",
        "      ollama pull qwen2.5:14b && ollama serve",
        "    OR drop the LLM-judge evaluators from your suite — see",
        "      `multivon-eval init -t quickstart` for a deterministic-only starter.",
    ]
    return "\n".join(lines)


def _wrap_provider_error(provider: str, model: str, exc: Exception) -> JudgeUnavailable:
    """Re-raise a provider SDK exception as JudgeUnavailable with __cause__ preserved.

    When the underlying exception looks like missing-credentials or
    can't-reach-server, the wrapped message includes plain-language
    setup hints — the #1 beginner mistake."""
    base_msg = f"Judge call failed: {type(exc).__name__}: {exc}"
    if _looks_like_auth_or_connection_error(exc):
        base_msg = base_msg + _setup_hint(provider)
    err = JudgeUnavailable(
        base_msg,
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
    _record_usage(
        provider="anthropic",
        model=config.model,
        input_tokens=getattr(response.usage, "input_tokens", 0) if getattr(response, "usage", None) else 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) if getattr(response, "usage", None) else 0,
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
    usage = getattr(response, "usage", None)
    _record_usage(
        provider="openai",
        model=config.model,
        input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
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
    usage = getattr(response, "usage", None) or {}
    _record_usage(
        provider="litellm",
        model=config.model,
        input_tokens=usage.get("prompt_tokens", 0) if hasattr(usage, "get") else getattr(usage, "prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0) if hasattr(usage, "get") else getattr(usage, "completion_tokens", 0),
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
    judge cache (see :mod:`multivon_eval.cache`). The cache is advisory:
    if it can't be initialised or a get/put fails (corrupt DB, unwritable
    path, sqlite lock), the call falls through to the live judge so the
    eval still completes.
    """
    if not config.cache:
        return _make_judge_call_uncached(prompt, config)

    # Imported lazily so installs that never enable the cache pay nothing.
    cache = None
    try:
        from .cache import get_cache
        cache = get_cache()
        cached = cache.get(prompt, config)
        if cached is not None:
            return cached
    except CacheError as exc:
        _warn_cache_degraded("read", exc)
        cache = None
    except Exception as exc:  # never let an unknown cache bug break an eval
        _warn_cache_degraded("read", exc)
        cache = None

    result = _make_judge_call_uncached(prompt, config)
    if cache is not None:
        try:
            cache.put(prompt, config, result)
        except CacheError as exc:
            _warn_cache_degraded("write", exc)
        except Exception as exc:
            _warn_cache_degraded("write", exc)
    return result


# Imported here to avoid a circular import at top of the module.
from .exceptions import CacheError  # noqa: E402


_CACHE_DEGRADATION_WARNED = False


def _warn_cache_degraded(direction: str, exc: BaseException) -> None:
    """Print a one-time warning when the cache fails. Don't crash the eval."""
    global _CACHE_DEGRADATION_WARNED
    if _CACHE_DEGRADATION_WARNED:
        return
    _CACHE_DEGRADATION_WARNED = True
    import sys
    print(
        f"  [multivon-eval] judge cache {direction} failed ({type(exc).__name__}: {exc}); "
        f"continuing without cache.",
        file=sys.stderr,
    )


# ── Async siblings ──────────────────────────────────────────────────────────


async def _async_anthropic_call(prompt: str, config: JudgeConfig) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    _record_usage(
        provider="anthropic",
        model=config.model,
        input_tokens=getattr(response.usage, "input_tokens", 0) if getattr(response, "usage", None) else 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) if getattr(response, "usage", None) else 0,
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
    usage = getattr(response, "usage", None)
    _record_usage(
        provider="openai",
        model=config.model,
        input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
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
    usage = getattr(response, "usage", None) or {}
    _record_usage(
        provider="litellm",
        model=config.model,
        input_tokens=usage.get("prompt_tokens", 0) if hasattr(usage, "get") else getattr(usage, "prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0) if hasattr(usage, "get") else getattr(usage, "completion_tokens", 0),
    )
    return response.choices[0].message.content or ""


# ── Cost / token accounting hook ────────────────────────────────────────────


def _record_usage(*, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Forward token counts to whichever cost tracker is currently active.

    Lazy import keeps the hot path free of an unused dependency for code
    that doesn't track costs.
    """
    if input_tokens == 0 and output_tokens == 0:
        return
    try:
        from .costs import record_call
        record_call(provider=provider, model=model,
                    input_tokens=int(input_tokens or 0),
                    output_tokens=int(output_tokens or 0))
    except Exception:
        # Cost accounting is observability, not correctness.
        # Never let a bug here propagate to the eval.
        pass


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
    AsyncOpenAI / litellm.acompletion).

    Cache is advisory: a CacheError on read or write degrades to an uncached
    call rather than failing the eval.
    """
    if not config.cache:
        return await _make_judge_call_async_uncached(prompt, config)

    cache = None
    try:
        from .cache import get_cache
        cache = get_cache()
        cached = cache.get(prompt, config)
        if cached is not None:
            return cached
    except CacheError as exc:
        _warn_cache_degraded("read", exc)
        cache = None
    except Exception as exc:
        _warn_cache_degraded("read", exc)
        cache = None

    result = await _make_judge_call_async_uncached(prompt, config)
    if cache is not None:
        try:
            cache.put(prompt, config, result)
        except CacheError as exc:
            _warn_cache_degraded("write", exc)
        except Exception as exc:
            _warn_cache_degraded("write", exc)
    return result
