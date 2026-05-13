"""
Pluggable secrets resolver.

Default: environment variables. Anything beyond that — Vault, AWS Secrets
Manager, GCP Secret Manager, Azure Key Vault, 1Password, sops, keyring — can
be plugged in without taking a hard dependency, because integrators register
their own ``SecretsResolver`` instances.

    from multivon_eval.secrets import get_secret, set_resolver, ChainedResolver

    # Use the default env resolver
    api_key = get_secret("ANTHROPIC_API_KEY")

    # Chain a custom resolver in front of env:
    class VaultResolver:
        def get(self, key: str) -> str | None:
            return _vault_client.read(f"secret/data/{key}")["data"]["value"]

    set_resolver(ChainedResolver([VaultResolver(), EnvResolver()]))

The resolver is module-scoped global state — set it once at process start,
typically before importing models.
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from .exceptions import SecretsError

__all__ = [
    "SecretsResolver",
    "EnvResolver",
    "ChainedResolver",
    "StaticResolver",
    "get_secret",
    "set_resolver",
    "get_resolver",
    "reset_resolver",
]


@runtime_checkable
class SecretsResolver(Protocol):
    """Anything with a ``get(key) -> str | None`` method is a resolver.

    Return ``None`` when the key is unknown so a chained resolver can fall
    through. Raise only on unrecoverable backend errors (network down,
    permission denied).
    """

    def get(self, key: str) -> str | None: ...


class EnvResolver:
    """Resolver that reads from ``os.environ``. Always the default."""

    name = "env"

    def get(self, key: str) -> str | None:
        return os.environ.get(key)


class StaticResolver:
    """In-memory mapping. Useful in tests and for one-off overrides."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values = dict(values or {})
        self.name = "static"

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def set(self, key: str, value: str) -> None:
        self._values[key] = value


class ChainedResolver:
    """Try each resolver in order; return the first non-None hit."""

    def __init__(self, resolvers: list[SecretsResolver]) -> None:
        if not resolvers:
            raise ValueError("ChainedResolver requires at least one resolver")
        self._resolvers = list(resolvers)
        self.name = "chain[" + ",".join(getattr(r, "name", type(r).__name__) for r in resolvers) + "]"

    def get(self, key: str) -> str | None:
        for resolver in self._resolvers:
            value = resolver.get(key)
            if value is not None:
                return value
        return None


_DEFAULT_RESOLVER: SecretsResolver = EnvResolver()
_RESOLVER: SecretsResolver = _DEFAULT_RESOLVER


def set_resolver(resolver: SecretsResolver) -> None:
    """Replace the process-wide secrets resolver."""
    global _RESOLVER
    if not isinstance(resolver, SecretsResolver):
        raise TypeError(f"resolver must implement SecretsResolver; got {type(resolver).__name__}")
    _RESOLVER = resolver


def get_resolver() -> SecretsResolver:
    """Return the current process-wide resolver."""
    return _RESOLVER


def reset_resolver() -> None:
    """Restore the default :class:`EnvResolver`. Primarily for tests."""
    global _RESOLVER
    _RESOLVER = _DEFAULT_RESOLVER


def get_secret(key: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Fetch ``key`` via the active resolver.

    Args:
        key:      Secret name (e.g. "ANTHROPIC_API_KEY").
        default:  Returned when neither the resolver nor ``required`` produces a value.
        required: When true, raise :class:`SecretsError` instead of returning ``default``.

    Returns the secret string, or ``default`` if missing and ``required`` is false.
    """
    value = _RESOLVER.get(key)
    if value is not None:
        return value
    if required:
        raise SecretsError(key, resolver=getattr(_RESOLVER, "name", type(_RESOLVER).__name__))
    return default
