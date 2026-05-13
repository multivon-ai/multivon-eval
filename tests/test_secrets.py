"""Tests for the pluggable secrets resolver."""
from __future__ import annotations

import pytest

from multivon_eval import (
    SecretsError,
    EnvResolver,
    ChainedResolver,
    StaticResolver,
    get_secret,
    set_resolver,
    reset_resolver,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_resolver()
    yield
    reset_resolver()


class TestEnvResolver:
    def test_env_resolver_reads_environ(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_X", "env-value")
        assert EnvResolver().get("TEST_KEY_X") == "env-value"

    def test_env_resolver_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("MISSING_K_Q", raising=False)
        assert EnvResolver().get("MISSING_K_Q") is None


class TestStaticResolver:
    def test_static_resolver_returns_stored_value(self):
        r = StaticResolver({"k": "v"})
        assert r.get("k") == "v"
        assert r.get("missing") is None

    def test_static_resolver_set(self):
        r = StaticResolver()
        r.set("api_key", "secret")
        assert r.get("api_key") == "secret"


class TestChainedResolver:
    def test_chained_returns_first_hit(self):
        r = ChainedResolver([
            StaticResolver({"key": "from-first"}),
            StaticResolver({"key": "from-second"}),
        ])
        assert r.get("key") == "from-first"

    def test_chained_falls_through_on_none(self):
        r = ChainedResolver([
            StaticResolver({}),
            StaticResolver({"only_in_second": "yes"}),
        ])
        assert r.get("only_in_second") == "yes"

    def test_chained_returns_none_when_all_miss(self):
        r = ChainedResolver([StaticResolver({}), StaticResolver({})])
        assert r.get("nothing") is None

    def test_chain_requires_at_least_one(self):
        with pytest.raises(ValueError):
            ChainedResolver([])

    def test_chained_name_reflects_components(self):
        r = ChainedResolver([StaticResolver(), EnvResolver()])
        assert "chain" in r.name
        assert "static" in r.name and "env" in r.name


class TestGetSecret:
    def test_default_uses_env_resolver(self, monkeypatch):
        monkeypatch.setenv("FOO_BAR_BAZ", "from-env")
        assert get_secret("FOO_BAR_BAZ") == "from-env"

    def test_custom_resolver_via_set_resolver(self, monkeypatch):
        set_resolver(StaticResolver({"my_key": "from-static"}))
        monkeypatch.delenv("my_key", raising=False)
        assert get_secret("my_key") == "from-static"

    def test_missing_returns_default(self):
        assert get_secret("not-a-real-key", default="fallback") == "fallback"

    def test_required_raises_secrets_error(self):
        with pytest.raises(SecretsError) as info:
            get_secret("absolutely-not-set-anywhere", required=True)
        assert info.value.key == "absolutely-not-set-anywhere"

    def test_set_resolver_validates_protocol(self):
        with pytest.raises(TypeError):
            set_resolver("not a resolver")  # type: ignore[arg-type]
