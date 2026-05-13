"""Tests for the judge-result cache (SQLite-backed)."""
from __future__ import annotations

import os
import time

import pytest

from multivon_eval import JudgeConfig, JudgeCache, set_cache, get_cache, CacheError
from multivon_eval.cache import _hash_call


@pytest.fixture
def fresh_cache(tmp_path):
    cache = JudgeCache(db_path=tmp_path / "judge.db")
    set_cache(cache)
    yield cache
    set_cache(None)


class TestHashing:
    def test_same_inputs_same_hash(self):
        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        h1 = _hash_call("hello", cfg)
        h2 = _hash_call("hello", cfg)
        assert h1 == h2

    def test_different_prompt_different_hash(self):
        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        assert _hash_call("hello", cfg) != _hash_call("world", cfg)

    def test_different_model_different_hash(self):
        a = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        b = JudgeConfig(provider="openai", model="gpt-4o").resolve()
        assert _hash_call("hello", a) != _hash_call("hello", b)

    def test_different_provider_different_hash(self):
        a = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        b = JudgeConfig(provider="anthropic", model="gpt-4o-mini").resolve()
        assert _hash_call("hello", a) != _hash_call("hello", b)

    def test_temperature_difference_matters(self):
        a = JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.0).resolve()
        b = JudgeConfig(provider="openai", model="gpt-4o-mini", temperature=0.7).resolve()
        assert _hash_call("hello", a) != _hash_call("hello", b)

    def test_timeout_does_not_affect_hash(self):
        """Timeout is a network concern, not a semantic call input."""
        a = JudgeConfig(provider="openai", model="gpt-4o-mini", timeout=10).resolve()
        b = JudgeConfig(provider="openai", model="gpt-4o-mini", timeout=60).resolve()
        assert _hash_call("hello", a) == _hash_call("hello", b)


class TestPutGet:
    def test_miss_returns_none(self, fresh_cache):
        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        assert fresh_cache.get("hello", cfg) is None
        assert fresh_cache.stats.misses == 1

    def test_put_then_get_returns_value(self, fresh_cache):
        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        fresh_cache.put("hello", cfg, "Yes.")
        assert fresh_cache.get("hello", cfg) == "Yes."
        assert fresh_cache.stats.hits == 1
        assert fresh_cache.stats.writes == 1

    def test_overwrite_same_key(self, fresh_cache):
        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        fresh_cache.put("k", cfg, "v1")
        fresh_cache.put("k", cfg, "v2")
        assert fresh_cache.get("k", cfg) == "v2"

    def test_different_config_is_a_miss(self, fresh_cache):
        cfg_a = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        cfg_b = JudgeConfig(provider="openai", model="gpt-4o").resolve()
        fresh_cache.put("hello", cfg_a, "A")
        assert fresh_cache.get("hello", cfg_b) is None

    def test_size_reflects_writes(self, fresh_cache):
        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        fresh_cache.put("a", cfg, "1")
        fresh_cache.put("b", cfg, "2")
        assert fresh_cache.size() == 2

    def test_clear_drops_everything(self, fresh_cache):
        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        fresh_cache.put("a", cfg, "1")
        fresh_cache.put("b", cfg, "2")
        n = fresh_cache.clear()
        assert n == 2
        assert fresh_cache.size() == 0


class TestTTL:
    def test_ttl_expiry_is_a_miss(self, tmp_path):
        cache = JudgeCache(db_path=tmp_path / "ttl.db", ttl=0.05)
        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        cache.put("k", cfg, "v")
        assert cache.get("k", cfg) == "v"
        time.sleep(0.1)
        assert cache.get("k", cfg) is None

    def test_within_ttl_is_a_hit(self, tmp_path):
        cache = JudgeCache(db_path=tmp_path / "ttl.db", ttl=60)
        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        cache.put("k", cfg, "v")
        assert cache.get("k", cfg) == "v"


class TestSingleton:
    def test_get_cache_returns_singleton(self):
        a = get_cache()
        b = get_cache()
        assert a is b
        set_cache(None)


class TestErrorHandling:
    def test_unwritable_path_raises_cache_error(self, tmp_path):
        # Create a file where we'd want the dir to be — init should fail cleanly.
        path = tmp_path / "blocked"
        path.write_text("not a dir")
        with pytest.raises(CacheError):
            JudgeCache(db_path=path / "subdir" / "judge.db")


class TestEvaluatorCachePropagation:
    """Codex P2 #1: `_call()` / `_judge_call()` previously rebuilt the
    JudgeConfig and dropped the `cache` + `base_url` fields, so opting in
    to caching via JudgeConfig(cache=True) on an evaluator was silently
    a no-op."""

    def test_with_max_tokens_preserves_every_field(self):
        from multivon_eval.evaluators.llm_judge import _with_max_tokens

        original = JudgeConfig(
            provider="openai",
            model="gpt-4o-mini",
            base_url="https://internal/v1",
            temperature=0.3,
            max_tokens=512,
            timeout=99,
            reliability_check=True,
            reliability_sample=7,
            cache=True,
            extra={"foo": "bar"},
        )
        out = _with_max_tokens(original, max_tokens=2048)
        # The one field we override:
        assert out.max_tokens == 2048
        # Everything else preserved — these are the regressions we're guarding:
        assert out.provider == "openai"
        assert out.model == "gpt-4o-mini"
        assert out.base_url == "https://internal/v1"
        assert out.temperature == 0.3
        assert out.timeout == 99
        assert out.reliability_check is True
        assert out.reliability_sample == 7
        assert out.cache is True
        assert out.extra == {"foo": "bar"}

    def test_with_max_tokens_no_override(self):
        from multivon_eval.evaluators.llm_judge import _with_max_tokens

        original = JudgeConfig(provider="openai", model="gpt-4o-mini", max_tokens=512, cache=True)
        out = _with_max_tokens(original, None)
        assert out.max_tokens == 512
        assert out.cache is True


class TestCacheFailureDegradesGracefully:
    """Codex P2 #2: A CacheError from get/put must not kill the eval."""

    def test_make_judge_call_falls_through_when_cache_read_fails(self, monkeypatch):
        from multivon_eval import judge as judge_mod
        from multivon_eval.exceptions import CacheError

        class _BrokenCache:
            def get(self, prompt, cfg):
                raise CacheError("simulated read failure")

            def put(self, prompt, cfg, value):
                raise CacheError("simulated write failure")

        monkeypatch.setattr(judge_mod, "_CACHE_DEGRADATION_WARNED", False)
        monkeypatch.setattr("multivon_eval.cache.get_cache", lambda: _BrokenCache())

        called = {"n": 0}

        def fake_uncached(prompt, cfg):
            called["n"] += 1
            return "live response"

        monkeypatch.setattr(judge_mod, "_make_judge_call_uncached", fake_uncached)

        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini", cache=True).resolve()
        result = judge_mod.make_judge_call("hello", cfg)
        assert result == "live response"
        assert called["n"] == 1  # cache failure didn't suppress the live call

    def test_make_judge_call_falls_through_when_get_cache_raises(self, monkeypatch):
        from multivon_eval import judge as judge_mod
        from multivon_eval.exceptions import CacheError

        def boom():
            raise CacheError("can't init cache")

        monkeypatch.setattr(judge_mod, "_CACHE_DEGRADATION_WARNED", False)
        monkeypatch.setattr("multivon_eval.cache.get_cache", boom)
        monkeypatch.setattr(judge_mod, "_make_judge_call_uncached", lambda p, c: "live")

        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini", cache=True).resolve()
        assert judge_mod.make_judge_call("hi", cfg) == "live"

    @pytest.mark.asyncio
    async def test_make_judge_call_async_falls_through_when_cache_fails(self, monkeypatch):
        from multivon_eval import judge as judge_mod
        from multivon_eval.exceptions import CacheError

        class _BrokenCache:
            def get(self, prompt, cfg):
                raise CacheError("read fail")

            def put(self, prompt, cfg, value):
                raise CacheError("write fail")

        monkeypatch.setattr(judge_mod, "_CACHE_DEGRADATION_WARNED", False)
        monkeypatch.setattr("multivon_eval.cache.get_cache", lambda: _BrokenCache())

        async def fake_uncached(prompt, cfg):
            return "live-async"

        monkeypatch.setattr(judge_mod, "_make_judge_call_async_uncached", fake_uncached)

        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini", cache=True).resolve()
        result = await judge_mod.make_judge_call_async("hello", cfg)
        assert result == "live-async"
