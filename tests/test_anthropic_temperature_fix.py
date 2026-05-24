"""Tests for the Opus 4-7 / Opus 5+ temperature-deprecation fix.

Anthropic's reasoning-tier models reject the ``temperature`` parameter
with a 400. The AnthropicAdapter must detect those models by name and
omit the field from the request kwargs. Older Anthropic models
(Haiku 4, Sonnet 4, Opus 3, etc.) still receive temperature unchanged.

Originally caught when every pdfhell mini-v4 leaderboard call against
Opus 4-7 silently failed — see pdfhell/research/CORRECTION_NOTICE.md.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from multivon_eval.adapters import AnthropicAdapter


def _make(model: str) -> AnthropicAdapter:
    return AnthropicAdapter(model=model, client=MagicMock())


# ─── _supports_temperature ──────────────────────────────────────────────


def test_opus_4_7_omits_temperature():
    assert _make("claude-opus-4-7")._supports_temperature() is False


def test_opus_4_7_dated_variant_omits_temperature():
    assert _make("claude-opus-4-7-20251101")._supports_temperature() is False


def test_opus_5_omits_temperature():
    assert _make("claude-opus-5")._supports_temperature() is False


def test_opus_5_dated_variant_omits_temperature():
    assert _make("claude-opus-5-1-20260301")._supports_temperature() is False


@pytest.mark.parametrize("model", [
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
])
def test_other_models_keep_temperature(model: str):
    assert _make(model)._supports_temperature() is True


def test_case_insensitive_match():
    # Adapter normalises model.lower() before prefix match — uppercase
    # input from a typo shouldn't accidentally re-enable temperature.
    assert _make("CLAUDE-OPUS-4-7")._supports_temperature() is False


def test_empty_model_falls_back_to_supporting_temperature():
    # If the model string is empty (e.g. default JudgeConfig), we don't
    # know which model will be used — default to "supports" so the
    # behaviour matches the SDK's own default.
    assert _make("")._supports_temperature() is True


# ─── _build_kwargs ──────────────────────────────────────────────────────


def test_build_kwargs_omits_temperature_for_opus_4_7():
    a = _make("claude-opus-4-7")
    kwargs = a._build_kwargs(messages=[{"role": "user", "content": "x"}])
    assert "temperature" not in kwargs, kwargs


def test_build_kwargs_includes_temperature_for_haiku():
    a = _make("claude-haiku-4-5")
    a._temperature = 0.5
    kwargs = a._build_kwargs(messages=[{"role": "user", "content": "x"}])
    assert kwargs["temperature"] == 0.5


def test_build_kwargs_includes_max_tokens_always():
    for model in ("claude-opus-4-7", "claude-haiku-4-5"):
        a = _make(model)
        kwargs = a._build_kwargs(messages=[{"role": "user", "content": "x"}])
        assert kwargs["max_tokens"] == 1024


def test_build_kwargs_includes_system_when_provided():
    a = _make("claude-opus-4-7")
    kwargs = a._build_kwargs(
        messages=[{"role": "user", "content": "x"}],
        system="be brief",
    )
    assert kwargs["system"] == "be brief"


def test_build_kwargs_excludes_system_when_empty():
    a = _make("claude-opus-4-7")
    kwargs = a._build_kwargs(messages=[{"role": "user", "content": "x"}])
    assert "system" not in kwargs


# ─── __call__ integration with mocked client ────────────────────────────


def test_call_sends_no_temperature_for_opus_4_7():
    """End-to-end: AnthropicAdapter("claude-opus-4-7")("hi") sends
    a client.messages.create() call with no `temperature` kwarg."""
    client = MagicMock()
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="ok")]
    client.messages.create.return_value = fake_response

    a = AnthropicAdapter(model="claude-opus-4-7", client=client)
    out = a("hello")

    assert out == "ok"
    client.messages.create.assert_called_once()
    call_kwargs = client.messages.create.call_args.kwargs
    assert "temperature" not in call_kwargs, (
        f"temperature must not be sent for Opus 4-7; got kwargs={list(call_kwargs.keys())}"
    )


def test_call_sends_temperature_for_haiku():
    """Sanity: non-reasoning models still get temperature."""
    client = MagicMock()
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="ok")]
    client.messages.create.return_value = fake_response

    a = AnthropicAdapter(model="claude-haiku-4-5", client=client, temperature=0.3)
    out = a("hello")

    assert out == "ok"
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs.get("temperature") == 0.3
