"""Tests for the experimental multimodal evaluators.

These tests exercise the public surface (imports, error paths, image
metadata parsing) without making real vision-model API calls. The
provider-specific call paths are mocked because (a) they need API keys
and (b) we want a deterministic CI signal — the vision providers'
behavior is exercised separately in `test_integrations_live.py`.
"""
from __future__ import annotations

import base64
import io
import json
from unittest.mock import patch

import pytest

from multivon_eval import (
    DocumentGrounding,
    EvalCase,
    JudgeConfig,
    VQAFaithfulness,
)
from multivon_eval.evaluators.multimodal import (
    _image_to_data_uri,
    _is_vision_capable,
    _parse_yes_no,
)


# --- helpers ---------------------------------------------------------------

def _png_bytes() -> bytes:
    """1x1 white PNG — small enough to inline."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP8//8/AwAI/AL+XSb9PgAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def png_path(tmp_path):
    p = tmp_path / "test.png"
    p.write_bytes(_png_bytes())
    return str(p)


# --- _is_vision_capable ----------------------------------------------------

def test_is_vision_capable_known_models():
    assert _is_vision_capable(JudgeConfig(provider="anthropic", model="claude-sonnet-4-6"))
    assert _is_vision_capable(JudgeConfig(provider="openai", model="gpt-4o-mini"))
    assert _is_vision_capable(JudgeConfig(provider="google", model="gemini-2.5-flash"))


def test_is_vision_capable_rejects_known_text_only():
    # gpt-3.5-turbo is not in the vision-capable map for openai.
    assert not _is_vision_capable(JudgeConfig(provider="openai", model="gpt-3.5-turbo"))


def test_is_vision_capable_unknown_provider_passes_through():
    # litellm is not in the capability map → we don't gate it (let provider error)
    assert _is_vision_capable(JudgeConfig(provider="litellm", model="anything"))


# --- _image_to_data_uri ----------------------------------------------------

def test_image_to_data_uri_local_path(png_path):
    uri, mime, b64 = _image_to_data_uri(png_path)
    assert uri.startswith("data:image/png;base64,")
    assert mime == "image/png"
    assert b64  # non-empty


def test_image_to_data_uri_http_url_passthrough():
    uri, mime, b64 = _image_to_data_uri("https://example.com/foo.jpg")
    assert uri == "https://example.com/foo.jpg"
    assert mime == "image/jpeg"
    assert b64 == ""


def test_image_to_data_uri_data_uri_passthrough():
    src = "data:image/png;base64,iVBORw0KGgoAAAANSU="
    uri, mime, b64 = _image_to_data_uri(src)
    assert uri == src
    assert mime == "image/png"
    assert b64 == "iVBORw0KGgoAAAANSU="


def test_image_to_data_uri_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _image_to_data_uri(str(tmp_path / "nope.png"))


# --- _parse_yes_no ---------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Yes", True),
    ("no, that's wrong", False),
    ("YES.", True),
    ("nope", False),
    ("", False),
    ("Maybe yes", True),
])
def test_parse_yes_no(text, expected):
    assert _parse_yes_no(text) is expected


# --- VQAFaithfulness -------------------------------------------------------

def test_vqa_faithfulness_requires_image():
    """Without an image in metadata, returns 0.0 with explanation."""
    e = VQAFaithfulness(judge=JudgeConfig(provider="google", model="gemini-2.5-flash"))
    case = EvalCase(input="what is in the image?")
    res = e.evaluate(case, "A cat sits on a mat.")
    assert res.score == 0.0
    assert "No image provided" in res.reason


def test_vqa_faithfulness_calls_vision_judge_twice(png_path):
    """Happy path: extract-claims call, then 1 verify call per claim.

    We mock _call_vision_judge entirely so the test is deterministic and
    requires no provider SDKs installed.
    """
    e = VQAFaithfulness(judge=JudgeConfig(provider="google", model="gemini-2.5-flash"))
    case = EvalCase(
        input="what's in the image?",
        metadata={"image_path": png_path},
    )
    # First call returns claims JSON; subsequent calls return yes/no
    call_results = [
        '["A cat is visible.", "The cat is orange."]',
        "Yes",
        "No",
    ]
    with patch(
        "multivon_eval.evaluators.multimodal._call_vision_judge",
        side_effect=call_results,
    ):
        res = e.evaluate(case, "A cat is visible. The cat is orange.")
    assert 0.0 < res.score < 1.0
    assert "1/2" in res.reason


def test_vqa_faithfulness_no_claims_is_trivially_faithful(png_path):
    """If the judge extracts no claims, score 1.0 (nothing to be wrong about)."""
    e = VQAFaithfulness(judge=JudgeConfig(provider="google", model="gemini-2.5-flash"))
    case = EvalCase(input="anything", metadata={"image_path": png_path})
    with patch(
        "multivon_eval.evaluators.multimodal._call_vision_judge",
        return_value="[]",
    ):
        res = e.evaluate(case, "I cannot tell from this image.")
    assert res.score == 1.0


def test_vqa_faithfulness_accepts_images_list(png_path):
    """case.metadata['images'] list is read."""
    e = VQAFaithfulness(judge=JudgeConfig(provider="google", model="gemini-2.5-flash"))
    case = EvalCase(input="anything", metadata={"images": [png_path]})
    with patch(
        "multivon_eval.evaluators.multimodal._call_vision_judge",
        return_value="[]",
    ):
        res = e.evaluate(case, "Nothing")
    assert res.score == 1.0


# --- DocumentGrounding -----------------------------------------------------

def test_document_grounding_requires_images():
    e = DocumentGrounding(judge=JudgeConfig(provider="google", model="gemini-2.5-flash"))
    case = EvalCase(input="summarize")
    res = e.evaluate(case, "The contract says X.")
    assert res.score == 0.0
    assert "No document pages" in res.reason


def test_document_grounding_parses_q1_q2_q3(png_path):
    """Three Q1/Q2/Q3 lines parsed; score = fraction yes."""
    e = DocumentGrounding(judge=JudgeConfig(provider="google", model="gemini-2.5-flash"))
    case = EvalCase(input="anything", metadata={"images": [png_path, png_path]})
    with patch(
        "multivon_eval.evaluators.multimodal._call_vision_judge",
        return_value="Q1: Yes\nQ2: No\nQ3: Yes",
    ):
        res = e.evaluate(case, "The contract is valid until 2027.")
    assert res.score == pytest.approx(2 / 3)
    assert "✓ Q1" in res.reason
    assert "✗ Q2" in res.reason


def test_document_grounding_missing_q_is_failing(png_path):
    e = DocumentGrounding(judge=JudgeConfig(provider="google", model="gemini-2.5-flash"))
    case = EvalCase(input="anything", metadata={"images": [png_path]})
    with patch(
        "multivon_eval.evaluators.multimodal._call_vision_judge",
        return_value="Q1: Yes",  # missing Q2 and Q3
    ):
        res = e.evaluate(case, "Something")
    # 1/3 yes, the rest fail
    assert res.score == pytest.approx(1 / 3)


# --- exports ---------------------------------------------------------------

def test_multimodal_evaluators_exported():
    """The package surface includes both classes (and they're importable from top level)."""
    import multivon_eval
    assert hasattr(multivon_eval, "VQAFaithfulness")
    assert hasattr(multivon_eval, "DocumentGrounding")
    assert "VQAFaithfulness" in multivon_eval.__all__
    assert "DocumentGrounding" in multivon_eval.__all__
