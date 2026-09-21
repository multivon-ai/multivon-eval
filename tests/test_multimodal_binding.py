"""Content-bound media reaching the vision evaluators.

The evaluators previously read unbound path/URL strings out of case.metadata,
so a regrade could not show which bytes were graded. These tests cover the
bound path, the failures it is supposed to produce, and the deprecation of the
unbound one. No vision model is called.
"""
from __future__ import annotations

import base64
import io
import warnings
import wave
from unittest.mock import patch

import pytest

pytest.importorskip('PIL')

from PIL import Image

from multivon_eval import (
    DocumentGrounding,
    EvalCase,
    JudgeConfig,
    MediaArtifact,
    VQAFaithfulness,
    case_media,
    media_sources,
    with_media,
)

MODULE = 'multivon_eval.evaluators.multimodal.call_vision'
JUDGE = JudgeConfig(provider='google', model='gemini-2.5-flash')


def png_bytes(color='white', width=8, height=8) -> bytes:
    stream = io.BytesIO()
    Image.new('RGB', (width, height), color).save(stream, format='PNG')
    return stream.getvalue()


def wav_bytes() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(bytes(800))
    return stream.getvalue()


def bound_case(content: bytes, media_type: str = 'image/png') -> tuple[EvalCase, MediaArtifact]:
    artifact = MediaArtifact.capture(content, media_type)
    return with_media(EvalCase(input='What is shown?'), artifact), artifact


def resolver_for(content: bytes):
    return lambda artifact: content


# --- the bound path --------------------------------------------------------

def test_bound_media_reaches_the_judge_as_a_verified_data_uri():
    content = png_bytes()
    case, _ = bound_case(content)
    expected = base64.b64encode(content).decode('ascii')

    evaluator = VQAFaithfulness(judge=JUDGE, media_resolver=resolver_for(content))
    with patch(MODULE, side_effect=['["a white square"]', 'Yes']) as call:
        result = evaluator.evaluate(case, 'It shows a white square.')

    sources = call.call_args_list[0].args[1]
    assert sources == [f'data:image/png;base64,{expected}']
    assert result.metadata['media_bound'] is True
    assert result.score == 1.0


def test_document_grounding_records_that_media_was_bound():
    content = png_bytes()
    case, _ = bound_case(content)
    evaluator = DocumentGrounding(judge=JUDGE, media_resolver=resolver_for(content))
    with patch(MODULE, return_value='Q1: Yes\nQ2: Yes\nQ3: No'):
        result = evaluator.evaluate(case, 'The total is 12.')
    assert result.metadata['media_bound'] is True
    assert result.score == pytest.approx(2 / 3)


def test_bound_media_does_not_warn():
    content = png_bytes()
    case, _ = bound_case(content)
    evaluator = VQAFaithfulness(judge=JUDGE, media_resolver=resolver_for(content))
    with patch(MODULE, side_effect=['["a white square"]', 'Yes']):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            evaluator.evaluate(case, 'It shows a white square.')
    assert not [w for w in recorded if issubclass(w.category, DeprecationWarning)]


# --- the failures binding is supposed to produce ---------------------------

def test_bytes_that_no_longer_match_the_descriptor_fail_before_the_judge():
    case, _ = bound_case(png_bytes('white'))
    evaluator = VQAFaithfulness(judge=JUDGE, media_resolver=resolver_for(png_bytes('black')))
    with patch(MODULE) as call, pytest.raises(ValueError, match='do not match the bound content'):
        evaluator.evaluate(case, 'It shows a white square.')
    call.assert_not_called()


def test_bound_media_without_a_resolver_refuses_to_grade():
    case, _ = bound_case(png_bytes())
    evaluator = VQAFaithfulness(judge=JUDGE)
    with patch(MODULE) as call, pytest.raises(ValueError, match='explicit bytes resolver'):
        evaluator.evaluate(case, 'It shows a white square.')
    call.assert_not_called()


def test_bound_audio_names_the_unsupported_type():
    content = wav_bytes()
    case, _ = bound_case(content, 'audio/wav')
    evaluator = VQAFaithfulness(judge=JUDGE, media_resolver=resolver_for(content))
    with patch(MODULE) as call, pytest.raises(ValueError, match='audio/wav'):
        evaluator.evaluate(case, 'It says hello.')
    call.assert_not_called()


def test_media_resolver_must_be_callable():
    with pytest.raises(TypeError, match='must be callable'):
        VQAFaithfulness(judge=JUDGE, media_resolver='not-callable')


# --- the unbound path is still accepted, and says so -----------------------

def test_unbound_metadata_references_warn_but_still_grade(tmp_path):
    path = tmp_path / 'page.png'
    path.write_bytes(png_bytes())
    case = EvalCase(input='What is shown?', metadata={'image_path': str(path)})
    evaluator = VQAFaithfulness(judge=JUDGE)
    with patch(MODULE, side_effect=['["a white square"]', 'Yes']):
        with pytest.warns(DeprecationWarning, match='do not bind file bytes'):
            result = evaluator.evaluate(case, 'It shows a white square.')
    assert result.metadata['media_bound'] is False


def test_no_media_at_all_is_unmeasured():
    evaluator = VQAFaithfulness(judge=JUDGE)
    with patch(MODULE) as call:
        result = evaluator.evaluate(EvalCase(input='What is shown?'), 'Anything.')
    call.assert_not_called()
    assert result.score is None or not result.passed


# --- media_sources itself --------------------------------------------------

def test_media_sources_is_empty_without_bound_media():
    assert media_sources(EvalCase(input='q'), None) == ()


def test_media_sources_preserves_bound_order():
    first, second = png_bytes('white'), png_bytes('black')
    case = with_media(
        EvalCase(input='q'),
        MediaArtifact.capture(first, 'image/png'),
        MediaArtifact.capture(second, 'image/png'),
    )
    by_digest = {MediaArtifact.capture(c, 'image/png').id: c for c in (first, second)}
    uris = media_sources(case, lambda artifact: by_digest[artifact.id])
    assert uris == tuple(
        f'data:image/png;base64,{base64.b64encode(c).decode("ascii")}' for c in (first, second))
    assert [a.id for a in case_media(case)] == list(by_digest)


def test_binding_api_is_public():
    import multivon_eval

    for name in ('MediaArtifact', 'with_media', 'case_media', 'media_sources', 'call_vision'):
        assert name in multivon_eval.__all__
        assert hasattr(multivon_eval, name)
