"""Tests for the structured exception hierarchy."""
from __future__ import annotations

import pytest

from multivon_eval import (
    MultivonError,
    JudgeUnavailable,
    CalibrationMissing,
    EvaluatorPrereqMissing,
    CacheError,
    SecretsError,
    ComplianceError,
    JudgeConfig,
)
from multivon_eval.calibration import calibrated_threshold


class TestHierarchy:
    def test_all_are_subclasses_of_multivon_error(self):
        for exc in (
            JudgeUnavailable,
            CalibrationMissing,
            EvaluatorPrereqMissing,
            CacheError,
            SecretsError,
            ComplianceError,
        ):
            assert issubclass(exc, MultivonError), exc

    def test_judge_unavailable_carries_provider_model(self):
        exc = JudgeUnavailable("bang", provider="anthropic", model="x")
        assert exc.provider == "anthropic"
        assert exc.model == "x"
        assert "bang" in str(exc)

    def test_evaluator_prereq_missing_message(self):
        exc = EvaluatorPrereqMissing("Faithfulness", "context")
        assert exc.evaluator == "Faithfulness"
        assert exc.missing == "context"
        assert "context" in str(exc)
        assert "Faithfulness" in str(exc)


class TestCalibrationStrict:
    def test_strict_raises_for_unknown_combination(self):
        cfg = JudgeConfig(provider="openai", model="frontier-model-9000").resolve()
        with pytest.raises(CalibrationMissing) as info:
            calibrated_threshold("hallucination", cfg, strict=True)
        assert info.value.evaluator == "hallucination"
        assert info.value.judge_model == "frontier-model-9000"

    def test_non_strict_falls_back_to_default(self):
        cfg = JudgeConfig(provider="openai", model="frontier-model-9000").resolve()
        assert calibrated_threshold("hallucination", cfg) == 0.7

    def test_strict_does_not_raise_for_known_combination(self):
        cfg = JudgeConfig(provider="anthropic", model="claude-haiku-4-5-20251001").resolve()
        thr = calibrated_threshold("hallucination", cfg, strict=True)
        assert 0 < thr < 1


def _raise(exc_cls):
    raise exc_cls("test message")


class TestCatching:
    def test_root_catches_all(self):
        for cls in (JudgeUnavailable, CacheError, ComplianceError):
            with pytest.raises(MultivonError):
                if cls is JudgeUnavailable:
                    raise cls("x", provider="p", model="m")
                _raise(cls)
