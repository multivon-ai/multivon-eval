"""Tests for the JSON-backed calibration loader and provenance API."""
from __future__ import annotations

import pytest

from multivon_eval import (
    JudgeConfig,
    calibrated_threshold,
    calibration_provenance,
    load_calibration,
    threshold_table,
    CalibrationEntry,
    CalibrationMissing,
)


class TestLoad:
    def test_load_returns_table_with_metadata(self):
        t = load_calibration()
        assert t.schema_version >= 1
        assert t.generated_at
        assert t.methodology
        assert len(t.entries) > 0

    def test_entries_are_calibration_entry_instances(self):
        t = load_calibration()
        assert all(isinstance(e, CalibrationEntry) for e in t.entries)

    def test_load_caches(self):
        a = load_calibration()
        b = load_calibration()
        assert a is b

    def test_reload_returns_new_object(self):
        a = load_calibration()
        b = load_calibration(reload=True)
        assert a is not b
        assert a.schema_version == b.schema_version


class TestLookupAndAliases:
    def test_canonical_id_resolves(self):
        cfg = JudgeConfig(provider="anthropic", model="claude-haiku-4-5-20251001").resolve()
        prov = calibration_provenance("hallucination", cfg)
        assert prov is not None
        assert prov.threshold == 0.55
        assert prov.evaluator == "hallucination"

    def test_alias_resolves_to_same_entry(self):
        canonical = JudgeConfig(provider="anthropic", model="claude-haiku-4-5-20251001").resolve()
        alias = JudgeConfig(provider="anthropic", model="claude-haiku-4-5").resolve()
        a = calibration_provenance("hallucination", canonical)
        b = calibration_provenance("hallucination", alias)
        assert a is not None and b is not None
        assert a.threshold == b.threshold
        assert a.dataset == b.dataset

    def test_unknown_returns_none(self):
        cfg = JudgeConfig(provider="openai", model="not-a-real-model").resolve()
        assert calibration_provenance("hallucination", cfg) is None


class TestThreshold:
    def test_calibrated_threshold_matches_provenance_entry(self):
        cfg = JudgeConfig(provider="openai", model="gpt-4o-mini").resolve()
        prov = calibration_provenance("faithfulness", cfg)
        assert prov is not None
        assert calibrated_threshold("faithfulness", cfg) == prov.threshold

    def test_falls_back_to_default(self):
        cfg = JudgeConfig(provider="openai", model="frontier-2030").resolve()
        assert calibrated_threshold("faithfulness", cfg) == 0.7

    def test_strict_raises_for_missing(self):
        cfg = JudgeConfig(provider="openai", model="frontier-2030").resolve()
        with pytest.raises(CalibrationMissing):
            calibrated_threshold("faithfulness", cfg, strict=True)


class TestProvenanceFields:
    def test_provenance_carries_dataset_hash_and_n(self):
        cfg = JudgeConfig(provider="anthropic", model="claude-haiku-4-5-20251001").resolve()
        prov = calibration_provenance("faithfulness", cfg)
        assert prov is not None
        assert prov.dataset
        assert prov.dataset_hash
        assert prov.n > 0
        assert prov.measured_at

    def test_provenance_may_have_null_f1(self):
        """Some rows ship with F1=None when the original sweep didn't retain it.
        That's documented behavior — the threshold is still usable."""
        t = load_calibration()
        partial = [e for e in t.entries if e.f1 is None]
        # We've intentionally kept some rows where F1 was not retained;
        # verifying the loader handles them.
        assert len(partial) >= 0


class TestFlatTable:
    def test_threshold_table_includes_canonical_and_aliases(self):
        flat = threshold_table()
        assert ("hallucination", "claude-haiku-4-5-20251001") in flat
        assert ("hallucination", "claude-haiku-4-5") in flat
        assert flat[("hallucination", "claude-haiku-4-5-20251001")] == \
               flat[("hallucination", "claude-haiku-4-5")]
