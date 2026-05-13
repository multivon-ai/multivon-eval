"""Tests for the per-run token + USD cost accounting."""
from __future__ import annotations

import asyncio

import pytest

from multivon_eval import EvalCase, EvalSuite
from multivon_eval._cost_models import ModelPricing, estimate_cost_usd, register_pricing
from multivon_eval.costs import (
    CostTracker,
    Costs,
    ProviderUsage,
    active_tracker,
    record_call,
    reset_token,
    set_active_tracker,
)
from multivon_eval.evaluators.deterministic import NotEmpty


class TestCostModels:
    def test_known_model_returns_cost(self):
        cost = estimate_cost_usd("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
        # gpt-4o-mini in catalog: 0.15 input + 0.60 output per million
        assert cost == pytest.approx(0.75, rel=1e-6)

    def test_unknown_model_returns_none(self):
        assert estimate_cost_usd("not-a-real-model", input_tokens=100, output_tokens=100) is None

    def test_zero_tokens_is_zero_cost(self):
        cost = estimate_cost_usd("gpt-4o-mini", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_register_pricing_overrides(self):
        register_pricing("custom-model", ModelPricing(input_per_million=1.0, output_per_million=2.0))
        assert estimate_cost_usd("custom-model", input_tokens=1_000_000, output_tokens=500_000) == pytest.approx(2.0)


class TestCostTracker:
    def test_initial_state_is_empty(self):
        t = CostTracker()
        snap = t.snapshot()
        assert snap.total_calls == 0
        assert snap.total_tokens == 0
        assert snap.total_cost_usd == 0.0
        assert snap.by_model == []

    def test_records_per_model(self):
        t = CostTracker()
        t.record(provider="openai", model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        t.record(provider="openai", model="gpt-4o-mini", input_tokens=200, output_tokens=80)
        t.record(provider="anthropic", model="claude-haiku-4-5-20251001", input_tokens=300, output_tokens=20)
        snap = t.snapshot()
        assert snap.total_calls == 3
        assert snap.total_input_tokens == 600
        assert snap.total_output_tokens == 150
        assert len(snap.by_model) == 2
        # by_model sorted by (provider, model)
        anthropic, openai = snap.by_model
        assert anthropic.provider == "anthropic"
        assert anthropic.calls == 1
        assert openai.calls == 2
        assert openai.input_tokens == 300

    def test_unknown_model_yields_none_cost(self):
        t = CostTracker()
        t.record(provider="custom", model="totally-unknown", input_tokens=100, output_tokens=50)
        snap = t.snapshot()
        assert snap.by_model[0].cost_usd is None
        assert snap.total_cost_usd is None  # any unknown model → total is None

    def test_snapshot_is_a_copy(self):
        t = CostTracker()
        t.record(provider="openai", model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        snap_a = t.snapshot()
        t.record(provider="openai", model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        # snap_a must not see the second record
        assert snap_a.total_calls == 1
        # but a fresh snapshot does
        snap_b = t.snapshot()
        assert snap_b.total_calls == 2


class TestActiveTrackerContextVar:
    def test_no_tracker_means_record_is_noop(self):
        # Clear any leftover tracker.
        tok = set_active_tracker(None)
        try:
            assert active_tracker() is None
            # Must not raise.
            record_call(provider="openai", model="gpt-4o-mini", input_tokens=10, output_tokens=5)
        finally:
            reset_token(tok)

    def test_set_active_tracker_routes_records(self):
        t = CostTracker()
        tok = set_active_tracker(t)
        try:
            record_call(provider="openai", model="gpt-4o-mini", input_tokens=10, output_tokens=5)
        finally:
            reset_token(tok)
        snap = t.snapshot()
        assert snap.total_calls == 1
        assert snap.total_input_tokens == 10

    def test_tracker_is_async_safe(self):
        """Concurrent async tasks see the SAME tracker (contextvar inherits)."""
        async def _run():
            t = CostTracker()
            tok = set_active_tracker(t)
            try:
                async def task():
                    record_call(provider="openai", model="gpt-4o-mini",
                                input_tokens=10, output_tokens=5)
                await asyncio.gather(task(), task(), task())
            finally:
                reset_token(tok)
            return t.snapshot()

        snap = asyncio.run(_run())
        assert snap.total_calls == 3


class TestSuiteIntegration:
    def test_report_has_costs_object(self):
        suite = EvalSuite("costs report")
        suite.add_cases([EvalCase(input="a"), EvalCase(input="b")])
        suite.add_evaluators(NotEmpty())
        report = suite.run(lambda p: "yes", verbose=False)
        assert report.costs is not None
        assert isinstance(report.costs, Costs)
        # NotEmpty makes no judge calls.
        assert report.costs.total_calls == 0

    def test_costs_serialize_in_json(self):
        suite = EvalSuite("costs json")
        suite.add_cases([EvalCase(input="a")])
        suite.add_evaluators(NotEmpty())
        report = suite.run(lambda p: "yes", verbose=False)
        import json
        data = json.loads(report.to_json())
        assert "costs" in data["summary"]
        assert data["summary"]["costs"]["total_calls"] == 0

    @pytest.mark.asyncio
    async def test_async_run_populates_costs(self):
        suite = EvalSuite("costs async")
        suite.add_cases([EvalCase(input="a")])
        suite.add_evaluators(NotEmpty())

        async def m(p): return "ok"
        report = await suite.run_async(m, verbose=False)
        assert report.costs is not None
        assert report.costs.total_calls == 0


class TestCostsStr:
    def test_empty_costs_str(self):
        c = Costs()
        assert "no judge calls" in str(c)

    def test_populated_costs_str(self):
        c = Costs(by_model=[
            ProviderUsage(
                provider="openai", model="gpt-4o-mini",
                input_tokens=1000, output_tokens=500,
                calls=3, cost_usd=0.00045,
            ),
        ])
        s = str(c)
        assert "gpt-4o-mini" in s
        assert "3 calls" in s
        assert "1,000" in s
        assert "$0.0005" in s or "$0.0004" in s
