"""Tests for the async evaluator path and concurrent suite.run_async()."""
from __future__ import annotations

import asyncio
import time

import pytest

from multivon_eval import EvalCase, EvalSuite
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.evaluators.deterministic import NotEmpty, MaxLatency
from multivon_eval.result import EvalResult


class _SyncEvaluator(Evaluator):
    """Sync-only evaluator that takes a measurable amount of time so we can
    assert that concurrent evaluation overlaps the wait."""
    name = "slow_sync"

    def __init__(self, *, sleep_seconds: float = 0.05):
        super().__init__(threshold=0.5)
        self._sleep = sleep_seconds

    def evaluate(self, case, output):
        time.sleep(self._sleep)
        return self._result(1.0, reason=f"slept {self._sleep}s")


class _AsyncOverrideEvaluator(Evaluator):
    """Evaluator that overrides aevaluate() to provide a true async impl."""
    name = "async_override"

    def evaluate(self, case, output):
        # Should not be hit by aevaluate(), but must exist.
        return self._result(0.0, reason="sync fallback hit")

    async def aevaluate(self, case, output, **kwargs):
        await asyncio.sleep(0.05)
        return self._result(1.0, reason="async path")


class TestAsyncBase:
    @pytest.mark.asyncio
    async def test_default_aevaluate_runs_sync_in_thread(self):
        ev = NotEmpty()
        case = EvalCase(input="x")
        result = await ev.aevaluate(case, "non-empty")
        assert result.passed
        assert result.evaluator == "not_empty"

    @pytest.mark.asyncio
    async def test_override_aevaluate_runs_instead_of_default(self):
        ev = _AsyncOverrideEvaluator()
        case = EvalCase(input="x")
        result = await ev.aevaluate(case, "y")
        assert result.passed
        assert result.reason == "async path"


class TestSuiteRunAsync:
    @pytest.mark.asyncio
    async def test_run_async_basic_flow(self):
        async def model(prompt: str) -> str:
            return f"echo: {prompt}"

        suite = EvalSuite("async basic")
        suite.add_cases([EvalCase(input="hello"), EvalCase(input="world")])
        suite.add_evaluators(NotEmpty())

        report = await suite.run_async(model, verbose=False)
        assert report.total == 2
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_run_async_concurrency_overlaps_evaluators(self):
        """Three slow evaluators on one case should not take 3 × sleep — they
        should overlap because aevaluate runs each in a thread."""
        async def model(prompt: str) -> str:
            return "hi"

        suite = EvalSuite("async concurrency")
        suite.add_cases([EvalCase(input="x")])
        suite.add_evaluators(
            _SyncEvaluator(sleep_seconds=0.1),
            _SyncEvaluator(sleep_seconds=0.1),
            _SyncEvaluator(sleep_seconds=0.1),
        )

        t0 = time.time()
        report = await suite.run_async(model, verbose=False)
        elapsed = time.time() - t0
        # Sequential would be ~0.3s. With overlap, well under 0.25s on any
        # machine that isn't pathologically slow.
        assert elapsed < 0.25, f"evaluators did not overlap (elapsed={elapsed:.3f}s)"
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_evaluator_concurrency_limit_serializes(self):
        async def model(prompt: str) -> str:
            return "hi"

        suite = EvalSuite("serialized")
        suite.add_cases([EvalCase(input="x")])
        suite.add_evaluators(
            _SyncEvaluator(sleep_seconds=0.05),
            _SyncEvaluator(sleep_seconds=0.05),
            _SyncEvaluator(sleep_seconds=0.05),
        )

        t0 = time.time()
        await suite.run_async(model, verbose=False, evaluator_concurrency=1)
        elapsed = time.time() - t0
        # 3 × 0.05 = 0.15 floor when strictly serialized.
        assert elapsed >= 0.13, f"evaluator_concurrency=1 should serialise (elapsed={elapsed:.3f}s)"

    @pytest.mark.asyncio
    async def test_run_async_passes_latency_to_max_latency(self):
        async def model(prompt: str) -> str:
            await asyncio.sleep(0.01)
            return "ok"

        suite = EvalSuite("latency")
        suite.add_cases([EvalCase(input="x")])
        suite.add_evaluators(MaxLatency(max_ms=10_000))

        report = await suite.run_async(model, verbose=False)
        assert report.pass_rate == 1.0
        # Verify the evaluator actually saw the latency (not default 0.0).
        assert "ms" in report.case_results[0].results[0].reason

    @pytest.mark.asyncio
    async def test_model_error_skips_non_latency_evaluators(self):
        async def model(prompt: str) -> str:
            raise RuntimeError("upstream down")

        suite = EvalSuite("model-error")
        suite.add_cases([EvalCase(input="x")])
        suite.add_evaluators(NotEmpty(), MaxLatency(max_ms=10_000))

        report = await suite.run_async(model, verbose=False)
        cr = report.case_results[0]
        assert cr.model_error == "upstream down"
        not_empty_result = next(r for r in cr.results if r.evaluator == "not_empty")
        assert "skipped" in not_empty_result.reason
