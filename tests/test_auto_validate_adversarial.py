"""Validation tests for multivon_eval.auto.validate_adversarial_cases.

N-shot aggregation, hardness_band filtering, and the resilience knobs
(baseline crash → counted as failure, evaluator crash → case dropped).

Uses a stub evaluator monkeypatched onto multivon_eval to avoid any LLM
calls — the tests are deterministic and run in <1 second.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest

import multivon_eval as m
from multivon_eval import EvalCase
from multivon_eval.auto import HardnessReport, validate_adversarial_cases
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.result import EvalResult


class _ScriptedEvaluator(Evaluator):
    """Evaluator whose ``evaluate`` calls return a pre-scripted sequence.

    Each call pops the next (score, passed) from the script. Lets a test
    say "shot 1 passes, shot 2 fails, shot 3 fails" deterministically.
    """

    name = "ScriptedEvaluator"
    _script: list[tuple[float, bool]] = []
    _calls: int = 0

    @classmethod
    def set_script(cls, script: list[tuple[float, bool]]) -> None:
        cls._script = list(script)
        cls._calls = 0

    @classmethod
    def call_count(cls) -> int:
        return cls._calls

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:  # noqa: ARG002
        idx = type(self)._calls
        type(self)._calls += 1
        if idx >= len(type(self)._script):
            # Default if script is exhausted — fail-safe pass
            score, passed = 1.0, True
        else:
            score, passed = type(self)._script[idx]
        return EvalResult(
            evaluator=self.name, score=score, passed=passed,
            reason="scripted", metadata={},
        )


class _AlwaysCrashEvaluator(Evaluator):
    """Evaluator that always raises — used to test evaluator-crash handling."""

    name = "AlwaysCrashEvaluator"

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:  # noqa: ARG002
        raise RuntimeError("simulated evaluator crash")


@pytest.fixture(autouse=True)
def register_stub_evaluators() -> Iterator[None]:
    """Make the stub evaluators discoverable by validate_adversarial_cases.

    The function looks up evaluator classes via ``getattr(multivon_eval, name)``
    so we attach the stubs to the module for the duration of the test.
    """
    m.ScriptedEvaluator = _ScriptedEvaluator  # type: ignore[attr-defined]
    m.AlwaysCrashEvaluator = _AlwaysCrashEvaluator  # type: ignore[attr-defined]
    _ScriptedEvaluator.set_script([])
    yield
    del m.ScriptedEvaluator  # type: ignore[attr-defined]
    del m.AlwaysCrashEvaluator  # type: ignore[attr-defined]


def _adversarial(input_text: str, evaluator_name: str = "ScriptedEvaluator") -> EvalCase:
    """Build a minimal case that looks like generate_adversarial_cases output."""
    return EvalCase(
        input=input_text,
        metadata={"stress_tests": [evaluator_name]},
        tags=["adversarial:test"],
    )


# ─── Aggregation math ─────────────────────────────────────────────────────

def test_n_shots_aggregates_failure_rate_correctly():
    # 3 shots: pass, fail, fail → failure_rate = 2/3
    _ScriptedEvaluator.set_script([(0.9, True), (0.1, False), (0.1, False)])
    case = _adversarial("q")
    _, reports = validate_adversarial_cases(
        [case], lambda _: "baseline output", n_shots=3,
        hardness_band=(0.0, 1.0),
    )
    assert len(reports) == 1
    assert reports[0].n_shots == 3
    assert reports[0].failure_rate == pytest.approx(2 / 3)
    assert reports[0].scores == [0.9, 0.1, 0.1]


def test_baseline_outputs_recorded_per_shot():
    _ScriptedEvaluator.set_script([(0.5, True)] * 3)
    case = _adversarial("q")
    outputs = iter(["a", "b", "c"])
    _, reports = validate_adversarial_cases(
        [case], lambda _: next(outputs), n_shots=3,
        hardness_band=(0.0, 1.0),
    )
    assert reports[0].baseline_outputs == ["a", "b", "c"]


# ─── Hardness band filtering ──────────────────────────────────────────────

def test_kept_when_failure_rate_in_band():
    # 3 shots all fail → failure_rate = 1.0, inside (0.5, 1.0)
    _ScriptedEvaluator.set_script([(0.1, False)] * 3)
    case = _adversarial("q")
    kept, reports = validate_adversarial_cases(
        [case], lambda _: "x", n_shots=3, hardness_band=(0.5, 1.0),
    )
    assert kept == [case]
    assert reports[0].in_hardness_band is True


def test_dropped_when_failure_rate_below_band():
    # 3 shots all pass → failure_rate = 0.0, below (0.5, 1.0)
    _ScriptedEvaluator.set_script([(0.9, True)] * 3)
    case = _adversarial("q")
    kept, reports = validate_adversarial_cases(
        [case], lambda _: "x", n_shots=3, hardness_band=(0.5, 1.0),
    )
    assert kept == []
    assert reports[0].in_hardness_band is False


def test_dropped_when_failure_rate_above_band():
    # 4 shots all fail → failure_rate = 1.0, above (0.2, 0.6)
    _ScriptedEvaluator.set_script([(0.1, False)] * 4)
    case = _adversarial("q")
    kept, reports = validate_adversarial_cases(
        [case], lambda _: "x", n_shots=4, hardness_band=(0.2, 0.6),
    )
    assert kept == []
    assert reports[0].in_hardness_band is False
    assert reports[0].failure_rate == 1.0


def test_discriminating_band_keeps_partial_failures():
    # 5 shots: 2 fails (rate=0.4) → inside (0.2, 0.6) discriminating band
    _ScriptedEvaluator.set_script([
        (0.9, True), (0.1, False), (0.9, True), (0.1, False), (0.9, True),
    ])
    case = _adversarial("q")
    kept, reports = validate_adversarial_cases(
        [case], lambda _: "x", n_shots=5, hardness_band=(0.2, 0.6),
    )
    assert kept == [case]
    assert reports[0].failure_rate == pytest.approx(0.4)


# ─── n_shots=1 degenerate case ────────────────────────────────────────────

def test_single_shot_still_works():
    # n_shots=1 collapses to single-shot behavior — band must be (0, 1) ish
    _ScriptedEvaluator.set_script([(0.0, False)])
    case = _adversarial("q")
    kept, reports = validate_adversarial_cases(
        [case], lambda _: "x", n_shots=1, hardness_band=(0.5, 1.0),
    )
    assert kept == [case]
    assert reports[0].n_shots == 1
    assert reports[0].failure_rate == 1.0


# ─── Derived properties ──────────────────────────────────────────────────

def test_baseline_failed_property_uses_majority_threshold():
    # 3 shots: 2 fails → failure_rate 0.67 → baseline_failed True
    _ScriptedEvaluator.set_script([(0.9, True), (0.1, False), (0.1, False)])
    case = _adversarial("q")
    _, reports = validate_adversarial_cases(
        [case], lambda _: "x", n_shots=3, hardness_band=(0.0, 1.0),
    )
    assert reports[0].baseline_failed is True

    # 3 shots: 1 fail → failure_rate 0.33 → baseline_failed False
    _ScriptedEvaluator.set_script([(0.9, True), (0.1, False), (0.9, True)])
    case2 = _adversarial("q2")
    _, reports2 = validate_adversarial_cases(
        [case2], lambda _: "x", n_shots=3, hardness_band=(0.0, 1.0),
    )
    assert reports2[0].baseline_failed is False


def test_baseline_score_property_is_mean():
    _ScriptedEvaluator.set_script([(0.2, False), (0.4, False), (0.9, True)])
    case = _adversarial("q")
    _, reports = validate_adversarial_cases(
        [case], lambda _: "x", n_shots=3, hardness_band=(0.0, 1.0),
    )
    assert reports[0].baseline_score == pytest.approx((0.2 + 0.4 + 0.9) / 3)


# ─── Resilience: baseline / evaluator crashes ─────────────────────────────

def test_baseline_crash_counts_as_failure():
    # Baseline raises on every call — counts as 3 failures, kept under (0.5, 1.0)
    _ScriptedEvaluator.set_script([])  # never called

    def crashing_baseline(_: str) -> str:
        raise RuntimeError("simulated baseline crash")

    case = _adversarial("q")
    kept, reports = validate_adversarial_cases(
        [case], crashing_baseline, n_shots=3, hardness_band=(0.5, 1.0),
    )
    assert kept == [case]
    assert reports[0].failure_rate == 1.0
    assert reports[0].scores == [0.0, 0.0, 0.0]
    assert reports[0].baseline_outputs == ["", "", ""]


def test_baseline_crash_mixed_with_passes():
    # Crash on shot 2 (counts as fail), pass on 1 and 3 → failure_rate = 1/3
    _ScriptedEvaluator.set_script([(0.9, True), (0.9, True)])  # only 2 calls
    call_count = {"n": 0}

    def flaky_baseline(_: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("flaky")
        return "ok"

    case = _adversarial("q")
    _, reports = validate_adversarial_cases(
        [case], flaky_baseline, n_shots=3, hardness_band=(0.0, 1.0),
    )
    assert reports[0].failure_rate == pytest.approx(1 / 3)
    # Crashed shot is recorded with empty output + 0.0 score
    assert reports[0].baseline_outputs[1] == ""
    assert reports[0].scores[1] == 0.0


def test_evaluator_crash_drops_case_entirely():
    case = _adversarial("q", evaluator_name="AlwaysCrashEvaluator")
    kept, reports = validate_adversarial_cases(
        [case], lambda _: "x", n_shots=3, hardness_band=(0.0, 1.0),
    )
    # Case is absent from BOTH kept and reports — tooling failures
    # shouldn't pollute either set.
    assert kept == []
    assert reports == []


# ─── Argument validation ─────────────────────────────────────────────────

def test_n_shots_zero_raises():
    with pytest.raises(ValueError, match="n_shots"):
        validate_adversarial_cases(
            [_adversarial("q")], lambda _: "x", n_shots=0,
        )


def test_invalid_hardness_band_raises():
    with pytest.raises(ValueError, match="hardness_band"):
        validate_adversarial_cases(
            [_adversarial("q")], lambda _: "x",
            hardness_band=(0.7, 0.3),  # lo > hi
        )

    with pytest.raises(ValueError, match="hardness_band"):
        validate_adversarial_cases(
            [_adversarial("q")], lambda _: "x",
            hardness_band=(-0.1, 0.5),  # out of [0, 1]
        )


# ─── Skip conditions ─────────────────────────────────────────────────────

def test_case_without_stress_tests_metadata_skipped():
    case = EvalCase(input="q")  # no metadata
    kept, reports = validate_adversarial_cases(
        [case], lambda _: "x", n_shots=3,
    )
    assert kept == []
    assert reports == []


def test_stress_test_evaluator_not_in_sdk_skipped():
    case = EvalCase(
        input="q",
        metadata={"stress_tests": ["EvaluatorThatDoesNotExist"]},
    )
    kept, reports = validate_adversarial_cases(
        [case], lambda _: "x", n_shots=3,
    )
    assert kept == []
    assert reports == []


def test_empty_case_list_returns_empty():
    kept, reports = validate_adversarial_cases(
        [], lambda _: "x", n_shots=3,
    )
    assert kept == []
    assert reports == []


# ─── Multiple cases with mixed outcomes ───────────────────────────────────

def test_mixed_cases_filtered_independently():
    # Two cases. Script feeds them in the order they're evaluated.
    # Case A: 3 fails → kept under (0.5, 1.0)
    # Case B: 3 passes → dropped under (0.5, 1.0)
    _ScriptedEvaluator.set_script([
        (0.1, False), (0.1, False), (0.1, False),  # case A
        (0.9, True), (0.9, True), (0.9, True),     # case B
    ])
    case_a = _adversarial("a")
    case_b = _adversarial("b")
    kept, reports = validate_adversarial_cases(
        [case_a, case_b], lambda _: "x", n_shots=3, hardness_band=(0.5, 1.0),
    )
    assert kept == [case_a]
    assert len(reports) == 2
    assert reports[0].in_hardness_band is True
    assert reports[1].in_hardness_band is False


def test_evaluator_called_exactly_n_shots_times_per_case():
    _ScriptedEvaluator.set_script([(0.5, True)] * 6)
    cases = [_adversarial(f"q{i}") for i in range(2)]
    validate_adversarial_cases(
        cases, lambda _: "x", n_shots=3, hardness_band=(0.0, 1.0),
    )
    assert _ScriptedEvaluator.call_count() == 6  # 2 cases × 3 shots
