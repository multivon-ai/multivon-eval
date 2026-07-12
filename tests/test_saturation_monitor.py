"""Saturation monitor — EvalReport.saturated / min_detectable_regression /
purpose plumbing + the terminal graduation / regression-triage warnings.

All offline: deterministic evaluators only, no judge calls.
"""
from __future__ import annotations

import json

import pytest

from multivon_eval import EvalCase, EvalSuite
from multivon_eval.result import CaseResult, EvalReport, EvalResult
from multivon_eval.reporters.terminal import print_report


def _cr(
    passed: bool = True,
    *,
    case_input: str = "in",
    model_error: str | None = None,
    runs: int = 1,
    pass_count: int = -1,
) -> CaseResult:
    results = [] if model_error else [
        EvalResult(evaluator="ev", score=1.0 if passed else 0.0, passed=passed)
    ]
    return CaseResult(
        case_input=case_input,
        actual_output="out",
        results=results,
        model_error=model_error,
        runs=runs,
        pass_count=pass_count,
    )


def _report(case_results, purpose: str = "") -> EvalReport:
    return EvalReport(suite_name="s", case_results=case_results, purpose=purpose)


# ─── (1) saturated property ─────────────────────────────────────────────────

class TestSaturatedProperty:
    def test_all_pass_is_saturated(self):
        assert _report([_cr(True) for _ in range(4)]).saturated is True

    def test_one_quality_failure_is_not_saturated(self):
        assert _report([_cr(True), _cr(True), _cr(False)]).saturated is False

    def test_model_error_case_excluded_from_denominator(self):
        # Intended: errors are NOT evaluated cases, so 3 passes + 1
        # model_error is still saturation — the quality signal that exists
        # is at ceiling. The error is surfaced separately via error_rate.
        report = _report([_cr(True), _cr(True), _cr(True), _cr(model_error="boom")])
        assert report.saturated is True

    def test_empty_report_is_not_saturated(self):
        assert _report([]).saturated is False

    def test_all_errors_is_not_saturated(self):
        # A "100%" built on judge/model outages must not count.
        assert _report([_cr(model_error="x"), _cr(model_error="y")]).saturated is False


# ─── (2) min_detectable_regression ──────────────────────────────────────────

class TestMinDetectableRegression:
    def test_shrinks_as_n_grows(self):
        small = _report([_cr(True) for _ in range(10)])
        large = _report([_cr(True) for _ in range(200)])
        assert small.min_detectable_regression > large.min_detectable_regression

    def test_returns_one_at_zero_evaluated(self):
        assert _report([]).min_detectable_regression == 1.0
        assert _report([_cr(model_error="x")]).min_detectable_regression == 1.0

    def test_baseline_capped_at_095_for_perfect_pass_rate(self):
        # At pass_rate 1.0 the variance term p(1-p) would be 0 and the MDE
        # would flatter itself to ~0; the cap keeps the claim honest.
        from multivon_eval.experiments import min_detectable_effect
        report = _report([_cr(True) for _ in range(50)])
        assert report.pass_rate == 1.0
        assert report.min_detectable_regression == min_detectable_effect(
            50, baseline=0.95
        )
        assert report.min_detectable_regression > 0.0

    def test_baseline_floored_at_050_for_low_pass_rate(self):
        from multivon_eval.experiments import min_detectable_effect
        report = _report([_cr(False) for _ in range(20)])
        assert report.min_detectable_regression == min_detectable_effect(
            20, baseline=0.5
        )


# ─── (3) purpose plumbing ───────────────────────────────────────────────────

class TestPurposePlumbing:
    def _suite(self, purpose):
        s = EvalSuite("p", purpose=purpose)
        s.add_cases([EvalCase(input=f"q{i}", expected_output="ok") for i in range(3)])
        from multivon_eval.evaluators.deterministic import ExactMatch
        s.add_evaluator(ExactMatch())
        return s

    def test_run_copies_purpose_onto_report(self):
        report = self._suite("capability").run(lambda x: "ok", verbose=False)
        assert report.purpose == "capability"

    def test_default_purpose_is_empty(self):
        report = self._suite("").run(lambda x: "ok", verbose=False)
        assert report.purpose == ""

    def test_invalid_purpose_raises(self):
        with pytest.raises(ValueError):
            EvalSuite("bad", purpose="benchmarking")

    def test_json_round_trip(self):
        report = self._suite("regression").run(lambda x: "ok", verbose=False)
        data = json.loads(report.to_json())
        assert data["summary"]["purpose"] == "regression"
        assert data["summary"]["saturated"] is True
        assert data["summary"]["min_detectable_regression"] == pytest.approx(
            report.min_detectable_regression, abs=1e-4
        )
        restored = EvalReport.from_dict(data)
        assert restored.purpose == "regression"
        assert restored.saturated is True
        assert restored.min_detectable_regression == pytest.approx(
            report.min_detectable_regression, abs=1e-4
        )


# ─── (4) terminal output ────────────────────────────────────────────────────

class TestTerminalSaturationWarnings:
    def test_saturated_capability_report_prints_graduation(self, capsys):
        report = _report([_cr(True, case_input=f"q{i}") for i in range(5)],
                         purpose="capability")
        print_report(report)
        out = capsys.readouterr().out
        assert "Graduate" in out
        wilson_lower = report.pass_rate_ci()[0]
        assert f"{wilson_lower:.1%}" in out

    def test_saturated_unset_purpose_also_warns(self, capsys):
        # Beginners never set purpose; the monitor still fires.
        report = _report([_cr(True, case_input=f"q{i}") for i in range(5)])
        print_report(report)
        assert "Saturated" in capsys.readouterr().out

    def test_regression_purpose_prints_triage_not_graduation(self, capsys):
        report = _report(
            [_cr(True), _cr(True), _cr(False, case_input="broken task")],
            purpose="regression",
        )
        print_report(report)
        out = capsys.readouterr().out
        assert "triage" in out
        assert "Graduate" not in out
        assert "broken task" in out

    def test_regression_purpose_all_passing_prints_nothing(self, capsys):
        report = _report([_cr(True) for _ in range(5)], purpose="regression")
        print_report(report)
        out = capsys.readouterr().out
        assert "triage" not in out
        assert "Graduate" not in out

    def test_two_case_saturated_suite_prints_only_power_warning(self, capsys):
        report = _report([_cr(True), _cr(True)])
        print_report(report)
        out = capsys.readouterr().out
        assert "Power warning" in out
        assert "Saturated" not in out

    def test_non_saturated_report_prints_neither(self, capsys):
        cases = [_cr(True, case_input=f"q{i}") for i in range(199)] + [_cr(False)]
        print_report(_report(cases))
        out = capsys.readouterr().out
        assert "Saturated" not in out
        assert "Power warning" not in out


# ─── (5) no-regression guard for the existing power warning ────────────────

class TestPowerWarningUnchanged:
    def test_power_warning_still_fires_on_small_unsaturated_suite(self, capsys):
        report = _report([_cr(True), _cr(True), _cr(False)])
        print_report(report)
        out = capsys.readouterr().out
        assert "Power warning" in out
        assert "80% power" in out
