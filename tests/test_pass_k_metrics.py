"""pass@k / pass^k estimators, cluster-bootstrap CIs, and honesty gates."""
import json
from math import comb

import pytest

from multivon_eval.passk import (
    PassKResult, pass_at_k, pass_hat_k, suite_pass_k,
)
from multivon_eval.result import (
    CaseResult, EvalGateFailure, EvalReport, EvalResult,
)


def _er(passed=True, score=1.0):
    return EvalResult(evaluator="check", score=score, passed=passed)


def _case(inp, runs, pass_count, **kw):
    return CaseResult(
        case_input=inp,
        actual_output="out",
        results=[_er(passed=(pass_count == runs))],
        runs=runs,
        pass_count=pass_count,
        all_scores=[1.0] * pass_count + [0.0] * (runs - pass_count),
        **kw,
    )


def _report(cases, name="passk-suite"):
    return EvalReport(suite_name=name, model_id="test-model", case_results=cases)


# ── (1) Exact estimator values by hand ──────────────────────────────────────

class TestEstimatorValues:
    def test_pass_at_k_hand_computed(self):
        assert pass_at_k(5, 3, 2) == pytest.approx(1 - comb(2, 2) / comb(5, 2))
        assert pass_at_k(5, 3, 2) == pytest.approx(0.9)

    def test_pass_hat_k_hand_computed(self):
        assert pass_hat_k(5, 3, 2) == pytest.approx(comb(3, 2) / comb(5, 2))
        assert pass_hat_k(5, 3, 2) == pytest.approx(0.3)

    def test_c_zero_gives_zero(self):
        assert pass_at_k(5, 0, 2) == 0.0
        assert pass_hat_k(5, 0, 2) == 0.0

    def test_c_equals_n_gives_one(self):
        assert pass_at_k(5, 5, 2) == 1.0
        assert pass_hat_k(5, 5, 2) == 1.0

    def test_k_equals_n_reduces_to_indicators(self):
        # pass^n == all-pass indicator; pass@n == any-pass indicator.
        assert pass_hat_k(4, 4, 4) == 1.0
        assert pass_hat_k(4, 3, 4) == 0.0
        assert pass_at_k(4, 1, 4) == 1.0
        assert pass_at_k(4, 0, 4) == 0.0

    def test_k_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            pass_at_k(3, 2, 4)
        with pytest.raises(ValueError):
            pass_hat_k(3, 2, 4)

    def test_k_below_one_raises(self):
        with pytest.raises(ValueError):
            pass_at_k(3, 2, 0)
        with pytest.raises(ValueError):
            pass_hat_k(3, 2, 0)


# ── (2) Anti-bias regression ────────────────────────────────────────────────

def test_pass_hat_k_is_not_plugin_estimate():
    # The plug-in (c/n)**k samples WITH replacement and is upward-biased
    # for finite n; the hypergeometric estimator is exact. Guard against
    # anyone "simplifying" pass_hat_k back into the vanity metric.
    assert pass_hat_k(5, 3, 2) != (3 / 5) ** 2
    assert pass_hat_k(5, 3, 2) < (3 / 5) ** 2


# ── (3) Honest UNKNOWN when k > runs ────────────────────────────────────────

class TestUnknown:
    def test_pass_at_k_beyond_runs_is_unknown(self):
        report = _report([_case("a", 5, 3), _case("b", 5, 5)])
        res = report.pass_at_k(10)
        assert res.value is None
        assert res.ci_low is None and res.ci_high is None
        assert "rerun with --runs" in res.unknown_reason
        assert "does not extrapolate" in res.unknown_reason
        assert res.k == 10 and res.runs == 5

    def test_gate_fails_on_unknown(self):
        report = _report([_case("a", 5, 5), _case("b", 5, 5)])
        with pytest.raises(EvalGateFailure) as exc:
            report.assert_pass_hat_k(10, 0.5)
        assert "rerun with --runs" in str(exc.value)

    def test_k_below_one_raises_on_report(self):
        report = _report([_case("a", 5, 3)])
        with pytest.raises(ValueError):
            report.pass_at_k(0)
        with pytest.raises(ValueError):
            report.pass_hat_k(-1)


# ── (4) Cluster bootstrap ───────────────────────────────────────────────────

class TestBootstrap:
    STATS = [(5, 3), (5, 5), (5, 1), (5, 0), (5, 4), (5, 2)]

    def test_deterministic_under_fixed_seed(self):
        a = suite_pass_k(self.STATS, 2, metric="pass@k", seed=7)
        b = suite_pass_k(self.STATS, 2, metric="pass@k", seed=7)
        assert (a.value, a.ci_low, a.ci_high) == (b.value, b.ci_low, b.ci_high)

    def test_ci_contains_point_estimate(self):
        for metric in ("pass@k", "pass^k"):
            res = suite_pass_k(self.STATS, 2, metric=metric)
            assert res.ci_low <= res.value <= res.ci_high

    def test_all_pass_suite_is_honest_at_ceiling(self):
        res = suite_pass_k([(5, 5)] * 10, 3, metric="pass^k")
        assert res.value == 1.0
        assert res.ci_low < 1.0

    def test_single_case_suite_wide_ci_not_crash(self):
        res = suite_pass_k([(5, 3)], 2, metric="pass^k")
        assert res.value == pytest.approx(0.3)
        assert res.ci_low <= res.value <= res.ci_high
        assert (res.ci_high - res.ci_low) > 0.5

    def test_bootstrap_resamples_cases_not_trials(self):
        # Two cases: always-pass and always-fail. Per-case pass^5 is 1.0
        # or 0.0, so every case-level resample mean lies in {0, 0.5, 1} —
        # the percentile CI must hit the extremes. Resampling raw trials
        # would mix the 5 passes and 5 fails into intermediate values and
        # fake precision.
        res = suite_pass_k([(5, 5), (5, 0)], 5, metric="pass^k")
        assert res.value == 0.5
        assert res.ci_low == 0.0
        assert res.ci_high == 1.0

    def test_empty_case_pool_is_unknown(self):
        res = suite_pass_k([], 2, metric="pass@k")
        assert res.value is None
        assert res.n_cases == 0
        assert "no evaluated cases" in res.unknown_reason

    def test_unknown_metric_rejected(self):
        with pytest.raises(ValueError):
            suite_pass_k([(5, 3)], 2, metric="pass~k")


# ── (5) Denominator honesty ─────────────────────────────────────────────────

def test_error_and_skipped_cases_excluded_from_denominator():
    cases = [
        _case("ok-1", 3, 2),
        _case("ok-2", 3, 3),
        _case("errored", 3, 0, model_error="model exploded"),
        _case("skipped", 3, 0, skipped=True),
    ]
    report = _report(cases)
    res = report.pass_hat_k(3)
    assert res.n_cases == 2
    # Same pool as pass_rate: only ok-1 and ok-2 contribute.
    assert res.value == pytest.approx((pass_hat_k(3, 2, 3) + pass_hat_k(3, 3, 3)) / 2)


# ── (6) JSON round trip ─────────────────────────────────────────────────────

class TestJsonRoundTrip:
    def test_from_dict_round_trip_preserves_pass_hat_k(self):
        report = _report([
            _case("a", 5, 3), _case("b", 5, 5), _case("c", 5, 1), _case("d", 5, 4),
        ])
        restored = EvalReport.from_dict(json.loads(report.to_json()))
        orig = report.pass_hat_k(3)
        back = restored.pass_hat_k(3)
        assert back.value == pytest.approx(orig.value)
        assert back.ci_low == pytest.approx(orig.ci_low, abs=0.05)
        assert back.ci_high == pytest.approx(orig.ci_high, abs=0.05)

    def test_summary_keys_present_only_for_multi_run(self):
        multi = _report([_case("a", 5, 3), _case("b", 5, 5)])
        summary = json.loads(multi.to_json())["summary"]
        assert summary["pass_at_k"]["k"] == 5
        assert summary["pass_at_k"]["estimator"] == "combinatorial-unbiased"
        assert summary["pass_hat_k"]["estimator"] == "hypergeometric-exact"
        assert len(summary["pass_hat_k"]["ci_95"]) == 2

        single = _report([
            CaseResult(case_input="x", actual_output="y", results=[_er()]),
        ])
        summary = json.loads(single.to_json())["summary"]
        assert "pass_at_k" not in summary
        assert "pass_hat_k" not in summary


# ── (7) Gate semantics ──────────────────────────────────────────────────────

class TestGate:
    def test_gate_raises_when_ci_low_below_threshold(self):
        report = _report([_case(f"c{i}", 5, 5) for i in range(10)])
        res = report.pass_hat_k(5)
        with pytest.raises(EvalGateFailure) as exc:
            report.assert_pass_hat_k(5, min_ci_low=res.ci_low + 0.01)
        msg = str(exc.value)
        assert f"{res.value:.3f}" in msg
        assert f"{res.ci_low:.3f}" in msg and f"{res.ci_high:.3f}" in msg
        assert "pass^5" in msg
        assert exc.value.threshold == pytest.approx(res.ci_low + 0.01)

    def test_gate_passes_when_ci_low_meets_threshold(self):
        report = _report([_case(f"c{i}", 5, 5) for i in range(10)])
        res = report.pass_hat_k(5)
        report.assert_pass_hat_k(5, min_ci_low=res.ci_low)  # must not raise


# ── Lottery cases ───────────────────────────────────────────────────────────

def test_lottery_cases_ranked_by_divergence():
    stable = _case("stable", 5, 5)
    coin_flip = _case("coin-flip", 5, 3)
    mostly = _case("mostly", 5, 4)
    never = _case("never", 5, 0)
    report = _report([stable, mostly, never, coin_flip])
    # k=2: coin-flip gap = 0.9-0.3 = 0.6 > mostly gap = 1.0-0.6 = 0.4.
    lottery = report.lottery_cases(2)
    assert [cr.case_input for cr in lottery] == ["coin-flip", "mostly"]
    # Default k = runs_per_case: only the flaky cases qualify.
    assert {cr.case_input for cr in report.lottery_cases()} == {"coin-flip", "mostly"}


# ── (8) Terminal reporter ───────────────────────────────────────────────────

class TestTerminalReporter:
    def test_multi_run_report_prints_reliability_block(self):
        from multivon_eval.reporters import terminal
        report = _report([
            _case("always works", 3, 3),
            _case("sometimes works", 3, 1),
            _case("never works", 3, 0),
        ])
        with terminal.console.capture() as cap:
            terminal.print_report(report)
        out = cap.get()
        assert "Reliability (3 runs/case)" in out
        assert "pass@3" in out and "pass^3" in out
        assert "capability" in out and "reliability" in out
        assert "passes sometimes, never reliably" in out

    def test_single_run_report_has_no_pass_k_block(self):
        from multivon_eval.reporters import terminal
        report = _report([
            CaseResult(case_input="x", actual_output="y", results=[_er()]),
        ])
        with terminal.console.capture() as cap:
            terminal.print_report(report)
        out = cap.get()
        assert "pass^" not in out
        assert "Reliability (" not in out


# ── (9) compare.py — values only, no winner call ────────────────────────────

class TestCompare:
    def test_multi_run_compare_shows_pass_hat_k_values_only(self):
        baseline = _report([_case("a", 5, 3), _case("b", 5, 5)], name="baseline")
        proposal = _report([_case("a", 5, 4), _case("b", 5, 5)], name="proposal")
        diff = baseline.compare(proposal)
        text = diff.to_text()
        phk_lines = [ln for ln in text.splitlines() if "pass^" in ln]
        assert len(phk_lines) == 1
        line = phk_lines[0]
        assert f"{diff.baseline_pass_hat_k.value:.3f}" in line
        assert f"{diff.proposal_pass_hat_k.value:.3f}" in line
        for verdict in ("winner", "significant", "p ="):
            assert verdict not in line

    def test_single_run_compare_has_no_pass_hat_k(self):
        baseline = _report([
            CaseResult(case_input="x", actual_output="y", results=[_er()]),
        ])
        proposal = _report([
            CaseResult(case_input="x", actual_output="y", results=[_er()]),
        ])
        diff = baseline.compare(proposal)
        assert diff.baseline_pass_hat_k is None
        assert diff.proposal_pass_hat_k is None
        assert "pass^" not in diff.to_text()


# ── Package exports ─────────────────────────────────────────────────────────

def test_top_level_exports():
    import multivon_eval as mv
    assert mv.pass_at_k(5, 3, 2) == pytest.approx(0.9)
    assert mv.pass_hat_k(5, 3, 2) == pytest.approx(0.3)
    assert mv.PassKResult is PassKResult
    for name in ("pass_at_k", "pass_hat_k", "PassKResult"):
        assert name in mv.__all__


# ── (10) Heterogeneous per-case run counts (early_stop) ─────────────────────
#
# Release blocker: SPRT early_stop records fewer trials for easy cases, so
# per-case n differs. Report surfaces (pass_at_k / pass_hat_k / to_json /
# terminal) must NEVER raise after a paid run — they degrade to an honest
# UNKNOWN naming the shortest-run case. Library functions keep ValueError.

class TestHeterogeneousRuns:
    def _hetero_cases(self):
        return [_case("stopped-early", 6, 6), _case("ran-full", 10, 5)]

    def test_report_does_not_raise_and_goes_unknown(self):
        report = _report(self._hetero_cases())
        res = report.pass_at_k(10)  # must not raise
        assert res.value is None
        assert res.ci_low is None and res.ci_high is None
        assert "6 trials" in res.unknown_reason
        assert "shortest-run case" in res.unknown_reason
        assert "early_stop" in res.unknown_reason
        assert "rerun with --runs >= 10 or pass k <= 6" in res.unknown_reason
        assert res.runs == 6  # min(n) over evaluated cases — one definition

    def test_case_order_independence(self):
        fwd = _report(self._hetero_cases())
        rev = _report(list(reversed(self._hetero_cases())))
        assert fwd.runs_per_case == rev.runs_per_case == 10
        f, r = fwd.pass_hat_k(10), rev.pass_hat_k(10)
        assert f.value is None and r.value is None
        assert f.unknown_reason == r.unknown_reason
        assert f.runs == r.runs == 6
        # And at a supported k both orders agree on the value.
        assert fwd.pass_hat_k(3).value == pytest.approx(rev.pass_hat_k(3).value)

    def test_supported_k_uses_each_cases_own_n(self):
        report = _report(self._hetero_cases())
        res = report.pass_at_k(3)
        expected = (pass_at_k(6, 6, 3) + pass_at_k(10, 5, 3)) / 2
        assert res.value == pytest.approx(expected)
        assert res.runs == 6  # min(n), same definition as the UNKNOWN branch

    def test_to_json_does_not_raise_and_discloses_unknown(self):
        report = _report(self._hetero_cases())
        summary = json.loads(report.to_json())["summary"]  # must not raise
        assert summary["runs_per_case"] == 10
        assert summary["pass_hat_k"]["value"] is None
        assert "6 trials" in summary["pass_hat_k"]["unknown_reason"]
        assert summary["pass_hat_k"]["runs"] == 6

    def test_terminal_report_prints_unknown_line_not_crash(self):
        from multivon_eval.reporters import terminal
        report = _report(self._hetero_cases())
        with terminal.console.capture() as cap:
            terminal.print_report(report)  # must not raise
        out = cap.get()
        assert "Reliability (10 runs/case)" in out
        assert "UNKNOWN" in out

    def test_gate_still_fails_loud_on_heterogeneous_unknown(self):
        # assert_pass_hat_k is a GATE, not a report — UNKNOWN must raise.
        report = _report(self._hetero_cases())
        with pytest.raises(EvalGateFailure, match="pass k <= 6"):
            report.assert_pass_hat_k(10, 0.5)

    def test_library_functions_keep_valueerror(self):
        with pytest.raises(ValueError, match="extrapolat"):
            pass_at_k(6, 6, 10)
        with pytest.raises(ValueError, match="extrapolat"):
            suite_pass_k([(6, 6), (10, 5)], 10, metric="pass^k")

    def test_end_to_end_early_stop_run_completes(self, capsys):
        # The original crash: runs=10, early_stop=True, workers=1 — SPRT
        # stops the stable case at 6 trials, the flaky one runs all 10.
        import itertools
        from multivon_eval import EvalCase, EvalSuite
        from multivon_eval.evaluators.deterministic import Contains

        flip = itertools.cycle(["ok", "bad"])

        def model(prompt):
            return next(flip) if prompt == "flaky" else "ok"

        suite = (
            EvalSuite("hetero-e2e")
            .add_cases([
                EvalCase(input="flaky", expected_output="ok"),
                EvalCase(input="stable", expected_output="ok"),
            ])
            .add_evaluator(Contains(["ok"]))
        )
        report = suite.run(model, runs=10, early_stop=True, workers=1,
                           verbose=True)  # must not raise
        run_counts = sorted(cr.runs for cr in report.case_results)
        assert run_counts[0] < 10 <= run_counts[1] or run_counts == [10, 10]
        json.loads(report.to_json())  # must not raise either


# ── (11) suite_pass_k argument validation ───────────────────────────────────

class TestSuitePassKArgValidation:
    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
    def test_confidence_must_be_in_open_unit_interval(self, confidence):
        with pytest.raises(ValueError, match="confidence"):
            suite_pass_k([(5, 3)], 2, metric="pass@k", confidence=confidence)

    @pytest.mark.parametrize("n_boot", [0, -5])
    def test_n_boot_must_be_positive(self, n_boot):
        with pytest.raises(ValueError, match="n_boot"):
            suite_pass_k([(5, 3)], 2, metric="pass@k", n_boot=n_boot)

    def test_valid_edges_accepted(self):
        res = suite_pass_k([(5, 3), (5, 4)], 2, metric="pass@k",
                           confidence=0.5, n_boot=1)
        assert res.value is not None


# ── (12) early_stop is serial-only: parallel path warns ─────────────────────

def test_early_stop_on_parallel_path_warns(capsys):
    from multivon_eval import EvalCase, EvalSuite
    from multivon_eval.evaluators.deterministic import NotEmpty

    suite = EvalSuite("par")
    for i in range(3):
        suite.add_case(EvalCase(input=f"q{i}"))
    suite.add_evaluator(NotEmpty())
    suite.run(lambda p: "out", runs=2, early_stop=True, workers=2, verbose=False)
    assert "early_stop requires workers=1; ignoring" in capsys.readouterr().err


def test_early_stop_serial_path_does_not_warn(capsys):
    from multivon_eval import EvalCase, EvalSuite
    from multivon_eval.evaluators.deterministic import NotEmpty

    suite = EvalSuite("ser").add_case(EvalCase(input="q")).add_evaluator(NotEmpty())
    suite.run(lambda p: "out", runs=2, early_stop=True, workers=1, verbose=False)
    assert "early_stop" not in capsys.readouterr().err
