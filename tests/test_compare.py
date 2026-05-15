"""D14: ``multivon-eval compare`` — diff two EvalReports.

The everyday prompt-engineering question: did this change help?
Test the structured diff (paired regressions / improvements / added /
removed), the McNemar wiring, the CLI exit codes, and the
``EvalReport.compare`` convenience.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from multivon_eval import (
    CaseDiff, EvalReport, EvalResult, EvalStatus, ReportDiff, compare_reports,
)
from multivon_eval.compare import _cli, _pair_by_input
from multivon_eval.result import CaseResult


def _case(
    inp: str, *,
    passed: bool = True,
    score: float = 1.0,
    judge_error: str | None = None,
    runs: int = 1,
    pass_count: int = -1,
) -> CaseResult:
    return CaseResult(
        case_input=inp,
        actual_output="ok",
        results=[EvalResult("e", score, passed)],
        judge_error=judge_error,
        runs=runs,
        pass_count=pass_count,
    )


def _report(name: str, cases: list[CaseResult]) -> EvalReport:
    return EvalReport(suite_name=name, case_results=cases)


# ─────────────────────────────────────────────────────────────────────────────
# Pairing: case_input matching, duplicates by occurrence, added / removed
# ─────────────────────────────────────────────────────────────────────────────

def test_pair_by_input_matches_identical_lists():
    a = [_case("x"), _case("y")]
    b = [_case("x"), _case("y")]
    paired, added, removed = _pair_by_input(a, b)
    assert len(paired) == 2
    assert added == []
    assert removed == []


def test_pair_by_input_handles_added_and_removed():
    a = [_case("x"), _case("only_in_baseline")]
    b = [_case("x"), _case("only_in_proposal")]
    paired, added, removed = _pair_by_input(a, b)
    assert len(paired) == 1
    assert paired[0][0].case_input == "x"
    assert [c.case_input for c in added] == ["only_in_proposal"]
    assert [c.case_input for c in removed] == ["only_in_baseline"]


def test_pair_by_input_pairs_duplicates_in_occurrence_order():
    """Three baseline cases all with input 'X' and three proposal cases
    with input 'X' get paired 1-1-1 — matches what an operator means
    when they rerun the same prompt three times in a suite."""
    a = [_case("X", score=0.1), _case("X", score=0.2), _case("X", score=0.3)]
    b = [_case("X", score=0.4), _case("X", score=0.5), _case("X", score=0.6)]
    paired, added, removed = _pair_by_input(a, b)
    assert len(paired) == 3
    for (b_cr, p_cr), expected in zip(paired, [(0.1, 0.4), (0.2, 0.5), (0.3, 0.6)]):
        assert b_cr.score == pytest.approx(expected[0])
        assert p_cr.score == pytest.approx(expected[1])


def test_pair_by_input_uneven_duplicates_count_extras_as_added_or_removed():
    a = [_case("X", score=0.1), _case("X", score=0.2)]
    b = [_case("X", score=0.7), _case("X", score=0.8), _case("X", score=0.9)]
    paired, added, removed = _pair_by_input(a, b)
    assert len(paired) == 2
    assert len(added) == 1     # third proposal-side X
    assert added[0].score == pytest.approx(0.9)
    assert removed == []


# ─────────────────────────────────────────────────────────────────────────────
# CaseDiff.direction — improved / regressed / unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_case_diff_improved():
    d = CaseDiff("x", EvalStatus.FAILED_QUALITY, EvalStatus.PASSED, 0.3, 0.95)
    assert d.direction == "improved"


def test_case_diff_regressed():
    d = CaseDiff("x", EvalStatus.PASSED, EvalStatus.FAILED_QUALITY, 0.95, 0.30)
    assert d.direction == "regressed"


def test_case_diff_unchanged_for_pass_to_pass():
    d = CaseDiff("x", EvalStatus.PASSED, EvalStatus.PASSED, 0.9, 0.95)
    assert d.direction == "unchanged"


def test_case_diff_unchanged_for_fail_to_fail():
    """Even if the score moved, fail→fail is not a status change."""
    d = CaseDiff("x", EvalStatus.FAILED_QUALITY, EvalStatus.FAILED_QUALITY, 0.1, 0.4)
    assert d.direction == "unchanged"


def test_case_diff_error_to_pass_is_improvement():
    d = CaseDiff("x", EvalStatus.JUDGE_ERROR, EvalStatus.PASSED, 0.0, 1.0)
    assert d.direction == "improved"


def test_case_diff_pass_to_error_is_regression():
    """If a judge outage replaced a previously-passing eval, that is a
    regression — the operator wants to see it, even though the
    underlying model behavior may be unchanged."""
    d = CaseDiff("x", EvalStatus.PASSED, EvalStatus.JUDGE_ERROR, 1.0, 0.0)
    assert d.direction == "regressed"


# ─────────────────────────────────────────────────────────────────────────────
# compare_reports — end-to-end summary
# ─────────────────────────────────────────────────────────────────────────────

def test_compare_reports_basic_deltas():
    base = _report("base", [_case("x", passed=False, score=0.3), _case("y", passed=True, score=1.0)])
    prop = _report("prop", [_case("x", passed=True, score=0.9), _case("y", passed=True, score=1.0)])
    d = compare_reports(base, prop)
    assert d.baseline_pass_rate == pytest.approx(0.5)
    assert d.proposal_pass_rate == pytest.approx(1.0)
    assert d.pass_rate_delta == pytest.approx(0.5)
    assert len(d.improvements) == 1
    assert d.improvements[0].case_input == "x"
    assert d.regressions == []
    assert d.unchanged[0].case_input == "y"


def test_compare_reports_records_added_and_removed():
    base = _report("base", [_case("kept"), _case("removed_only")])
    prop = _report("prop", [_case("kept"), _case("added_only")])
    d = compare_reports(base, prop)
    assert len(d.paired) == 1
    assert [c.case_input for c in d.added] == ["added_only"]
    assert [c.case_input for c in d.removed] == ["removed_only"]


def test_compare_reports_mcnemar_p_none_for_no_paired_cases():
    """Compare two reports with totally disjoint case inputs → no
    paired cases → McNemar p is None (not 1.0, which would falsely
    imply 'tested and found no difference')."""
    base = _report("a", [_case("only-in-a")])
    prop = _report("b", [_case("only-in-b")])
    d = compare_reports(base, prop)
    assert d.paired == []
    assert d.mcnemar_p is None


def test_compare_reports_mcnemar_runs_on_paired_cases():
    """McNemar fires when there are paired cases AND at least one
    discordant pair. With matched pass/pass everywhere, p = 1.0."""
    base = _report("a", [_case("x", passed=True), _case("y", passed=True)])
    prop = _report("b", [_case("x", passed=True), _case("y", passed=True)])
    d = compare_reports(base, prop)
    assert d.mcnemar_p == pytest.approx(1.0)


def test_compare_reports_mcnemar_flags_significant_change():
    """A reasonably large swing (many discordant pairs in one direction)
    produces a p-value below 0.05."""
    # 20 cases: 19 went from fail → pass, 1 unchanged. McNemar's |b-c|
    # with b=0, c=19, applies the continuity correction.
    base_cases = [_case(f"x{i}", passed=False) for i in range(19)] + [_case("z", passed=True)]
    prop_cases = [_case(f"x{i}", passed=True) for i in range(19)] + [_case("z", passed=True)]
    d = compare_reports(_report("a", base_cases), _report("b", prop_cases))
    assert d.mcnemar_p is not None
    assert d.mcnemar_p < 0.05


def test_compare_reports_uses_status_aware_pass_rate():
    """Pass rate / errors / flaky deltas come from the EvalReport
    properties (which already exclude error cases from the denominator).
    Just verify the wiring matches what EvalReport reports."""
    base = _report("a", [_case("x", passed=True), _case("y", judge_error="429")])
    assert base.pass_rate == pytest.approx(1.0)
    assert base.errors == 1
    prop = _report("b", [_case("x", passed=True), _case("y", passed=True)])
    d = compare_reports(base, prop)
    assert d.baseline_errors == 1
    assert d.proposal_errors == 0
    assert d.errors_delta == -1


# ─────────────────────────────────────────────────────────────────────────────
# Rendering: to_text, to_markdown, to_dict
# ─────────────────────────────────────────────────────────────────────────────

def test_to_text_includes_pass_rate_delta_and_per_case():
    base = _report("base", [_case("x", passed=False, score=0.3), _case("y", passed=True, score=1.0)])
    prop = _report("prop", [_case("x", passed=True, score=0.9), _case("y", passed=False, score=0.2)])
    d = compare_reports(base, prop)
    txt = d.to_text()
    assert "Pass rate" in txt
    assert "0.500 -> 0.500" in txt  # one improved, one regressed → net zero
    assert "Regressions" in txt
    assert "Improvements" in txt


def test_to_text_regressions_only_hides_improvements():
    base = _report("a", [_case("x", passed=False), _case("y", passed=True)])
    prop = _report("b", [_case("x", passed=True), _case("y", passed=False)])
    d = compare_reports(base, prop)
    txt = d.to_text(regressions_only=True)
    assert "Regressions" in txt
    assert "Improvements" not in txt


def test_to_markdown_renders_table_and_lists():
    base = _report("a", [_case("x", passed=False), _case("y", passed=True)])
    prop = _report("b", [_case("x", passed=True), _case("y", passed=True)])
    d = compare_reports(base, prop)
    md = d.to_markdown()
    assert "| Metric |" in md
    assert "Pass rate" in md
    assert "### Improvements" in md
    assert "`x`" in md


def test_to_dict_round_trips_via_json():
    """The dict form is JSON-serializable so callers can pipe the diff
    into other tools or store it in a CI artifact."""
    base = _report("a", [_case("x", passed=False), _case("y", passed=True)])
    prop = _report("b", [_case("x", passed=True), _case("y", passed=True)])
    d = compare_reports(base, prop)
    blob = json.dumps(d.to_dict(), default=str)  # default=str for EvalStatus enum
    parsed = json.loads(blob)
    assert parsed["deltas"]["pass_rate"] == pytest.approx(0.5)
    assert len(parsed["improvements"]) == 1
    assert parsed["mcnemar_p"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# EvalReport.compare convenience method
# ─────────────────────────────────────────────────────────────────────────────

def test_evalreport_compare_method():
    a = _report("a", [_case("x", passed=False)])
    b = _report("b", [_case("x", passed=True)])
    d = a.compare(b)
    assert isinstance(d, ReportDiff)
    assert len(d.improvements) == 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI — file loading, exit codes, formats
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(path: Path, cases: list[CaseResult]) -> None:
    path.write_text(_report(path.stem, cases).to_json(), encoding="utf-8")


def test_cli_default_text_mode(tmp_path, capsys):
    base = tmp_path / "base.json"
    prop = tmp_path / "prop.json"
    _write_report(base, [_case("x", passed=False, score=0.3)])
    _write_report(prop, [_case("x", passed=True, score=0.9)])

    rc = _cli([str(base), str(prop)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Pass rate" in out
    assert "Improvements" in out


def test_cli_markdown_mode(tmp_path, capsys):
    base = tmp_path / "base.json"
    prop = tmp_path / "prop.json"
    _write_report(base, [_case("x", passed=False), _case("y", passed=True)])
    _write_report(prop, [_case("x", passed=True), _case("y", passed=False)])

    rc = _cli([str(base), str(prop), "--markdown"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "| Metric |" in out
    assert "### Regressions" in out


def test_cli_json_mode(tmp_path, capsys):
    base = tmp_path / "base.json"
    prop = tmp_path / "prop.json"
    _write_report(base, [_case("x", passed=False)])
    _write_report(prop, [_case("x", passed=True)])

    rc = _cli([str(base), str(prop), "--json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["deltas"]["pass_rate"] == pytest.approx(1.0)


def test_cli_fail_on_regression_exits_1(tmp_path, capsys):
    """CI gate: --fail-on-regression must return 1 if ANY case regressed,
    even if the aggregate pass-rate improved."""
    base = tmp_path / "base.json"
    prop = tmp_path / "prop.json"
    # Three cases improve, one regresses → aggregate moves up but the
    # gate fires because there IS a regression.
    base_cases = [_case("a", passed=False), _case("b", passed=False),
                  _case("c", passed=False), _case("d", passed=True)]
    prop_cases = [_case("a", passed=True), _case("b", passed=True),
                  _case("c", passed=True), _case("d", passed=False)]
    _write_report(base, base_cases)
    _write_report(prop, prop_cases)

    rc = _cli([str(base), str(prop), "--fail-on-regression"])
    assert rc == 1


def test_cli_fail_on_regression_does_not_fire_without_regressions(tmp_path):
    base = tmp_path / "base.json"
    prop = tmp_path / "prop.json"
    _write_report(base, [_case("x", passed=False)])
    _write_report(prop, [_case("x", passed=True)])

    rc = _cli([str(base), str(prop), "--fail-on-regression"])
    assert rc == 0


def test_cli_missing_file_returns_2(tmp_path, capsys):
    rc = _cli([str(tmp_path / "missing.json"), str(tmp_path / "also.json")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error" in err.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Integration: invoke through `multivon-eval compare`
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Codex round-1 regressions
# ─────────────────────────────────────────────────────────────────────────────

def test_pair_by_input_handles_aliased_objects():
    """Codex ISSUE 1: tracking consumption by ``id()`` broke when the
    same CaseResult object appeared more than once (e.g. ``[cr] * 3``).
    Switched to positional index — all aliased occurrences now pair.
    """
    shared = _case("X", score=0.5)
    base = [shared, shared, shared]
    prop = [_case("X", score=0.7), _case("X", score=0.8), _case("X", score=0.9)]
    paired, added, removed = _pair_by_input(base, prop)
    assert len(paired) == 3, "all three aliased occurrences must pair"
    assert added == []
    assert removed == []


def test_skipped_pair_is_unchanged_not_regressed_or_improved():
    """Codex ISSUE 2: skipped on either side is not the same as a quality
    failure. Direction must be 'unchanged' so skipped→pass doesn't show
    up as an improvement (which would falsely tell the user the model
    got better when in fact the case wasn't evaluated before)."""
    sk = CaseDiff("x", EvalStatus.SKIPPED, EvalStatus.PASSED, 0.0, 1.0)
    assert sk.direction == "unchanged"
    sk2 = CaseDiff("x", EvalStatus.PASSED, EvalStatus.SKIPPED, 1.0, 0.0)
    assert sk2.direction == "unchanged"


def test_mcnemar_excludes_skipped_pairs(monkeypatch):
    """If every paired case has SKIPPED on one side, McNemar has no
    valid pairs to test → returns None, NOT 1.0 (which would mean
    'tested and found no difference')."""
    base = _report("a", [
        _case("x", passed=False),  # skipped on baseline
    ])
    # Force SKIPPED status on the baseline case (the helper doesn't expose it).
    base.case_results[0].skipped = True
    prop = _report("b", [_case("x", passed=True)])
    d = compare_reports(base, prop)
    assert d.mcnemar_p is None


def test_mcnemar_still_runs_on_mixed_skipped_and_paired():
    """A mix of skipped and real pairs should still produce a McNemar
    p-value from the REAL pairs only."""
    base = _report("a", [
        _case("x", passed=True),
        _case("y", passed=False),
        _case("z", passed=True),  # skipped, excluded
    ])
    base.case_results[2].skipped = True
    prop = _report("b", [
        _case("x", passed=True),
        _case("y", passed=True),
        _case("z", passed=True),
    ])
    d = compare_reports(base, prop)
    assert d.mcnemar_p is not None
    # The non-skipped pair "y" went False→True — one discordant pair
    # out of two non-skipped paired cases.


def test_cli_subcommand_works_end_to_end(tmp_path):
    """The top-level CLI dispatches 'compare' to the compare submodule.
    Run the real binary to make sure argparse + dispatch wiring is intact."""
    base = tmp_path / "base.json"
    prop = tmp_path / "prop.json"
    _write_report(base, [_case("x", passed=False)])
    _write_report(prop, [_case("x", passed=True)])

    result = subprocess.run(
        [sys.executable, "-m", "multivon_eval", "compare", str(base), str(prop)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Improvements" in result.stdout
