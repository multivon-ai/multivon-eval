"""multivon-eval validate — grade-your-grader audit (reference-only,
never calls the model under test) + ZERO_PASS_SUSPECT detection.

All offline: deterministic evaluators + spy judges, no network.
"""
from __future__ import annotations

import json
from argparse import Namespace

import pytest

from multivon_eval import EvalCase, EvalSuite, validate_suite
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.evaluators.deterministic import Contains, ExactMatch, NotEmpty
from multivon_eval.result import CaseResult, EvalReport, EvalResult
from multivon_eval.validate import (
    STATUS_BROKEN,
    STATUS_NO_DISCRIMINATION,
    STATUS_OK,
    STATUS_UNVALIDATABLE,
)


class SpyJudge(Evaluator):
    """Fake LLM-judge evaluator with a call-count spy."""
    name = "spy_judge"
    uses_llm_judge = True

    def __init__(self):
        super().__init__(0.5)
        self.calls = 0

    def evaluate(self, case, output):
        self.calls += 1
        return self._result(1.0, "judged")


# ─── (1) happy path ─────────────────────────────────────────────────────────

class TestHappyPath:
    def test_all_ok_when_references_pass(self):
        suite = EvalSuite("happy")
        suite.add_cases([
            EvalCase(input="a", expected_output="hello world"),
            EvalCase(input="b", expected_output="hello there"),
        ])
        suite.add_evaluators(NotEmpty(), Contains(["hello"]))
        report = validate_suite(suite)
        assert all(r.status == STATUS_OK for r in report.results)
        assert report.passed is True
        assert report.effective_informative_cases == (2, 2)

    def test_suite_convenience_method(self):
        suite = EvalSuite("conv")
        suite.add_case(EvalCase(input="a", expected_output="x"))
        suite.add_evaluator(NotEmpty())
        assert suite.validate().passed is True


# ─── (2) broken grader ──────────────────────────────────────────────────────

class TestBrokenGrader:
    def test_impossible_grader_flags_broken(self):
        suite = EvalSuite("broken")
        suite.add_case(EvalCase(input="a", expected_output="the answer is 4"))
        suite.add_evaluator(Contains(["unicorn"]))  # reference can never contain this
        report = validate_suite(suite)
        (cv,) = report.results
        assert cv.status == STATUS_BROKEN
        assert cv.failed_graders[0].reason  # carries the grader's own explanation
        assert "unicorn" in cv.failed_graders[0].reason.lower() or cv.failed_graders[0].reason
        assert report.passed is False

    def test_grader_raising_on_reference_is_broken(self):
        class Crashy(Evaluator):
            name = "crashy"

            def evaluate(self, case, output):
                raise RuntimeError("kaboom")

        suite = EvalSuite("crash")
        suite.add_case(EvalCase(input="a", expected_output="x"))
        suite.add_evaluator(Crashy())
        report = validate_suite(suite)
        (cv,) = report.results
        assert cv.status == STATUS_BROKEN
        assert "kaboom" in cv.failed_graders[0].reason

    def test_zero_evaluator_suite_is_all_unvalidatable(self):
        suite = EvalSuite("empty")
        suite.add_case(EvalCase(input="a", expected_output="x"))
        report = validate_suite(suite)
        (cv,) = report.results
        assert cv.status == STATUS_UNVALIDATABLE
        assert cv.reason == "no evaluators"


# ─── (3) reference_output resolution ─────────────────────────────────────────

class TestReferenceOutput:
    def test_reference_output_takes_precedence(self):
        suite = EvalSuite("prec")
        suite.add_case(EvalCase(
            input="a",
            expected_output="wrong label",
            reference_output="the good answer",
        ))
        suite.add_evaluator(Contains(["good"]))
        report = validate_suite(suite)
        assert report.results[0].status == STATUS_OK

    def test_callable_reference_invoked_exactly_once(self):
        calls = []

        def make_ref(case):
            calls.append(case.input)
            return "generated reference"

        suite = EvalSuite("callable")
        suite.add_case(EvalCase(input="a", reference_output=make_ref))
        suite.add_evaluators(Contains(["generated"]), NotEmpty())
        report = validate_suite(suite)
        assert report.results[0].status == STATUS_OK
        assert calls == ["a"]

    def test_callable_raising_marks_broken_with_traceback(self):
        def bad_ref(case):
            raise ValueError("reference builder exploded")

        suite = EvalSuite("callable-crash")
        suite.add_case(EvalCase(input="a", reference_output=bad_ref))
        suite.add_evaluator(NotEmpty())
        report = validate_suite(suite)
        (cv,) = report.results
        assert cv.status == STATUS_BROKEN
        assert "reference builder exploded" in cv.reason
        assert "Traceback" in cv.reason


# ─── (4) UNVALIDATABLE ──────────────────────────────────────────────────────

class TestUnvalidatable:
    def test_no_reference_listed_with_nudge_and_not_a_failure(self):
        suite = EvalSuite("unval")
        suite.add_cases([
            EvalCase(input="a", expected_output="ok"),
            EvalCase(input="b"),  # no reference of any kind
        ])
        suite.add_evaluator(NotEmpty())
        report = validate_suite(suite)
        unval = report.unvalidatable
        assert len(unval) == 1
        assert "expected_output or reference_output" in unval[0].reason
        assert report.passed is True  # warning, not failure
        assert report.effective_informative_cases == (1, 1)


# ─── (5) judge gating ───────────────────────────────────────────────────────

class TestJudgeGating:
    def test_judge_skipped_by_default(self):
        spy = SpyJudge()
        suite = EvalSuite("judged")
        suite.add_case(EvalCase(input="a", expected_output="x"))
        suite.add_evaluators(NotEmpty(), spy)
        report = validate_suite(suite)
        assert spy.calls == 0
        assert "spy_judge" in report.results[0].skipped_graders
        assert report.results[0].status == STATUS_OK

    def test_include_judges_runs_them(self):
        spy = SpyJudge()
        suite = EvalSuite("judged")
        suite.add_case(EvalCase(input="a", expected_output="x"))
        suite.add_evaluators(spy)
        report = validate_suite(suite, include_judges=True)
        assert spy.calls == 1
        assert report.results[0].skipped_graders == []


# ─── (6) NO_DISCRIMINATION via contrast twins ───────────────────────────────

def _pair(bad_output: str):
    """Hand-built contrast pair mirroring _contrast.py's metadata shape."""
    original = EvalCase(
        input="capital?",
        expected_output="Paris is the capital of France",
        metadata={"pair_id": "p1"},
    )
    twin = EvalCase(
        input="capital?",
        metadata={
            "pair_id": "p1",
            "unfaithful_answer": bad_output,
            "expected_behavior": "fail: minimally-edited false fact",
        },
    )
    return original, twin


class TestNoDiscrimination:
    def test_lenient_grader_passing_both_is_flagged(self):
        original, twin = _pair("Paris is the capital of Germany")
        suite = EvalSuite("contrast")
        suite.add_cases([original, twin])
        suite.add_evaluator(Contains(["Paris"]))  # passes reference AND twin
        report = validate_suite(suite)
        cv = report.results[0]
        assert cv.status == STATUS_NO_DISCRIMINATION
        assert "zero information" in cv.reason
        # ok_count excludes the flagged case: twin itself is unvalidatable
        assert report.effective_informative_cases == (0, 1)

    def test_strict_grader_failing_twin_is_not_flagged(self):
        original, twin = _pair("Paris is the capital of Germany")
        suite = EvalSuite("contrast")
        suite.add_cases([original, twin])
        suite.add_evaluator(Contains(["France"]))  # fails the twin — discriminates
        report = validate_suite(suite)
        assert report.results[0].status == STATUS_OK

    def test_no_pairs_is_a_noop(self):
        suite = EvalSuite("nopairs")
        suite.add_case(EvalCase(input="a", expected_output="hello"))
        suite.add_evaluator(Contains(["hello"]))
        report = validate_suite(suite)
        assert report.results[0].status == STATUS_OK

    def test_contrast_false_disables_check(self):
        original, twin = _pair("Paris is the capital of Germany")
        suite = EvalSuite("contrast-off")
        suite.add_cases([original, twin])
        suite.add_evaluator(Contains(["Paris"]))
        report = validate_suite(suite, contrast=False)
        assert report.results[0].status == STATUS_OK


# ─── (7) zero_pass_cases + terminal footer ──────────────────────────────────

def _cr(pass_count: int, runs: int, case_input: str = "in") -> CaseResult:
    passed = pass_count == runs
    return CaseResult(
        case_input=case_input,
        actual_output="out",
        results=[EvalResult(evaluator="ev", score=1.0 if passed else 0.0, passed=passed)],
        runs=runs,
        pass_count=pass_count,
    )


class TestZeroPassCases:
    def test_multi_run_zero_pass_case_listed(self):
        report = EvalReport("s", [_cr(3, 3), _cr(0, 3, "never passes")])
        assert [c.case_input for c in report.zero_pass_cases] == ["never passes"]

    def test_flaky_case_not_listed(self):
        report = EvalReport("s", [_cr(1, 3, "flaky")])
        assert report.zero_pass_cases == []

    def test_single_run_all_fail_lists_all(self):
        cases = [
            CaseResult(
                case_input=f"q{i}", actual_output="o",
                results=[EvalResult(evaluator="ev", score=0.0, passed=False)],
            )
            for i in range(3)
        ]
        report = EvalReport("s", cases)
        assert len(report.zero_pass_cases) == 3

    def test_single_run_partial_fail_lists_none(self):
        cases = [
            CaseResult(
                case_input="p", actual_output="o",
                results=[EvalResult(evaluator="ev", score=1.0, passed=True)],
            ),
            CaseResult(
                case_input="f", actual_output="o",
                results=[EvalResult(evaluator="ev", score=0.0, passed=False)],
            ),
        ]
        assert EvalReport("s", cases).zero_pass_cases == []

    def test_terminal_footer_recommends_validate(self, capsys):
        from multivon_eval.reporters.terminal import print_report
        report = EvalReport("s", [_cr(3, 3), _cr(0, 3, "never passes")])
        print_report(report)
        assert "multivon-eval validate" in capsys.readouterr().out


# ─── (8) CLI ────────────────────────────────────────────────────────────────

_CLEAN_EVAL = """\
from multivon_eval import EvalCase, EvalSuite
from multivon_eval.evaluators.deterministic import Contains

suite = EvalSuite("cli-clean")
suite.add_case(EvalCase(input="a", expected_output="hello world"))
suite.add_evaluator(Contains(["hello"]))

if __name__ == "__main__":
    raise SystemExit("model must NEVER run during validate")
"""

_BROKEN_EVAL = """\
from multivon_eval import EvalCase, EvalSuite
from multivon_eval.evaluators.deterministic import Contains

suite = EvalSuite("cli-broken")
suite.add_case(EvalCase(input="a", expected_output="hello world"))
suite.add_evaluator(Contains(["unicorn"]))
"""


def _args(file, judges=False, no_contrast=False, json_path=None):
    return Namespace(file=str(file), judges=judges,
                     no_contrast=no_contrast, json=json_path)


class TestCmdValidate:
    def test_exit_zero_on_clean_and_model_never_runs(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_validate
        f = tmp_path / "eval_clean.py"
        f.write_text(_CLEAN_EVAL)
        # The __main__ guard raising proves the model path never executes.
        assert cmd_validate(_args(f)) == 0
        assert "OK" in capsys.readouterr().out

    def test_exit_one_on_broken(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_validate
        f = tmp_path / "eval_broken.py"
        f.write_text(_BROKEN_EVAL)
        assert cmd_validate(_args(f)) == 1
        assert "BROKEN_TASK_OR_GRADER" in capsys.readouterr().out

    def test_json_output_is_parseable(self, tmp_path):
        from multivon_eval.cli import cmd_validate
        f = tmp_path / "eval_broken.py"
        f.write_text(_BROKEN_EVAL)
        out = tmp_path / "validation.json"
        assert cmd_validate(_args(f, json_path=str(out))) == 1
        data = json.loads(out.read_text())
        assert data["passed"] is False
        assert data["summary"]["broken"] == 1
        assert data["results"][0]["status"] == "BROKEN_TASK_OR_GRADER"

    def test_no_suite_found_exits_two(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_validate
        f = tmp_path / "nothing.py"
        f.write_text("x = 1\n")
        assert cmd_validate(_args(f)) == 2
        assert "no EvalSuite" in capsys.readouterr().err

    def test_report_to_json_round_trip(self):
        suite = EvalSuite("json")
        suite.add_case(EvalCase(input="a", expected_output="x"))
        suite.add_evaluator(NotEmpty())
        report = validate_suite(suite)
        data = json.loads(report.to_json())
        assert data["suite"] == "json"
        assert data["passed"] is True
        assert str(report)  # rich-free plain rendering


# ─── (9) additive-field safety: lockfile stability ──────────────────────────

class TestLockfileInvariant:
    # Golden hash computed with the PRE-reference_output _cases_hash on the
    # exact case list below. If this test fails, adding reference_output
    # changed historical lock hashes — the most dangerous invariant in the
    # repo (Track A dissent).
    GOLDEN = "d9727989f8e4a29c776f83322eeda80c45e7e47eea6e9325afd51d0b6d1eedf7"

    def _cases(self, with_reference=False):
        ref = "the known good answer" if with_reference else None
        return [
            EvalCase(input="What is 2+2?", expected_output="4",
                     context="arithmetic", reference_output=ref),
            EvalCase(input="Capital of France?", expected_output="Paris",
                     reference_output=ref),
            EvalCase(input="no-reference case", reference_output=ref),
        ]

    def test_old_style_cases_hash_matches_golden(self):
        from multivon_eval.lockfile import _cases_hash
        assert _cases_hash(self._cases()) == self.GOLDEN

    def test_reference_output_does_not_change_cases_hash(self):
        from multivon_eval.lockfile import _cases_hash
        assert _cases_hash(self._cases(with_reference=True)) == self.GOLDEN

    def test_suite_hash_identical_with_and_without_reference(self):
        def lock(with_ref):
            s = EvalSuite("lockstable").add_evaluators(NotEmpty(), ExactMatch())
            s.add_cases(self._cases(with_reference=with_ref))
            return s.lock()

        a, b = lock(False), lock(True)
        assert a.cases_hash == b.cases_hash
        assert a.suite_hash == b.suite_hash


# ─── (10) NOTHING_VALIDATED: zero graders executed is never green ───────────

from multivon_eval.validate import STATUS_NOTHING_VALIDATED


class TestNothingValidated:
    def _judge_only_suite(self):
        suite = EvalSuite("judge-only")
        suite.add_cases([
            EvalCase(input="q1", expected_output="good"),
            EvalCase(input="q2", expected_output="also good"),
        ])
        suite.add_evaluator(SpyJudge())
        return suite

    def test_judge_only_suite_offline_is_not_green(self):
        report = validate_suite(self._judge_only_suite())
        assert all(r.status == STATUS_UNVALIDATABLE for r in report.results)
        assert all(
            "all graders are judge-backed; rerun with --judges" == r.reason
            for r in report.results
        )
        assert report.nothing_validated is True
        assert report.status == STATUS_NOTHING_VALIDATED
        assert report.passed is False
        # Nothing counted as informative either.
        assert report.effective_informative_cases == (0, 0)

    def test_str_says_nothing_validated_plainly(self):
        text = str(validate_suite(self._judge_only_suite()))
        assert "NOTHING_VALIDATED" in text
        assert "zero graders executed" in text
        assert "PASSED" not in text

    def test_to_dict_carries_status(self):
        data = validate_suite(self._judge_only_suite()).to_dict()
        assert data["passed"] is False
        assert data["status"] == STATUS_NOTHING_VALIDATED

    def test_cli_exit_nonzero_for_judge_only_suite(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_validate
        f = tmp_path / "eval_judge_only.py"
        f.write_text(
            "from multivon_eval import EvalCase, EvalSuite\n"
            "from multivon_eval.evaluators.llm_judge import Relevance\n"
            'suite = EvalSuite("cli-judge-only")\n'
            'suite.add_case(EvalCase(input="a", expected_output="x"))\n'
            "suite.add_evaluator(Relevance())\n"
        )
        assert cmd_validate(_args(f)) == 1
        assert "NOTHING_VALIDATED" in capsys.readouterr().out

    def test_ok_plus_unvalidatable_still_passes(self):
        # Partial coverage stays a warning — only TOTAL blindness fails.
        suite = EvalSuite("partial")
        suite.add_cases([
            EvalCase(input="a", expected_output="hello"),
            EvalCase(input="b"),
        ])
        suite.add_evaluators(Contains(["hello"]), SpyJudge())
        report = validate_suite(suite)
        assert report.passed is True
        assert report.status == STATUS_OK


# ─── (11) contrast rerun makes zero judge calls offline ─────────────────────

class TestContrastOffline:
    def test_contrast_rerun_never_calls_judge_offline(self):
        spy = SpyJudge()
        original, twin = _pair("Paris is the capital of Germany")
        suite = EvalSuite("contrast-judge")
        suite.add_cases([original, twin])
        suite.add_evaluators(Contains(["Paris"]), spy)
        report = validate_suite(suite, contrast=True)  # offline default
        assert spy.calls == 0
        assert report.results[0].status == STATUS_NO_DISCRIMINATION

    def test_contrast_filter_holds_even_if_ref_passing_leaks(self):
        # Defense-in-depth: force a judge-backed grader into the contrast
        # pool via _validate_case and assert the offline filter drops it.
        from multivon_eval.validate import _validate_case

        spy = SpyJudge()
        original, twin = _pair("Paris is the capital of Germany")
        cv = _validate_case(
            0, original, [original, twin], [Contains(["Paris"]), spy],
            include_judges=False, contrast=True,
        )
        assert spy.calls == 0
        assert cv.status == STATUS_NO_DISCRIMINATION
        assert "spy_judge" not in cv.reason

    def test_include_judges_contrast_calls_are_tracked_in_costs(self):
        spy = SpyJudge()
        original, twin = _pair("Paris is the capital of Germany")
        suite = EvalSuite("contrast-judges-on")
        suite.add_cases([original, twin])
        suite.add_evaluators(spy)
        report = validate_suite(suite, include_judges=True, contrast=True)
        # ref grading on the original (the twin carries no reference) +
        # the contrast rerun against the twin's known-bad output.
        assert spy.calls == 2
        # costs snapshot travels on the report and into to_dict/to_json.
        data = report.to_dict()
        assert "costs" in data
        assert json.loads(report.to_json())["skipped_judge_graders"] == []

    def test_to_dict_discloses_skipped_judge_graders_offline(self):
        suite = EvalSuite("disclose")
        suite.add_case(EvalCase(input="a", expected_output="hello"))
        suite.add_evaluators(Contains(["hello"]), SpyJudge())
        data = validate_suite(suite).to_dict()
        assert data["skipped_judge_graders"] == ["spy_judge"]
        report = validate_suite(suite)
        assert "spy_judge" in str(report)
        assert "--judges" in str(report)


# ─── (12) JudgeUnavailable routes to UNVALIDATABLE, not BROKEN ──────────────

class TestJudgeUnavailableRouting:
    def test_judge_outage_is_unvalidatable_infrastructure(self):
        from multivon_eval import JudgeUnavailable

        class OutageJudge(Evaluator):
            name = "outage_judge"
            uses_llm_judge = True

            def evaluate(self, case, output):
                raise JudgeUnavailable("503 from provider", provider="p", model="m")

        suite = EvalSuite("outage")
        suite.add_case(EvalCase(input="a", expected_output="x"))
        suite.add_evaluator(OutageJudge())
        report = validate_suite(suite, include_judges=True)
        (cv,) = report.results
        assert cv.status == STATUS_UNVALIDATABLE
        assert "judge unavailable" in cv.reason
        assert "not a grader verdict" in cv.reason
        assert report.broken == []

    def test_genuine_grader_failure_still_broken_despite_outage(self):
        from multivon_eval import JudgeUnavailable

        class OutageJudge(Evaluator):
            name = "outage_judge"
            uses_llm_judge = True

            def evaluate(self, case, output):
                raise JudgeUnavailable("503", provider="p", model="m")

        suite = EvalSuite("outage+broken")
        suite.add_case(EvalCase(input="a", expected_output="the answer"))
        suite.add_evaluators(Contains(["unicorn"]), OutageJudge())
        report = validate_suite(suite, include_judges=True)
        (cv,) = report.results
        # Only the genuine verdict speaks: BROKEN via Contains, and the
        # outage never appears among the failed graders.
        assert cv.status == STATUS_BROKEN
        assert [g.evaluator for g in cv.failed_graders] == ["contains"]

    def test_generic_grader_exception_still_broken(self):
        class Crashy(Evaluator):
            name = "crashy2"

            def evaluate(self, case, output):
                raise RuntimeError("actual bug")

        suite = EvalSuite("crash2")
        suite.add_case(EvalCase(input="a", expected_output="x"))
        suite.add_evaluator(Crashy())
        report = validate_suite(suite)
        assert report.results[0].status == STATUS_BROKEN
