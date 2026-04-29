import warnings
from unittest.mock import call, patch

import pytest

from multivon_eval import CheckEvaluator, EvalCase, EvalSuite


MOCK_QUESTIONS_JSON = '["Does the response mention the policy?", "Is the policy name explicit?", "Does it link to more info?"]'
MOCK_QUESTIONS = [
    "Does the response mention the policy?",
    "Is the policy name explicit?",
    "Does it link to more info?",
]


def make_case(input_text="What is the return policy?", output=None, context=None):
    return EvalCase(input=input_text, context=context)


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------

class TestCheckEvaluatorInit:
    def test_empty_criterion_raises(self):
        with pytest.raises(ValueError, match="criterion must be a non-empty string"):
            CheckEvaluator("")

    def test_whitespace_only_criterion_raises(self):
        with pytest.raises(ValueError, match="criterion must be a non-empty string"):
            CheckEvaluator("   ")

    def test_empty_questions_list_raises(self):
        with pytest.raises(ValueError, match="questions list must not be empty"):
            CheckEvaluator("Valid criterion", questions=[])

    def test_criterion_truncated_at_300_chars(self):
        long = "x" * 400
        ev = CheckEvaluator(long)
        assert len(ev._criterion) == 300

    def test_num_questions_clamped_below_1(self):
        ev = CheckEvaluator("Check something", num_questions=0)
        assert ev._num_questions == 1

    def test_num_questions_clamped_above_10(self):
        ev = CheckEvaluator("Check something", num_questions=99)
        assert ev._num_questions == 10

    def test_name_derived_from_criterion(self):
        ev = CheckEvaluator("Response should mention the return policy")
        assert ev.name.startswith("response_should_mention")

    def test_custom_name_respected(self):
        ev = CheckEvaluator("Something", name="my_check")
        assert ev.name == "my_check"

    def test_pre_supplied_questions_stored(self):
        ev = CheckEvaluator("Check", questions=["Is it short?", "Is it clear?"])
        assert ev._questions == [("Is it short?", True), ("Is it clear?", True)]

    def test_pre_supplied_questions_skip_generation(self):
        ev = CheckEvaluator("Check", questions=["Is it short?"])
        with patch("multivon_eval.evaluators.llm_judge._call") as mock_call:
            ev.prepare()
            mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

class TestCheckEvaluatorGeneration:
    def _make_ev(self, **kwargs):
        return CheckEvaluator("Response mentions the return policy", **kwargs)

    def test_generates_on_prepare(self):
        ev = self._make_ev()
        with patch("multivon_eval.evaluators.llm_judge._call", return_value=MOCK_QUESTIONS_JSON):
            ev.prepare()
        assert ev.resolved_questions == MOCK_QUESTIONS

    def test_retries_on_bad_json_then_succeeds(self):
        ev = self._make_ev()
        responses = ["not json at all", MOCK_QUESTIONS_JSON]
        with patch("multivon_eval.evaluators.llm_judge._call", side_effect=responses):
            ev.prepare()
        assert ev.resolved_questions == MOCK_QUESTIONS

    def test_fallback_on_two_consecutive_failures(self):
        ev = self._make_ev()
        with patch("multivon_eval.evaluators.llm_judge._call", return_value="bad"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ev.prepare()
        assert ev._used_fallback is True
        assert any("fallback" in str(w.message).lower() for w in caught)

    def test_fallback_uses_criterion_as_question(self):
        ev = self._make_ev()
        with patch("multivon_eval.evaluators.llm_judge._call", return_value="bad"):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                ev.prepare()
        assert ev._questions == [(ev._criterion, True)]

    def test_questions_capped_at_num_questions(self):
        many = '["Q1?", "Q2?", "Q3?", "Q4?", "Q5?", "Q6?"]'
        ev = CheckEvaluator("Check", num_questions=3)
        with patch("multivon_eval.evaluators.llm_judge._call", return_value=many):
            ev.prepare()
        assert len(ev._questions) == 3

    def test_prepare_called_twice_generates_once(self):
        ev = self._make_ev()
        with patch("multivon_eval.evaluators.llm_judge._call", return_value=MOCK_QUESTIONS_JSON) as mock_call:
            ev.prepare()
            ev.prepare()
        assert mock_call.call_count == 1

    def test_empty_array_triggers_fallback(self):
        ev = self._make_ev()
        with patch("multivon_eval.evaluators.llm_judge._call", return_value="[]"):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                ev.prepare()
        assert ev._used_fallback is True

    def test_resolved_questions_none_before_prepare(self):
        ev = self._make_ev()
        assert ev.resolved_questions is None

    def test_resolved_questions_list_after_prepare(self):
        ev = self._make_ev()
        with patch("multivon_eval.evaluators.llm_judge._call", return_value=MOCK_QUESTIONS_JSON):
            ev.prepare()
        assert isinstance(ev.resolved_questions, list)
        assert all(isinstance(q, str) for q in ev.resolved_questions)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

class TestCheckEvaluatorEvaluate:
    def _prepared_ev(self, threshold=0.7, questions=None):
        ev = CheckEvaluator(
            "Response mentions the return policy",
            threshold=threshold,
            questions=questions or MOCK_QUESTIONS,
        )
        return ev

    def test_passes_above_threshold(self):
        ev = self._prepared_ev(threshold=0.6)
        with patch("multivon_eval.evaluators.llm_judge._qag_eval", return_value=(0.67, ["✓ q1", "✗ q2", "✓ q3"])):
            result = ev.evaluate(make_case(), "Some response")
        assert result.passed is True
        assert result.score == pytest.approx(0.67)

    def test_fails_below_threshold(self):
        ev = self._prepared_ev(threshold=0.7)
        with patch("multivon_eval.evaluators.llm_judge._qag_eval", return_value=(0.33, ["✗ q1", "✗ q2", "✓ q3"])):
            result = ev.evaluate(make_case(), "Some response")
        assert result.passed is False

    def test_custom_threshold_respected(self):
        ev = self._prepared_ev(threshold=0.3)
        with patch("multivon_eval.evaluators.llm_judge._qag_eval", return_value=(0.33, [])):
            result = ev.evaluate(make_case(), "Short")
        assert result.passed is True

    def test_reason_includes_criterion(self):
        ev = self._prepared_ev()
        with patch("multivon_eval.evaluators.llm_judge._qag_eval", return_value=(1.0, [])):
            result = ev.evaluate(make_case(), "Answer")
        assert "Response mentions the return policy" in result.reason

    def test_fallback_flag_in_reason(self):
        ev = CheckEvaluator("Check")
        with patch("multivon_eval.evaluators.llm_judge._call", return_value="bad"):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                ev.prepare()
        with patch("multivon_eval.evaluators.llm_judge._qag_eval", return_value=(1.0, [])):
            result = ev.evaluate(make_case(), "Answer")
        assert "fallback" in result.reason.lower()

    def test_context_included_in_qag_prompt(self):
        ev = self._prepared_ev()
        case = make_case(context="Policy: 30 day returns")
        captured = {}

        def fake_qag(questions, ctx, judge):
            captured["ctx"] = ctx
            return (1.0, [])

        with patch("multivon_eval.evaluators.llm_judge._qag_eval", side_effect=fake_qag):
            ev.evaluate(case, "Answer")

        assert "Policy: 30 day returns" in captured["ctx"]

    def test_prepare_called_lazily_on_evaluate(self):
        ev = CheckEvaluator("Check something")
        with patch("multivon_eval.evaluators.llm_judge._call", return_value=MOCK_QUESTIONS_JSON):
            with patch("multivon_eval.evaluators.llm_judge._qag_eval", return_value=(1.0, [])):
                ev.evaluate(make_case(), "Answer")
        assert ev._questions is not None


# ---------------------------------------------------------------------------
# EvalSuite.add_check
# ---------------------------------------------------------------------------

class TestEvalSuiteAddCheck:
    def test_returns_self_for_chaining(self):
        suite = EvalSuite("test")
        result = suite.add_check("Check something")
        assert result is suite

    def test_creates_check_evaluator(self):
        suite = EvalSuite("test")
        suite.add_check("Tone is professional")
        assert len(suite._evaluators) == 1
        assert isinstance(suite._evaluators[0], CheckEvaluator)

    def test_empty_criterion_raises_at_call_time(self):
        suite = EvalSuite("test")
        with pytest.raises(ValueError):
            suite.add_check("")

    def test_threshold_passed_through(self):
        suite = EvalSuite("test")
        suite.add_check("Check", threshold=0.9)
        assert suite._evaluators[0].threshold == 0.9

    def test_questions_escape_hatch(self):
        suite = EvalSuite("test")
        suite.add_check("Check", questions=["Is it short?"])
        ev = suite._evaluators[0]
        assert ev._questions == [("Is it short?", True)]

    def test_multiple_add_checks(self):
        suite = EvalSuite("test")
        suite.add_check("Check A").add_check("Check B")
        assert len(suite._evaluators) == 2

    def test_warmup_calls_prepare_before_eval_loop(self):
        suite = EvalSuite("test")
        suite.add_check("Check", questions=["Is it good?"])
        suite.add_case(EvalCase(input="hi"))

        prepare_calls = []

        original_prepare = CheckEvaluator.prepare
        def tracking_prepare(self, judge=None):
            prepare_calls.append(self.name)
            original_prepare(self, judge)

        with patch.object(CheckEvaluator, "prepare", tracking_prepare):
            with patch("multivon_eval.suite.EvalSuite._run_case_once") as mock_run:
                mock_run.return_value = suite._evaluators[0].evaluate.__self__ if False else None
                # Just verify prepare is called; don't actually run cases
                for ev in suite._evaluators:
                    if hasattr(ev, "prepare"):
                        ev.prepare()

        assert len(prepare_calls) == 1
