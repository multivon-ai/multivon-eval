"""Judge-integrity fixes: parser Unknown out (CR-04), no error laundering in
QAG evaluators (CR-03), error-budget gate (CR-02), None-sentinel judge config
merge (CR-09)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from multivon_eval import (
    EvalCase, EvalStatus, EvalSuite, JudgeUnavailable,
)
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.evaluators.llm_judge import (
    CustomRubric, Faithfulness, _extract_json_array, _parse_yes_no,
)
from multivon_eval.judge import JudgeConfig, configure, get_global_judge, resolve_judge
from multivon_eval.result import EvalGateFailure


@pytest.fixture()
def restore_global_judge():
    saved = get_global_judge()
    yield
    configure(saved)


# ─────────────────────────────────────────────────────────────────────────────
# CR-04 — _parse_yes_no verdict table
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reply,expected", [
    # Clear affirmatives/negatives — identical to the pre-fix parser.
    ("Yes", True),
    ("Yes, it is correct", True),
    ("YES definitely", True),
    ("yes.", True),
    ("No", False),
    ("No, it is wrong", False),
    ("NO not at all", False),
    ("The answer is yes.", True),
    ("I believe the answer is no.", False),
    ('"Yes"', True),
    # Previously mis-scored ambiguous tail — now UNKNOWN, never a Yes.
    ("I cannot say yes or no with certainty", None),
    ("unclear", None),
    ("", None),
    ("Maybe", None),
    ("It depends on the context and the situation at hand", None),
    ("eyes have it", None),  # substring "yes" must not match
])
def test_parse_yes_no_verdict_table(reply, expected):
    assert _parse_yes_no(reply) is expected


def test_parse_yes_no_never_yes_on_hedge():
    hedges = [
        "I cannot say yes or no with certainty",
        "It could be yes or it could be no",
        "Neither yes nor no applies here",
    ]
    for hedge in hedges:
        assert _parse_yes_no(hedge) is not True


def test_extract_json_array_with_bracket_in_claim():
    raw = 'Here are the claims: ["uses arr[0] indexing", "second claim"] hope that helps'
    assert _extract_json_array(raw) == ["uses arr[0] indexing", "second claim"]


def test_extract_json_array_plain_json():
    assert _extract_json_array('["a", "b"]') == ["a", "b"]


def test_extract_json_array_unparseable_returns_none():
    assert _extract_json_array("I could not find any claims, sorry.") is None


# ─────────────────────────────────────────────────────────────────────────────
# CR-04 — unknown exclusion arithmetic in QAG scoring
# ─────────────────────────────────────────────────────────────────────────────

def _rubric3():
    return CustomRubric(
        criteria=[("Q1?", True), ("Q2?", True), ("Q3?", True)],
        threshold=0.5,
    )


def test_qag_unknown_excluded_from_denominator():
    answers = iter(["Yes", "I really cannot tell", "No"])
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        side_effect=lambda prompt, cfg: next(answers),
    ):
        result = _rubric3().evaluate(EvalCase(input="q"), "out")
    # 1 pass + 1 fail scored, 1 unknown excluded: 1/2, not 1/3 or 2/3.
    assert result.score == pytest.approx(0.5)
    assert "1 of 3 question(s) UNKNOWN" in result.reason


def test_qag_clean_replies_score_unchanged():
    answers = iter(["Yes", "Yes", "No"])
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        side_effect=lambda prompt, cfg: next(answers),
    ):
        result = _rubric3().evaluate(EvalCase(input="q"), "out")
    assert result.score == pytest.approx(2 / 3)
    assert "UNKNOWN" not in result.reason


def test_qag_all_unknown_raises_judge_unavailable():
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        return_value="beats me",
    ):
        with pytest.raises(JudgeUnavailable, match="no parseable Yes/No verdict"):
            _rubric3().evaluate(EvalCase(input="q"), "out")


def test_qag_all_unknown_surfaces_as_judge_error_status():
    suite = EvalSuite("t").add_case(EvalCase(input="q")).add_evaluator(_rubric3())
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        return_value="beats me",
    ):
        report = suite.run(lambda x: "out", verbose=False, workers=1)
    assert report.case_results[0].status == EvalStatus.JUDGE_ERROR
    assert report.evaluated == 0
    assert report.errors == 1


# ─────────────────────────────────────────────────────────────────────────────
# CR-03 — judge errors surface as statuses, not fake zeros/Falses
# ─────────────────────────────────────────────────────────────────────────────

def test_per_question_judge_exception_yields_judge_error_status():
    suite = EvalSuite("t").add_case(EvalCase(input="q")).add_evaluator(_rubric3())
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        side_effect=JudgeUnavailable("transient 503", provider="p", model="m"),
    ):
        report = suite.run(lambda x: "out", verbose=False, workers=1)
    cr = report.case_results[0]
    assert cr.status == EvalStatus.JUDGE_ERROR
    # Excluded from the pass-rate denominator.
    assert report.evaluated == 0
    assert report.pass_rate == 0.0


def test_per_question_generic_exception_yields_evaluator_error_status():
    suite = EvalSuite("t").add_case(EvalCase(input="q")).add_evaluator(_rubric3())
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        side_effect=RuntimeError("bug in prompt builder"),
    ):
        report = suite.run(lambda x: "out", verbose=False, workers=1)
    cr = report.case_results[0]
    assert cr.status == EvalStatus.EVALUATOR_ERROR
    assert report.evaluated == 0


def test_claim_extraction_failure_is_evaluator_error_not_zero():
    ev = Faithfulness(threshold=0.7)
    case = EvalCase(input="q", context="ctx")
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        return_value="Sorry, I could not identify claims.",
    ):
        with pytest.raises(ValueError, match="could not extract"):
            ev.evaluate(case, "The sky is blue.")

    suite = EvalSuite("t").add_case(case).add_evaluator(Faithfulness(threshold=0.7))
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        return_value="Sorry, I could not identify claims.",
    ):
        report = suite.run(lambda x: "The sky is blue.", verbose=False, workers=1)
    assert report.case_results[0].status == EvalStatus.EVALUATOR_ERROR


def test_faithfulness_claim_cap_disclosed():
    claims = [f"claim {i}" for i in range(25)]
    replies = iter([str(claims).replace("'", '"')] + ["Yes"] * 10)
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        side_effect=lambda prompt, cfg: next(replies),
    ):
        result = Faithfulness(threshold=0.7).evaluate(
            EvalCase(input="q", context="ctx"), "long output"
        )
    assert result.score == pytest.approx(1.0)
    assert "verified 10 of 25 claims (capped)" in result.reason


def test_faithfulness_uncapped_no_disclosure():
    replies = iter(['["a", "b"]', "Yes", "No"])
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        side_effect=lambda prompt, cfg: next(replies),
    ):
        result = Faithfulness(threshold=0.7).evaluate(
            EvalCase(input="q", context="ctx"), "output"
        )
    assert result.score == pytest.approx(0.5)
    assert "capped" not in result.reason


def test_faithfulness_adversarial_bracket_claims_verified_fully():
    replies = iter(['["uses arr[0]", "plain claim"]', "Yes", "Yes"])
    with patch(
        "multivon_eval.evaluators.llm_judge.make_judge_call",
        side_effect=lambda prompt, cfg: next(replies),
    ):
        result = Faithfulness(threshold=0.7).evaluate(
            EvalCase(input="q", context="ctx"), "output"
        )
    assert result.score == pytest.approx(1.0)
    assert "2/2 claims grounded" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# CR-02 — error budget on the CI gate
# ─────────────────────────────────────────────────────────────────────────────

class _AlwaysPass(Evaluator):
    name = "always_pass"

    def evaluate(self, case, output):
        return self._result(1.0, "ok")


def _suite_9_errors_1_pass() -> EvalSuite:
    suite = EvalSuite("gate")
    for i in range(10):
        suite.add_case(EvalCase(input=f"q{i}"))
    suite.add_evaluator(_AlwaysPass())
    return suite


def _model_fn(prompt: str) -> str:
    if prompt == "q0":
        return "fine"
    raise ConnectionError("model down")


def test_error_budget_gate_fails_on_9_of_10_errors():
    suite = _suite_9_errors_1_pass()
    with pytest.raises(EvalGateFailure) as exc_info:
        suite.run(
            _model_fn, verbose=False, workers=1,
            fail_threshold=0.5, max_error_rate=0.1,
        )
    msg = str(exc_info.value)
    assert "Eval gate INDETERMINATE" in msg
    assert "error rate 90.0% exceeds error budget 10.0%" in msg
    assert "model_error=9" in msg
    # The blind pass_rate the budget protects against is disclosed.
    assert exc_info.value.pass_rate == pytest.approx(1.0)


def test_error_budget_enforced_without_fail_threshold():
    suite = _suite_9_errors_1_pass()
    with pytest.raises(EvalGateFailure, match="error budget"):
        suite.run(_model_fn, verbose=False, workers=1, max_error_rate=0.1)


def test_default_gate_unchanged_but_warns_at_10_percent(capsys):
    suite = _suite_9_errors_1_pass()
    report = suite.run(_model_fn, verbose=False, workers=1, fail_threshold=0.5)
    assert report.pass_rate == pytest.approx(1.0)
    err = capsys.readouterr().err
    assert "error budget" in err
    assert "9/10" in err


def test_no_warning_when_no_errors(capsys):
    suite = EvalSuite("clean").add_case(EvalCase(input="q")).add_evaluator(_AlwaysPass())
    suite.run(lambda x: "fine", verbose=False, workers=1, fail_threshold=0.5)
    assert "error budget" not in capsys.readouterr().err


def test_error_rate_within_budget_passes():
    suite = EvalSuite("ok")
    for i in range(10):
        suite.add_case(EvalCase(input=f"q{i}"))
    suite.add_evaluator(_AlwaysPass())

    def one_error(prompt: str) -> str:
        if prompt == "q9":
            raise ConnectionError("blip")
        return "fine"

    report = suite.run(
        one_error, verbose=False, workers=1,
        fail_threshold=0.5, max_error_rate=0.5,
    )
    assert report.errors == 1
    assert report.error_rate == pytest.approx(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# CR-09 — resolve_judge None-sentinel merge
# ─────────────────────────────────────────────────────────────────────────────

def test_explicit_zero_temperature_overrides_nonzero_global(restore_global_judge):
    configure(JudgeConfig(provider="openai", model="m", temperature=0.7))
    resolved = resolve_judge(JudgeConfig(temperature=0.0))
    assert resolved.temperature == 0.0


def test_explicit_default_valued_max_tokens_and_timeout_override(restore_global_judge):
    configure(JudgeConfig(provider="openai", model="m", max_tokens=2048, timeout=60))
    resolved = resolve_judge(JudgeConfig(max_tokens=1024, timeout=30))
    assert resolved.max_tokens == 1024
    assert resolved.timeout == 30


def test_unset_fields_inherit_globals(restore_global_judge):
    configure(JudgeConfig(
        provider="openai", model="m",
        temperature=0.7, max_tokens=2048, timeout=60, reliability_sample=9,
    ))
    resolved = resolve_judge(JudgeConfig(provider="anthropic"))
    assert resolved.provider == "anthropic"
    assert resolved.temperature == 0.7
    assert resolved.max_tokens == 2048
    assert resolved.timeout == 60
    assert resolved.reliability_sample == 9


def test_no_args_config_resolves_historical_defaults(restore_global_judge, monkeypatch):
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    configure(JudgeConfig())
    resolved = resolve_judge(None)
    assert resolved.temperature == 0.0
    assert resolved.max_tokens == 1024
    assert resolved.timeout == 30
    assert resolved.reliability_sample == 5
    assert resolved.provider == "anthropic"


def test_env_var_precedence_unchanged(restore_global_judge, monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-mini")
    configure(JudgeConfig())
    resolved = resolve_judge(None)
    assert resolved.provider == "openai"
    assert resolved.model == "gpt-4o-mini"


def test_resolve_fills_nones_on_direct_resolve():
    resolved = JudgeConfig(provider="openai", model="m").resolve()
    assert resolved.temperature == 0.0
    assert resolved.max_tokens == 1024
    assert resolved.timeout == 30
