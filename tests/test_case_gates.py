"""Tests for ``multivon_eval.case_gates`` — per-case acceptance gates.

All deterministic, no network: the well-formed + duplicate gates are free
by design, and the hardness gate is exercised via a stub baseline +
deterministic evaluator (NotEmpty) so no judge is ever called.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from multivon_eval import EvalCase
from multivon_eval.case_gates import (
    JACCARD_DUPLICATE_THRESHOLD,
    GateResult,
    GenerationReport,
    _input_tokens,
    digest_inputs,
    gate_duplicate,
    gate_hardness,
    gate_well_formed,
    token_jaccard,
)


# ─── gate_well_formed ─────────────────────────────────────────────────────


def test_well_formed_passes_with_expected_output():
    case = EvalCase(input="What is the refund window?", expected_output="30 days")
    res = gate_well_formed(case)
    assert isinstance(res, GateResult)
    assert res.gate == "well_formed"
    assert res.passed


def test_well_formed_passes_with_metadata_expected_behavior():
    case = EvalCase(
        input="Ask about pricing",
        metadata={"expected_behavior": "Model should refuse to guess prices."},
    )
    assert gate_well_formed(case).passed


def test_well_formed_fails_on_empty_input():
    res = gate_well_formed(EvalCase(input="", expected_output="something"))
    assert not res.passed
    assert "input" in res.reason

    # Whitespace-only input is empty too.
    assert not gate_well_formed(EvalCase(input="   \n ", expected_output="x")).passed


def test_well_formed_fails_without_any_expected_behavior():
    res = gate_well_formed(EvalCase(input="A question with no answer key"))
    assert not res.passed
    assert "expected" in res.reason


def test_well_formed_ignores_blank_expected_output_and_blank_metadata():
    # Blank strings don't count as an expected behavior.
    assert not gate_well_formed(
        EvalCase(input="q", expected_output="  ", metadata={"expected_behavior": ""})
    ).passed


# ─── gate_duplicate ───────────────────────────────────────────────────────


def test_duplicate_loose_normalized_identical_inputs():
    accepted = [EvalCase(input="What is  the refund\nwindow?", expected_output="a")]
    dup = EvalCase(input="What is the refund window?", expected_output="b")
    res = gate_duplicate(dup, accepted)
    assert not res.passed
    assert "identical" in res.reason


def test_duplicate_token_jaccard_near_identical():
    # 6 shared tokens / 7 union tokens = 0.857 ≥ 0.85 → duplicate.
    accepted = [EvalCase(input="what is the standard refund window policy")]
    near = EvalCase(input="what is the refund window policy")
    res = gate_duplicate(near, accepted)
    assert not res.passed
    assert "Jaccard" in res.reason


def test_duplicate_distinct_inputs_pass():
    accepted = [EvalCase(input="What is the refund window?")]
    fresh = EvalCase(input="Can I ship internationally to Brazil?")
    assert gate_duplicate(fresh, accepted).passed


def test_duplicate_empty_accepted_list_passes():
    assert gate_duplicate(EvalCase(input="anything"), []).passed


def test_token_jaccard_math():
    a = _input_tokens("what is the refund window")
    b = _input_tokens("what is the refund window please")
    assert token_jaccard(a, b) == pytest.approx(5 / 6)
    assert token_jaccard(a, a) == 1.0
    assert token_jaccard(frozenset(), frozenset()) == 1.0
    assert 0.0 < JACCARD_DUPLICATE_THRESHOLD <= 1.0


def test_tokenization_is_casefolded_and_punctuation_free():
    assert _input_tokens("What's THE Refund-Window?") == _input_tokens(
        "what s the refund window"
    )


# ─── gate_hardness (delegation) ───────────────────────────────────────────


def test_gate_hardness_delegates_to_validate_adversarial_cases():
    cases = [EvalCase(input="q", metadata={"stress_tests": ["NotEmpty"]})]
    baseline = lambda _x: ""  # noqa: E731

    with patch(
        "multivon_eval.auto.validate_adversarial_cases",
        return_value=(["kept"], ["report"]),
    ) as mocked:
        kept, reports = gate_hardness(
            cases, baseline, n_shots=5, hardness_band=(0.2, 0.8), judge=None,
        )

    assert kept == ["kept"]
    assert reports == ["report"]
    mocked.assert_called_once_with(
        cases, baseline, n_shots=5, hardness_band=(0.2, 0.8), judge=None,
    )


def test_gate_hardness_real_band_filtering_with_deterministic_evaluator():
    # NotEmpty is deterministic (no judge): an empty baseline output fails
    # every shot (failure_rate 1.0 → in band), a non-empty one never fails
    # (failure_rate 0.0 → out of band).
    hard = EvalCase(input="hard question", metadata={"stress_tests": ["NotEmpty"]})
    easy = EvalCase(input="easy question", metadata={"stress_tests": ["NotEmpty"]})

    def baseline(text: str) -> str:
        return "" if "hard" in text else "a perfectly fine answer"

    kept, reports = gate_hardness([hard, easy], baseline, n_shots=3)
    assert kept == [hard]
    by_case = {id(r.case): r for r in reports}
    assert by_case[id(hard)].failure_rate == 1.0
    assert by_case[id(easy)].failure_rate == 0.0
    assert by_case[id(hard)].in_hardness_band
    assert not by_case[id(easy)].in_hardness_band


# ─── digest_inputs ────────────────────────────────────────────────────────


def test_digest_truncates_to_first_words():
    case = EvalCase(input="one two three four five six seven eight nine ten")
    digest = digest_inputs([case], max_words=8)
    assert digest == "- one two three four five six seven eight …"


def test_digest_short_input_has_no_ellipsis():
    assert digest_inputs([EvalCase(input="short question")]) == "- short question"


def test_digest_caps_entries_to_most_recent():
    cases = [EvalCase(input=f"question number {i}") for i in range(100)]
    digest = digest_inputs(cases, max_entries=10)
    lines = digest.splitlines()
    assert len(lines) == 10
    assert lines[0] == "- question number 90"
    assert lines[-1] == "- question number 99"


# ─── GenerationReport ─────────────────────────────────────────────────────


def test_generation_report_summary_line_with_hardness():
    report = GenerationReport(
        requested=500, generated=500, accepted=431,
        dropped_malformed=12, dropped_duplicate=38, dropped_hardness=19,
        hardness_skipped=False, hardness_band=(0.5, 1.0),
    )
    assert report.summary_line() == (
        "generated 500, accepted 431 — dropped 38 duplicates, 12 malformed, "
        "19 outside hardness band [0.5, 1.0]"
    )


def test_generation_report_summary_line_when_hardness_skipped():
    report = GenerationReport(
        requested=30, generated=30, accepted=27,
        dropped_malformed=1, dropped_duplicate=2,
    )
    assert report.summary_line() == (
        "generated 30, accepted 27 — dropped 2 duplicates, 1 malformed"
    )
    assert report.hardness_skipped
    assert report.hardness_skip_reason == "no --validate-cases / baseline model"
