"""Tests for multivon_eval.mutate — deterministic mutators + template grids.

Everything here is pure-Python: no LLM calls, no network, no mocking
needed (the whole point of the deterministic generation toolkit).
"""
from __future__ import annotations

import json
import random

import pytest

from multivon_eval.case import EvalCase
from multivon_eval.mutate import (
    FLIP_MUTATIONS,
    MUTATIONS,
    TEMPLATE_PRODUCT_CAP,
    case_noise,
    cases_from_template,
    mutate_cases,
    negation_flip,
    punctuation_strip,
    typo_noise,
    unicode_confusable,
    whitespace_noise,
)


def _rng(seed=0):
    return random.Random(seed)


def _strip_provenance(case: EvalCase) -> str:
    """Serialize the deterministic part of a case (everything except the
    provenance stamp's fresh case_uid / authored_at)."""
    meta = {k: v for k, v in case.metadata.items() if k != "_provenance"}
    return json.dumps(
        [case.input, case.expected_output, case.context, meta, case.tags],
        ensure_ascii=False, sort_keys=True,
    )


# ─── Individual mutators ─────────────────────────────────────────────────────

class TestMutators:
    def test_typo_noise_changes_a_long_word(self):
        out = typo_noise("The refund window is thirty days.", _rng(1))
        assert out is not None and out != "The refund window is thirty days."

    def test_typo_noise_never_touches_numbers(self):
        # Only pure-letter runs of >=5 chars are eligible; digits never are.
        assert typo_noise("1234567890 999 42", _rng(0)) is None
        text = "Order 1234567890 today"
        for seed in range(20):
            out = typo_noise(text, _rng(seed))
            if out is not None:
                assert "1234567890" in out  # the number is never mutated

    def test_typo_noise_inapplicable_short_words(self):
        assert typo_noise("a b c d", _rng(0)) is None

    def test_whitespace_noise_only_whitespace_differs(self):
        text = "hello world again"
        out = whitespace_noise(text, _rng(3))
        assert out is not None
        assert "".join(out.split()) == "".join(text.split())

    def test_whitespace_noise_blank_inapplicable(self):
        assert whitespace_noise("   ", _rng(0)) is None

    def test_case_noise_changes_case_only(self):
        text = "please cancel my order"
        out = case_noise(text, _rng(5))
        assert out is not None
        assert out != text
        assert out.lower() == text.lower()

    def test_case_noise_inapplicable_without_letters(self):
        assert case_noise("123 456", _rng(0)) is None

    def test_unicode_confusable_preserves_visual_text(self):
        text = "payment account"
        out = unicode_confusable(text, _rng(2))
        assert out is not None
        assert out != text
        assert len(out) == len(text)  # confusables are 1:1 substitutions

    def test_unicode_confusable_never_maps_digits(self):
        # The borrowed homoglyph table's "0" entry was dropped on purpose.
        assert unicode_confusable("000 111", _rng(0)) is None

    def test_punctuation_strip_terminal(self):
        out = punctuation_strip("Is this refundable?", _rng(0))
        assert out is not None
        assert "?" not in out or "," not in out

    def test_punctuation_strip_inapplicable(self):
        assert punctuation_strip("no punctuation here", _rng(0)) is None

    def test_negation_flip_single_site(self):
        assert negation_flip("The plan is active.", _rng(0)) == "The plan is not active."
        assert negation_flip("The plan is not active.", _rng(0)) == "The plan is active."
        assert negation_flip("You cannot do this thing", _rng(0)) == "You can do this thing"

    def test_negation_flip_conservative_on_multiple_sites(self):
        # Two negatable sites — inapplicable, never a guess.
        assert negation_flip("This is fine and that is not fine.", _rng(0)) is None

    def test_negation_flip_inapplicable_without_sites(self):
        assert negation_flip("Hello there friend", _rng(0)) is None

    def test_negation_flip_is_not_wins_over_is_and_not(self):
        # "is not" must be consumed as ONE site (longest-first alternation),
        # not double-counted as "is" + "not".
        out = negation_flip("The order is not refundable today", _rng(0))
        assert out == "The order is refundable today"


# ─── mutate_cases ────────────────────────────────────────────────────────────

CASES = [
    EvalCase(input="The refund window is 30 days for returns.",
             expected_output="30 days", context="policy doc"),
    EvalCase(input="Please cancel my subscription, thanks a lot.",
             expected_output="cancelled"),
]


class TestMutateCases:
    def test_deterministic_per_seed_byte_equal(self):
        a, _ = mutate_cases(CASES, seed=42)
        b, _ = mutate_cases(CASES, seed=42)
        assert [_strip_provenance(c) for c in a] == [_strip_provenance(c) for c in b]
        assert len(a) > 0

    def test_different_seed_differs(self):
        a, _ = mutate_cases(CASES, mutations=["typo_noise"], seed=1)
        b, _ = mutate_cases(CASES, mutations=["typo_noise"], seed=2)
        assert [c.input for c in a] != [c.input for c in b]

    def test_invariant_mutants_carry_expected_output_and_context(self):
        mutants, _ = mutate_cases(CASES, mutations=["whitespace_noise"], seed=0)
        assert mutants
        for m in mutants:
            assert m.metadata["generation"]["expectation"] == "invariant"
        assert mutants[0].expected_output == "30 days"
        assert mutants[0].context == "policy doc"

    def test_flip_mutant_drops_label_and_explains(self):
        src = [EvalCase(input="The plan is active.", expected_output="active")]
        mutants, report = mutate_cases(src, mutations=["negation_flip"], seed=0)
        (m,) = mutants
        assert m.input == "The plan is not active."
        assert m.expected_output is None  # the old label no longer applies
        assert m.metadata["generation"]["expectation"] == "flip"
        assert "relabel" in m.metadata["expected_behavior"]
        assert report.accepted == 1

    def test_metadata_contract(self):
        mutants, _ = mutate_cases(CASES, mutations=["typo_noise"], seed=7)
        g = mutants[0].metadata["generation"]
        assert g["kind"] == "mutation"
        assert g["mutation"] == "typo_noise"
        assert g["seed"] == 7
        assert g["source_case_uid"] is None  # sources were unstamped
        prov = mutants[0].metadata["_provenance"]
        assert prov["authored_by"] == "generator:mutation"

    def test_source_case_uid_recorded_when_source_is_stamped(self):
        from multivon_eval.provenance import stamp_metadata_inplace
        src = EvalCase(input="Please cancel my subscription now.",
                       expected_output="ok")
        stamp_metadata_inplace(src.metadata, authored_by="human",
                               git={}, targets=[])
        mutants, _ = mutate_cases([src], mutations=["typo_noise"], seed=0)
        assert (mutants[0].metadata["generation"]["source_case_uid"]
                == src.metadata["_provenance"]["case_uid"])

    def test_unknown_mutation_raises(self):
        with pytest.raises(ValueError, match="unknown mutation"):
            mutate_cases(CASES, mutations=["nope"])

    def test_invariant_mutant_without_label_dropped_malformed(self):
        # Source has no expected_output and no expected-behavior text —
        # an invariant mutant inherits that gap and the gate drops it.
        src = [EvalCase(input="Please refund my whole order today.")]
        mutants, report = mutate_cases(src, mutations=["typo_noise"], seed=0)
        assert mutants == []
        assert report.dropped_malformed == 1
        assert report.generated == 1

    def test_exact_duplicate_mutants_dropped_and_counted(self):
        # Only terminal punctuation exists, so both per_case attempts
        # produce the identical mutant — second one is a duplicate.
        src = [EvalCase(input="Hi there friend.", expected_output="hello")]
        mutants, report = mutate_cases(
            src, mutations=["punctuation_strip"], seed=0, per_case=2,
        )
        assert len(mutants) == 1
        assert report.generated == 2
        assert report.dropped_duplicate == 1
        assert report.accepted == 1

    def test_report_accounting_invariant(self):
        _, report = mutate_cases(CASES, seed=3, per_case=2)
        assert report.kind == "mutation"
        assert report.generated == (
            report.accepted + report.dropped_malformed
            + report.dropped_duplicate + report.dropped_hardness
            + report.dropped_unverified
        )
        assert report.requested == len(CASES) * len(MUTATIONS) * 2

    def test_flip_registry(self):
        assert FLIP_MUTATIONS == {"negation_flip"}
        assert set(FLIP_MUTATIONS) <= set(MUTATIONS)


# ─── cases_from_template ─────────────────────────────────────────────────────

AXES = {"item": ["a laptop", "a phone", "some shoes"],
        "when": ["yesterday", "in March"]}


class TestCasesFromTemplate:
    def test_full_product(self):
        cases, report = cases_from_template(
            "Refund for {item} bought {when}", AXES,
            expected_output="refund decision",
        )
        assert report.generated == 6
        assert report.accepted == 6
        assert report.kind == "template"
        inputs = {c.input for c in cases}
        assert "Refund for a laptop bought yesterday" in inputs
        assert "Refund for some shoes bought in March" in inputs

    def test_expected_output_formatted_per_row(self):
        cases, _ = cases_from_template(
            "Refund for {item} bought {when}", AXES,
            expected_output="policy for {item}",
        )
        assert any(c.expected_output == "policy for a phone" for c in cases)

    def test_metadata_records_axis_values_and_seed(self):
        cases, _ = cases_from_template(
            "Refund for {item} bought {when}", AXES, seed=9,
            expected_output="x",
        )
        g = cases[0].metadata["generation"]
        assert g["kind"] == "template"
        assert g["seed"] == 9
        assert g["axes"] == {"item": "a laptop", "when": "yesterday"}
        assert cases[0].metadata["_provenance"]["authored_by"] == "generator:template"

    def test_product_cap_errors_clearly(self):
        big = {"a": list(range(13)), "b": list(range(13)), "c": list(range(13))}
        with pytest.raises(ValueError, match="cap"):
            cases_from_template("{a} {b} {c}", big, expected_output="x")
        assert 13 ** 3 > TEMPLATE_PRODUCT_CAP

    def test_pairwise_covers_every_pair(self):
        axes = {"a": ["1", "2", "3"], "b": ["x", "y", "z"], "c": ["p", "q", "r"]}
        cases, report = cases_from_template(
            "{a} {b} {c}", axes, sample="pairwise", seed=4, expected_output="ok",
        )
        assert report.generated < 27  # smaller than the full product
        rows = [c.metadata["generation"]["axes"] for c in cases]
        keys = list(axes)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                seen = {(r[keys[i]], r[keys[j]]) for r in rows}
                want = {(vi, vj) for vi in axes[keys[i]] for vj in axes[keys[j]]}
                assert seen == want, f"uncovered pairs for ({keys[i]}, {keys[j]})"

    def test_pairwise_deterministic_per_seed(self):
        axes = {"a": ["1", "2", "3"], "b": ["x", "y", "z"], "c": ["p", "q"]}
        a, _ = cases_from_template("{a} {b} {c}", axes, sample="pairwise",
                                   seed=2, expected_output="ok")
        b, _ = cases_from_template("{a} {b} {c}", axes, sample="pairwise",
                                   seed=2, expected_output="ok")
        assert [c.input for c in a] == [c.input for c in b]

    def test_n_subsamples_deterministically(self):
        a, ra = cases_from_template("Refund for {item} bought {when}", AXES,
                                    n=3, seed=1, expected_output="x")
        b, _ = cases_from_template("Refund for {item} bought {when}", AXES,
                                   n=3, seed=1, expected_output="x")
        assert ra.requested == 3
        assert [c.input for c in a] == [c.input for c in b]

    def test_missing_axis_raises(self):
        with pytest.raises(ValueError, match="no axis values"):
            cases_from_template("Refund for {item} on {date}",
                                {"item": ["x"]}, expected_output="y")

    def test_unused_axis_raises(self):
        with pytest.raises(ValueError, match="do not appear"):
            cases_from_template("Refund for {item}",
                                {"item": ["x"], "extra": ["y"]},
                                expected_output="y")

    def test_bad_sample_raises(self):
        with pytest.raises(ValueError, match="sample"):
            cases_from_template("{item}", {"item": ["x"]}, sample="grid")

    def test_no_label_rows_are_kept_and_gated_on_input_only(self):
        # Rows without expected_output/expected_behavior are valid — judge
        # evaluators need no reference answer. Gate is non-empty input +
        # dedupe; no label is invented.
        cases, report = cases_from_template(
            "Refund for {item} bought {when}", AXES,
        )
        assert len(cases) == report.accepted == report.generated
        assert all((c.input or "").strip() for c in cases)
        assert all(c.expected_output is None for c in cases)
        assert report.dropped_malformed == 0

    def test_expected_behavior_satisfies_gate_without_label(self):
        cases, report = cases_from_template(
            "Refund for {item} bought {when}", AXES,
            expected_behavior="answers politely with the refund policy",
        )
        assert report.accepted == 6
        assert cases[0].expected_output is None
        assert cases[0].metadata["expected_behavior"].startswith("answers")
