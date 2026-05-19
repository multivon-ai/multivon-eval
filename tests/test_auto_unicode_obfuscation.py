"""Validation tests for multivon_eval.auto.generate_unicode_obfuscation_cases.

Deterministic — character-level transforms, no LLM.
"""
from __future__ import annotations

from multivon_eval.auto import generate_unicode_obfuscation_cases


def test_homoglyph_substitutes_visually_identical_letters():
    cases = generate_unicode_obfuscation_cases(
        base_strings=["Aadhaar"],
        obfuscation_kinds=("homoglyph",),
    )
    assert len(cases) == 1
    obfuscated_input = cases[0].input
    # "Aadhaar" contains 'A', 'a', 'a', 'a' — at least one Cyrillic 'А'/'а' substitute
    assert "А" in obfuscated_input or "а" in obfuscated_input
    # Visually it should still look right — but the bytes differ
    assert "Aadhaar" not in obfuscated_input  # original ASCII gone for substituted chars
    # Original preserved as expected_output for evaluator comparison
    assert cases[0].expected_output == "Aadhaar"


def test_zero_width_inserts_invisible_chars_between_every_char():
    cases = generate_unicode_obfuscation_cases(
        base_strings=["PAN"],
        obfuscation_kinds=("zero_width",),
    )
    text = cases[0].input
    # U+200B should appear between each character
    assert "​" in text
    # Stripping zero-width should restore the original
    assert text.replace("​", "").endswith("PAN")


def test_rtlo_prepends_right_to_left_override():
    cases = generate_unicode_obfuscation_cases(
        base_strings=["1234"],
        obfuscation_kinds=("rtlo",),
    )
    text = cases[0].input
    assert "‮" in text


def test_cross_product_of_bases_and_kinds():
    cases = generate_unicode_obfuscation_cases(
        base_strings=["one", "two", "three"],
        obfuscation_kinds=("homoglyph", "zero_width", "rtlo"),
    )
    assert len(cases) == 9  # 3 base × 3 kinds


def test_each_case_carries_metadata_and_tag():
    cases = generate_unicode_obfuscation_cases(
        base_strings=["secret"],
        obfuscation_kinds=("homoglyph",),
    )
    case = cases[0]
    assert case.tags == ["adversarial:unicode_obfuscation:homoglyph"]
    assert case.metadata["target_failure_mode"] == "unicode_obfuscation"
    assert case.metadata["obfuscation_kind"] == "homoglyph"
    assert case.metadata["original_string"] == "secret"
    # stress_tests metadata is what validate_adversarial_cases needs
    assert "PIIEvaluator" in case.metadata["stress_tests"]


def test_unknown_kind_is_silently_skipped():
    cases = generate_unicode_obfuscation_cases(
        base_strings=["x"],
        obfuscation_kinds=("homoglyph", "made_up"),  # bogus kind
    )
    # Only the valid kind produces a case
    assert len(cases) == 1
    assert cases[0].metadata["obfuscation_kind"] == "homoglyph"


def test_empty_inputs_produce_empty_output():
    cases = generate_unicode_obfuscation_cases(
        base_strings=[],
        obfuscation_kinds=("homoglyph",),
    )
    assert cases == []
