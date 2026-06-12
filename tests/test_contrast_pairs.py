"""Tests for generate_contrast_pairs — fully mocked, zero LLM calls.

The proposal call (`discover._call_judge`) and the verification judge
(`evaluators.llm_judge.Faithfulness`) are both patched, mirroring the
test_discover.py approach.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from multivon_eval.case import EvalCase
from multivon_eval.generate import generate_contrast_pairs
from multivon_eval.judge import JudgeConfig
from multivon_eval.result import EvalResult

JUDGE = JudgeConfig(provider="anthropic", model="claude-haiku-4-5-20251001")

PROPOSAL = json.dumps({
    "unfaithful_answer": "X is Z, according to the documentation.",
    "changed_fact": "Y -> Z",
})


def _cases():
    return [
        EvalCase(input="What is X?", expected_output="X is Y.",
                 context="The docs say X is Y. The sky is blue."),
        EvalCase(input="Tell me about the launch date please",
                 expected_output="It launched in March.",
                 context="The product launched in March 2024."),
    ]


class _FakeFaithfulness:
    """Stand-in judge: returns the configured score, no LLM."""
    score = 0.1  # below the 0.7 threshold => flip confirmed

    def __init__(self, threshold=None, judge=None):
        self.threshold = 0.7

    def evaluate(self, case, output):
        return EvalResult(evaluator="faithfulness", score=type(self).score,
                          passed=type(self).score >= 0.7)


def _run(cases, *, proposal=PROPOSAL, faith=_FakeFaithfulness, **kwargs):
    with patch("multivon_eval.discover._call_judge", return_value=proposal), \
         patch("multivon_eval.evaluators.llm_judge.Faithfulness", faith):
        return generate_contrast_pairs(cases, judge=JUDGE, **kwargs)


class TestContrastPairs:
    def test_verified_twin_accepted_with_shared_pair_id(self, capsys):
        cases = _cases()
        twins, report = _run(cases)
        assert report.kind == "contrast"
        assert report.requested == 2
        assert report.accepted == len(twins) == 2
        for src, twin in zip(cases, twins):
            # pair_id written into BOTH cases' metadata
            assert twin.metadata["pair_id"] == src.metadata["pair_id"]
            assert twin.input == src.input
            assert twin.expected_output is None  # no gold answer for a twin
            assert twin.metadata["unfaithful_answer"].startswith("X is Z")
            g = twin.metadata["generation"]
            assert g["kind"] == "contrast"
            assert g["expectation"] == "fail"
            assert g["verified"] is True
            assert g["judge_score"] == pytest.approx(0.1)
            assert (twin.metadata["_provenance"]["authored_by"]
                    == "generator:contrast")
        # twins differ, so distinct pair_ids
        assert twins[0].metadata["pair_id"] != twins[1].metadata["pair_id"]

    def test_unconfirmed_flip_dropped_and_counted(self):
        class StillFaithful(_FakeFaithfulness):
            score = 0.9  # above threshold => the flip is NOT real

        cases = _cases()
        twins, report = _run(cases, faith=StillFaithful)
        assert twins == []
        assert report.dropped_unverified == 2
        assert report.accepted == 0
        assert "unverified" in report.summary_line()
        # rejected twins never contaminate the source cases
        assert all("pair_id" not in c.metadata for c in cases)

    def test_skipped_judge_verdict_counts_as_unverified(self):
        class Skips(_FakeFaithfulness):
            def evaluate(self, case, output):
                return EvalResult(evaluator="faithfulness", score=1.0,
                                  passed=True, metadata={"skipped": True})

        twins, report = _run(_cases(), faith=Skips)
        assert twins == []
        assert report.dropped_unverified == 2

    def test_verify_false_skips_judge_and_marks_unverified(self):
        fake = MagicMock()
        twins, report = _run(_cases(), faith=fake, verify=False)
        fake.assert_not_called()  # no judge spend at all
        assert report.accepted == 2
        assert all(t.metadata["generation"]["verified"] is False for t in twins)
        assert all(t.metadata["generation"]["judge_score"] is None for t in twins)

    def test_budget_hard_stop_preserves_partials(self):
        # Make every proposal "cost" more than the whole budget: the first
        # case completes, the second is never attempted — partials kept.
        with patch("multivon_eval.discover._call_judge", return_value=PROPOSAL), \
             patch("multivon_eval.discover._estimate_cost", return_value=10.0), \
             patch("multivon_eval.evaluators.llm_judge.Faithfulness",
                   _FakeFaithfulness):
            twins, report = generate_contrast_pairs(
                _cases(), judge=JUDGE, budget_usd=1.0,
            )
        assert len(twins) == 1
        assert report.requested == 2
        assert report.generated == 1
        assert report.accepted == 1

    def test_cases_without_context_or_label_not_eligible(self):
        cases = [
            EvalCase(input="no context here", expected_output="x"),
            EvalCase(input="no label here", context="some context"),
        ]
        twins, report = _run(cases)
        assert report.requested == 0
        assert twins == []

    def test_malformed_proposal_not_generated(self):
        twins, report = _run(_cases(), proposal="utterly not json")
        assert twins == []
        assert report.generated == 0
        assert report.requested == 2  # visible as requested - generated

    def test_proposal_call_failure_is_survivable(self):
        with patch("multivon_eval.discover._call_judge",
                   side_effect=RuntimeError("API down")), \
             patch("multivon_eval.evaluators.llm_judge.Faithfulness",
                   _FakeFaithfulness):
            twins, report = generate_contrast_pairs(_cases(), judge=JUDGE)
        assert twins == []
        assert report.generated == 0
