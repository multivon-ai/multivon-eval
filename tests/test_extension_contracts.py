"""Contributor fixtures for custom graders and imported output pairing."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from multivon_eval import EvalCase, EvalSuite, ExactMatch, MaxLatency, declare_dependencies
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.integrations import CaseImporter
from multivon_eval.result import EvalStatus


class MemoryImporter(CaseImporter):
    def load(self, **kwargs):
        return [EvalCase("same prompt", answer, metadata={"_output": answer}, case_id=answer)
                for answer in ("first", "second", "third")]


def replay(cases):
    with pytest.warns(DeprecationWarning, match="run_on_cases"):
        return MemoryImporter().as_model_fn(cases)


def suite(cases):
    return EvalSuite("imported outputs").add_cases(cases).add_evaluators(ExactMatch())


@pytest.mark.parametrize("workers", [1, 3])
def test_replay_pairs_duplicate_prompts_reordering_and_repeats(workers):
    cases = MemoryImporter().load()
    target = replay(cases)
    report = suite(list(reversed(cases))).run(target, runs=3, workers=workers, verbose=False)
    assert report.passed == 3
    assert [r.actual_output for r in report.case_results] == ["third", "second", "first"]
    assert all([t.data["output"] for t in r.trials] == [r.actual_output] * 3
               for r in report.case_results)


def test_async_replay_preserves_case_pairing():
    cases = MemoryImporter().load()
    report = asyncio.run(suite(cases).run_async(replay(cases), runs=2, verbose=False))
    assert report.passed == 3


def test_replay_rejects_ambiguous_unknown_and_changed_inputs():
    cases = MemoryImporter().load()
    target = replay(cases)
    for text in ("same prompt", "absent"):
        with pytest.raises(ValueError, match="Unknown or ambiguous"):
            target(text)
    changed = replace(cases[0], expected_output="changed")
    report = suite([changed]).run(target, verbose=False)
    assert report.case_results[0].status == EvalStatus.MODEL_ERROR


@pytest.mark.parametrize("metadata", [{}, {"_output": None}, {"_output": 42}])
def test_replay_requires_recorded_string_output(metadata):
    with pytest.raises(ValueError, match="string"):
        replay([EvalCase("prompt", metadata=metadata)])


def test_recorded_empty_output_is_valid_evidence():
    assert replay([EvalCase("prompt", metadata={"_output": ""})])("prompt") == ""


def test_saved_output_path_retains_origin_and_unknown_target_latency():
    cases = MemoryImporter().load()
    report = suite([]).add_evaluators(MaxLatency(100)).run_on_cases(
        [(case, case.metadata["_output"]) for case in cases], verbose=False,
    )
    assert report.passed == 3
    for result in report.case_results:
        trial = result.trials[0].data
        assert trial["origin"] == "import" and trial["latency_ms"] is None
        assert result.results[-1].metadata["skipped"]


class RecordedState(Evaluator):
    name = "recorded_state"

    def evaluate(self, case, output):
        if "committed" not in case.metadata:
            return self._skipped("No committed state recorded")
        return self._result(float(case.metadata["committed"]), "Recorded commit state")


@pytest.mark.parametrize("mode", ["sync", "async", "saved"])
def test_custom_evaluator_extension_across_execution_paths(mode):
    cases = [EvalCase("yes", metadata={"committed": True}),
             EvalCase("no", metadata={"committed": False}), EvalCase("unknown")]
    grader = declare_dependencies(RecordedState(), version="fixture/v1", dependencies={})
    configured = EvalSuite("extension contract").add_cases(cases).add_evaluators(grader)
    if mode == "sync":
        report = configured.run(lambda _: "ok", workers=3, verbose=False)
    elif mode == "async":
        async def target(_):
            return "ok"
        report = asyncio.run(configured.run_async(target, verbose=False))
    else:
        report = configured.run_on_cases([(c, "ok") for c in cases], verbose=False)
    assert [r.status for r in report.case_results] == [
        EvalStatus.PASSED, EvalStatus.FAILED_QUALITY, EvalStatus.SKIPPED,
    ]
