"""Tests for simulate → dataset export: results_to_cases + the
`simulate --export-cases` CLI flag. No LLM calls — SimulationResults are
constructed directly and the CLI's simulate/score machinery is patched.
"""
from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

from multivon_eval.case import EvalCase
from multivon_eval.dataset import load_jsonl
from multivon_eval.simulate import Persona, SimulationResult, results_to_cases


def _result(name="alice", goal="Get a refund.", transcript=None,
            stop_reason="goal_reached"):
    persona = Persona(
        name=name, profile="A customer.", goal=goal,
        success_criteria="Refund confirmed.", traits=["terse"],
    )
    if transcript is None:
        transcript = [
            {"role": "user", "content": "I want a refund"},
            {"role": "assistant", "content": "Refund initiated."},
        ]
    metadata = {
        "simulated": True, "persona": name, "persona_traits": ["terse"],
        "stop_reason": stop_reason,
        "_provenance": {"schema_version": 1, "case_uid": f"uid-{name}",
                        "authored_by": "simulator", "targets": []},
    }
    case = EvalCase(input=goal, conversation=transcript, metadata=metadata)
    return SimulationResult(
        persona=persona, transcript=transcript, turns=len(transcript) // 2,
        stop_reason=stop_reason, goal_achieved=True, cost_usd=0.0, case=case,
    )


class TestResultsToCases:
    def test_conversation_case_shape(self):
        cases, report = results_to_cases([_result()])
        assert report.kind == "simulate_export"
        (case,) = cases
        assert case.input == "Get a refund."
        assert case.conversation == [
            {"role": "user", "content": "I want a refund"},
            {"role": "assistant", "content": "Refund initiated."},
        ]
        assert case.metadata["simulated"] is True
        assert case.metadata["persona"] == "alice"
        assert case.metadata["persona_traits"] == ["terse"]
        assert case.metadata["stop_reason"] == "goal_reached"
        # success criteria becomes the expected behavior (gate + judges)
        assert case.metadata["expected_behavior"] == "Refund confirmed."
        # simulator provenance is kept, not restamped
        assert case.metadata["_provenance"]["authored_by"] == "simulator"

    def test_empty_transcripts_skipped_and_counted(self):
        results = [
            _result(name="ok"),
            _result(name="dead", transcript=[], stop_reason="driver_error"),
        ]
        cases, report = results_to_cases(results)
        assert len(cases) == 1
        assert report.requested == report.generated == 2
        assert report.dropped_malformed == 1
        assert report.accepted == 1

    def test_duplicate_goals_deduped(self):
        results = [_result(name="a"), _result(name="b")]  # identical goals
        cases, report = results_to_cases(results)
        assert len(cases) == 1
        assert report.dropped_duplicate == 1

    def test_empty_input_is_honest(self):
        cases, report = results_to_cases([])
        assert cases == [] and report.requested == 0


class TestCliExportCases:
    def _ns(self, tmp_path, export):
        return Namespace(
            model_cmd=str(tmp_path / "model.py"),
            personas=str(tmp_path / "personas.jsonl"), propose_from=None,
            n_personas=5, max_turns=4, budget=1.0,
            out=str(tmp_path / "results.jsonl"), seed=0,
            judge_model="claude-haiku-4-5-20251001",
            judge_provider="anthropic",
            export_cases=export,
        )

    def _setup_files(self, tmp_path):
        (tmp_path / "model.py").write_text(
            "def model_fn(prompt):\n    return 'ok'\n", encoding="utf-8")
        (tmp_path / "personas.jsonl").write_text(json.dumps({
            "name": "alice", "profile": "Customer.", "goal": "Get a refund.",
            "success_criteria": "Refund confirmed.", "traits": ["terse"],
        }) + "\n", encoding="utf-8")

    def _summary(self):
        return {
            "per_persona": {}, "total_cost_usd": 0.0,
            "goal_completion": {"achieved": 1, "judged": 1, "rate": 1.0},
        }

    def test_export_cases_writes_loadable_jsonl(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_simulate

        self._setup_files(tmp_path)
        export = tmp_path / "cases.jsonl"
        results = [
            _result(name="alice"),
            _result(name="empty", goal="Different goal entirely, truly.",
                    transcript=[], stop_reason="driver_error"),
        ]
        with patch("multivon_eval.simulate.simulate", return_value=results), \
             patch("multivon_eval.simulate.score_simulations",
                   return_value=self._summary()):
            rcode = cmd_simulate(self._ns(tmp_path, str(export)))
        assert rcode == 0
        out = capsys.readouterr().out
        assert "exported cases" in out

        # Round-trips through the standard loader, conversation intact.
        (case,) = load_jsonl(str(export))
        assert case.input == "Get a refund."
        assert case.conversation[0]["role"] == "user"
        assert case.metadata["simulated"] is True
        assert case.metadata["expected_behavior"] == "Refund confirmed."

    def test_no_flag_no_export(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_simulate

        self._setup_files(tmp_path)
        with patch("multivon_eval.simulate.simulate",
                   return_value=[_result()]), \
             patch("multivon_eval.simulate.score_simulations",
                   return_value=self._summary()):
            rcode = cmd_simulate(self._ns(tmp_path, None))
        assert rcode == 0
        assert "exported cases" not in capsys.readouterr().out
