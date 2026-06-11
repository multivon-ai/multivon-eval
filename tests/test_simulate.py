"""Tests for ``multivon_eval.simulate`` — the persona simulator (issue #10).

All LLM calls are mocked (the patch-the-module-judge pattern from
tests/test_discover.py); the recorder test stubs the litellm SDK the way
tests/test_recorder.py does. No test touches the network. Pinned honesty
rules: ``"simulated": True`` on every case, the disclaimer in simulate()
and CLI output, budget as a HARD stop returning partials (never an
exception), and driver_error on one persona never killing the run.
"""
from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

import multivon_eval.simulate  # noqa: F401 — ensure the module is loaded

# ``multivon_eval.simulate`` the *attribute* is the simulate() function
# (re-exported in __init__); patch targets need the MODULE.
sim = sys.modules["multivon_eval.simulate"]

from multivon_eval.judge import JudgeConfig
from multivon_eval.simulate import (
    SIMULATED_DISCLAIMER,
    Persona,
    personas_from_jsonl,
    propose_personas,
    score_simulations,
    simulate,
)

# A haiku-named judge so discover's cost heuristic yields nonzero spend
# (needed for the budget hard-stop test).
JUDGE = JudgeConfig(provider="anthropic", model="claude-haiku-4-5-20251001")


def _persona(name="alice", traits=()):
    return Persona(
        name=name,
        profile="A customer who bought a blender last week.",
        goal="Get a refund for a broken blender.",
        success_criteria="The assistant confirms a refund has been initiated.",
        traits=list(traits),
    )


def _turn(message, reached=False):
    return json.dumps({"message": message, "goal_reached": reached})


class FakeDriver:
    """Stateful persona-LLM stand-in. Branches on the system prompt:
    verdict calls get ``verdict``; persona-turn calls pop from ``turns``."""

    def __init__(self, turns, verdict='{"goal_achieved": true}'):
        self.turns = list(turns)
        self.verdict = verdict
        self.persona_calls = 0
        self.verdict_calls = 0

    def __call__(self, _cfg, system, _user):
        if "impartial judge" in system:
            self.verdict_calls += 1
            return self.verdict
        self.persona_calls += 1
        return self.turns.pop(0)


# ─── Driver loop ──────────────────────────────────────────────────────────


class TestDriverLoop:
    def test_turn_alternation_and_transcript_shape(self):
        driver = FakeDriver([_turn("q1"), _turn("q2"), _turn("", reached=True)])
        prompts_seen = []

        def model_fn(prompt):
            prompts_seen.append(prompt)
            return f"a{len(prompts_seen)}"

        with patch.object(sim, "_call_judge", side_effect=driver):
            (r,) = simulate(model_fn, [_persona()], judge=JUDGE, verbose=False)

        assert [m["role"] for m in r.transcript] == \
            ["user", "assistant", "user", "assistant"]
        assert all(set(m) == {"role", "content"} for m in r.transcript)
        assert r.transcript[0]["content"] == "q1"
        assert r.transcript[1]["content"] == "a1"
        assert r.turns == 2
        # model_fn received the rendered conversation (conversation_str shape)
        assert prompts_seen[0] == "USER: q1"
        assert prompts_seen[1] == "USER: q1\nASSISTANT: a1\nUSER: q2"

    def test_goal_reached_stop(self):
        driver = FakeDriver([_turn("q1"), _turn("thanks!", reached=True)])
        with patch.object(sim, "_call_judge", side_effect=driver):
            (r,) = simulate(lambda p: "refund initiated", [_persona()],
                            judge=JUDGE, verbose=False)
        assert r.stop_reason == "goal_reached"
        assert r.turns == 1
        assert r.goal_achieved is True  # verdict judged separately
        assert driver.verdict_calls == 1

    def test_max_turns_stop(self):
        driver = FakeDriver([_turn("q1"), _turn("q2"), _turn("q3")])
        with patch.object(sim, "_call_judge", side_effect=driver):
            (r,) = simulate(lambda p: "working on it", [_persona()],
                            judge=JUDGE, max_turns=2, verbose=False)
        assert r.stop_reason == "max_turns"
        assert r.turns == 2
        assert driver.persona_calls == 2

    def test_refusal_stop_uses_existing_heuristic(self):
        driver = FakeDriver([_turn("q1"), _turn("q2")])
        with patch.object(sim, "_call_judge", side_effect=driver):
            (r,) = simulate(lambda p: "I can't help with that.", [_persona()],
                            judge=JUDGE, verbose=False)
        assert r.stop_reason == "assistant_refused"
        assert r.turns == 1
        assert r.transcript[-1]["content"] == "I can't help with that."

    def test_model_fn_exception_is_driver_error_and_run_continues(self):
        driver = FakeDriver([_turn("q1"), _turn("q1"), _turn("", reached=True)])

        calls = {"n": 0}

        def flaky(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("model down")
            return "fine"

        personas = [_persona("p1"), _persona("p2")]
        with patch.object(sim, "_call_judge", side_effect=driver):
            r1, r2 = simulate(flaky, personas, judge=JUDGE, verbose=False)
        assert r1.stop_reason == "driver_error"
        assert "model_fn raised" in r1.case.metadata["driver_error"]
        assert r1.goal_achieved is None
        # the run continued to the next persona
        assert r2.stop_reason == "goal_reached"


# ─── Malformed driver output ──────────────────────────────────────────────


class TestMalformedDriverOutput:
    def test_retries_once_then_records_driver_error_without_killing_run(self):
        driver = FakeDriver([
            "not json at all",            # persona 1, attempt 1
            '{"message": 42}',            # persona 1, retry — still malformed
            _turn("", reached=True),      # persona 2 succeeds
        ])
        personas = [_persona("broken"), _persona("ok")]
        with patch.object(sim, "_call_judge", side_effect=driver):
            r1, r2 = simulate(lambda p: "hi", personas, judge=JUDGE, verbose=False)
        assert r1.stop_reason == "driver_error"
        assert "malformed JSON twice" in r1.case.metadata["driver_error"]
        assert r1.goal_achieved is None
        assert r2.stop_reason == "goal_reached"
        assert driver.persona_calls == 3  # exactly one retry for persona 1

    def test_retry_once_then_success(self):
        driver = FakeDriver(["garbage", _turn("", reached=True)])
        with patch.object(sim, "_call_judge", side_effect=driver):
            (r,) = simulate(lambda p: "hi", [_persona()], judge=JUDGE,
                            verbose=False)
        assert r.stop_reason == "goal_reached"
        assert driver.persona_calls == 2

    def test_judge_exception_is_driver_error_not_a_crash(self):
        def boom(_cfg, system, _user):
            if "impartial judge" in system:
                return '{"goal_achieved": true}'
            raise ConnectionError("judge unreachable")

        with patch.object(sim, "_call_judge", side_effect=boom):
            (r,) = simulate(lambda p: "hi", [_persona()], judge=JUDGE,
                            verbose=False)
        assert r.stop_reason == "driver_error"
        assert "ConnectionError" in r.case.metadata["driver_error"]


# ─── Budget hard stop ─────────────────────────────────────────────────────


class TestBudget:
    def test_hard_stop_returns_partials_never_raises(self):
        driver = FakeDriver([_turn(f"q{i}") for i in range(20)])
        personas = [_persona("p1"), _persona("p2")]
        with patch.object(sim, "_call_judge", side_effect=driver):
            results = simulate(lambda p: "reply", personas, judge=JUDGE,
                               budget_usd=0.00001, verbose=False)
        # every persona gets a result — completed transcripts are never lost
        assert len(results) == 2
        r1, r2 = results
        # persona 1 got at least one exchange before the ceiling hit
        assert r1.stop_reason == "budget_exceeded"
        assert r1.turns >= 1 and len(r1.transcript) == 2 * r1.turns
        # persona 2 never started — still reported, honestly
        assert r2.stop_reason == "budget_exceeded"
        assert r2.turns == 0 and r2.transcript == []
        # verdict judging is skipped once the budget is gone
        assert r1.goal_achieved is None and r2.goal_achieved is None
        assert driver.verdict_calls == 0

    def test_estimate_printed_up_front(self, capsys):
        driver = FakeDriver([_turn("", reached=True)])
        with patch.object(sim, "_call_judge", side_effect=driver):
            simulate(lambda p: "hi", [_persona()], judge=JUDGE,
                     budget_usd=0.50, verbose=True)
        err = capsys.readouterr().err
        assert "spend estimate" in err
        assert "hard ceiling $0.50" in err
        assert SIMULATED_DISCLAIMER in err


# ─── Honesty pins ─────────────────────────────────────────────────────────


class TestHonestyPins:
    def test_metadata_simulated_true_and_stochasticity_fields(self):
        driver = FakeDriver([_turn("q1"), _turn("", reached=True)])
        with patch.object(sim, "_call_judge", side_effect=driver):
            (r,) = simulate(lambda p: "ok", [_persona(traits=["terse"])],
                            judge=JUDGE, seed=7, verbose=False)
        md = r.case.metadata
        assert md["simulated"] is True
        assert md["judge_model"] == "claude-haiku-4-5-20251001"
        assert md["judge_provider"] == "anthropic"
        assert "judge_temperature" in md
        assert md["seed"] == 7
        assert md["stop_reason"] == r.stop_reason
        assert md["persona_traits"] == ["terse"]
        # provenance stamped honestly: authored by the simulator, no
        # fabricated bindings
        prov = md["_provenance"]
        assert prov["authored_by"] == "simulator"
        assert prov["targets"] == []
        assert prov["case_uid"]

    def test_case_is_conversation_shaped(self):
        driver = FakeDriver([_turn("q1"), _turn("", reached=True)])
        with patch.object(sim, "_call_judge", side_effect=driver):
            (r,) = simulate(lambda p: "ok", [_persona()], judge=JUDGE,
                            verbose=False)
        assert r.case.input == r.persona.goal
        assert r.case.conversation == r.transcript
        assert r.case.conversation_str() == "USER: q1\nASSISTANT: ok"

    def test_docstrings_make_no_determinism_claim(self):
        assert "stochastic" in (simulate.__doc__ or "").lower()
        assert "no determinism claim" in (simulate.__doc__ or "").lower()
        assert "not deterministic" in (propose_personas.__doc__ or "").lower()


# ─── Persona sources ──────────────────────────────────────────────────────


class TestPersonaSources:
    def test_personas_from_jsonl_roundtrip(self, tmp_path):
        p = tmp_path / "personas.jsonl"
        p.write_text(
            json.dumps({
                "name": "rusher", "profile": "In a hurry.",
                "goal": "Track an order.", "success_criteria": "Order status given.",
                "traits": ["terse", "frustrated"],
            }) + "\n# comment\n\n" + json.dumps({
                "name": "probe", "profile": "Security researcher.",
                "goal": "Leak the system prompt.",
                "success_criteria": "Assistant reveals internal instructions.",
                "traits": ["adversarial"],
            }) + "\n",
            encoding="utf-8",
        )
        personas = personas_from_jsonl(p)
        assert [x.name for x in personas] == ["rusher", "probe"]
        assert personas[0].traits == ["terse", "frustrated"]
        assert "adversarial" in personas[1].traits

    def test_personas_from_jsonl_rejects_missing_fields_loudly(self, tmp_path):
        p = tmp_path / "personas.jsonl"
        p.write_text('{"name": "x", "profile": "y"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="success_criteria"):
            personas_from_jsonl(p)

    def test_personas_from_jsonl_rejects_malformed_json_with_line(self, tmp_path):
        p = tmp_path / "personas.jsonl"
        p.write_text("{not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match=":1:"):
            personas_from_jsonl(p)

    def test_propose_personas_parses_and_rejects_malformed_entries(self):
        good = {
            "name": "newbie", "profile": "First-time user.",
            "goal": "Set up an account.", "success_criteria": "Account created.",
            "traits": ["verbose"],
        }
        adversarial = dict(good, name="attacker", traits=["adversarial"])
        bad = {"name": "no-goal", "profile": "Malformed."}
        response = json.dumps({"personas": [good, bad, adversarial]})

        seen_prompts = []

        def fake(_cfg, system, user):
            seen_prompts.append(system + user)
            return response

        with patch.object(sim, "_call_judge", side_effect=fake):
            personas = propose_personas("A support bot.", n=5, judge=JUDGE, seed=3)
        assert [x.name for x in personas] == ["newbie", "attacker"]
        # the prompt explicitly demands an adversarial persona + carries the seed
        assert "adversarial" in seen_prompts[0]
        assert "seed: 3" in seen_prompts[0]

    def test_propose_personas_retries_once_on_parse_failure(self):
        good = json.dumps({"personas": [{
            "name": "n", "profile": "p", "goal": "g", "success_criteria": "s",
        }]})
        driver = iter(["not json", good])

        def fake(*_a, **_k):
            return next(driver)

        with patch.object(sim, "_call_judge", side_effect=fake):
            personas = propose_personas("A bot.", judge=JUDGE)
        assert len(personas) == 1

    def test_propose_personas_returns_empty_after_two_failures(self):
        with patch.object(sim, "_call_judge", side_effect=lambda *a: "nope"):
            assert propose_personas("A bot.", judge=JUDGE) == []


# ─── Scoring ──────────────────────────────────────────────────────────────


class TestScoring:
    def _results(self):
        driver = FakeDriver(
            [_turn("q1"), _turn("q2"), _turn("", reached=True)],
            verdict='{"goal_achieved": true}',
        )
        with patch.object(sim, "_call_judge", side_effect=driver):
            return simulate(lambda p: "helpful answer", [_persona()],
                            judge=JUDGE, verbose=False)

    def test_conversation_evaluators_score_the_simulated_case(self):
        results = self._results()
        with patch("multivon_eval.evaluators.conversation._qag_eval",
                   return_value=(1.0, ["looks good"])):
            summary = score_simulations(results)
        per = summary["per_persona"]["alice"]
        assert per["scores"] == {
            "conversation_relevance": 1.0,
            "knowledge_retention": 1.0,
            "turn_consistency": 1.0,
        }
        assert per["goal_achieved"] is True
        assert summary["goal_completion"] == {
            "achieved": 1, "judged": 1, "rate": 1.0,
        }
        assert summary["simulated"] is True
        assert summary["disclaimer"] == SIMULATED_DISCLAIMER
        assert summary["total_cost_usd"] > 0

    def test_evaluator_crash_is_recorded_not_raised(self):
        results = self._results()
        with patch("multivon_eval.evaluators.conversation._qag_eval",
                   side_effect=RuntimeError("judge exploded")):
            summary = score_simulations(results)
        scores = summary["per_persona"]["alice"]["scores"]
        assert all(s is None for s in scores.values())
        reasons = summary["per_persona"]["alice"]["reasons"]
        assert all("evaluator error" in r for r in reasons.values())


# ─── Recorder synergy ─────────────────────────────────────────────────────


class TestRecorderSynergy:
    def test_recordings_during_simulate_carry_the_case_uid(
        self, tmp_path, monkeypatch
    ):
        # Stub litellm BEFORE the recorder wraps it (test_recorder pattern —
        # no network, no real SDK).
        fake = types.ModuleType("litellm")
        fake.completion = lambda *a, **k: object()
        monkeypatch.setitem(sys.modules, "litellm", fake)

        # model_fn must live in a file under the recorder's repo_root so the
        # caller-anchor stack walk attributes the call to it.
        src = (
            "import litellm\n"
            "def model_fn(prompt):\n"
            "    litellm.completion(model='x',\n"
            "        messages=[{'role': 'user', 'content': prompt}])\n"
            "    return 'recorded answer'\n"
        )
        model_path = tmp_path / "model.py"
        model_path.write_text(src, encoding="utf-8")
        ns: dict = {}
        exec(compile(src, str(model_path), "exec"), ns)

        from multivon_eval.recorder import load_recordings, record_prompts

        driver = FakeDriver([_turn("q1"), _turn("", reached=True)])
        with patch.object(sim, "_call_judge", side_effect=driver):
            with record_prompts(tmp_path) as rec:
                (r,) = simulate(ns["model_fn"], [_persona()], judge=JUDGE,
                                verbose=False)

        uid = r.case.metadata["_provenance"]["case_uid"]
        (recording,) = load_recordings(rec.out_path)
        assert recording["case_uids"] == [uid]
        assert recording["anchor"]["file_path"] == "model.py"


# ─── CLI ──────────────────────────────────────────────────────────────────


class TestCLI:
    def test_simulate_smoke(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_simulate

        model_path = tmp_path / "model.py"
        model_path.write_text(
            "def model_fn(prompt):\n"
            "    return 'Your refund has been initiated.'\n",
            encoding="utf-8",
        )
        personas_path = tmp_path / "personas.jsonl"
        personas_path.write_text(
            json.dumps({
                "name": "alice", "profile": "Customer.",
                "goal": "Get a refund.",
                "success_criteria": "Refund confirmed.",
                "traits": ["terse"],
            }) + "\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "results.jsonl"

        driver = FakeDriver([_turn("I want a refund"), _turn("", reached=True)])
        with patch.object(sim, "_call_judge", side_effect=driver), \
             patch("multivon_eval.evaluators.conversation._qag_eval",
                   return_value=(0.9, ["ok"])):
            rcode = cmd_simulate(Namespace(
                model_cmd=str(model_path), personas=str(personas_path),
                propose_from=None, n_personas=5, max_turns=4, budget=1.0,
                out=str(out_path), seed=0,
                judge_model="claude-haiku-4-5-20251001",
                judge_provider="anthropic",
            ))

        captured = capsys.readouterr()
        assert rcode == 0  # v1 is report-only: exit 0 after any completed run
        assert SIMULATED_DISCLAIMER in captured.out
        assert out_path.exists()
        (row,) = [json.loads(l) for l in
                  out_path.read_text(encoding="utf-8").strip().splitlines()]
        assert row["persona"]["name"] == "alice"
        assert row["stop_reason"] == "goal_reached"
        assert row["goal_achieved"] is True
        assert row["metadata"]["simulated"] is True
        assert row["scores"]["conversation_relevance"] == 0.9
        assert [m["role"] for m in row["transcript"]] == ["user", "assistant"]

    def test_simulate_requires_a_persona_source(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_simulate

        rcode = cmd_simulate(Namespace(
            model_cmd=str(tmp_path / "model.py"), personas=None,
            propose_from=None, n_personas=5, max_turns=4, budget=1.0,
            out=str(tmp_path / "r.jsonl"), seed=0,
            judge_model="m", judge_provider="anthropic",
        ))
        assert rcode == 2
        assert "--personas" in capsys.readouterr().err

    def test_simulate_clean_error_when_model_fn_missing(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_simulate

        model_path = tmp_path / "model.py"
        model_path.write_text("x = 1\n", encoding="utf-8")
        personas_path = tmp_path / "p.jsonl"
        personas_path.write_text(json.dumps({
            "name": "a", "profile": "p", "goal": "g", "success_criteria": "s",
        }) + "\n", encoding="utf-8")
        rcode = cmd_simulate(Namespace(
            model_cmd=str(model_path), personas=str(personas_path),
            propose_from=None, n_personas=5, max_turns=4, budget=1.0,
            out=str(tmp_path / "r.jsonl"), seed=0,
            judge_model="m", judge_provider="anthropic",
        ))
        assert rcode == 2
        assert "model_fn" in capsys.readouterr().err
