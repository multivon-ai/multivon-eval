"""The `python -m multivon_eval` demo is the "30 seconds, no API key" promise.
It must NEVER end in a traceback, regardless of what judge (if any) is detected.

Regression tests for the crash where a detected-but-unreachable judge (e.g. Ollama
listening on :11434 but the model isn't pulled) propagated a JudgeUnavailable out
of suite.run().prepare() and exited 1 with a stack trace.
"""
import multivon_eval.__main__ as demo
import multivon_eval.judge as judge_mod
import multivon_eval.evaluators.llm_judge as llm_judge_mod
from multivon_eval.exceptions import JudgeUnavailable


def test_demo_runs_deterministic_when_no_judge(monkeypatch, capsys):
    monkeypatch.setattr(demo, "_detect_judge", lambda: ("", "", ""))
    demo._run_demo()  # must not raise
    out = capsys.readouterr().out
    assert "not detected" in out.lower()


def test_demo_degrades_when_judge_detected_but_unreachable(monkeypatch, capsys):
    # A judge is detected (port open / key set) but every call fails — the
    # original crash scenario (Ollama up, model not pulled).
    monkeypatch.setattr(
        demo, "_detect_judge", lambda: ("openai", "llama3", "http://localhost:11434/v1")
    )

    def unreachable(prompt, config):
        raise JudgeUnavailable("simulated unreachable judge", provider="openai", model="llama3")

    monkeypatch.setattr(judge_mod, "make_judge_call", unreachable)

    demo._run_demo()  # must not raise — the liveness probe catches it
    out = capsys.readouterr().out
    assert "deterministic" in out.lower()


def test_demo_degrades_when_judge_fails_midrun(monkeypatch, capsys):
    # The judge passes the one-word liveness probe but then fails during the
    # actual run (prepare/QAG generation). The safety net around suite.run()
    # must catch it and finish on the deterministic tier.
    monkeypatch.setattr(
        demo, "_detect_judge", lambda: ("openai", "llama3", "http://localhost:11434/v1")
    )

    state = {"calls": 0}

    def flaky(prompt, config):
        state["calls"] += 1
        if state["calls"] == 1:
            return "ok"  # probe passes
        raise JudgeUnavailable("simulated mid-run outage", provider="openai", model="llama3")

    monkeypatch.setattr(judge_mod, "make_judge_call", flaky)
    monkeypatch.setattr(llm_judge_mod, "make_judge_call", flaky)

    demo._run_demo()  # must not raise
    out = capsys.readouterr().out
    assert "deterministic" in out.lower()
    assert state["calls"] >= 2  # probe + at least one failed in-run call
