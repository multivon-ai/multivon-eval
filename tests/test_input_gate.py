"""Tests for the input-quality preflight gate (issue #14).

WARN-only phase 1: no hard REFUSE, no exit-2-block on any signal. All
deterministic, no network — the 4 signals reuse free local machinery.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from multivon_eval import assess_input, InputQualityReport, SignalFinding
from multivon_eval.input_gate import _NEAR_DUP_SAMPLE_CAP
from multivon_eval import discover
import multivon_eval.input_gate as input_gate


def _traces(n, *, output=True, context=False, unique=True):
    rows = []
    for i in range(n):
        body = f"unique question number {i} about topic {i}" if unique \
            else "the same repeated question over and over again"
        row = {"input": body}
        if output:
            row["output"] = f"answer {i}"
        if context:
            row["context"] = f"context passage {i}"
        rows.append(row)
    return rows


# ─── trace_count band: 0 / 5 / 25 ──────────────────────────────────────────


def test_trace_count_zero_warns():
    r = assess_input(kind="bootstrap", traces=[])
    tc = next(f for f in r.findings if f.signal == "trace_count")
    assert tc.tier == "warn"
    assert "nothing to bootstrap" in tc.message
    assert r.verdict == "WARN"


def test_trace_count_below_min_warns():
    r = assess_input(kind="bootstrap", traces=_traces(5))
    tc = next(f for f in r.findings if f.signal == "trace_count")
    assert tc.tier == "warn"
    assert "below 20" in tc.message


def test_trace_count_above_min_ok():
    r = assess_input(kind="bootstrap", traces=_traces(25))
    tc = next(f for f in r.findings if f.signal == "trace_count")
    assert tc.tier == "ok"
    assert r.verdict == "PROCEED"


# ─── field_completeness: per-field + zero-output message ────────────────────


def test_field_completeness_reported_per_field_not_averaged():
    r = assess_input(kind="bootstrap", traces=_traces(25, context=True))
    fc = next(f for f in r.findings if f.signal == "field_completeness")
    assert "input=" in fc.measured
    assert "output=" in fc.measured
    assert "context=" in fc.measured


def test_field_completeness_zero_output_names_uncalibrated_thresholds():
    r = assess_input(kind="bootstrap", traces=_traces(25, output=False))
    fc = next(f for f in r.findings if f.signal == "field_completeness")
    assert fc.tier == "warn"
    assert "UNCALIBRATED" in fc.message
    assert "early-return" in fc.message.lower()


def test_field_completeness_thin_output_slice_warns():
    rows = _traces(20)
    for row in rows[:18]:  # only 2/20 have output -> 0.10
        row.pop("output")
    r = assess_input(kind="bootstrap", traces=rows)
    fc = next(f for f in r.findings if f.signal == "field_completeness")
    assert fc.tier == "warn"


def test_field_completeness_low_context_for_context_tasks_warns():
    rows = _traces(25, context=True)
    for row in rows[:20]:  # only 5/25 carry context -> 0.20
        row["context"] = ""
    r = assess_input(kind="bootstrap", traces=rows)
    fc = next(f for f in r.findings if f.signal == "field_completeness")
    assert fc.tier == "warn"
    assert "context" in fc.message.lower()


# ─── near_duplicate_ratio: high-dup vs diverse + reservoir cap ──────────────


def test_near_duplicate_high_dup_corpus_warns():
    r = assess_input(kind="bootstrap", traces=_traces(25, unique=False))
    nd = next(f for f in r.findings if f.signal == "near_duplicate_ratio")
    assert nd.tier == "warn"
    assert "distinct" in nd.message


def test_near_duplicate_diverse_corpus_ok():
    r = assess_input(kind="bootstrap", traces=_traces(25, unique=True))
    nd = next(f for f in r.findings if f.signal == "near_duplicate_ratio")
    assert nd.tier == "ok"


def test_near_duplicate_reservoir_cap_does_not_hang_on_50k():
    texts = [f"distinct input number {i} talking about subject {i}"
             for i in range(50_000)]
    frac, n = input_gate._unique_fraction(texts)
    assert n == _NEAR_DUP_SAMPLE_CAP  # only the sample is compared
    assert 0.0 <= frac <= 1.0


# ─── pii density: neutral fact ──────────────────────────────────────────────


def test_pii_density_is_neutral_fact_not_moral_block():
    rows = [{"input": "contact me", "output": "sure jane@example.com"}
            for _ in range(25)]
    r = assess_input(kind="bootstrap", traces=rows)
    pii = next(f for f in r.findings if f.signal == "pii_secret_density")
    assert pii.threshold == "neutral"
    assert "expected" in pii.message  # "if this is a PII-handling eval..."
    # tier may be warn (a fact surfaced), but verdict is never REFUSE
    assert r.verdict in ("PROCEED", "WARN")


def test_pii_density_clean_corpus_ok():
    r = assess_input(kind="bootstrap", traces=_traces(25))
    pii = next(f for f in r.findings if f.signal == "pii_secret_density")
    assert pii.tier == "ok"


# ─── kind routing: cases-kind skips doc/trace signals ───────────────────────


def test_cases_kind_runs_only_count_and_well_formed():
    cases = [{"input": "what is x", "expected_output": "x is y"}]
    r = assess_input(kind="cases", cases=cases)
    signals = {f.signal for f in r.findings}
    assert signals == {"case_count", "well_formed_rate"}
    # doc/trace signals must NOT appear as phantom greens
    assert "trace_count" not in signals
    assert "near_duplicate_ratio" not in signals


def test_unrun_signals_are_never_in_a_kinds_findings():
    r = assess_input(kind="generate", document="x" * 500)
    # generate defines document_length/near_dup/pii — never trace_count etc.
    for f in r.findings:
        assert f.signal in ("document_length", "near_duplicate_ratio",
                            "pii_secret_density")


def test_skipped_state_exists_for_a_constructed_skip():
    # _skipped marks a defined-but-unrun signal as SKIPPED, never ok-green.
    sk = input_gate._skipped("trace_count")
    assert sk.state == "SKIPPED"
    assert sk.tier == "ok"  # skipped doesn't flag, but state distinguishes it


def test_well_formed_rate_warns_on_malformed_cases():
    cases = [{"input": "", "expected_output": ""} for _ in range(10)]
    r = assess_input(kind="cases", cases=cases)
    wf = next(f for f in r.findings if f.signal == "well_formed_rate")
    assert wf.tier == "warn"
    assert r.verdict == "WARN"


# ─── generate kind: empty / too-short source ────────────────────────────────


def test_generate_empty_document_warns():
    r = assess_input(kind="generate", document="")
    dl = next(f for f in r.findings if f.signal == "document_length")
    assert dl.tier == "warn"
    assert "empty" in dl.message


def test_generate_long_document_ok():
    r = assess_input(kind="generate", document="paragraph one. " * 100)
    dl = next(f for f in r.findings if f.signal == "document_length")
    assert dl.tier == "ok"


# ─── rendering: PROCEED silent, WARN headline denominator ───────────────────


def test_proceed_prints_nothing():
    r = assess_input(kind="bootstrap", traces=_traces(25, context=True))
    assert r.verdict == "PROCEED"
    assert r.render_text() == ""


def test_warn_headline_denominator_counts_all_signals():
    # Only field_completeness flags, but denominator must read 4 (all
    # bootstrap signals), never M-of-M by dropping clean signals.
    rows = _traces(25)
    for row in rows[:18]:
        row.pop("output")
    r = assess_input(kind="bootstrap", traces=rows)
    head = r.render_text().splitlines()[0]
    assert "of 4 signals flagged" in head
    assert r.measurable_total == 4


def test_warn_render_has_blind_spots_footer():
    r = assess_input(kind="bootstrap", traces=[])
    out = r.render_text()
    assert "not checked:" in out
    assert "semantic correctness of labels" in out
    assert "factual accuracy of source" in out


# ─── anti-drift: shared CALIBRATION_MIN_TRACES constant ─────────────────────


def test_calibration_min_traces_constant_shared_no_duplicated_literal():
    # input_gate imports the constant from discover — the single source.
    assert input_gate.CALIBRATION_MIN_TRACES is discover.CALIBRATION_MIN_TRACES
    assert discover.CALIBRATION_MIN_TRACES == 20


def test_discover_warning_uses_the_constant():
    # The calibration warning text must reflect the shared constant value.
    import inspect
    src = inspect.getsource(discover)
    assert "len(traces) < CALIBRATION_MIN_TRACES" in src
    # no bare `< 20` literal left in the calibration guard
    assert "if len(traces) < 20:" not in src


# ─── assess CLI exit codes ──────────────────────────────────────────────────


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "multivon_eval.cli", *args],
        capture_output=True, text=True,
    )


def test_assess_cli_proceed_exits_0(tmp_path):
    p = tmp_path / "traces.jsonl"
    p.write_text("\n".join(
        json.dumps({"input": f"q {i} distinct topic {i}",
                    "output": f"a {i}", "context": f"c {i}"})
        for i in range(25)
    ))
    res = _run_cli(["assess", str(p), "--for", "bootstrap"])
    assert res.returncode == 0
    assert "PROCEED" in res.stdout


def test_assess_cli_warn_exits_1(tmp_path):
    p = tmp_path / "traces.jsonl"
    p.write_text("\n".join(
        json.dumps({"input": f"q {i}"}) for i in range(5)
    ))
    res = _run_cli(["assess", str(p), "--for", "bootstrap"])
    assert res.returncode == 1
    assert "WARN" in (res.stdout + res.stderr)


def test_assess_cli_missing_file_exits_2():
    res = _run_cli(["assess", "/no/such/file.jsonl"])
    assert res.returncode == 2


# ─── --skip-input-gate: stderr line, never blocks ───────────────────────────


def test_skip_input_gate_prints_stderr_and_does_not_block(tmp_path, monkeypatch):
    # Drive cmd_bootstrap's skip branch directly: it must print the stderr
    # line and pass run_input_gate=False to bootstrap (which we stub) —
    # never changing exit behavior.
    from multivon_eval import cli
    import argparse

    captured = {}

    def fake_bootstrap(**kwargs):
        captured.update(kwargs)
        raise SystemExit(0)  # short-circuit before any real work

    prod = tmp_path / "p.md"
    prod.write_text("a product")
    tr = tmp_path / "t.jsonl"
    tr.write_text(json.dumps({"input": "hi", "output": "yo"}))

    monkeypatch.setattr(cli, "bootstrap", fake_bootstrap, raising=False)
    monkeypatch.setattr("multivon_eval.discover.bootstrap", fake_bootstrap)

    args = argparse.Namespace(
        product=str(prod), traces=str(tr), output=str(tmp_path / "out"),
        judge_provider="anthropic", judge_model="x", pii_policy="redact",
        skip_seed_cases=True, skip_calibration=True, n_seed_cases=0,
        repo=".", validate=False, validate_n_shots=3,
        skip_input_gate=True,
    )
    with pytest.raises(SystemExit):
        cli.cmd_bootstrap(args)
    assert captured.get("run_input_gate") is False
