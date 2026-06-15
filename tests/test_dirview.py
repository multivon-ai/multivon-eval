"""Tests for directory-mode `multivon-eval view` (multivon_eval.dirview).

Rendering functions are pure and tested without starting the server.
Fixtures are produced with the REAL EvalSuite/EvalCase API (a tiny suite
run with a stub model + deterministic evaluator, save_json, reload) so
the dicts under test are genuine to_json() shapes, not hand-built.

One end-to-end smoke test starts the actual server (--no-browser, on an
ephemeral port), curls the three routes, and shuts down cleanly.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from multivon_eval import EvalCase, EvalSuite
from multivon_eval.evaluators.deterministic import ExactMatch
from multivon_eval import dirview


# ── Real fixtures via the suite API ─────────────────────────────────────────

def _make_report(model_fn, *, name="demo", model_id="stub-model"):
    suite = EvalSuite(name, model_id)
    suite.add_case(EvalCase(input="2+2?", expected_output="4"))
    suite.add_case(EvalCase(input="capital of France?", expected_output="Paris"))
    suite.add_case(EvalCase(input="color of the sky?", expected_output="blue"))
    suite.add_evaluator(ExactMatch())
    return suite.run(model_fn, verbose=False)


def _good_model(x: str) -> str:
    if "2+2" in x:
        return "4"
    if "France" in x:
        return "Paris"
    return "blue"


def _bad_model(x: str) -> str:
    # Regresses two of the three answers vs _good_model.
    if "2+2" in x:
        return "4"          # still passes
    if "France" in x:
        return "Berlin"     # regressed
    return "green"          # regressed


@pytest.fixture
def report_dir(tmp_path):
    """A temp dir with two valid reports + several junk JSON files."""
    a = _make_report(_good_model, name="suite-a")
    b = _make_report(_bad_model, name="suite-b")
    a.save_json(str(tmp_path / "run_a.json"))
    b.save_json(str(tmp_path / "run_b.json"))
    # Junk files that must be skipped.
    (tmp_path / "SECURITY_ANALYSIS.json").write_text(json.dumps({
        "vulnerabilities": [], "riskScore": 3, "recommendations": [],
        "summary": "no critical issues",  # str summary, not dict
    }))
    (tmp_path / "empty.json").write_text("{}")
    (tmp_path / "notjson.json").write_text("{ this is not valid json ")
    return tmp_path


def _load(path: Path):
    return dirview.load_report(path)


# ── Validator ────────────────────────────────────────────────────────────────

def test_validator_accepts_real_report(tmp_path):
    r = _make_report(_good_model)
    r.save_json(str(tmp_path / "r.json"))
    data = json.loads((tmp_path / "r.json").read_text())
    assert dirview.is_eval_report(data) is True


def test_validator_rejects_security_junk():
    junk = {
        "vulnerabilities": [{"id": "V1", "severity": "high"}],
        "riskScore": 7,
        "riskAssessment": "moderate",
        "recommendations": ["patch X"],
        "summary": "some prose summary",  # present but not a dict-with-pass_rate
    }
    assert dirview.is_eval_report(junk) is False


def test_validator_rejects_empty_and_nondict():
    assert dirview.is_eval_report({}) is False
    assert dirview.is_eval_report([]) is False
    assert dirview.is_eval_report("nope") is False
    # summary dict but no pass_rate / no cases
    assert dirview.is_eval_report({"summary": {"foo": 1}}) is False
    # summary+pass_rate but empty cases
    assert dirview.is_eval_report({"summary": {"pass_rate": 1.0}, "cases": []}) is False
    # cases present but not case-shaped
    assert dirview.is_eval_report(
        {"summary": {"pass_rate": 1.0}, "cases": [{"nope": 1}]}
    ) is False


# ── Discovery ────────────────────────────────────────────────────────────────

def test_discover_splits_valid_and_skipped(report_dir):
    valid, skipped = dirview.discover(report_dir, recursive=False)
    assert len(valid) == 2
    assert len(skipped) == 3  # SECURITY, empty, notjson
    stems = {e.stem for e in valid}
    assert stems == {"run_a", "run_b"}


def test_discover_recursive_vs_nonrecursive(report_dir):
    sub = report_dir / "nested"
    sub.mkdir()
    _make_report(_good_model, name="nested-c").save_json(str(sub / "run_c.json"))

    valid_flat, _ = dirview.discover(report_dir, recursive=False)
    assert len(valid_flat) == 2  # nested not seen

    valid_rec, _ = dirview.discover(report_dir, recursive=True)
    assert len(valid_rec) == 3
    nested = [e for e in valid_rec if e.stem == "run_c"][0]
    assert nested.parent_prefix == "nested"


# ── INDEX rendering ──────────────────────────────────────────────────────────

def test_index_lists_valid_and_collapses_skipped(report_dir):
    valid, skipped = dirview.discover(report_dir, recursive=False)
    html = dirview.render_index(valid, skipped, base_dir=report_dir)
    assert "run_a" in html
    assert "run_b" in html
    # Skipped collapse to a single muted footer count with an [expand].
    assert "3 file(s) skipped (not eval reports)" in html
    assert "[expand]" in html
    # Each row links to OPEN and offers a per-row diff dropdown.
    assert 'href="/r/0"' in html
    assert "diff vs" in html
    assert "/diff?a=" in html


def test_index_sort_param_reorders(report_dir):
    valid, skipped = dirview.discover(report_dir, recursive=False)
    asc = dirview.render_index(valid, skipped, sort="run", direction="asc")
    desc = dirview.render_index(valid, skipped, sort="run", direction="desc")
    # run_a should appear before run_b ascending, and after descending.
    assert asc.index("run_a") < asc.index("run_b")
    assert desc.index("run_b") < desc.index("run_a")


def test_index_flags_high_error_rate_red():
    # Build a report whose cases mostly error → error_rate >= 0.10.
    def erroring_model(x):
        raise RuntimeError("model down")
    r = _make_report(erroring_model)
    import tempfile, os
    d = tempfile.mkdtemp()
    r.save_json(os.path.join(d, "err.json"))
    valid, skipped = dirview.discover(Path(d), recursive=False)
    assert len(valid) == 1
    assert valid[0].error_rate >= 0.10
    html = dirview.render_index(valid, skipped)
    assert 'class="badge errflag"' in html  # red flag applied to the row


def test_index_no_error_flag_when_clean(report_dir):
    valid, _ = dirview.discover(report_dir, recursive=False)
    # Both fixtures have zero error cases.
    assert all(e.error_rate == 0.0 for e in valid)
    html = dirview.render_index(valid, [])
    assert 'class="badge errflag"' not in html  # no row carries the red flag


# ── OPEN rendering ───────────────────────────────────────────────────────────

def test_open_serves_to_html_verbatim_with_breadcrumb(report_dir):
    valid, _ = dirview.discover(report_dir, recursive=False)
    entry = valid[0]
    report = _load(entry.path)
    html = dirview.render_open(report, entry)
    # Breadcrumb back to the index.
    assert 'href="/"' in html
    assert "all reports" in html
    # Verbatim report body markers from to_html() are present.
    assert report.to_html()[:40] in html or "multivon-eval" in html


# ── DIFF rendering ───────────────────────────────────────────────────────────

def test_diff_buckets_match_reportdiff(report_dir):
    valid, _ = dirview.discover(report_dir, recursive=False)
    a = _load([e for e in valid if e.stem == "run_a"][0].path)
    b = _load([e for e in valid if e.stem == "run_b"][0].path)
    diff = a.compare(b)
    # Sanity: good→bad regressed two cases, none fixed.
    assert len(diff.regressions) == 2
    assert len(diff.improvements) == 0

    html = dirview.render_diff(a, b, name_a="run_a", name_b="run_b")
    assert f"Regressed ({len(diff.regressions)})" in html
    assert "Fixed (0)" in html
    assert "Still failing" in html
    assert "Unchanged" in html
    # Pass-rate delta is negative (good → bad).
    assert "delta down" in html


def test_diff_pulls_both_judge_reasons_for_regressed(report_dir):
    valid, _ = dirview.discover(report_dir, recursive=False)
    a = _load([e for e in valid if e.stem == "run_a"][0].path)
    b = _load([e for e in valid if e.stem == "run_b"][0].path)
    html = dirview.render_diff(a, b, name_a="run_a", name_b="run_b")
    # Both runs' judge reasons are stacked for a regressed case (France→Berlin).
    # ExactMatch lowercases; the proposal reason carries expected + wrong answer.
    assert "paris" in html   # expected value, from the proposal's failing reason
    assert "berlin" in html  # the wrong answer the proposal returned
    # The baseline passed this case, so its reason block reads "Exact match".
    assert "Exact match" in html
    # Both run names appear as stacked reason-block headers in the diff.
    assert html.count("run_a") >= 1 and html.count("run_b") >= 1


def test_diff_still_failing_bucket():
    # A case that fails on BOTH sides with the same wrong answer → still failing.
    def model_x(inp):
        if "2+2" in inp:
            return "4"
        return "wrong"
    import tempfile, os
    d = tempfile.mkdtemp()
    _make_report(model_x, name="x1").save_json(os.path.join(d, "x1.json"))
    _make_report(model_x, name="x2").save_json(os.path.join(d, "x2.json"))
    valid, _ = dirview.discover(Path(d), recursive=False)
    a = _load(valid[0].path)
    b = _load(valid[1].path)
    html = dirview.render_diff(a, b)
    # Two cases fail on both sides → "Still failing (2)".
    assert "Still failing (2)" in html


# ── End-to-end smoke: real server, three routes ─────────────────────────────

def test_server_smoke_routes(report_dir):
    code = (
        "import sys; from pathlib import Path;"
        "from multivon_eval.dirview_server import serve_directory;"
        f"serve_directory(Path({str(report_dir)!r}), recursive=False, port=0, no_browser=True)"
    )
    # We need the bound port — run a tiny harness that prints it. The CLI
    # prints the URL line; capture stdout to learn the ephemeral port.
    # -u: unbuffered stdout so the URL line reaches our readline() promptly
    # (block-buffering on a pipe would otherwise withhold it past startup).
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        url = None
        deadline = time.time() + 15
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            if "http://127.0.0.1:" in line:
                url = line.split("→")[-1].strip()
                break
        assert url, "server did not report a URL"

        def get(path):
            with urllib.request.urlopen(url.rstrip("/") + path, timeout=5) as r:
                return r.status, r.read().decode("utf-8")

        st, body = get("/")
        assert st == 200 and "run_a" in body and "skipped" in body

        st, body = get("/r/0")
        assert st == 200 and "all reports" in body

        st, body = get("/diff?a=0&b=1")
        assert st == 200 and "Regressed" in body
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
