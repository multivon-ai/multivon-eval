"""Tests for multivon_eval.recorder — the runtime prompt recorder (v1).

Each test pins one of the issue-#9 design constraints: patch-and-restore
leaves the SDKs byte-identical (critical), zero overhead when off,
fingerprint parity with the static scanner, case binding by observation,
append-safe idempotent storage, the source:"runtime" baseline tier that
never touches static records, the k-of-N OBSERVED report language, and
propose-only binding stamps. No test touches the network — SDK methods
are stubbed before the recorder wraps them.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import types
from argparse import Namespace
from pathlib import Path

import pytest

import anthropic.resources.messages as _anthropic_messages
import openai.resources.chat.completions as _openai_completions

from multivon_eval.attribution.fingerprint import (
    fingerprint_text, loose_fingerprint_text,
)
from multivon_eval.recorder import (
    DEFAULT_RECORDINGS_NAME,
    PromptRecorder,
    apply_bindings,
    bind_case,
    compare_observed,
    load_recordings,
    merge_recording_dicts,
    merge_recordings_into_baseline,
    propose_bindings,
    record_prompts,
    recording_active,
    runtime_records_from_recordings,
    set_active_case,
    reset_active_case,
    unbind_case,
    write_recordings,
)
from multivon_eval.staleness import (
    DEFAULT_BASELINE_NAME,
    build_staleness_report,
    load_baseline,
    render_json,
    render_markdown,
    render_text,
    write_baseline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_Messages = _anthropic_messages.Messages
_Completions = _openai_completions.Completions

_SENTINEL = object()


def _stub(*_a, **_k):
    """Network-free stand-in for an SDK create method."""
    return _SENTINEL


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _run_from_file(tmp_path: Path, rel: str, src: str, ns: dict) -> None:
    """Exec ``src`` with a filename under tmp_path so the recorder's
    stack walk anchors the call to a repo-relative file."""
    p = _write(tmp_path, rel, src)
    code = compile(p.read_text(encoding="utf-8"), str(p), "exec")
    exec(code, ns)


@pytest.fixture
def fake_litellm(monkeypatch):
    """A stub litellm module (the real one is optional and not installed)."""
    mod = types.ModuleType("litellm")

    def completion(*_a, **_k):
        return _SENTINEL

    async def acompletion(*_a, **_k):
        return _SENTINEL

    mod.completion = completion
    mod.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return mod


@pytest.fixture
def stubbed_sdks(monkeypatch, fake_litellm):
    """Replace the real SDK create methods with network-free stubs BEFORE
    the recorder wraps them; monkeypatch restores them after the test."""
    monkeypatch.setattr(_Messages, "create", _stub)
    monkeypatch.setattr(_Completions, "create", _stub)
    return fake_litellm


# ── 1. patch-and-restore (critical) ────────────────────────────────────


class TestPatchAndRestore:
    def test_all_sdk_surfaces_restored_byte_identical_after_exit(
        self, tmp_path, stubbed_sdks
    ):
        ll = stubbed_sdks
        originals = {
            "anthropic": _Messages.create,
            "openai": _Completions.create,
            "litellm.completion": ll.completion,
            "litellm.acompletion": ll.acompletion,
        }
        with record_prompts(tmp_path) as rec:
            assert recording_active()
            # wrapped: different objects, marked
            assert _Messages.create is not originals["anthropic"]
            assert getattr(_Messages.create, "__multivon_recorder__", False)
            assert getattr(ll.completion, "__multivon_recorder__", False)
            assert getattr(ll.acompletion, "__multivon_recorder__", False)
            assert "anthropic.messages.create" in rec.patched_sdks
            assert "litellm.completion" in rec.patched_sdks
        # restored: the exact same objects, no markers left behind
        assert _Messages.create is originals["anthropic"]
        assert _Completions.create is originals["openai"]
        assert ll.completion is originals["litellm.completion"]
        assert ll.acompletion is originals["litellm.acompletion"]
        assert not recording_active()

    def test_exception_inside_context_still_restores(self, tmp_path, stubbed_sdks):
        orig = _Messages.create
        with pytest.raises(RuntimeError, match="boom"):
            with record_prompts(tmp_path):
                raise RuntimeError("boom")
        assert _Messages.create is orig
        assert not recording_active()

    def test_recorders_do_not_nest(self, tmp_path, stubbed_sdks):
        with record_prompts(tmp_path):
            with pytest.raises(RuntimeError, match="already active"):
                PromptRecorder(tmp_path).start()
        assert not recording_active()


# ── 2. zero overhead when off ──────────────────────────────────────────


class TestZeroOverheadOff:
    def test_importing_multivon_eval_performs_no_patching(self):
        # Fresh interpreter: import must not wrap anything or pull litellm in.
        code = (
            "import multivon_eval\n"
            "import sys\n"
            "import anthropic.resources.messages as m\n"
            "import openai.resources.chat.completions as o\n"
            "assert not hasattr(m.Messages.create, '__multivon_recorder__')\n"
            "assert not hasattr(o.Completions.create, '__multivon_recorder__')\n"
            "assert 'litellm' not in sys.modules\n"
            "from multivon_eval.recorder import recording_active\n"
            "assert recording_active() is False\n"
            "print('clean')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        assert "clean" in proc.stdout

    def test_set_active_case_is_cheap_and_safe_when_off(self):
        token = set_active_case("uid-off")
        reset_active_case(token)
        assert bind_case({"_provenance": {"case_uid": "x"}}) is None  # off → no-op
        unbind_case(None)  # tolerated


# ── 3. interception produces correct fingerprints (no network) ─────────


class TestCapture:
    def test_anthropic_system_and_messages_recorded(self, tmp_path, stubbed_sdks):
        src = """
            import anthropic.resources.messages as m
            m.Messages.create(
                None, model="x",
                system="You are an invoice parser. Return JSON only.",
                messages=[{"role": "user", "content": "parse invoice 42"}],
            )
        """
        with record_prompts(tmp_path) as rec:
            _run_from_file(tmp_path, "app.py", src, {})
        assert rec.record_errors == 0
        recs = load_recordings(rec.out_path)
        assert len(recs) == 2
        by_role = {r["anchor"]["role"]: r for r in recs}
        sys_rec = by_role["system"]
        text = "You are an invoice parser. Return JSON only."
        assert sys_rec["fingerprint"] == fingerprint_text(text)
        assert sys_rec["loose_fingerprint"] == loose_fingerprint_text(text)
        assert sys_rec["anchor"]["file_path"] == "app.py"
        assert sys_rec["anchor"]["qualname"] == "<module>"
        assert sys_rec["anchor"]["sdk"] == "anthropic"
        assert sys_rec["anchor"]["call_site"] == "messages.create"
        assert sys_rec["count"] == 1
        assert sys_rec["recorder_version"] == 1
        usr = by_role["user"]
        assert usr["fingerprint"] == fingerprint_text("parse invoice 42")
        # fingerprints only by default — no rendered text in the artifact
        assert "text" not in sys_rec and "text" not in usr

    def test_openai_and_litellm_sync_and_async_recorded(
        self, tmp_path, stubbed_sdks
    ):
        src = """
            import asyncio
            import litellm
            import openai.resources.chat.completions as o

            def call_openai():
                o.Completions.create(
                    None, model="x",
                    messages=[{"role": "system", "content": "openai sys prompt"}],
                )

            async def call_litellm():
                litellm.completion(
                    model="x", messages=[{"role": "user", "content": "sync q"}],
                )
                await litellm.acompletion(
                    model="x", messages=[{"role": "user", "content": "async q"}],
                )

            call_openai()
            asyncio.run(call_litellm())
        """
        with record_prompts(tmp_path) as rec:
            _run_from_file(tmp_path, "svc/llm.py", src, {})
        recs = load_recordings(rec.out_path)
        by_fp = {r["fingerprint"]: r for r in recs}
        oai = by_fp[fingerprint_text("openai sys prompt")]
        assert oai["anchor"]["sdk"] == "openai"
        assert oai["anchor"]["call_site"] == "chat.completions.create"
        assert oai["anchor"]["qualname"].endswith("call_openai")
        assert oai["anchor"]["file_path"] == "svc/llm.py"
        sync_rec = by_fp[fingerprint_text("sync q")]
        assert (sync_rec["anchor"]["sdk"], sync_rec["anchor"]["call_site"]) == \
            ("litellm", "completion")
        async_rec = by_fp[fingerprint_text("async q")]
        assert (async_rec["anchor"]["sdk"], async_rec["anchor"]["call_site"]) == \
            ("litellm", "acompletion")
        assert async_rec["anchor"]["qualname"].endswith("call_litellm")

    def test_record_text_flag_stores_rendered_text(self, tmp_path, stubbed_sdks):
        src = """
            import litellm
            litellm.completion(model="x",
                               messages=[{"role": "user", "content": "keep me"}])
        """
        with record_prompts(tmp_path, record_text=True) as rec:
            _run_from_file(tmp_path, "app.py", src, {})
        (r,) = load_recordings(rec.out_path)
        assert r["text"] == "keep me"

    def test_calls_from_outside_repo_root_are_skipped(self, tmp_path, stubbed_sdks):
        import litellm  # the fake

        with record_prompts(tmp_path) as rec:
            # caller frame = this test file, which is NOT under tmp_path
            litellm.completion(model="x",
                               messages=[{"role": "user", "content": "hi"}])
        assert not rec.out_path.exists()


# ── 4. **kwargs calls capture real rendered text ───────────────────────


class TestKwargsUnpack:
    def test_kwargs_unpacked_call_captures_rendered_text(
        self, tmp_path, stubbed_sdks
    ):
        # The static scanner can only emit <dynamic:KwargsUnpack> here; at
        # call time they are real kwargs and the recorder sees the text.
        src = """
            import anthropic.resources.messages as m
            doc = "invoices"
            kwargs = {
                "model": "x",
                "system": f"You handle {doc}.",
                "messages": [{"role": "user", "content": f"process {doc} now"}],
            }
            m.Messages.create(None, **kwargs)
        """
        with record_prompts(tmp_path) as rec:
            _run_from_file(tmp_path, "app.py", src, {})
        recs = load_recordings(rec.out_path)
        fps = {r["fingerprint"] for r in recs}
        assert fingerprint_text("You handle invoices.") in fps
        assert fingerprint_text("process invoices now") in fps


# ── 5. case binding by observation (contextvar) ────────────────────────


class TestCaseBinding:
    def test_active_case_uid_lands_on_recordings(self, tmp_path, stubbed_sdks):
        src = """
            import litellm
            litellm.completion(model="x",
                               messages=[{"role": "user", "content": "same q"}])
        """
        with record_prompts(tmp_path) as rec:
            tok = set_active_case("case-aaa")
            _run_from_file(tmp_path, "app.py", src, {})
            reset_active_case(tok)
            tok = set_active_case("case-bbb")
            _run_from_file(tmp_path, "app.py", src, {})
            reset_active_case(tok)
            _run_from_file(tmp_path, "app.py", src, {})  # no active case
        (r,) = load_recordings(rec.out_path)
        assert r["case_uids"] == ["case-aaa", "case-bbb"]
        assert r["count"] == 3

    def test_suite_run_binds_provenance_case_uid(self, tmp_path, stubbed_sdks):
        from multivon_eval import EvalCase, EvalSuite, NotEmpty

        src = """
            import litellm

            def ask(q):
                litellm.completion(
                    model="x", messages=[{"role": "user", "content": q}],
                )
                return "answer: " + q
        """
        ns: dict = {}
        _run_from_file(tmp_path, "model.py", src, ns)

        suite = EvalSuite("recorder-binding")
        suite.add_cases([EvalCase(
            input="what is drift?",
            metadata={"_provenance": {"schema_version": 1,
                                      "case_uid": "case-xyz", "targets": []}},
        )])
        suite.add_evaluators(NotEmpty())
        with record_prompts(tmp_path) as rec:
            suite.run(ns["ask"], verbose=False)
        (r,) = load_recordings(rec.out_path)
        assert r["case_uids"] == ["case-xyz"]
        assert r["anchor"]["file_path"] == "model.py"
        assert r["anchor"]["qualname"].endswith("ask")


# ── 6. storage: append-safe, idempotent merge ──────────────────────────


class TestStorageMerge:
    def _record_once(self, tmp_path, uid):
        src = """
            import litellm
            litellm.completion(model="x",
                               messages=[{"role": "user", "content": "stable q"}])
        """
        with record_prompts(tmp_path) as rec:
            tok = set_active_case(uid)
            _run_from_file(tmp_path, "app.py", src, {})
            reset_active_case(tok)
        return rec.out_path

    def test_sessions_merge_one_line_per_key(self, tmp_path, stubbed_sdks):
        out = self._record_once(tmp_path, "u1")
        self._record_once(tmp_path, "u2")
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1  # same (anchor, role, fingerprint) key merged
        rec = json.loads(lines[0])
        assert rec["count"] == 2
        assert rec["case_uids"] == ["u1", "u2"]

    def test_flush_with_nothing_new_leaves_file_untouched(
        self, tmp_path, stubbed_sdks
    ):
        out = self._record_once(tmp_path, "u1")
        before = out.read_bytes()
        with record_prompts(tmp_path) as rec:
            rec.flush()  # nothing recorded — must not rewrite
        assert out.read_bytes() == before

    def test_merge_recording_dicts_identity_and_union(self, tmp_path):
        rec = {
            "schema_version": 1,
            "anchor": {"file_path": "a.py", "qualname": "f", "sdk": "openai",
                       "call_site": "chat.completions.create", "role": "user",
                       "line": 3},
            "fingerprint": "f" * 64, "loose_fingerprint": "l" * 64,
            "case_uids": ["c1"], "count": 2,
            "first_seen": "2026-06-01T00:00:00Z",
            "last_seen": "2026-06-02T00:00:00Z", "recorder_version": 1,
        }
        assert merge_recording_dicts([rec], []) == [rec]  # identity
        other = dict(rec, case_uids=["c2"], count=3,
                     first_seen="2026-05-30T00:00:00Z",
                     last_seen="2026-06-03T00:00:00Z")
        (merged,) = merge_recording_dicts([rec], [other])
        assert merged["count"] == 5
        assert merged["case_uids"] == ["c1", "c2"]
        assert merged["first_seen"] == "2026-05-30T00:00:00Z"
        assert merged["last_seen"] == "2026-06-03T00:00:00Z"

    def test_load_recordings_skips_malformed_and_future_lines(self, tmp_path):
        p = tmp_path / DEFAULT_RECORDINGS_NAME
        good = json.dumps({"schema_version": 1, "anchor": {"file_path": "a.py"},
                           "fingerprint": "x" * 64})
        future = json.dumps({"schema_version": 99, "anchor": {}, "fingerprint": "y"})
        p.write_text(f"{good}\nnot json\n{future}\n", encoding="utf-8")
        recs = load_recordings(p)
        assert len(recs) == 1 and recs[0]["fingerprint"] == "x" * 64


# ── 7-8. baseline merge (source:"runtime") + OBSERVED report tier ──────


_STATIC_APP = '''
    import anthropic

    def extract(doc):
        return anthropic.Anthropic().messages.create(
            model="m",
            system="You are an invoice parser. Return JSON only.",
            messages=[{"role": "user", "content": "parse this"}],
        )
'''


def _make_recording(fp_text: str, *, role="user", file_path="runner.py",
                    qualname="ask", case_uids=(), count=1) -> dict:
    return {
        "schema_version": 1,
        "anchor": {"file_path": file_path, "qualname": qualname,
                   "sdk": "litellm", "call_site": "completion",
                   "role": role, "line": 7},
        "fingerprint": fingerprint_text(fp_text),
        "loose_fingerprint": loose_fingerprint_text(fp_text),
        "case_uids": sorted(case_uids),
        "count": count,
        "first_seen": "2026-06-10T00:00:00Z",
        "last_seen": "2026-06-11T00:00:00Z",
        "recorder_version": 1,
    }


class TestBaselineMerge:
    def test_merge_adds_runtime_records_without_touching_static(self, tmp_path):
        _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        bpath = tmp_path / DEFAULT_BASELINE_NAME
        static_before = json.loads(bpath.read_text(encoding="utf-8"))["records"]

        rpath = tmp_path / DEFAULT_RECORDINGS_NAME
        write_recordings(rpath, [
            _make_recording("rendered v1", case_uids=["c1"]),
            _make_recording("rendered v2", case_uids=["c1", "c2"]),
        ])
        n_sites, n_fps = merge_recordings_into_baseline(bpath, rpath)
        assert (n_sites, n_fps) == (1, 2)

        payload = json.loads(bpath.read_text(encoding="utf-8"))
        assert payload["records"] == static_before  # NEVER touched
        (rr,) = payload["runtime_records"]
        assert rr["source"] == "runtime"
        assert sorted(rr["fingerprints"]) == sorted([
            fingerprint_text("rendered v1"), fingerprint_text("rendered v2"),
        ])  # fingerprint SET — variable renderings
        assert rr["case_uids"] == ["c1", "c2"]

        # re-merge is idempotent (recordings file is the count source of truth)
        before = bpath.read_bytes()
        merge_recordings_into_baseline(bpath, rpath)
        assert bpath.read_bytes() == before

        # loadable through the typed reader too
        baseline = load_baseline(bpath)
        assert len(baseline.runtime_records) == 1

    def test_static_rescan_preserves_runtime_records(self, tmp_path):
        _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        bpath = tmp_path / DEFAULT_BASELINE_NAME
        rpath = tmp_path / DEFAULT_RECORDINGS_NAME
        write_recordings(rpath, [_make_recording("rendered v1")])
        merge_recordings_into_baseline(bpath, rpath)
        write_baseline(tmp_path)  # refresh static tier
        baseline = load_baseline(bpath)
        assert len(baseline.runtime_records) == 1  # runtime tier survives

    def test_cli_baseline_merge_recordings(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_staleness

        _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        write_recordings(tmp_path / DEFAULT_RECORDINGS_NAME,
                         [_make_recording("rendered v1")])
        rcode = cmd_staleness(Namespace(
            staleness_cmd="baseline", path=str(tmp_path), out=None,
            dry_run=False, merge_recordings="",
        ))
        out = capsys.readouterr().out
        assert rcode == 0
        assert 'source:"runtime"' in out
        assert "static records untouched" in out


class TestObservedTier:
    def _merged_repo(self, tmp_path) -> Path:
        _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        bpath = tmp_path / DEFAULT_BASELINE_NAME
        rpath = tmp_path / DEFAULT_RECORDINGS_NAME
        write_recordings(rpath, [
            _make_recording("rendered v1", case_uids=["c1"]),
            _make_recording("rendered v2", case_uids=["c2"]),
        ])
        merge_recordings_into_baseline(bpath, rpath)
        return rpath

    def test_report_renders_k_of_n_never_fresh(self, tmp_path):
        rpath = self._merged_repo(tmp_path)
        # current recordings see only ONE of the two known renderings,
        # plus one never-seen-before rendering.
        write_recordings(rpath, [
            _make_recording("rendered v1"),
            _make_recording("rendered v3 brand new"),
        ])
        report = build_staleness_report(tmp_path)
        assert report.determinacy["observed_runtime"] == 1
        (ov,) = report.observed
        assert (ov.matched, ov.baseline_renderings) == (1, 2)
        assert ov.new_renderings == 1
        text = render_text(report)
        assert "OBSERVED at runtime (1)" in text
        assert "matched 1 of 2 previously observed renderings" in text
        assert "1 new rendering(s) not in the baseline" in text
        assert "recordings-vs-recordings" in text
        assert "observed at runtime (recorder" in text  # determinacy clause
        # never collapsed into the static freshness language
        assert "trust tiers (never collapsed)" in text
        # the runtime site is never called unchanged/fresh
        assert "runner.py" not in text.split("OBSERVED")[0]

    def test_report_without_current_recordings_says_so(self, tmp_path):
        rpath = self._merged_repo(tmp_path)
        rpath.unlink()  # baseline knows the site; nothing current to compare
        report = build_staleness_report(tmp_path)
        (ov,) = report.observed
        assert ov.has_current is False
        text = render_text(report)
        assert "no current recordings to compare" in text

    def test_json_and_markdown_carry_observed_tier(self, tmp_path):
        rpath = self._merged_repo(tmp_path)
        write_recordings(rpath, [_make_recording("rendered v1")])
        report = build_staleness_report(tmp_path)
        payload = json.loads(render_json(report))
        assert payload["determinacy"]["observed_runtime"] == 1
        (obs,) = payload["observed"]
        assert obs["source"] == "runtime"
        assert (obs["matched"], obs["baseline_renderings"]) == (1, 2)
        assert "renderings observed" in obs["caveat"]
        assert payload["trust_tiers"]
        md = render_markdown(report)
        assert "observed (runtime)" in md
        assert "matched 1 of 2 previously observed renderings" in md

    def test_unmerged_runtime_sites_are_counted(self, tmp_path):
        rpath = self._merged_repo(tmp_path)
        write_recordings(rpath, [
            _make_recording("rendered v1"),
            _make_recording("other site", file_path="other.py", qualname="go"),
        ])
        report = build_staleness_report(tmp_path)
        assert report.unmerged_runtime == 1
        assert "not yet in the baseline" in render_text(report)

    def test_runtime_bound_case_target_is_unverifiable_vs_static(self, tmp_path):
        from multivon_eval.staleness import match_target

        status, labels = match_target(
            {"source": "runtime", "fingerprint": "x" * 64,
             "anchor": {"file_path": "runner.py"}}, [],
        )
        assert status == "unverifiable"
        assert "runtime" in labels


# ── 9. stamp --from-recordings: propose-only ───────────────────────────


class TestStampFromRecordings:
    def _setup(self, tmp_path):
        from multivon_eval.provenance import new_provenance

        cases = tmp_path / "seed_cases.jsonl"
        prov = new_provenance(
            authored_by="bootstrap", git={"sha": "abc1234", "dirty": False},
            targets=[], case_uid="case-xyz",
        )
        cases.write_text(
            json.dumps({"input": "q", "metadata": {"_provenance": prov}}) + "\n",
            encoding="utf-8",
        )
        rpath = tmp_path / DEFAULT_RECORDINGS_NAME
        write_recordings(rpath, [
            _make_recording("rendered v1", case_uids=["case-xyz"], count=4),
        ])
        return cases, rpath

    def _ns(self, tmp_path, rpath, cases, apply=False):
        return Namespace(
            staleness_cmd="stamp", cases=str(cases) if cases else None,
            site=None, index=None, tag=None, all=False, evidence=None,
            repo=str(tmp_path), dry_run=False, force=False,
            from_recordings=str(rpath), apply=apply,
        )

    def test_propose_only_prints_bindings_and_writes_nothing(
        self, tmp_path, capsys
    ):
        from multivon_eval.cli import cmd_staleness

        cases, rpath = self._setup(tmp_path)
        before = cases.read_bytes()
        rcode = cmd_staleness(self._ns(tmp_path, rpath, cases, apply=False))
        out = capsys.readouterr().out
        assert rcode == 0
        assert "case case-xyz" in out
        assert "observed 4×" in out
        assert "propose-only" in out and "--apply" in out
        assert cases.read_bytes() == before  # NOTHING written

    def test_apply_writes_observed_runtime_targets(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_staleness

        cases, rpath = self._setup(tmp_path)
        rcode = cmd_staleness(self._ns(tmp_path, rpath, cases, apply=True))
        assert rcode == 0
        assert "applied observed bindings to 1 case(s)" in capsys.readouterr().out
        data = json.loads(cases.read_text(encoding="utf-8").strip())
        (target,) = data["metadata"]["_provenance"]["targets"]
        assert target["source"] == "runtime"
        assert target["bound"] == "observed"
        assert target["fingerprint"] == fingerprint_text("rendered v1")
        # the case keeps its identity — observation binds, never rewrites
        assert data["metadata"]["_provenance"]["case_uid"] == "case-xyz"

    def test_apply_without_cases_errors(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_staleness

        _cases, rpath = self._setup(tmp_path)
        rcode = cmd_staleness(self._ns(tmp_path, rpath, cases=None, apply=True))
        assert rcode == 2
        assert "--cases" in capsys.readouterr().err

    def test_apply_bindings_function_matches_by_case_uid(self, tmp_path):
        cases, rpath = self._setup(tmp_path)
        proposals = propose_bindings(load_recordings(rpath))
        assert len(proposals) == 1 and proposals[0].case_uid == "case-xyz"
        updated = apply_bindings(cases, proposals, repo=tmp_path)
        assert updated == 1
        # idempotent: same targets restamped → byte-identical file
        before = cases.read_bytes()
        assert apply_bindings(cases, proposals, repo=tmp_path) == 0
        assert cases.read_bytes() == before


# ── grouping helpers ───────────────────────────────────────────────────


class TestRuntimeRecordGrouping:
    def test_variable_renderings_group_into_fingerprint_set(self):
        recs = [
            _make_recording("v1", case_uids=["a"], count=2),
            _make_recording("v2", case_uids=["b"], count=3),
            _make_recording("elsewhere", file_path="z.py", qualname="g"),
        ]
        grouped = runtime_records_from_recordings(recs)
        assert len(grouped) == 2
        main = next(g for g in grouped if g["anchor"]["file_path"] == "runner.py")
        assert len(main["fingerprints"]) == 2
        assert main["observations"] == 5
        assert main["case_uids"] == ["a", "b"]
        assert all(g["source"] == "runtime" for g in grouped)

    def test_compare_observed_with_no_current_is_honest(self):
        grouped = runtime_records_from_recordings([_make_recording("v1")])
        (ov,) = compare_observed(grouped, None)
        assert ov.has_current is False and ov.matched == 0
