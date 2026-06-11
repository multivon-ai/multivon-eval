"""Tests for multivon_eval.staleness — prompt-drift staleness detection.

Each numbered test pins a specific fatal-flaw fix demanded by the
adversarial design round (see the spec's test_plan). The lock-orthogonality
test is load-bearing: it pins the storage assumption that stamping
metadata._provenance never perturbs suite.lock.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from multivon_eval import EvalCase, EvalSuite, NotEmpty
from multivon_eval.attribution import SCANNER_VERSION, scan
from multivon_eval.lockfile import (
    _cases_hash, build_suite_lock, verify_suite_against_lock,
)
from multivon_eval.provenance import (
    stamp_metadata_inplace, stamp_jsonl, target_from_record,
)
from multivon_eval.staleness import (
    BaselineError,
    DEFAULT_BASELINE_NAME,
    build_staleness_report,
    load_baseline,
    match_records,
    render_json,
    render_markdown,
    render_text,
    scan_repo,
    write_baseline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


_STATIC_APP = '''
    import anthropic

    class Extractor:
        def extract(self, doc):
            return anthropic.Anthropic().messages.create(
                model="m",
                system="You are an invoice parser. Return JSON only.",
                messages=[{"role": "user", "content": "parse this"}],
            )
'''


# ── 1. DYNAMIC GATE ────────────────────────────────────────────────────


class TestDynamicGate:
    def test_rewritten_dynamic_constant_is_never_reported_fresh(self, tmp_path):
        # Conditionally-assigned name → stays dynamic even under scanner v2.
        src = '''
            import os
            import anthropic
            if os.environ.get("X"):
                SYSTEM = "{text}"
            anthropic.Anthropic().messages.create(
                model="m", system=SYSTEM, messages=[],
            )
        '''
        _write(tmp_path, "app.py", src.format(text="original prompt text"))
        write_baseline(tmp_path)
        # Rewrite the referenced constant ENTIRELY.
        _write(tmp_path, "app.py", src.format(text="completely different prompt"))
        report = build_staleness_report(tmp_path)
        assert len(report.verdicts) == 1
        v = report.verdicts[0]
        # The kill condition: a placeholder fingerprint ('<dynamic:Name>')
        # would compare equal — the verdict must NOT be unchanged/fresh.
        assert v.status != "unchanged"
        assert v.status == "unknown"
        assert "dynamic" in v.labels
        assert report.counts()["unknown"] == 1
        assert report.counts()["changed"] == 0

    def test_static_record_becoming_dynamic_is_unknown_not_changed(self, tmp_path):
        _write(tmp_path, "app.py", '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m", system="literal prompt", messages=[],
            )
        ''')
        write_baseline(tmp_path)
        _write(tmp_path, "app.py", '''
            import anthropic
            def build():
                return "x"
            SYSTEM = build()
            anthropic.Anthropic().messages.create(
                model="m", system=SYSTEM, messages=[],
            )
        ''')
        report = build_staleness_report(tmp_path)
        v = report.verdicts[0]
        assert v.status == "unknown"
        assert "became-dynamic" in v.labels

    def test_dynamic_baseline_whose_anchor_disappears_degrades_to_removed(self, tmp_path):
        _write(tmp_path, "app.py", '''
            import anthropic
            def go(prompt):
                anthropic.Anthropic().messages.create(
                    model="m", system=prompt, messages=[],
                )
        ''')
        write_baseline(tmp_path)
        _write(tmp_path, "app.py", "x = 1\n")
        report = build_staleness_report(tmp_path)
        assert report.verdicts[0].status == "removed"


# ── 2. LOCK ORTHOGONALITY (load-bearing) ───────────────────────────────


class TestLockOrthogonality:
    def test_stamping_provenance_never_perturbs_suite_lock(self):
        suite = EvalSuite("orthogonality")
        suite.add_cases([
            EvalCase(input="q1", expected_output="a1"),
            EvalCase(input="q2", expected_output="a2", metadata={"difficulty": "hard"}),
        ])
        suite.add_evaluators(NotEmpty())
        lock = build_suite_lock(suite)
        hash_before = _cases_hash(suite._cases)

        for case in suite._cases:
            stamp_metadata_inplace(
                case.metadata,
                authored_by="human",
                git={"sha": "a1b2c3d", "dirty": False},
                targets=[],
            )
        # _provenance is present...
        assert all("_provenance" in c.metadata for c in suite._cases)
        # ...and the lock does not move: _cases_hash excludes metadata by
        # design (lockfile.py) — the two drift detectors stay orthogonal.
        verify_suite_against_lock(suite, lock)  # must not raise
        assert _cases_hash(suite._cases) == hash_before


# ── 5. LINE-SHIFT IMMUNITY ─────────────────────────────────────────────


class TestLineShiftImmunity:
    def test_inserting_imports_above_every_call_site_yields_zero_findings(self, tmp_path):
        f = _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        original = f.read_text(encoding="utf-8")
        f.write_text("import os\nimport sys\nimport json\n" + original,
                     encoding="utf-8")
        report = build_staleness_report(tmp_path)
        c = report.counts()
        assert c["changed"] == 0
        assert c["removed"] == 0
        assert c["added"] == 0
        assert c["unknown"] == 0
        assert report.exit_code == 0


# ── 6. MOVE / RENAME ───────────────────────────────────────────────────


class TestMoveAndRename:
    def test_file_rename_with_unchanged_prompt_is_unchanged_moved(self, tmp_path):
        f = _write(tmp_path, "old_name.py", _STATIC_APP)
        write_baseline(tmp_path)
        content = f.read_text(encoding="utf-8")
        f.unlink()
        _write(tmp_path, "new_name.py", content)
        report = build_staleness_report(tmp_path)
        c = report.counts()
        assert c["changed"] == 0 and c["removed"] == 0 and c["added"] == 0
        moved = [v for v in report.verdicts if "moved" in v.labels]
        assert len(moved) == 2  # system + user message at the same call
        assert all(v.status == "unchanged" for v in moved)

    def test_rename_plus_edit_in_one_commit_is_removed_with_caveat(self, tmp_path):
        f = _write(tmp_path, "old_name.py", _STATIC_APP)
        write_baseline(tmp_path)
        f.unlink()
        _write(tmp_path, "new_name.py", _STATIC_APP.replace(
            "Return JSON only.", "Return YAML, always."))
        report = build_staleness_report(tmp_path)
        system_verdicts = [v for v in report.verdicts
                           if v.baseline and v.baseline.role == "system"]
        assert len(system_verdicts) == 1
        # statically unbridgeable — never a fuzzy CHANGED
        assert system_verdicts[0].status == "removed"
        text = render_text(report)
        # mandatory three-way caveat
        assert "renamed+edited" in text
        assert "moved beyond static reach" in text
        # the new site surfaces as ADDED
        assert any(r.role == "system" for r in report.added)

    def test_rename_plus_reformat_only_is_changed_file_renamed(self, tmp_path):
        src = '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m",
                system="""line one
                line two""",
                messages=[],
            )
        '''
        f = _write(tmp_path, "old_name.py", src)
        write_baseline(tmp_path)
        f.unlink()
        _write(tmp_path, "new_name.py", src.replace(
            '                line two', '            line two'))
        report = build_staleness_report(tmp_path)
        v = report.verdicts[0]
        assert v.status == "changed"
        assert "file-renamed" in v.labels
        assert "formatting-only" in v.labels


# ── 7. DUPLICATES ──────────────────────────────────────────────────────


_DUP_A = '''
    import anthropic
    def alpha():
        anthropic.Anthropic().messages.create(
            model="m", system="shared duplicate prompt", messages=[],
        )
'''
_DUP_B = '''
    import anthropic
    def beta():
        anthropic.Anthropic().messages.create(
            model="m", system="shared duplicate prompt", messages=[],
        )
'''


class TestDuplicates:
    def test_deleting_one_duplicate_reports_that_site(self, tmp_path):
        _write(tmp_path, "a.py", _DUP_A)
        f_b = _write(tmp_path, "b.py", _DUP_B)
        write_baseline(tmp_path)
        f_b.unlink()
        report = build_staleness_report(tmp_path)
        removed = [v for v in report.verdicts if v.status == "removed"]
        unchanged = [v for v in report.verdicts if v.status == "unchanged"]
        assert len(removed) == 1
        assert removed[0].baseline.file_path == "b.py"
        assert len(unchanged) == 1
        assert unchanged[0].baseline.file_path == "a.py"
        assert unchanged[0].confidence == "exact"

    def test_one_bound_case_marks_all_duplicate_sites_covered(self, tmp_path):
        _write(tmp_path, "a.py", _DUP_A)
        _write(tmp_path, "b.py", _DUP_B)
        write_baseline(tmp_path)
        records = scan(str(tmp_path))
        site = next(r for r in records if r.file_path == "a.py")
        cases = tmp_path / "seed_cases.jsonl"
        cases.write_text(json.dumps({"input": "q"}) + "\n", encoding="utf-8")
        stamp_jsonl(cases, [target_from_record(site)], select_all=True,
                    repo=tmp_path)
        report = build_staleness_report(tmp_path)
        covered, denom = report.coverage()
        # the coverage join marks ALL N duplicate sites covered by that
        # fingerprint, or the numbers lie.
        assert (covered, denom) == (2, 2)


# ── 8. FORMATTING-ONLY ─────────────────────────────────────────────────


class TestFormattingOnly:
    def test_reindented_triple_quoted_prompt_is_changed_with_label(self, tmp_path):
        src = '''
            import anthropic
            def run():
                anthropic.Anthropic().messages.create(
                    model="m",
                    system="""You are a router.
                    Pick a queue.""",
                    messages=[],
                )
        '''
        _write(tmp_path, "app.py", src)
        write_baseline(tmp_path)
        # Re-indent the continuation line: normalize_text preserves leading
        # indentation, so the strict fingerprint flips.
        _write(tmp_path, "app.py", src.replace(
            "                    Pick a queue.", "            Pick a queue."))
        report = build_staleness_report(tmp_path)
        v = report.verdicts[0]
        assert v.status == "changed"  # tagged, never suppressed
        assert "formatting-only" in v.labels
        assert report.counts()["changed"] == 1
        assert report.counts()["formatting_only"] == 1
        assert "formatting-only" in render_text(report)
        # --fail-on changed still fires: label-only, not a suppression.
        report.fail_on = ("changed",)
        assert report.exit_code == 1


# ── 9. SELF-SCAN ───────────────────────────────────────────────────────


class TestSelfScan:
    def test_default_ignores_keep_test_fixtures_out_of_the_corpus(self):
        records, ignores, _skipped = scan_repo(REPO_ROOT)
        assert "tests" in ignores and "examples" in ignores
        assert not any(r.file_path.startswith("tests/") for r in records)
        assert not any(r.file_path.startswith("examples/") for r in records)


# ── 10. EXIT-CODE MATRIX ───────────────────────────────────────────────


class TestExitCodeMatrix:
    def _changed_repo(self, tmp_path) -> Path:
        f = _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        f.write_text(f.read_text(encoding="utf-8").replace(
            "Return JSON only.", "Return XML only."), encoding="utf-8")
        return tmp_path

    def test_report_only_is_exit_0_even_with_findings(self, tmp_path):
        report = build_staleness_report(self._changed_repo(tmp_path))
        assert report.counts()["changed"] == 1
        assert report.exit_code == 0

    def test_fail_on_changed_fires_exit_1(self, tmp_path):
        report = build_staleness_report(
            self._changed_repo(tmp_path), fail_on=("changed",))
        assert report.exit_code == 1

    def test_fail_on_removed_does_not_fire_on_changed(self, tmp_path):
        report = build_staleness_report(
            self._changed_repo(tmp_path), fail_on=("removed",))
        assert report.exit_code == 0

    def test_no_baseline_is_exit_2_with_friendly_message(self, tmp_path):
        _write(tmp_path, "app.py", _STATIC_APP)
        report = build_staleness_report(tmp_path)
        assert report.no_baseline is True
        assert report.exit_code == 2
        text = render_text(report)
        assert "no baseline found" in text
        assert "staleness baseline" in text
        # no findings wall
        assert "CHANGED" not in text

    def test_future_case_schema_version_is_counted_not_fatal(self, tmp_path):
        _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        cases = tmp_path / "seed_cases.jsonl"
        cases.write_text(json.dumps({
            "input": "q",
            "metadata": {"_provenance": {"schema_version": 99, "case_uid": "x"}},
        }) + "\n", encoding="utf-8")
        report = build_staleness_report(tmp_path)
        assert report.case_stats()["unreadable"] == 1
        assert report.exit_code == 0  # counted, exit unaffected
        assert "newer multivon-eval" in render_json(report)

    def test_scanner_version_mismatch_is_exit_2(self, tmp_path):
        _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        bpath = tmp_path / DEFAULT_BASELINE_NAME
        payload = json.loads(bpath.read_text(encoding="utf-8"))
        payload["scanner_version"] = SCANNER_VERSION - 1
        bpath.write_text(json.dumps(payload), encoding="utf-8")
        report = build_staleness_report(tmp_path)
        assert report.scanner_mismatch is True
        assert report.exit_code == 2
        assert "rescan recommended" in render_text(report)

    def test_future_baseline_schema_is_warn_and_skip_exit_2(self, tmp_path):
        _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        bpath = tmp_path / DEFAULT_BASELINE_NAME
        payload = json.loads(bpath.read_text(encoding="utf-8"))
        payload["schema_version"] = 99
        bpath.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(BaselineError):
            load_baseline(bpath)
        report = build_staleness_report(tmp_path)  # never crashes
        assert report.baseline is None
        assert report.baseline_warning is not None
        assert report.exit_code == 2


# ── 12. BASELINE REFRESH ───────────────────────────────────────────────


class TestBaselineRefresh:
    def test_dry_run_prints_diff_and_writes_nothing(self, tmp_path):
        f = _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        bpath = tmp_path / DEFAULT_BASELINE_NAME
        before = bpath.read_bytes()
        f.write_text(f.read_text(encoding="utf-8").replace(
            "Return JSON only.", "Return TOML only."), encoding="utf-8")
        _, diff_lines = write_baseline(tmp_path, dry_run=True)
        assert any("changed" in ln for ln in diff_lines)
        assert bpath.read_bytes() == before  # nothing written

    def test_refresh_writes_atomically_and_clears_findings(self, tmp_path):
        f = _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        f.write_text(f.read_text(encoding="utf-8").replace(
            "Return JSON only.", "Return TOML only."), encoding="utf-8")
        assert build_staleness_report(tmp_path).counts()["changed"] == 1
        _, diff_lines = write_baseline(tmp_path)
        assert any("changed" in ln for ln in diff_lines)
        report = build_staleness_report(tmp_path)
        assert report.counts()["changed"] == 0
        assert report.exit_code == 0
        # no stray temp files left behind by the atomic write
        assert not [p for p in tmp_path.iterdir()
                    if p.name.startswith(f".{DEFAULT_BASELINE_NAME}.")]


# ── 13. BOOTSTRAP E2E ──────────────────────────────────────────────────


def _fake_judge_response(*_a, **_k):
    return json.dumps({
        "evaluators": [
            {"name": "Faithfulness", "tier": "primary", "threshold": 0.75,
             "rationale": "RAG shape"},
            {"name": "NotEmpty", "tier": "guardrail", "threshold": 1.0,
             "rationale": "sanity"},
        ],
        "discussion": "fixture discussion",
    })


class TestBootstrapE2E:
    def test_bootstrap_emits_baseline_and_unbound_stamps(self, tmp_path):
        from multivon_eval import discover

        _write(tmp_path, "app/extract.py", _STATIC_APP)
        product = _write(tmp_path, "product.md",
                         "# Product\nInvoice parser bot.\n")
        traces = tmp_path / "traces.jsonl"
        traces.write_text(json.dumps({
            "input": "parse?", "context": "ctx", "output": "done",
        }) + "\n", encoding="utf-8")

        seed = [EvalCase(input="adversarial q", expected_output="a")]
        with patch.object(discover, "_call_judge",
                          side_effect=_fake_judge_response), \
                patch.object(discover, "generate_seed_cases",
                             return_value=(seed, 0.0)):
            result = discover.bootstrap(
                description_path=product,
                traces_path=traces,
                output_dir=tmp_path / "eval-bootstrap",
                skip_calibration=True,
                repo=tmp_path,
            )

        bpath = tmp_path / DEFAULT_BASELINE_NAME
        assert bpath.exists()
        assert result.artifacts["prompt_baseline"] == bpath
        baseline = load_baseline(bpath)
        assert baseline.scanner_version == SCANNER_VERSION
        assert len(baseline.records) == 2  # system + user message

        # generated cases carry repo-state provenance with targets=[] —
        # bindings are NEVER fabricated.
        lines = (tmp_path / "eval-bootstrap" / "seed_cases.jsonl") \
            .read_text(encoding="utf-8").strip().splitlines()
        assert lines
        for line in lines:
            prov = json.loads(line)["metadata"]["_provenance"]
            assert prov["authored_by"] == "bootstrap"
            assert prov["targets"] == []
            assert prov["case_uid"]

        # immediate staleness run: all unchanged, exit 0
        report = build_staleness_report(tmp_path)
        c = report.counts()
        assert c["changed"] == 0 and c["removed"] == 0 and c["added"] == 0
        assert report.exit_code == 0
        stats = report.case_stats()
        assert stats["total"] == 1
        assert stats["stamped"] == 1
        assert stats["bound"] == 0


# ── matcher units + renderers ──────────────────────────────────────────


class TestMatcherUnits:
    def test_tier_a_ambiguity_breaks_ties_by_nearest_line(self, tmp_path):
        # two calls, SAME anchor (file/qualname/sdk/role/role_position)
        src = '''
            import anthropic
            def run():
                anthropic.Anthropic().messages.create(
                    model="m", system="{a}", messages=[],
                )
                anthropic.Anthropic().messages.create(
                    model="m", system="{b}", messages=[],
                )
        '''
        _write(tmp_path, "app.py", src.format(a="first text", b="second text"))
        write_baseline(tmp_path)
        _write(tmp_path, "app.py", src.format(a="first text EDITED",
                                              b="second text EDITED"))
        report = build_staleness_report(tmp_path)
        assert [v.status for v in report.verdicts] == ["changed", "changed"]
        # the first record sees two Tier-A candidates → ambiguous, surfaced
        # not silently resolved; the second sees the one remaining candidate.
        assert report.verdicts[0].confidence == "ambiguous"
        # nearest-line: first baseline ↔ first live
        assert report.verdicts[0].live.line < report.verdicts[1].live.line

    def test_reverted_prompt_is_automatically_unchanged_again(self, tmp_path):
        f = _write(tmp_path, "app.py", _STATIC_APP)
        original = f.read_text(encoding="utf-8")
        write_baseline(tmp_path)
        f.write_text(original.replace("JSON", "XML"), encoding="utf-8")
        assert build_staleness_report(tmp_path).counts()["changed"] == 1
        f.write_text(original, encoding="utf-8")  # revert — no tombstones
        assert build_staleness_report(tmp_path).counts()["changed"] == 0

    def test_added_site_appears_with_no_case_reference(self, tmp_path):
        _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        _write(tmp_path, "new.py", '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m", system="brand new prompt", messages=[],
            )
        ''')
        report = build_staleness_report(tmp_path)
        assert report.counts()["added"] == 1
        assert "no cases reference this prompt" in render_text(report)


class TestRenderers:
    def _report(self, tmp_path):
        f = _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        f.write_text(f.read_text(encoding="utf-8").replace(
            "Return JSON only.", "Return CSV only."), encoding="utf-8")
        return build_staleness_report(tmp_path)

    def test_text_report_opens_with_determinacy_headline(self, tmp_path):
        text = render_text(self._report(tmp_path))
        head = "\n".join(text.splitlines()[:2])
        assert "statically resolvable; verdicts below cover only those" in head

    def test_text_report_has_standing_blind_spots_footer(self, tmp_path):
        text = render_text(self._report(tmp_path))
        assert "blind spots:" in text
        assert "Responses API" in text
        assert "non-Python services" in text

    def test_markdown_report_carries_determinacy_and_blind_spots(self, tmp_path):
        md = render_markdown(self._report(tmp_path))
        assert "statically resolvable" in md
        assert "blind spots" in md
        assert "**changed**" in md

    def test_json_report_schema(self, tmp_path):
        payload = json.loads(render_json(self._report(tmp_path)))
        assert payload["schema_version"] == 1
        assert payload["determinacy"]["call_sites"] == 2
        assert payload["summary"]["changed"] == 1
        assert payload["exit_code"] == 0
        assert payload["blind_spots"]
        site = next(s for s in payload["sites"] if s["status"] == "changed")
        assert site["old_fingerprint"] != site["new_fingerprint"]
        assert site["confidence"] in ("exact", "structural", "moved", "ambiguous")

    def test_coverage_labeled_lower_bound_and_binding_hint_when_unbound(self, tmp_path):
        text = render_text(self._report(tmp_path))
        assert "lower bound, static sites only" in text
        assert "binding" in text  # zero bound cases → hint, not a 100%-uncovered wall


# ── 11. UNSCANNABLE FILES (audit W1/W3) ────────────────────────────────


class TestUnscannableFiles:
    """A baselined site whose file still EXISTS but no longer parses must
    never read REMOVED — false removed poisons trust. It reads UNKNOWN
    with an "unscannable" label, never trips --fail-on removed, and all
    three renderers carry the unreliability warning."""

    def _broken_repo(self, tmp_path) -> Path:
        _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        # File still exists, but a syntax error makes it unscannable.
        (tmp_path / "app.py").write_text("def broken(: pass", encoding="utf-8")
        return tmp_path

    def test_unscannable_site_is_unknown_not_removed(self, tmp_path):
        report = build_staleness_report(self._broken_repo(tmp_path))
        c = report.counts()
        assert c["removed"] == 0
        assert c["unscannable"] == 2  # system + user message sites
        for v in report.verdicts:
            assert v.status == "unknown"
            assert "unscannable" in v.labels

    def test_unscannable_never_trips_fail_on_removed(self, tmp_path):
        report = build_staleness_report(
            self._broken_repo(tmp_path), fail_on=("removed",),
        )
        assert report.exit_code == 0

    def test_text_renderer_warns_and_labels_unscannable(self, tmp_path):
        text = render_text(build_staleness_report(self._broken_repo(tmp_path)))
        assert "1 file unscannable (syntax/encoding)" in text
        assert "verdicts for sites in those files are unreliable" in text
        assert "UNSCANNABLE (2)" in text
        assert "NOT removed" in text
        assert "REMOVED" not in text.replace("NOT removed", "")

    def test_json_renderer_carries_skipped_files(self, tmp_path):
        payload = json.loads(
            render_json(build_staleness_report(self._broken_repo(tmp_path)))
        )
        assert payload["summary"]["removed"] == 0
        assert payload["summary"]["unscannable"] == 2
        (entry,) = payload["skipped_files"]
        assert entry["path"] == "app.py"
        assert "syntax error" in entry["reason"]
        assert any("unreliable" in w for w in payload["warnings"])

    def test_markdown_renderer_warns(self, tmp_path):
        md = render_markdown(build_staleness_report(self._broken_repo(tmp_path)))
        assert "unscannable" in md
        assert "unreliable" in md

    def test_skipped_files_never_written_into_baseline(self, tmp_path):
        _write(tmp_path, "app.py", _STATIC_APP)
        _write(tmp_path, "bad.py", "def broken(: pass")
        write_baseline(tmp_path)
        payload = json.loads(
            (tmp_path / DEFAULT_BASELINE_NAME).read_text(encoding="utf-8")
        )
        assert "skipped_files" not in payload
        assert all("bad.py" != r["file_path"] for r in payload["records"])

    def test_truly_removed_still_reads_removed(self, tmp_path):
        # The honest-removed path stays intact: file deleted → REMOVED.
        _write(tmp_path, "app.py", _STATIC_APP)
        write_baseline(tmp_path)
        (tmp_path / "app.py").unlink()
        report = build_staleness_report(tmp_path)
        assert report.counts()["removed"] == 2
        assert report.counts()["unscannable"] == 0


# ── 12. CLI PATH VALIDATION + SIGPIPE (audit C4/U1) ────────────────────


class TestStalenessCLIPathValidation:
    def _report_ns(self, path, **over):
        from argparse import Namespace
        ns = dict(
            staleness_cmd="report", path=str(path), baseline=None, cases=None,
            suite=None, ignore=None, include_tests=False, fail_on=None,
            format="text", recordings=None,
        )
        ns.update(over)
        return Namespace(**ns)

    def test_report_nonexistent_path_is_clean_exit_2(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_staleness

        rcode = cmd_staleness(self._report_ns(tmp_path / "no-such-dir"))
        assert rcode == 2
        assert "does not exist" in capsys.readouterr().err

    def test_baseline_nonexistent_path_is_clean_exit_2(self, tmp_path, capsys):
        from argparse import Namespace
        from multivon_eval.cli import cmd_staleness

        rcode = cmd_staleness(Namespace(
            staleness_cmd="baseline", path=str(tmp_path / "no-such-dir"),
            out=None, dry_run=False, merge_recordings=None,
        ))
        assert rcode == 2
        assert "does not exist" in capsys.readouterr().err

    def test_baseline_out_into_missing_dir_is_clean_exit_2(self, tmp_path, capsys):
        from argparse import Namespace
        from multivon_eval.cli import cmd_staleness

        _write(tmp_path, "app.py", _STATIC_APP)
        rcode = cmd_staleness(Namespace(
            staleness_cmd="baseline", path=str(tmp_path),
            out=str(tmp_path / "missing-dir" / "baseline.json"),
            dry_run=False, merge_recordings=None,
        ))
        assert rcode == 2
        assert "--out directory does not exist" in capsys.readouterr().err


def test_cli_broken_pipe_exits_0_quietly():
    # `multivon-eval staleness . | head` must not traceback when the pager
    # closes the pipe. Run in a subprocess so the devnull dup2 in the
    # handler can't disturb pytest's capture fds.
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "from multivon_eval import cli\n"
        "cli.cmd_discover = "
        "lambda args: (_ for _ in ()).throw(BrokenPipeError())\n"
        "sys.argv = ['multivon-eval', 'discover']\n"
        "cli.main()\n"
    )
    proc = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "Traceback" not in proc.stderr
