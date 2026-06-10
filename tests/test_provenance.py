"""Tests for multivon_eval.provenance — case provenance stamping.

Pins the two adversarial-round fatal-flaw fixes that live here:
  - JSONL preservation (test plan #3): the stamp write path is a raw-dict
    round-trip; it must never destroy expected_tool_calls / conversation /
    agent_trace / unknown user keys (load_jsonl drops fields the bootstrap
    writer emits — the asymmetry the raw path exists to avoid).
  - Stamp idempotency (test plan #11): identical restamps are byte-identical
    (no git churn), authored_at and case_uid preserved.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from multivon_eval.attribution import scan
from multivon_eval.provenance import (
    PROVENANCE_KEY,
    PROVENANCE_SCHEMA_VERSION,
    AmbiguousSiteError,
    NonConformingProvenanceError,
    merge_targets,
    new_provenance,
    parse_site_spec,
    read_provenance,
    resolve_site_spec,
    stamp,
    stamp_jsonl,
    target_from_record,
)


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


_APP = '''
    import anthropic

    class Extractor:
        def extract(self, doc):
            return anthropic.Anthropic().messages.create(
                model="m",
                system="You are an invoice parser.",
                messages=[{"role": "user", "content": "parse"}],
            )
'''


@pytest.fixture
def repo(tmp_path):
    _write(tmp_path, "app/extract.py", _APP)
    return tmp_path


def _system_target(repo: Path) -> dict:
    records = scan(str(repo))
    rec = next(r for r in records if r.role == "system")
    return target_from_record(rec)


# ── 3. JSONL PRESERVATION ──────────────────────────────────────────────


class TestJsonlPreservation:
    def test_stamp_preserves_every_field_and_unknown_keys(self, repo, tmp_path):
        rich = {
            "input": "do the thing",
            "expected_output": "done",
            "expected_tool_calls": ["search", "fetch"],
            "conversation": [{"role": "user", "content": "hi"}],
            "agent_trace": [{"thought": "t", "tool_calls": [], "output": "o"}],
            "user_custom_key": {"nested": [1, 2, 3]},
            "tags": ["x"],
        }
        other = {"input": "untouched case", "expected_tool_calls": ["a"]}
        cases = tmp_path / "cases.jsonl"
        cases.write_text(
            json.dumps(rich) + "\n" + json.dumps(other) + "\n",
            encoding="utf-8",
        )
        other_line_before = cases.read_text(encoding="utf-8").splitlines()[1]

        stamp_jsonl(cases, [_system_target(repo)], indices=[0], repo=repo)

        lines = cases.read_text(encoding="utf-8").splitlines()
        # unselected line: byte-identical
        assert lines[1] == other_line_before
        # selected line: identical except the injected key
        stamped = json.loads(lines[0])
        prov = stamped["metadata"].pop(PROVENANCE_KEY)
        if not stamped["metadata"]:
            stamped.pop("metadata")
        assert stamped == rich
        assert prov["schema_version"] == PROVENANCE_SCHEMA_VERSION
        assert prov["targets"][0]["bound"] == "manual"

    def test_stamp_never_touches_lines_when_nothing_selected(self, repo, tmp_path):
        cases = tmp_path / "cases.jsonl"
        cases.write_text('{"input": "q"}\n', encoding="utf-8")
        before = cases.read_bytes()
        result = stamp_jsonl(cases, [_system_target(repo)], indices=[5], repo=repo)
        assert result.selected == 0
        assert cases.read_bytes() == before

    def test_dry_run_writes_nothing(self, repo, tmp_path):
        cases = tmp_path / "cases.jsonl"
        cases.write_text('{"input": "q"}\n', encoding="utf-8")
        before = cases.read_bytes()
        result = stamp_jsonl(cases, [_system_target(repo)], select_all=True,
                             repo=repo, dry_run=True)
        assert result.updated == 1
        assert cases.read_bytes() == before


# ── 11. STAMP IDEMPOTENCY ──────────────────────────────────────────────


class TestStampIdempotency:
    def test_identical_restamp_is_byte_identical(self, repo, tmp_path):
        cases = tmp_path / "cases.jsonl"
        cases.write_text('{"input": "q1"}\n{"input": "q2"}\n', encoding="utf-8")
        target = _system_target(repo)

        stamp_jsonl(cases, [target], select_all=True, repo=repo)
        first = cases.read_bytes()
        prov_first = json.loads(first.decode().splitlines()[0])["metadata"][PROVENANCE_KEY]

        result = stamp_jsonl(cases, [target], select_all=True, repo=repo)
        assert cases.read_bytes() == first  # byte-identical — no git churn
        assert result.updated == 0
        assert result.unchanged == 2

        prov_again = json.loads(
            cases.read_text(encoding="utf-8").splitlines()[0]
        )["metadata"][PROVENANCE_KEY]
        assert prov_again["authored_at"] == prov_first["authored_at"]
        assert prov_again["case_uid"] == prov_first["case_uid"]

    def test_new_target_restamp_preserves_authorship_updates_stamped_at(
            self, repo, tmp_path):
        cases = tmp_path / "cases.jsonl"
        cases.write_text('{"input": "q1"}\n', encoding="utf-8")
        records = scan(str(repo))
        sys_t = target_from_record(next(r for r in records if r.role == "system"))
        usr_t = target_from_record(next(r for r in records if r.role == "user"))

        stamp_jsonl(cases, [sys_t], select_all=True, repo=repo,
                    _now="2026-01-01T00:00:00Z")
        prov1 = json.loads(cases.read_text().splitlines()[0])["metadata"][PROVENANCE_KEY]
        stamp_jsonl(cases, [usr_t], select_all=True, repo=repo,
                    _now="2026-02-02T00:00:00Z")
        prov2 = json.loads(cases.read_text().splitlines()[0])["metadata"][PROVENANCE_KEY]

        assert prov2["case_uid"] == prov1["case_uid"]
        assert prov2["authored_at"] == "2026-01-01T00:00:00Z"
        assert prov2["stamped_at"] == "2026-02-02T00:00:00Z"
        assert len(prov2["targets"]) == 2  # one case may bind several prompts

    def test_restamp_without_evidence_is_visible_in_report(self, repo, tmp_path):
        # honest-uncertainty rule 9: self-attestation is visible, not silent
        from multivon_eval.staleness import build_staleness_report, write_baseline

        write_baseline(repo)
        cases = repo / "seed_cases.jsonl"
        cases.write_text('{"input": "q1"}\n', encoding="utf-8")
        records = scan(str(repo))
        sys_t = target_from_record(next(r for r in records if r.role == "system"))
        usr_t = target_from_record(next(r for r in records if r.role == "user"))
        stamp_jsonl(cases, [sys_t], select_all=True, repo=repo,
                    _now="2026-01-01T00:00:00Z")
        stamp_jsonl(cases, [usr_t], select_all=True, repo=repo,
                    _now="2026-02-02T00:00:00Z")  # restamp, no evidence
        report = build_staleness_report(repo)
        assert report.case_stats()["restamped_no_evidence"] == 1


# ── non-conforming / versioning ────────────────────────────────────────


class TestNonConforming:
    def test_refuses_to_overwrite_malformed_provenance_without_force(
            self, repo, tmp_path):
        cases = tmp_path / "cases.jsonl"
        cases.write_text(json.dumps({
            "input": "q", "metadata": {PROVENANCE_KEY: "not-a-dict"},
        }) + "\n", encoding="utf-8")
        with pytest.raises(NonConformingProvenanceError):
            stamp_jsonl(cases, [_system_target(repo)], select_all=True, repo=repo)
        # --force replaces wholesale
        stamp_jsonl(cases, [_system_target(repo)], select_all=True, repo=repo,
                    force=True)
        _, prov = read_provenance(
            json.loads(cases.read_text().splitlines()[0])["metadata"])
        assert prov is not None

    def test_refuses_newer_schema_without_force(self, repo, tmp_path):
        cases = tmp_path / "cases.jsonl"
        cases.write_text(json.dumps({
            "input": "q",
            "metadata": {PROVENANCE_KEY: {"schema_version": 99}},
        }) + "\n", encoding="utf-8")
        with pytest.raises(NonConformingProvenanceError):
            stamp_jsonl(cases, [_system_target(repo)], select_all=True, repo=repo)


class TestReadProvenance:
    def test_statuses(self):
        assert read_provenance(None)[0] == "unstamped"
        assert read_provenance({})[0] == "unstamped"
        assert read_provenance({PROVENANCE_KEY: "junk"})[0] == "unreadable"
        assert read_provenance({PROVENANCE_KEY: {"schema_version": "x"}})[0] \
            == "unreadable"
        assert read_provenance(
            {PROVENANCE_KEY: {"schema_version": 99}})[0] == "unreadable_newer"
        ok = new_provenance(authored_by="human",
                            git={"sha": None, "dirty": False}, targets=[])
        assert read_provenance({PROVENANCE_KEY: ok})[0] == "ok"

    def test_v1_reader_ignores_unknown_keys(self):
        prov = new_provenance(authored_by="human",
                              git={"sha": None, "dirty": False}, targets=[])
        prov["future_field_from_v1_dot_5"] = {"anything": True}
        status, out = read_provenance({PROVENANCE_KEY: prov})
        assert status == "ok"
        assert out is prov


# ── site-spec parsing + resolution ─────────────────────────────────────


class TestSiteSpec:
    def test_parse_full_spec(self):
        parsed = parse_site_spec("app/extract.py::Extractor.extract.system")
        assert parsed["file_path"] == "app/extract.py"
        assert parsed["qualname"] == "Extractor.extract"
        assert parsed["role"] == "system"
        assert parsed["role_position"] is None

    def test_parse_role_with_position(self):
        parsed = parse_site_spec("a.py::run.user#2")
        assert parsed == {"file_path": "a.py", "qualname": "run",
                          "role": "user", "role_position": 2}

    def test_parse_bare_file(self):
        parsed = parse_site_spec("app/extract.py")
        assert parsed["file_path"] == "app/extract.py"
        assert parsed["qualname"] is None and parsed["role"] is None

    def test_parse_file_dot_role(self):
        parsed = parse_site_spec("app/extract.py.system")
        assert parsed["file_path"] == "app/extract.py"
        assert parsed["role"] == "system"

    def test_resolve_unique(self, repo):
        records = scan(str(repo))
        rec = resolve_site_spec(records, "app/extract.py.system")
        assert rec.role == "system"

    def test_resolve_no_match_errors(self, repo):
        records = scan(str(repo))
        with pytest.raises(AmbiguousSiteError):
            resolve_site_spec(records, "nonexistent.py.system")

    def test_resolve_ambiguous_errors_listing_candidates(self, repo):
        records = scan(str(repo))
        with pytest.raises(AmbiguousSiteError) as exc:
            resolve_site_spec(records, "app/extract.py")  # system AND user match
        assert len(exc.value.candidates) == 2

    def test_duplicated_fingerprint_requires_explicit_qualname(self, tmp_path):
        _write(tmp_path, "a.py", '''
            import anthropic
            def alpha():
                anthropic.Anthropic().messages.create(
                    model="m", system="dup text", messages=[],
                )
        ''')
        _write(tmp_path, "b.py", '''
            import anthropic
            def beta():
                anthropic.Anthropic().messages.create(
                    model="m", system="dup text", messages=[],
                )
        ''')
        records = scan(str(tmp_path))
        with pytest.raises(AmbiguousSiteError):
            resolve_site_spec(records, "a.py.system")
        rec = resolve_site_spec(records, "a.py::alpha.system")
        assert rec.qualname == "alpha"


# ── Python-inline helper + merge ───────────────────────────────────────


class TestStampHelper:
    def test_stamp_with_no_sites_is_unbound_repo_state_provenance(self, repo):
        metadata = stamp(repo=repo)
        status, prov = read_provenance(metadata)
        assert status == "ok"
        assert prov["targets"] == []
        assert prov["authored_by"] == "human"

    def test_stamp_resolves_sites_against_live_scan(self, repo):
        metadata = stamp(
            sites=["app/extract.py::Extractor.extract.system"], repo=repo,
        )
        _, prov = read_provenance(metadata)
        assert len(prov["targets"]) == 1
        t = prov["targets"][0]
        assert t["anchor"]["qualname"] == "Extractor.extract"
        assert t["is_dynamic"] is False
        assert t["source"] == "scan"


class TestMergeTargets:
    def test_same_key_replaces_different_key_appends(self):
        t1 = {"fingerprint": "f1", "anchor": {"file_path": "a.py"}, "x": 1}
        t1b = {"fingerprint": "f1", "anchor": {"file_path": "a.py"}, "x": 2}
        t2 = {"fingerprint": "f2", "anchor": {"file_path": "b.py"}}
        merged = merge_targets([t1], [t1b, t2])
        assert merged == [t1b, t2]
