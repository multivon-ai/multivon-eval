"""Tests for multivon_eval.attribution."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from multivon_eval.attribution import (
    PromptDiff,
    PromptRecord,
    diff_records,
    fingerprint_text,
    normalize_text,
    render_markdown,
    scan,
    scan_file,
)


# ── fingerprint ────────────────────────────────────────────────────────


class TestNormalizeText:
    def test_strips_trailing_whitespace_per_line(self):
        assert normalize_text("hello   \nworld   ") == "hello\nworld"

    def test_preserves_internal_blank_lines(self):
        # blank line between paragraphs preserved — it's part of the prompt
        assert normalize_text("line one\n\nline three") == "line one\n\nline three"

    def test_strips_surrounding_whitespace(self):
        assert normalize_text("  \n  hello  \n  ") == "hello"

    def test_preserves_case(self):
        # Case is sometimes load-bearing in prompts ("DO NOT" vs "do not").
        assert normalize_text("DO NOT") != normalize_text("do not")


class TestFingerprintText:
    def test_deterministic(self):
        assert fingerprint_text("hello") == fingerprint_text("hello")

    def test_different_text_different_fingerprint(self):
        assert fingerprint_text("hello") != fingerprint_text("world")

    def test_normalization_collapses_trailing_whitespace_differences(self):
        # "hello   " and "hello" normalize to the same thing → same fingerprint
        assert fingerprint_text("hello   ") == fingerprint_text("hello")

    def test_normalization_does_not_collapse_internal_whitespace(self):
        # Two spaces between words is different from one — a real edit.
        assert fingerprint_text("a  b") != fingerprint_text("a b")


# ── ast_extractor ──────────────────────────────────────────────────────


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content))
    return p


class TestExtractAnthropic:
    def test_system_kwarg(self, tmp_path):
        f = _write(tmp_path, "x.py", '''
            import anthropic
            client = anthropic.Anthropic()
            r = client.messages.create(
                model="claude-haiku-4-5",
                system="You are an invoice parser. Return JSON only.",
                messages=[{"role": "user", "content": "hi"}],
            )
        ''')
        recs = scan_file(str(f), repo_root=str(tmp_path))
        # system + 1 user message
        assert len(recs) == 2
        system = [r for r in recs if r.role == "system"][0]
        assert system.sdk == "anthropic"
        assert system.call_site == "messages.create"
        assert system.role_position == -1
        assert "invoice parser" in system.text
        assert system.is_dynamic is False

    def test_messages_list_extraction(self, tmp_path):
        f = _write(tmp_path, "x.py", '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m",
                messages=[
                    {"role": "user", "content": "what is 2+2?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "are you sure?"},
                ],
            )
        ''')
        recs = scan_file(str(f), repo_root=str(tmp_path))
        roles_and_positions = sorted([(r.role, r.role_position) for r in recs])
        assert roles_and_positions == [
            ("assistant", 1), ("user", 0), ("user", 2)
        ]

    def test_call_site_id_format(self, tmp_path):
        f = _write(tmp_path, "extractors/invoice.py", '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m",
                system="parse",
                messages=[{"role": "user", "content": "x"}],
            )
        ''')
        recs = scan_file(str(f), repo_root=str(tmp_path))
        ids = {r.call_site_id for r in recs}
        # system has role_position=-1 → suffix is just "system"
        # user has role_position=0 → suffix is "user#0"
        assert any(i.endswith(":anthropic.system") for i in ids)
        assert any(i.endswith(":anthropic.user#0") for i in ids)


class TestExtractOpenAI:
    def test_chat_completions_create(self, tmp_path):
        f = _write(tmp_path, "x.py", '''
            from openai import OpenAI
            OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "hi"},
                ],
            )
        ''')
        recs = scan_file(str(f), repo_root=str(tmp_path))
        sdks = {r.sdk for r in recs}
        assert sdks == {"openai"}
        assert {r.role for r in recs} == {"system", "user"}


class TestExtractLiteLLM:
    def test_completion(self, tmp_path):
        f = _write(tmp_path, "x.py", '''
            import litellm
            litellm.completion(
                model="claude-haiku-4-5",
                messages=[{"role": "user", "content": "hello"}],
            )
        ''')
        recs = scan_file(str(f), repo_root=str(tmp_path))
        assert len(recs) == 1
        assert recs[0].sdk == "litellm"
        assert recs[0].call_site == "completion"

    def test_acompletion(self, tmp_path):
        f = _write(tmp_path, "x.py", '''
            import litellm
            async def run():
                return await litellm.acompletion(
                    model="m",
                    messages=[{"role": "user", "content": "hi"}],
                )
        ''')
        recs = scan_file(str(f), repo_root=str(tmp_path))
        assert len(recs) == 1
        assert recs[0].call_site == "acompletion"


class TestExtractDynamicVsLiteral:
    def test_pure_literal_fstring_is_not_dynamic(self, tmp_path):
        f = _write(tmp_path, "x.py", '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m",
                system=f"static prompt",
                messages=[],
            )
        ''')
        recs = scan_file(str(f), repo_root=str(tmp_path))
        assert len(recs) == 1
        assert recs[0].is_dynamic is False
        assert recs[0].text == "static prompt"

    def test_runtime_fstring_is_dynamic(self, tmp_path):
        f = _write(tmp_path, "x.py", '''
            import anthropic
            user_name = "alice"
            anthropic.Anthropic().messages.create(
                model="m",
                system=f"hello {user_name}",
                messages=[],
            )
        ''')
        recs = scan_file(str(f), repo_root=str(tmp_path))
        assert len(recs) == 1
        assert recs[0].is_dynamic is True
        # placeholder shape
        assert recs[0].text.startswith("<dynamic:")

    def test_name_reference_is_dynamic(self, tmp_path):
        f = _write(tmp_path, "x.py", '''
            import anthropic
            SYSTEM_PROMPT = "I'm a constant"
            anthropic.Anthropic().messages.create(
                model="m",
                system=SYSTEM_PROMPT,
                messages=[],
            )
        ''')
        recs = scan_file(str(f), repo_root=str(tmp_path))
        # Per the v1 adversarial-fix discipline: named constants are NOT
        # captured by their name. Only kwarg literals count. So system=
        # being a Name → dynamic placeholder.
        assert len(recs) == 1
        assert recs[0].is_dynamic is True


class TestExtractSkipsNonSdkCalls:
    def test_unrelated_calls_ignored(self, tmp_path):
        f = _write(tmp_path, "x.py", '''
            import json
            json.dumps({"system": "this is not an SDK call"})
            print("system: hello")
        ''')
        recs = scan_file(str(f), repo_root=str(tmp_path))
        assert recs == []

    def test_syntax_error_returns_empty_not_raise(self, tmp_path):
        f = _write(tmp_path, "x.py", "def broken(: pass")
        # broken syntax → empty list, no exception
        assert scan_file(str(f), repo_root=str(tmp_path)) == []


class TestScanDirectory:
    def test_walks_subdirs_and_skips_ignored(self, tmp_path):
        _write(tmp_path, "app.py", '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m", system="app prompt", messages=[],
            )
        ''')
        _write(tmp_path, "sub/mod.py", '''
            import litellm
            litellm.completion(
                model="m", messages=[{"role": "user", "content": "sub prompt"}],
            )
        ''')
        # ignored — should not appear in results
        _write(tmp_path, ".venv/lib/skip.py", '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m", system="should be ignored", messages=[],
            )
        ''')
        _write(tmp_path, "node_modules/x.py", '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m", system="also ignored", messages=[],
            )
        ''')
        recs = scan(str(tmp_path))
        texts = {r.text for r in recs}
        assert "app prompt" in texts
        assert "sub prompt" in texts
        assert "should be ignored" not in texts
        assert "also ignored" not in texts

    def test_deterministic_order(self, tmp_path):
        _write(tmp_path, "b.py", '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m", system="B", messages=[],
            )
        ''')
        _write(tmp_path, "a.py", '''
            import anthropic
            anthropic.Anthropic().messages.create(
                model="m", system="A", messages=[],
            )
        ''')
        r1 = scan(str(tmp_path))
        r2 = scan(str(tmp_path))
        assert [r.text for r in r1] == [r.text for r in r2]
        assert [r.text for r in r1] == ["A", "B"]  # alphabetical by file


# ── diff ───────────────────────────────────────────────────────────────


def _rec(call_site_id: str, text: str, *, is_dynamic: bool = False) -> PromptRecord:
    # Parse call_site_id back into the components — used for compact test setup.
    file_path, rest = call_site_id.split(":", 1)
    line_str, rest2 = rest.split(":", 1)
    sdk, role_part = rest2.split(".", 1)
    if "#" in role_part:
        role, pos = role_part.split("#")
        role_position = int(pos)
    else:
        role, role_position = role_part, -1
    return PromptRecord(
        file_path=file_path, line=int(line_str), sdk=sdk,
        call_site="x.y", role=role, role_position=role_position,
        qualname="<module>", text=text, is_dynamic=is_dynamic,
        fingerprint=fingerprint_text(text),
    )


class TestDiffRecords:
    def test_unchanged_returns_empty(self):
        r = _rec("a.py:5:anthropic.system", "hello")
        assert diff_records([r], [r]) == []

    def test_modified(self):
        b = _rec("a.py:5:anthropic.system", "before")
        h = _rec("a.py:5:anthropic.system", "after")
        diffs = diff_records([b], [h])
        assert len(diffs) == 1
        assert diffs[0].change_type == "modified"
        assert diffs[0].before.text == "before"
        assert diffs[0].after.text == "after"

    def test_added(self):
        h = _rec("a.py:5:anthropic.system", "new")
        diffs = diff_records([], [h])
        assert [(d.change_type, d.after.text) for d in diffs] == [("added", "new")]

    def test_removed(self):
        b = _rec("a.py:5:anthropic.system", "gone")
        diffs = diff_records([b], [])
        assert [(d.change_type, d.before.text) for d in diffs] == [("removed", "gone")]

    def test_dynamic_change(self):
        b = _rec("a.py:5:anthropic.system", "<dynamic:JoinedStr>", is_dynamic=True)
        h = _rec("a.py:5:anthropic.system", "<dynamic:Name>", is_dynamic=True)
        diffs = diff_records([b], [h])
        assert len(diffs) == 1
        assert diffs[0].change_type == "dynamic"

    def test_dynamic_unchanged_is_no_diff(self):
        # same dynamic placeholder on both sides → no diff
        b = _rec("a.py:5:anthropic.system", "<dynamic:Name>", is_dynamic=True)
        h = _rec("a.py:5:anthropic.system", "<dynamic:Name>", is_dynamic=True)
        assert diff_records([b], [h]) == []

    def test_ordering_modified_then_added_then_removed(self):
        b = [
            _rec("a.py:1:anthropic.system", "x"),  # will be modified
            _rec("a.py:2:anthropic.system", "to-remove"),
        ]
        h = [
            _rec("a.py:1:anthropic.system", "x-new"),  # modified
            _rec("a.py:3:anthropic.system", "to-add"),
        ]
        diffs = diff_records(b, h)
        assert [d.change_type for d in diffs] == ["modified", "added", "removed"]


# ── render ─────────────────────────────────────────────────────────────


class TestRenderMarkdown:
    def test_no_diffs_returns_friendly_message(self):
        out = render_markdown([])
        assert "## Prompt changes" in out
        assert "No prompt changes" in out

    def test_modified_includes_both_before_and_after(self):
        b = _rec("a.py:1:anthropic.system", "old text here")
        h = _rec("a.py:1:anthropic.system", "new text here")
        diffs = diff_records([b], [h])
        out = render_markdown(diffs)
        assert "before" in out
        assert "after" in out
        assert "old text here" in out
        assert "new text here" in out

    def test_summary_line_counts_each_kind(self):
        b = [
            _rec("a.py:1:anthropic.system", "x"),
            _rec("a.py:2:anthropic.system", "remove-me"),
        ]
        h = [
            _rec("a.py:1:anthropic.system", "x-new"),
            _rec("a.py:3:anthropic.system", "add-me"),
        ]
        out = render_markdown(diff_records(b, h))
        assert "1 modified" in out
        assert "1 added" in out
        assert "1 removed" in out

    def test_no_impact_or_attribution_claims_in_output(self):
        # The whole point of v1: descriptive, not causal. No "Impact:",
        # no "root cause", no "confidence", no "likely cause" allowed.
        b = _rec("a.py:1:anthropic.system", "before")
        h = _rec("a.py:1:anthropic.system", "after")
        out = render_markdown(diff_records([b], [h]))
        out_lower = out.lower()
        for forbidden in ["impact:", "root cause", "confidence", "likely cause", "attributed to"]:
            assert forbidden not in out_lower, f"forbidden phrase in render: {forbidden!r}"

    def test_truncation_marker_on_long_prompt(self):
        long_text = "\n".join([f"line {i}" for i in range(20)])
        b = _rec("a.py:1:anthropic.system", "short")
        h = _rec("a.py:1:anthropic.system", long_text)
        out = render_markdown(diff_records([b], [h]))
        assert "truncated" in out

    def test_dynamic_unscanned_count_surfaces(self):
        out = render_markdown([], dynamic_unscanned_count=3)
        assert "3 prompt source" in out
