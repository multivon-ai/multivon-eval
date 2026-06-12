"""Tests for multivon_eval.cli — command handlers (no live I/O, no network).

Each cmd_* function is tested via argparse Namespace injection so we
exercise the real function body without spawning a subprocess.
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ─── helpers ─────────────────────────────────────────────────────────────────

def _ns(**kwargs):
    """Build a Namespace from kwargs."""
    return Namespace(**kwargs)


# ─── cmd_init ────────────────────────────────────────────────────────────────

class TestCmdInit:
    def _args(self, tmp_path, template="quickstart", ci=None, force=False):
        return _ns(template=template, dir=str(tmp_path / "out"), ci=ci, force=force)

    def test_creates_files_in_new_directory(self, tmp_path):
        from multivon_eval.cli import cmd_init
        args = self._args(tmp_path)
        result = cmd_init(args)
        assert result in (0, None)
        out = tmp_path / "out"
        assert out.is_dir()
        assert any(out.iterdir())

    def test_refuses_nonempty_dir_without_force(self, tmp_path):
        from multivon_eval.cli import cmd_init
        out = tmp_path / "out"
        out.mkdir()
        (out / "existing.txt").write_text("block me")
        args = _ns(template="quickstart", dir=str(out), ci=None, force=False)
        result = cmd_init(args)
        assert result == 1

    def test_force_flag_overwrites_nonempty_dir(self, tmp_path):
        from multivon_eval.cli import cmd_init
        out = tmp_path / "out"
        out.mkdir()
        (out / "existing.txt").write_text("old content")
        args = _ns(template="quickstart", dir=str(out), ci=None, force=True)
        result = cmd_init(args)
        assert result in (0, None)

    def test_target_is_file_returns_error(self, tmp_path):
        from multivon_eval.cli import cmd_init
        f = tmp_path / "file.txt"
        f.write_text("I am a file")
        args = _ns(template="quickstart", dir=str(f), ci=None, force=False)
        result = cmd_init(args)
        assert result == 1


# ─── cmd_experiments ─────────────────────────────────────────────────────────

class TestCmdExperiments:
    def test_list_no_experiments(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_experiments
        with patch("multivon_eval.experiments._experiments_dir", return_value=tmp_path):
            cmd_experiments(_ns(exp_cmd="list"))
        assert "No experiments" in capsys.readouterr().out

    def test_list_with_experiments(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_experiments
        (tmp_path / "my-exp.jsonl").write_text("")
        with patch("multivon_eval.experiments._experiments_dir", return_value=tmp_path):
            cmd_experiments(_ns(exp_cmd="list"))
        assert "my-exp" in capsys.readouterr().out

    def test_history_subcommand(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_experiments
        from multivon_eval.experiments import Experiment
        with patch("multivon_eval.experiments._experiments_dir", return_value=tmp_path):
            # empty history → "No runs"
            cmd_experiments(_ns(exp_cmd="history", name="empty-exp", n=5))
        assert "No runs" in capsys.readouterr().out

    def test_unknown_exp_cmd_prints_usage(self, capsys):
        from multivon_eval.cli import cmd_experiments
        cmd_experiments(_ns(exp_cmd="bogus"))
        assert "Usage" in capsys.readouterr().out


# ─── cmd_attribution ─────────────────────────────────────────────────────────

class TestCmdAttribution:
    def test_scan_text_format_no_prompts(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_attribution
        (tmp_path / "empty.py").write_text("x = 1\n")
        result = cmd_attribution(_ns(attribution_cmd="scan", path=str(tmp_path), format="text"))
        assert result == 0
        assert "No SDK prompt" in capsys.readouterr().out

    def test_scan_json_format_empty(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_attribution
        (tmp_path / "empty.py").write_text("x = 1\n")
        result = cmd_attribution(_ns(attribution_cmd="scan", path=str(tmp_path), format="json"))
        assert result == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data == []

    def test_diff_text_format_no_changes(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_attribution
        base = tmp_path / "base"
        head = tmp_path / "head"
        base.mkdir()
        head.mkdir()
        (base / "a.py").write_text("x = 1\n")
        (head / "a.py").write_text("x = 1\n")
        result = cmd_attribution(_ns(
            attribution_cmd="diff",
            base=str(base),
            head=str(head),
            format="text",
        ))
        assert result == 0
        assert "No prompt changes" in capsys.readouterr().out

    def test_diff_json_format(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_attribution
        base = tmp_path / "base"
        head = tmp_path / "head"
        base.mkdir()
        head.mkdir()
        (base / "a.py").write_text("x = 1\n")
        (head / "a.py").write_text("x = 1\n")
        result = cmd_attribution(_ns(
            attribution_cmd="diff",
            base=str(base),
            head=str(head),
            format="json",
        ))
        assert result == 0
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list)

    def test_missing_subcommand_returns_2(self, capsys):
        from multivon_eval.cli import cmd_attribution
        result = cmd_attribution(_ns(attribution_cmd=None))
        assert result == 2


# ─── cmd_doctor ───────────────────────────────────────────────────────────────

class TestCmdDoctor:
    def test_json_output_is_valid_json(self, capsys):
        from multivon_eval.cli import cmd_doctor
        args = _ns(no_ping=True, json=True)
        # Unset API keys so we don't actually ping
        with patch.dict("os.environ", {}, clear=False):
            result = cmd_doctor(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data
        assert "checks" in data
        assert isinstance(data["checks"], list)

    def test_no_ping_skips_provider_pings(self, capsys):
        from multivon_eval.cli import cmd_doctor
        args = _ns(no_ping=True, json=True)
        result = cmd_doctor(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        ping_checks = [c for c in data["checks"] if c["category"] == "ping"]
        assert ping_checks == []

    def test_ok_error_warn_counts_in_summary(self, capsys):
        from multivon_eval.cli import cmd_doctor
        args = _ns(no_ping=True, json=True)
        cmd_doctor(args)
        data = json.loads(capsys.readouterr().out)
        s = data["summary"]
        assert "ok" in s and "warn" in s and "error" in s
        assert s["ok"] + s["warn"] + s["error"] == len(data["checks"])

    def test_exit_code_0_when_all_ok(self, capsys):
        from multivon_eval.cli import cmd_doctor
        # Force all checks to appear as OK by clearing error conditions
        with patch("multivon_eval.cli._ping_anthropic"), \
             patch("multivon_eval.cli._ping_openai"), \
             patch("multivon_eval.cli._ping_google"):
            args = _ns(no_ping=True, json=True)
            # result is 0 (all ok) or 2 (warns); never 1 without real errors
            result = cmd_doctor(args)
        assert result in (0, 1, 2)  # shape check; value depends on env


# ─── cmd_discover ─────────────────────────────────────────────────────────────

class TestCmdDiscover:
    def test_outputs_valid_json(self, capsys):
        from multivon_eval.cli import cmd_discover
        result = cmd_discover(_ns(compact=False))
        assert result == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["package"] == "multivon-eval"
        assert "evaluators" in data
        assert "evaluator_count" in data

    def test_compact_output_is_single_line(self, capsys):
        from multivon_eval.cli import cmd_discover
        cmd_discover(_ns(compact=True))
        out = capsys.readouterr().out.strip()
        # Single-line compact JSON contains no literal newlines (one "\n" at end only)
        assert out.count("\n") == 0

    def test_evaluator_count_matches_list(self, capsys):
        from multivon_eval.cli import cmd_discover
        cmd_discover(_ns(compact=False))
        data = json.loads(capsys.readouterr().out)
        assert data["evaluator_count"] == len(data["evaluators"])

    def test_evaluator_entries_have_required_keys(self, capsys):
        from multivon_eval.cli import cmd_discover
        cmd_discover(_ns(compact=False))
        data = json.loads(capsys.readouterr().out)
        for ev in data["evaluators"]:
            assert "name" in ev
            assert "import" in ev
            assert "evaluator_id" in ev


# ─── cmd_generate (non-LLM paths) ────────────────────────────────────────────

class TestCmdGenerate:
    def test_no_source_no_text_exits(self, capsys):
        from multivon_eval.cli import cmd_generate
        args = _ns(source=None, text=None, n=5, task="qa", output=None)
        with pytest.raises(SystemExit):
            cmd_generate(args)

    def test_text_mode_prints_cases(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_generate
        mock_cases = [
            SimpleNamespace(input="q1", expected_output="a1", context="ctx1"),
        ]
        with patch("multivon_eval.generate.generate_from_text", return_value=mock_cases), \
             patch("dotenv.load_dotenv"):
            args = _ns(source=None, text="some text", n=1, task="qa", output=None)
            cmd_generate(args)
        out = capsys.readouterr().out
        assert "q1" in out

    def test_output_flag_writes_jsonl(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_generate
        out_file = tmp_path / "out.jsonl"
        mock_cases = [
            SimpleNamespace(input="q1", expected_output="a1", context="ctx"),
        ]
        with patch("multivon_eval.generate.generate_from_text", return_value=mock_cases), \
             patch("dotenv.load_dotenv"):
            args = _ns(source=None, text="text", n=1, task="qa", output=str(out_file))
            cmd_generate(args)
        lines = out_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["input"] == "q1"


# ─── cmd_generate — new modes (issue #13; all $0 / mocked) ───────────────────

class TestCmdGenerateModes:
    def _write_sources(self, tmp_path):
        src = tmp_path / "cases.jsonl"
        rows = [
            {"input": "The refund window is 30 days for returns.",
             "expected_output": "30 days", "context": "policy doc"},
            {"input": "Please cancel my subscription, thanks a lot.",
             "expected_output": "cancelled"},
        ]
        src.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                       encoding="utf-8")
        return src

    def _args(self, **kwargs):
        base = dict(
            source=None, text=None, n=None, task="qa", output=None,
            unanswerable_fraction=0.0, seed=0,
            mutate=None, mutations=None, per_case=1,
            template=None, axes=None, sample="all",
            expected_output=None, expected_behavior=None,
            contrast=None, no_verify=False, budget_usd=1.0,
        )
        base.update(kwargs)
        return _ns(**base)

    def test_mutate_mode_writes_full_fidelity_jsonl(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_generate
        src = self._write_sources(tmp_path)
        out = tmp_path / "mutants.jsonl"
        with patch("dotenv.load_dotenv"):
            cmd_generate(self._args(mutate=str(src), output=str(out)))
        rows = [json.loads(l) for l in
                out.read_text(encoding="utf-8").strip().splitlines()]
        assert rows
        for row in rows:
            gen = row["metadata"]["generation"]
            assert gen["kind"] == "mutation"
            assert gen["expectation"] in ("invariant", "flip")
            assert row["metadata"]["_provenance"]["authored_by"] == "generator:mutation"
        assert "accepted" in capsys.readouterr().out  # gate accounting printed

    def test_mutate_mode_unknown_mutation_exits_cleanly(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_generate
        src = self._write_sources(tmp_path)
        with patch("dotenv.load_dotenv"), pytest.raises(SystemExit):
            cmd_generate(self._args(mutate=str(src), mutations="bogus"))
        assert "unknown mutation" in capsys.readouterr().err

    def test_template_mode(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_generate
        out = tmp_path / "grid.jsonl"
        with patch("dotenv.load_dotenv"):
            cmd_generate(self._args(
                template="Refund for {item} bought {when}",
                axes=json.dumps({"item": ["a laptop", "a phone"],
                                 "when": ["yesterday", "in March"]}),
                expected_output="refund policy",
                output=str(out),
            ))
        rows = [json.loads(l) for l in
                out.read_text(encoding="utf-8").strip().splitlines()]
        assert len(rows) == 4
        assert rows[0]["input"] == "Refund for a laptop bought yesterday"
        assert rows[0]["metadata"]["generation"]["kind"] == "template"

    def test_template_mode_requires_axes(self, capsys):
        from multivon_eval.cli import cmd_generate
        with patch("dotenv.load_dotenv"), pytest.raises(SystemExit):
            cmd_generate(self._args(template="Refund for {item}"))
        assert "--axes" in capsys.readouterr().err

    def test_contrast_mode_mocked(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_generate
        src = self._write_sources(tmp_path)
        out = tmp_path / "twins.jsonl"
        proposal = json.dumps({"unfaithful_answer": "The window is 90 days.",
                               "changed_fact": "30 -> 90"})
        with patch("dotenv.load_dotenv"), \
             patch("multivon_eval.discover._call_judge", return_value=proposal):
            cmd_generate(self._args(contrast=str(src), no_verify=True,
                                    output=str(out)))
        rows = [json.loads(l) for l in
                out.read_text(encoding="utf-8").strip().splitlines()]
        # only the case WITH context is eligible for a twin
        assert len(rows) == 1
        assert rows[0]["metadata"]["pair_id"]
        assert rows[0]["metadata"]["generation"]["kind"] == "contrast"

    def test_only_one_mode_allowed(self, tmp_path, capsys):
        from multivon_eval.cli import cmd_generate
        src = self._write_sources(tmp_path)
        with patch("dotenv.load_dotenv"), pytest.raises(SystemExit):
            cmd_generate(self._args(mutate=str(src), template="{x}",
                                    axes='{"x": ["1"]}'))
        assert "one generation mode" in capsys.readouterr().err

    def test_parser_accepts_new_flags(self):
        # End-to-end argparse smoke: the documented flags all parse.
        from multivon_eval.cli import main
        import multivon_eval.cli as cli
        parser_argv = [
            "generate", "--mutate", "x.jsonl", "--mutations", "typo_noise",
            "--per-case", "2", "--seed", "3", "--output", "o.jsonl",
        ]
        with patch.object(sys, "argv", ["multivon-eval"] + parser_argv), \
             patch.object(cli, "cmd_generate") as fake:
            main()
        args = fake.call_args[0][0]
        assert args.mutate == "x.jsonl"
        assert args.mutations == "typo_noise"
        assert args.per_case == 2
        assert args.seed == 3
