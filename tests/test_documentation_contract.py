"""Offline checks for release facts and executable commands in the docs."""
from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

import multivon_eval
from multivon_eval.evaluators.base import Evaluator


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _help(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "multivon_eval", *args, "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_readme_commands_match_cli_contract() -> None:
    readme = _read("README.md")
    assert "view results.json [--open]" not in readme
    assert 'generate_from_file("docs/")' not in readme
    assert "generate --from docs/ --n" not in readme
    assert "staleness . [baseline|stamp]" not in readme

    assert "--no-browser" in _help("view")
    assert "--open" not in _help("view")
    assert "--from" in _help("generate")
    assert "baseline" in _help("staleness")


def test_quickstart_imports_every_used_evaluator() -> None:
    quickstart = _read("docs/quickstart.mdx")
    manual_block = quickstart.split("## Option B", 1)[1].split(
        "## Load cases", 1
    )[0]
    import_line = next(
        line for line in manual_block.splitlines()
        if line.startswith("from multivon_eval import")
    )
    assert "Faithfulness" in import_line


def test_evaluator_count_claim_matches_public_catalog() -> None:
    evaluator_count = sum(
        1
        for name in dir(multivon_eval)
        if inspect.isclass(getattr(multivon_eval, name))
        and issubclass(getattr(multivon_eval, name), Evaluator)
        and getattr(multivon_eval, name) is not Evaluator
    )
    assert evaluator_count == 44
    assert "Evaluators — 44 across 7 tiers" in _read("README.md")


def test_tool_ordering_language_matches_subsequence_behavior() -> None:
    docs = "\n".join(
        _read(path)
        for path in (
            "README.md",
            "docs/evaluators/agent.mdx",
            "docs/guides/agent-trace.mdx",
        )
    )
    assert "ordered subsequence" in docs
    assert "positional match instead of set match" not in docs
    assert "must match in exact order" not in docs


def test_bundled_mcp_reference_covers_all_22_tools() -> None:
    tool_reference = _read("docs/mcp/tool-reference.mdx")
    headings = set(re.findall(r"^### `([^`]+)`$", tool_reference, re.MULTILINE))
    expected = {
        "eval_discover", "pdfhell_make", "pdfhell_run", "eval_audit_pack",
        "eval_faithfulness", "eval_hallucination", "eval_relevance",
        "eval_answer_accuracy", "eval_context_precision", "eval_context_recall",
        "eval_toxicity", "eval_bias", "eval_pii_detection",
        "eval_schema_compliance", "eval_tool_call_accuracy",
        "eval_vqa_faithfulness", "eval_document_grounding", "eval_g_eval",
        "eval_custom_rubric", "eval_compare_runs", "eval_generate_cases",
        "eval_ingest_trace",
    }
    assert headings == expected
    assert "multivon-mcp 0.3.2 exposes **22 tools**" in tool_reference


def test_bundled_pdfhell_reference_matches_061_registry() -> None:
    quickstart = _read("docs/pdfhell/quickstart.mdx")
    traps = _read("docs/pdfhell/trap-families.mdx")
    cli = _read("docs/pdfhell/cli-reference.mdx")
    assert "PDF Hell 0.6.1" in quickstart
    family_table = traps.split("## Reproduce and inspect", 1)[0]
    assert len(re.findall(r"^\| `[^`]+` \|", family_table, re.MULTILINE)) == 17
    for suite_hash in (
        "8cb2f6ab", "8ad87b8d", "a0385d55", "6407e7bb",
        "9707abe1", "a451ce10",
    ):
        assert suite_hash in traps
    for flag in (
        "--model", "--suite", "--cases-dir", "--workers", "--pixels",
        "--dpi", "--quiet", "--out", "--junit", "--audit-pack",
        "--fail-threshold",
    ):
        assert flag in cli


def test_bootstrap_docs_account_for_all_artifacts() -> None:
    for path in (
        "docs/skills/eval-bootstrap.mdx",
        "multivon_eval/_skills/eval-bootstrap/SKILL.md",
    ):
        text = _read(path)
        assert "prompt_baseline.json" in text
        assert "four" in text.lower()
