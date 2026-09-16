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


def test_readme_leads_with_current_release_and_public_surfaces() -> None:
    readme = _read("README.md")
    assert f"Current release: {multivon_eval.__version__}" in readme
    assert f"Current release — {multivon_eval.__version__}" in readme
    assert "September 17, 2026" in readme
    assert "eval-framework-benchmark" not in readme
    assert len(readme.splitlines()) < 350
    assert readme.index("## Start in 30 seconds") < readme.index("## Why use it")
    assert readme.index("## Why use it") < readme.index("## Pick your path")
    assert "## Earlier release highlights" not in readme


def test_readme_relative_links_resolve() -> None:
    readme = _read("README.md")
    links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", readme)
    missing = []
    for href in links:
        if href.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative = href.split("#", 1)[0]
        if relative and not (ROOT / relative).exists():
            missing.append(href)
    assert not missing, f"README links missing local targets: {missing}"


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


def test_scaffolded_quickstart_validates_every_case() -> None:
    from multivon_eval.templates import TEMPLATES

    source = TEMPLATES["quickstart"]["eval.py"]
    assert source.count("expected_output=") == 3


def test_doctor_exit_codes_are_explained() -> None:
    docs = _read("README.md") + _read("docs/quickstart.mdx")
    assert "exits 0" in docs
    assert "2 when it finds warnings" in docs
    assert "1 when it finds an error" in docs


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


def _python_blocks(text: str):
    import textwrap

    for match in re.finditer(
        r"^(`{3,})(?:python|py)[^\n]*\n(.*?)^\1\s*$", text, re.MULTILINE | re.DOTALL
    ):
        yield textwrap.dedent(match.group(2))


def test_documented_python_imports_and_call_keywords_exist() -> None:
    import ast
    import importlib

    known_receivers = {
        "suite": multivon_eval.EvalSuite,
        "report": multivon_eval.EvalReport,
        "case": multivon_eval.EvalCase,
        "exp": multivon_eval.Experiment,
    }
    for path in sorted((ROOT / "docs").rglob("*.mdx")):
        for source in _python_blocks(path.read_text()):
            tree = ast.parse(source, filename=str(path))
            imported = {}
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith("multivon_eval"):
                    continue
                module = importlib.import_module(node.module)
                for alias in node.names:
                    imported[alias.asname or alias.name] = getattr(module, alias.name)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = None
                if isinstance(node.func, ast.Name):
                    target = imported.get(node.func.id)
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    receiver = known_receivers.get(node.func.value.id)
                    if receiver:
                        assert hasattr(receiver, node.func.attr), (path, node.func.attr)
                        target = getattr(receiver, node.func.attr)
                if not callable(target):
                    continue
                parameters = inspect.signature(target).parameters
                if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
                    continue
                for keyword in node.keywords:
                    if keyword.arg:
                        assert keyword.arg in parameters, (path, keyword.arg)


def test_docs_navigation_and_internal_links_resolve() -> None:
    import json

    docs = ROOT / "docs"

    def check_navigation(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "pages":
                    for page in child:
                        if isinstance(page, str):
                            assert (docs / f"{page}.mdx").exists(), page
                        else:
                            check_navigation(page)
                else:
                    check_navigation(child)
        elif isinstance(value, list):
            for child in value:
                check_navigation(child)

    check_navigation(json.loads((docs / "docs.json").read_text()))
    for path in docs.rglob("*.mdx"):
        for href in re.findall(r'(?:\]\(|href=")(/[^)#"\s]+)', path.read_text()):
            target = docs / href.lstrip("/")
            assert target.exists() or target.with_suffix(".mdx").exists(), (path, href)


def test_report_reference_fields_exist() -> None:
    from dataclasses import fields

    from multivon_eval.costs import Costs
    from multivon_eval.result import CaseResult, EvalReport, EvalResult

    text = _read("docs/reference/eval-report.mdx")
    sections = [
        ("## Quick reference", "## Common gotchas", EvalReport),
        ("## `CaseResult` shape", "## `EvalResult` shape", CaseResult),
        ("## `EvalResult` shape", "## `Costs` shape", EvalResult),
        ("## `Costs` shape", "## CI examples", Costs),
    ]
    for start, end, cls in sections:
        section = text.split(start, 1)[1].split(end, 1)[0]
        known = set(dir(cls)) | {field.name for field in fields(cls)}
        for field in re.findall(r"^\| `([a-z_]+)(?:\([^`]*\))?` \|", section, re.MULTILINE):
            assert field in known, (cls.__name__, field)


def test_offline_documented_examples_execute(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for path, end in [
        ("README.md", "## Why use it"),
        ("docs/quickstart.mdx", "## Propose a suite"),
        ("docs/guides/task-success.mdx", None),
    ]:
        text = _read(path)
        if end:
            text = text.split(end, 1)[0]
        blocks = list(_python_blocks(text))
        assert blocks, path
        for source in blocks:
            namespace = {}
            exec(compile(source, path, "exec"), namespace)  # noqa: S102 — selected repository examples
            report = namespace["report"]
            assert report.evaluated > 0 and report.pass_rate == 1
            assert report.errors == 0 and report.skipped == 0


def test_reference_comparison_example_uses_baseline_first(tmp_path) -> None:
    import json

    from multivon_eval.result import CaseResult, EvalReport, EvalResult

    prev = EvalReport("baseline", [CaseResult("same input", "bad", [EvalResult("check", 0, False)])])
    current = EvalReport("proposal", [CaseResult("same input", "good", [EvalResult("check", 1, True)])])
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(prev.to_json())
    text = _read("docs/reference/eval-report.mdx").split("Compare vs a baseline run:", 1)[1]
    source = next(_python_blocks(text)).replace('Path("baseline.json")', 'Path(baseline_path)')
    namespace = {"EvalReport": EvalReport, "json": json, "Path": Path,
                 "baseline_path": baseline_path, "report": current}
    exec(compile(source, "comparison example", "exec"), namespace)  # noqa: S102
    assert namespace["delta"].pass_rate_delta == 1.0
