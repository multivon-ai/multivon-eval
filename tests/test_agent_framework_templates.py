"""D16: end-to-end smoke tests for the agent-langgraph and
agent-openai-sdk templates.

These tests prove the persona walkthrough findings are fixed:

  - Templates SCAFFOLD without errors via the CLI.
  - eval.py IMPORTS cleanly without an API key set — so a user can
    read the file before they've signed up for an API.
  - requirements.txt + .env.example are produced.

We do NOT run the eval (that needs a real LLM). The optional live
test file (test_live_agent_frameworks.py) handles that under
``MULTIVON_EVAL_LIVE=1``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from multivon_eval.templates import TEMPLATES, list_templates


# ─────────────────────────────────────────────────────────────────────────────
# Both templates are registered
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_langgraph_is_registered():
    assert "agent-langgraph" in TEMPLATES
    files = TEMPLATES["agent-langgraph"]
    assert "eval.py" in files and "README.md" in files
    assert "requirements.txt" in files and ".env.example" in files


def test_agent_openai_sdk_is_registered():
    assert "agent-openai-sdk" in TEMPLATES
    files = TEMPLATES["agent-openai-sdk"]
    assert "eval.py" in files and "README.md" in files
    assert "requirements.txt" in files and ".env.example" in files


def test_list_templates_includes_new_templates():
    names = list_templates()
    assert "agent-langgraph" in names
    assert "agent-openai-sdk" in names
    # Display order: the framework-specific templates come right after
    # the generic ``agent`` so the user sees them as alternatives.
    assert names.index("agent") < names.index("agent-langgraph") < names.index("agent-openai-sdk")


# ─────────────────────────────────────────────────────────────────────────────
# CLI scaffold produces a usable project
# ─────────────────────────────────────────────────────────────────────────────

def test_cli_scaffolds_agent_langgraph(tmp_path):
    target = tmp_path / "lg_proj"
    rc = subprocess.run(
        [sys.executable, "-m", "multivon_eval", "init",
         "-t", "agent-langgraph", "-d", str(target)],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, f"scaffold failed: {rc.stderr}"
    assert (target / "eval.py").exists()
    assert (target / "README.md").exists()
    # requirements.txt names the right extra
    reqs = (target / "requirements.txt").read_text()
    assert "multivon-eval[langgraph]" in reqs


def test_cli_scaffolds_agent_openai_sdk(tmp_path):
    target = tmp_path / "oai_proj"
    rc = subprocess.run(
        [sys.executable, "-m", "multivon_eval", "init",
         "-t", "agent-openai-sdk", "-d", str(target)],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, f"scaffold failed: {rc.stderr}"
    assert (target / "eval.py").exists()
    reqs = (target / "requirements.txt").read_text()
    assert "multivon-eval[openai-agents]" in reqs


# ─────────────────────────────────────────────────────────────────────────────
# Persona finding: eval.py must IMPORT without an API key
# (was a hard-fail in OpenAI Agents SDK template; codex D16 cycle 3 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _can_import(file_path: Path, env: dict) -> tuple[bool, str]:
    """Try to `import` the eval.py file's module body without
    executing the __main__ block. Returns (success, stderr)."""
    rc = subprocess.run(
        [sys.executable, "-c",
         f"import importlib.util, sys; "
         f"spec = importlib.util.spec_from_file_location('eval_under_test', {str(file_path)!r}); "
         f"mod = importlib.util.module_from_spec(spec); "
         f"spec.loader.exec_module(mod); print('OK')"],
        capture_output=True, text=True, env=env, timeout=20,
    )
    return rc.returncode == 0, rc.stderr


@pytest.mark.skipif(
    "openai-agents" not in (subprocess.run(
        [sys.executable, "-m", "pip", "list"], capture_output=True, text=True
    ).stdout.lower()),
    reason="openai-agents SDK not installed in this environment",
)
def test_openai_sdk_template_imports_without_api_key(tmp_path):
    """Codex D16 cycle 3 persona B finding: the OpenAI Agents SDK
    template used to raise RuntimeError at IMPORT time when
    OPENAI_API_KEY was missing — preventing the user from even
    reading eval.py to learn how it works. The check must now be
    deferred to __main__."""
    target = tmp_path / "oai_import_test"
    subprocess.run(
        [sys.executable, "-m", "multivon_eval", "init",
         "-t", "agent-openai-sdk", "-d", str(target)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    ok, err = _can_import(target / "eval.py", env)
    assert ok, (
        "eval.py raised on import without OPENAI_API_KEY — the check "
        f"must be deferred to runtime. stderr:\n{err}"
    )


def test_openai_sdk_template_source_has_no_module_body_key_check():
    """Codex D16 cycle 4 ISSUE 5: the import-without-key test skips
    when the openai-agents SDK isn't installed, so a regression that
    moves the key check back to module body could slip past CI on
    envs without the extra.

    Lock the contract via AST inspection (robust to docstrings /
    comments / nested raises): no ``raise`` statement may live at
    module level. The check belongs inside ``_check_key()`` or the
    ``__main__`` block.
    """
    import ast
    eval_src = TEMPLATES["agent-openai-sdk"]["eval.py"]
    tree = ast.parse(eval_src)

    # Find any top-level ``raise`` (module.body, not nested in defs/ifs/...)
    top_level_raises = [
        node for node in tree.body if isinstance(node, ast.Raise)
    ]
    assert top_level_raises == [], (
        f"template eval.py raises at module body — should be deferred to "
        f"_check_key() or __main__. Found {len(top_level_raises)} raise(s)."
    )

    # The check function must exist and be called inside __main__.
    func_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "_check_key" in func_names, "template missing _check_key() function"

    # Look for `if __name__ == "__main__":` and verify _check_key() is invoked inside.
    main_blocks = [
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in node.test.comparators)
    ]
    assert main_blocks, "template has no __main__ guard"
    main_src = "\n".join(ast.unparse(node) for node in main_blocks[0].body)
    assert "_check_key()" in main_src, (
        "_check_key() must be invoked inside the __main__ block"
    )


# ─────────────────────────────────────────────────────────────────────────────
# README updates from persona C: framework templates are discoverable
# ─────────────────────────────────────────────────────────────────────────────

def test_readme_pick_your_path_includes_both_new_templates():
    """Persona C finding + codex cycle 4 ISSUE 6 strengthening:
    "Pick your path" table must show the exact init command for both
    new templates, not just a stray mention. Discoverability is the
    whole point."""
    readme = Path(__file__).parent.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    # The actual init command must appear so a user can copy-paste.
    assert "multivon-eval init -t agent-langgraph" in text, (
        "Pick-your-path table must show the exact `init -t agent-langgraph` command"
    )
    assert "multivon-eval init -t agent-openai-sdk" in text, (
        "Pick-your-path table must show the exact `init -t agent-openai-sdk` command"
    )
    # And the path-selection table itself must mention LangGraph and
    # OpenAI Agents SDK by name.
    pick_section = text.split("Pick your path")[-1].split("##")[0]
    assert "LangGraph" in pick_section, "Pick-your-path missing LangGraph row"
    assert "OpenAI Agents SDK" in pick_section, "Pick-your-path missing OpenAI Agents SDK row"


def test_langgraph_template_readme_shows_callback_wiring_snippet():
    """Codex cycle 4 ISSUE 7: substring 'callbacks' was too loose —
    lock the actual snippet a user must copy.

    The template README must show the `config={"callbacks": kwargs.get(...)}`
    pattern in a code block, because that's the wiring that decides
    whether the trace captures anything."""
    lg_readme = TEMPLATES["agent-langgraph"]["README.md"]
    # The specific snippet (not just the word "callbacks").
    assert 'kwargs.get("callbacks"' in lg_readme, (
        "agent-langgraph README must show `kwargs.get(\"callbacks\", [])`"
    )
    assert "config={" in lg_readme, (
        "agent-langgraph README must show config={\"callbacks\": ...} usage"
    )


def test_openai_sdk_template_readme_shows_capture_wiring_snippet():
    """Codex cycle 4 ISSUE 7: lock the `TRACER.capture(result)` snippet,
    not just the word 'capture'."""
    oai_readme = TEMPLATES["agent-openai-sdk"]["README.md"]
    assert "TRACER.capture(result)" in oai_readme, (
        "agent-openai-sdk README must show the literal TRACER.capture(result) call"
    )
    # And the surrounding model_fn pattern.
    assert "Runner.run_sync" in oai_readme, (
        "agent-openai-sdk README must show Runner.run_sync(...) usage"
    )


def test_template_readme_has_migration_note():
    """Persona C finding: existing `agent` template users need a
    'how to migrate' tip."""
    lg_readme = TEMPLATES["agent-langgraph"]["README.md"]
    assert "migrat" in lg_readme.lower(), "agent-langgraph README needs a migration section"

    oai_readme = TEMPLATES["agent-openai-sdk"]["README.md"]
    assert "migrat" in oai_readme.lower(), "agent-openai-sdk README needs a migration section"
