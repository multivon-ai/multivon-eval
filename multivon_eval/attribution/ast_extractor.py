"""AST-based extraction of prompt literals from SDK call sites.

Detects three SDK call shapes (kwarg-only — per the v1 adversarial-fix
discipline that drops fuzzy name-regex capture entirely)::

    anthropic.messages.create(system=<literal>, messages=[{"role": ..., "content": <literal>}, ...])
    client.messages.create(...)              # any object.messages.create
    openai.chat.completions.create(messages=[...])
    client.chat.completions.create(...)
    litellm.completion(messages=[...])
    litellm.acompletion(messages=[...])

The matcher is method-name-based, not type-inferred — so it will catch any
call ending in `.messages.create(...)` or `.chat.completions.create(...)`.
That trades some recall (an obscure SDK with the same method name will be
captured) for simplicity. False matches without a `system` kwarg or
`messages` kwarg are silently dropped.

Literals captured:
    - Plain string literals: `system="..."`.
    - f-string literals that contain ZERO runtime interpolation: treated as
      literal (their string content is fully known at parse time).
    - All other expressions (Name, Attribute, runtime f-strings, joined
      strings with variables, etc.) are recorded as PromptRecord with
      is_dynamic=True and a placeholder text so the count is right and the
      gap is visible in the eventual PR comment.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Iterator, Optional

from .fingerprint import fingerprint_text
from .schema import PromptRecord


# ── SDK call-site detection ────────────────────────────────────────────


def _attr_chain(node: ast.AST) -> list[str]:
    """Return the attribute-access chain leading up to a call's func, in source order.

    For ``client.messages.create``: ``["client", "messages", "create"]``.
    For ``anthropic.Anthropic().messages.create``: ``["messages", "create"]`` —
    we cannot traverse through the Call, so the chain stops at the first
    non-Attribute ancestor. Detection downstream is suffix-based, which
    accepts both shapes.
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    # If cur is a Call / Subscript / etc., we leave it out and rely on the
    # suffix matcher below.
    return list(reversed(parts))


def _identify_sdk(call: ast.Call) -> Optional[tuple[str, str]]:
    """Return (sdk_name, call_site_label) for a recognized SDK call, else None.

    Matching is suffix-based on the trailing method/attribute names, so it
    accepts both ``client.messages.create`` and ``anthropic.Anthropic().messages.create``.
    """
    chain = _attr_chain(call.func)
    if len(chain) < 2:
        return None
    # Anthropic: any expression ending in .messages.create(...)
    if chain[-2:] == ["messages", "create"]:
        return ("anthropic", "messages.create")
    # OpenAI: any expression ending in .chat.completions.create(...)
    if len(chain) >= 3 and chain[-3:] == ["chat", "completions", "create"]:
        return ("openai", "chat.completions.create")
    # LiteLLM: litellm.completion / litellm.acompletion (bare or via attribute).
    if chain[-2] == "litellm" and chain[-1] in ("completion", "acompletion"):
        return ("litellm", chain[-1])
    return None


# ── Literal extraction ────────────────────────────────────────────────


def _try_constant_str(node: ast.AST) -> Optional[str]:
    """Return the string value of a string Constant, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _try_pure_literal_fstring(node: ast.AST) -> Optional[str]:
    """If node is an f-string with ZERO runtime parts, return the resolved string.

    Catches things like f"hello world" (which the AST encodes as a JoinedStr with
    a single FormattedValue-free Constant child). Returns None for any f-string
    with FormattedValue components (runtime interpolation).
    """
    if not isinstance(node, ast.JoinedStr):
        return None
    parts: list[str] = []
    for child in node.values:
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            parts.append(child.value)
        else:
            return None  # has a FormattedValue or other runtime part
    return "".join(parts)


def _resolve_literal(node: ast.AST) -> tuple[Optional[str], bool]:
    """Return (text, is_dynamic).

    If the expression resolves to a known string at parse time, text is the
    string and is_dynamic is False. Otherwise text is None and is_dynamic is
    True; the caller can synthesize a placeholder for the PromptRecord.
    """
    s = _try_constant_str(node)
    if s is not None:
        return (s, False)
    s = _try_pure_literal_fstring(node)
    if s is not None:
        return (s, False)
    return (None, True)


def _placeholder_for_dynamic(node: ast.AST) -> str:
    """A stable placeholder string for dynamic prompt values.

    Same expression shape → same placeholder, so a PR that doesn't touch the
    dynamic call site still fingerprints to the same value.
    """
    return f"<dynamic:{type(node).__name__}>"


# ── Walking calls ─────────────────────────────────────────────────────


def _find_kwarg(call: ast.Call, name: str) -> Optional[ast.AST]:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _qualname_stack(stack: list[ast.AST]) -> str:
    parts: list[str] = []
    for node in stack:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append(node.name)
    return ".".join(parts) if parts else "<module>"


def _extract_messages_list(messages_node: ast.AST) -> Iterator[tuple[int, str, ast.AST]]:
    """Yield (position, role, content_node) for each dict in a messages= list.

    Skips entries whose role or content can't be resolved at parse time.
    """
    if not isinstance(messages_node, ast.List):
        return
    for pos, item in enumerate(messages_node.elts):
        if not isinstance(item, ast.Dict):
            continue
        role_text: Optional[str] = None
        content_node: Optional[ast.AST] = None
        for k, v in zip(item.keys, item.values):
            key = _try_constant_str(k) if k is not None else None
            if key == "role":
                role_text = _try_constant_str(v)
            elif key == "content":
                content_node = v
        if role_text is not None and content_node is not None:
            yield (pos, role_text, content_node)


def _build_record(
    *,
    file_path: str,
    line: int,
    sdk: str,
    call_site: str,
    role: str,
    role_position: int,
    qualname: str,
    node: ast.AST,
) -> PromptRecord:
    text, is_dynamic = _resolve_literal(node)
    if text is None:
        text = _placeholder_for_dynamic(node)
    return PromptRecord(
        file_path=file_path,
        line=line,
        sdk=sdk,
        call_site=call_site,
        role=role,
        role_position=role_position,
        qualname=qualname,
        text=text,
        is_dynamic=is_dynamic,
        fingerprint=fingerprint_text(text),
    )


def scan_file(file_path: str, repo_root: Optional[str] = None) -> list[PromptRecord]:
    """Return all PromptRecords discoverable in a single .py file.

    file_path may be absolute or relative to repo_root. The PromptRecord
    file_path is stored relative to repo_root (or unchanged if no root given).
    """
    abs_path = Path(file_path).resolve()
    if repo_root is not None:
        try:
            rel_path = str(abs_path.relative_to(Path(repo_root).resolve()))
        except ValueError:
            rel_path = str(abs_path)
    else:
        rel_path = str(abs_path)

    try:
        source = abs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source, filename=str(abs_path))
    except SyntaxError:
        return []

    records: list[PromptRecord] = []

    # Walk with a stack to compute qualname.
    def visit(node: ast.AST, stack: list[ast.AST]) -> None:
        push = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if push:
            stack.append(node)
        if isinstance(node, ast.Call):
            ident = _identify_sdk(node)
            if ident is not None:
                sdk, call_site = ident
                qualname = _qualname_stack(stack)
                # system= kwarg
                system_node = _find_kwarg(node, "system")
                if system_node is not None:
                    records.append(_build_record(
                        file_path=rel_path,
                        line=getattr(system_node, "lineno", getattr(node, "lineno", 0)),
                        sdk=sdk, call_site=call_site,
                        role="system", role_position=-1,
                        qualname=qualname, node=system_node,
                    ))
                # messages= kwarg
                messages_node = _find_kwarg(node, "messages")
                if messages_node is not None:
                    for pos, role, content_node in _extract_messages_list(messages_node):
                        records.append(_build_record(
                            file_path=rel_path,
                            line=getattr(content_node, "lineno", getattr(node, "lineno", 0)),
                            sdk=sdk, call_site=call_site,
                            role=role, role_position=pos,
                            qualname=qualname, node=content_node,
                        ))
        for child in ast.iter_child_nodes(node):
            visit(child, stack)
        if push:
            stack.pop()

    visit(tree, [])
    return records


# ── Repo-level scan ────────────────────────────────────────────────────


# Default directories to skip — virtualenvs, caches, build artifacts.
DEFAULT_IGNORE_DIRS = frozenset({
    ".venv", "venv", "env", ".env",
    "node_modules", ".git", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "build", "dist", ".tox", ".nox",
    "site-packages",
})


def scan(repo_root: str, ignore_dirs: Optional[frozenset[str]] = None) -> list[PromptRecord]:
    """Walk repo_root and return all PromptRecords across every .py file.

    Skips DEFAULT_IGNORE_DIRS (plus any extra names passed via ignore_dirs).
    Records are stable-sorted by (file_path, line, role_position) for
    deterministic output.
    """
    skip = (ignore_dirs or frozenset()) | DEFAULT_IGNORE_DIRS
    root = Path(repo_root).resolve()
    all_records: list[PromptRecord] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # mutate dirnames in place to prune the walk
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            all_records.extend(scan_file(full, repo_root=str(root)))
    all_records.sort(key=lambda r: (r.file_path, r.line, r.role_position))
    return all_records
