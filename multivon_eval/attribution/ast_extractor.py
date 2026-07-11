"""AST-based extraction of prompt literals from SDK call sites.

Detects three SDK call shapes (kwarg-only; fuzzy name-regex capture is
deliberately not attempted)::

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
    - Scanner v2: a bare Name that resolves — one hop, same file, module
      scope — to `X = "literal"` is treated as that literal (is_dynamic=False).
      Conditionally-reassigned names, function-scope names, names declared
      `global` anywhere, cross-module imports, and X = Y = "..." chains all
      stay dynamic.
    - All other expressions (Attribute, runtime f-strings, joined
      strings with variables, etc.) are recorded as PromptRecord with
      is_dynamic=True and a placeholder text so the count is right and the
      gap is visible in the eventual PR comment.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Iterator, Optional

from .fingerprint import fingerprint_text, loose_fingerprint_text
from .schema import PromptRecord


# Bumped whenever extraction semantics change in a way that shifts what is
# statically resolvable (v2: one-hop module-level constant resolution +
# loose_fingerprint; v4: NFC-normalized fingerprints + match-statement
# capture patterns disqualify module constants). A prompt_baseline.json
# written by an older scanner triggers a "rescan recommended" warning
# instead of fake drift.
SCANNER_VERSION = 4


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


def _identify_sdk(
    call: ast.Call,
    litellm_aliases: Optional[dict[str, str]] = None,
) -> Optional[tuple[str, str]]:
    """Return (sdk_name, call_site_label) for a recognized SDK call, else None.

    Matching is suffix-based on the trailing method/attribute names, so it
    accepts both ``client.messages.create`` and ``anthropic.Anthropic().messages.create``.

    ``litellm_aliases`` maps module-level imported names to their litellm
    originals (``from litellm import acompletion as ac`` → {"ac": "acompletion"})
    so bare aliased calls — the dominant shape in real repos like pr-agent —
    are detected.
    """
    chain = _attr_chain(call.func)
    if not chain:
        return None
    # Bare aliased litellm call: `completion(...)` after `from litellm import completion`.
    if len(chain) == 1 and litellm_aliases and chain[0] in litellm_aliases:
        return ("litellm", litellm_aliases[chain[0]])
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


def _litellm_import_aliases(tree: ast.Module) -> dict[str, str]:
    """Module-level ``from litellm import completion [as X]`` aliases.

    Only top-level imports count — a function-local import is rare enough
    that missing it is acceptable, and tracking it would need scope analysis
    for marginal gain. Star imports are ignored (cannot know what they bind).
    """
    aliases: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "litellm" and stmt.level == 0:
            for name in stmt.names:
                if name.name in ("completion", "acompletion"):
                    aliases[name.asname or name.name] = name.name
    return aliases


def _has_kwargs_unpack(call: ast.Call) -> bool:
    """True when the call passes ``**something`` — prompts exist but are
    invisible to static analysis (aider's ``litellm.completion(**kwargs)``)."""
    return any(kw.arg is None for kw in call.keywords)


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


def _resolve_literal(
    node: ast.AST,
    constants: Optional[dict[str, str]] = None,
    shadowed: frozenset[str] = frozenset(),
) -> tuple[Optional[str], bool]:
    """Return (text, is_dynamic).

    If the expression resolves to a known string at parse time, text is the
    string and is_dynamic is False. Otherwise text is None and is_dynamic is
    True; the caller can synthesize a placeholder for the PromptRecord.

    Scanner v2: a bare ``Name`` resolves iff it appears in the one-hop,
    module-scope ``constants`` map AND no enclosing function/class scope
    binds the same name (shadowing → dynamic, conservatively).
    """
    s = _try_constant_str(node)
    if s is not None:
        return (s, False)
    s = _try_pure_literal_fstring(node)
    if s is not None:
        return (s, False)
    if (
        constants
        and isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in shadowed
        and node.id in constants
    ):
        return (constants[node.id], False)
    return (None, True)


# ── Module-level constant resolution (scanner v2) ─────────────────────


def _stored_names(stmt: ast.AST) -> set[str]:
    """All names bound (Store context) under stmt, NOT descending into
    nested function/class/lambda scopes — those bindings are local."""
    out: set[str] = set()

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef, ast.Lambda)):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                    out.add(child.name)
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                out.add(child.id)
            elif isinstance(child, ast.alias):
                out.add((child.asname or child.name).split(".")[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                out.add(child.name)
            elif isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
                # `case PROMPT:` / `case [*PROMPT]:` — capture patterns bind
                # via a str field, not a Name(Store) node. Missing them lets
                # a rebound module constant read as static (false "static"
                # poisons trust; honest dynamic is preferred).
                out.add(child.name)
            walk(child)

    walk(stmt)
    return out


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """One-hop, same-file, module-scope string constants: name → text.

    Rules (deliberately conservative — false "dynamic" is honest, false
    "static" poisons every downstream verdict):
      - Only plain ``X = "literal"`` / ``X: str = "literal"`` at module
        top level (pure-literal f-strings count as literals).
      - A name assigned more than once at module level, assigned under any
        conditional/loop/try at module level, augmented, tuple-unpacked,
        or declared ``global`` anywhere in the file is disqualified.
      - One hop only: ``X = Y`` (Name → Name) does not resolve, so chains
        stay dynamic. Cross-module imports never enter the map.
    """
    constants: dict[str, str] = {}
    disqualified: set[str] = set()

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                # tuple-unpacking / chained X = Y = "..." / attribute targets
                for t in stmt.targets:
                    names = {t.id} if isinstance(t, ast.Name) else _stored_names(t)
                    disqualified |= names
                    for n in names:
                        constants.pop(n, None)
                continue
            name = stmt.targets[0].id
            text, is_dynamic = _resolve_literal(stmt.value)
            if name in constants or name in disqualified:
                disqualified.add(name)
                constants.pop(name, None)
            elif not is_dynamic and text is not None:
                constants[name] = text
            else:
                disqualified.add(name)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            text, is_dynamic = (
                _resolve_literal(stmt.value) if stmt.value is not None else (None, True)
            )
            if name in constants or name in disqualified:
                disqualified.add(name)
                constants.pop(name, None)
            elif not is_dynamic and text is not None:
                constants[name] = text
            else:
                disqualified.add(name)
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name):
                disqualified.add(stmt.target.id)
                constants.pop(stmt.target.id, None)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Different scope — module names are only at risk via `global`,
            # handled below.
            continue
        else:
            # If / For / While / Try / With / Match at module level: any
            # name bound inside is conditionally assigned → dynamic.
            names = _stored_names(stmt)
            disqualified |= names
            for n in names:
                constants.pop(n, None)

    # `global X` anywhere means a function may rebind the module name at
    # runtime — disqualify, conservatively.
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            for n in node.names:
                disqualified.add(n)
                constants.pop(n, None)

    return constants


def _scope_bound_names(scope: ast.AST) -> set[str]:
    """Names bound directly in a function/class scope (params, stores,
    imports, def/class names) — used to detect shadowing of module constants."""
    names: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = scope.args
        for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
            names.add(arg.arg)
        if a.vararg:
            names.add(a.vararg.arg)
        if a.kwarg:
            names.add(a.kwarg.arg)
        for stmt in scope.body:
            names |= _stored_names(stmt)
    elif isinstance(scope, ast.ClassDef):
        for stmt in scope.body:
            names |= _stored_names(stmt)
    return names


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
    constants: Optional[dict[str, str]] = None,
    shadowed: frozenset[str] = frozenset(),
) -> PromptRecord:
    text, is_dynamic = _resolve_literal(node, constants=constants, shadowed=shadowed)
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
        loose_fingerprint=loose_fingerprint_text(text),
    )


def scan_file(file_path: str, repo_root: Optional[str] = None) -> list[PromptRecord]:
    """Return all PromptRecords discoverable in a single .py file.

    file_path may be absolute or relative to repo_root. The PromptRecord
    file_path is stored relative to repo_root (or unchanged if no root given).

    Unscannable files (syntax/encoding errors, paths resolving outside
    repo_root) yield ``[]`` here; use :func:`scan_file_with_reason` when the
    caller needs to distinguish "no call sites" from "could not scan" —
    silent ``[]`` turns a syntax error into a false REMOVED downstream.
    """
    records, _reason = scan_file_with_reason(file_path, repo_root=repo_root)
    return records


def scan_file_with_reason(
    file_path: str, repo_root: Optional[str] = None
) -> tuple[list[PromptRecord], Optional[str]]:
    """Like :func:`scan_file` but returns ``(records, skip_reason)``.

    ``skip_reason`` is None when the file was scanned; otherwise a short
    human-readable reason ("syntax error: …", "encoding error: …",
    "resolves outside repo root") and ``records`` is empty.
    """
    abs_path = Path(file_path).resolve()
    if repo_root is not None:
        try:
            rel_path = str(abs_path.relative_to(Path(repo_root).resolve()))
        except ValueError:
            # A symlink (or junction) escaping the repo root. Recording the
            # machine-specific ABSOLUTE path would poison the baseline with
            # false REMOVED+ADDED churn on every other checkout — skip it
            # and let the caller surface the gap honestly.
            return ([], "resolves outside repo root")
    else:
        rel_path = str(abs_path)

    try:
        source = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return ([], f"encoding error: {exc.reason}")
    except OSError as exc:
        return ([], f"unreadable: {exc.strerror or exc}")

    try:
        tree = ast.parse(source, filename=str(abs_path))
    except SyntaxError as exc:
        return ([], f"syntax error: line {exc.lineno}: {exc.msg}")

    records: list[PromptRecord] = []
    constants = _module_constants(tree)
    litellm_aliases = _litellm_import_aliases(tree)
    _bound_cache: dict[int, set[str]] = {}

    def _shadowed(stack: list[ast.AST]) -> frozenset[str]:
        out: set[str] = set()
        for scope in stack:
            cached = _bound_cache.get(id(scope))
            if cached is None:
                cached = _scope_bound_names(scope)
                _bound_cache[id(scope)] = cached
            out |= cached
        return frozenset(out)

    # Walk with a stack to compute qualname.
    def visit(node: ast.AST, stack: list[ast.AST]) -> None:
        push = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if push:
            stack.append(node)
        if isinstance(node, ast.Call):
            ident = _identify_sdk(node, litellm_aliases)
            if ident is not None:
                sdk, call_site = ident
                qualname = _qualname_stack(stack)
                shadowed = _shadowed(stack)
                emitted = 0
                # system= kwarg
                system_node = _find_kwarg(node, "system")
                if system_node is not None:
                    records.append(_build_record(
                        file_path=rel_path,
                        line=getattr(system_node, "lineno", getattr(node, "lineno", 0)),
                        sdk=sdk, call_site=call_site,
                        role="system", role_position=-1,
                        qualname=qualname, node=system_node,
                        constants=constants, shadowed=shadowed,
                    ))
                    emitted += 1
                # messages= kwarg
                messages_node = _find_kwarg(node, "messages")
                if messages_node is not None:
                    before = len(records)
                    for pos, role, content_node in _extract_messages_list(messages_node):
                        records.append(_build_record(
                            file_path=rel_path,
                            line=getattr(content_node, "lineno", getattr(node, "lineno", 0)),
                            sdk=sdk, call_site=call_site,
                            role=role, role_position=pos,
                            qualname=qualname, node=content_node,
                            constants=constants, shadowed=shadowed,
                        ))
                    extracted = len(records) - before
                    emitted += extracted
                    statically_empty = (
                        isinstance(messages_node, ast.List)
                        and len(messages_node.elts) == 0
                    )
                    if extracted == 0 and not statically_empty:
                        # messages= exists but isn't a parseable literal list
                        # (a variable, a helper call, a comprehension) — or is
                        # a literal list whose entries can't be resolved. The
                        # call site is real; the prompts aren't statically
                        # visible. Emit one honest dynamic record instead of
                        # vanishing — invisible call sites made real repos
                        # report zero sites.
                        # A literal EMPTY list is statically known to carry no
                        # prompts: no record (that would be dishonest the
                        # other way).
                        records.append(_build_record(
                            file_path=rel_path,
                            line=getattr(messages_node, "lineno", getattr(node, "lineno", 0)),
                            sdk=sdk, call_site=call_site,
                            role="messages", role_position=-1,
                            qualname=qualname, node=messages_node,
                            constants=None, shadowed=frozenset(),
                        ))
                        emitted += 1
                if emitted == 0 and _has_kwargs_unpack(node):
                    # `litellm.completion(**kwargs)` — prompts exist, statically
                    # invisible. Surface the call site as UNKNOWN rather than
                    # omitting it; placeholder is shape-stable.
                    text = "<dynamic:KwargsUnpack>"
                    records.append(PromptRecord(
                        file_path=rel_path,
                        line=getattr(node, "lineno", 0),
                        sdk=sdk, call_site=call_site,
                        role="messages", role_position=-1,
                        qualname=qualname,
                        text=text,
                        is_dynamic=True,
                        fingerprint=fingerprint_text(text),
                        loose_fingerprint=loose_fingerprint_text(text),
                    ))
        for child in ast.iter_child_nodes(node):
            visit(child, stack)
        if push:
            stack.pop()

    visit(tree, [])
    return (records, None)


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
    deterministic output. Unscannable files are silently dropped here —
    use :func:`scan_with_skips` when the caller must surface them.
    """
    records, _skipped = scan_with_skips(repo_root, ignore_dirs=ignore_dirs)
    return records


def scan_with_skips(
    repo_root: str, ignore_dirs: Optional[frozenset[str]] = None
) -> tuple[list[PromptRecord], list[tuple[str, str]]]:
    """Like :func:`scan` but also returns the unscannable files.

    Returns ``(records, skipped)`` where ``skipped`` is a sorted list of
    ``(relative_path, reason)`` for every .py file the walk found but could
    not scan (syntax error, encoding error, symlink resolving outside the
    repo root). Verdicts for sites in those files are unreliable — callers
    rendering reports must say so rather than letting them read as REMOVED.
    """
    skip = (ignore_dirs or frozenset()) | DEFAULT_IGNORE_DIRS
    root = Path(repo_root).resolve()
    all_records: list[PromptRecord] = []
    skipped: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # mutate dirnames in place to prune the walk
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            records, reason = scan_file_with_reason(full, repo_root=str(root))
            if reason is not None:
                # The walk path is always under root (only the resolved
                # target may escape), so the relative path is stable.
                skipped.append((os.path.relpath(full, str(root)), reason))
                continue
            all_records.extend(records)
    all_records.sort(key=lambda r: (r.file_path, r.line, r.role_position))
    skipped.sort()
    return all_records, skipped
