"""multivon_eval.attribution — structured prompt-diff for AI eval CI.

Scope (v1):
    Walks a Python repo, finds LLM SDK call sites
    (anthropic.messages.create / openai.chat.completions.create / litellm.completion),
    extracts string-literal prompts from kwargs, fingerprints them, and computes
    diffs across two refs. The output is intentionally descriptive
    ("these N prompts changed"), not causal — no attribution claims.

    A hardened calibration spike showed Haiku-based attribution failing
    catastrophically on mixed-cause regressions (HIGH-confidence-and-wrong,
    14% rate). This package
    ships the structured-diff substrate; attribution itself is gated
    on a future non-prompt-change sidecar signal.

Public API::

    from multivon_eval.attribution import scan, diff_records, render_markdown

    base = scan("/path/to/repo_base")
    head = scan("/path/to/repo_head")
    diffs = diff_records(base, head)
    md = render_markdown(diffs)

What this package does NOT do (deliberate):
    - Causal attribution between a regression and a prompt change.
    - Detect prompts inside Jinja templates, LangChain ChatPromptTemplates,
      database-loaded prompts, or runtime-assembled strings.
    - Multi-hop or cross-module constant resolution. Scanner v2 resolves
      exactly one hop: a module-level ``X = "literal"`` in the SAME file
      used as ``system=X``. Conditionally-reassigned names, function-scope
      names, and imports stay dynamic.
"""
from __future__ import annotations

from .schema import PromptRecord, PromptDiff
from .fingerprint import (
    normalize_text, fingerprint_text,
    loose_normalize_text, loose_fingerprint_text,
)
from .ast_extractor import (
    scan, scan_file, scan_file_with_reason, scan_with_skips, SCANNER_VERSION,
)
from .diff import diff_records
from .render import render_markdown

__all__ = [
    "PromptRecord",
    "PromptDiff",
    "SCANNER_VERSION",
    "normalize_text",
    "fingerprint_text",
    "loose_normalize_text",
    "loose_fingerprint_text",
    "scan",
    "scan_file",
    "scan_file_with_reason",
    "scan_with_skips",
    "diff_records",
    "render_markdown",
]
