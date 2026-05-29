"""multivon_eval.attribution — structured prompt-diff for AI eval CI.

Scope (v1):
    Walks a Python repo, finds LLM SDK call sites
    (anthropic.messages.create / openai.chat.completions.create / litellm.completion),
    extracts string-literal prompts from kwargs, fingerprints them, and computes
    diffs across two refs. The output is intentionally descriptive
    ("these N prompts changed"), not causal — no attribution claims.

    The hardened calibration spike of 2026-05-30 documented in
    multivon-strategy/positioning/feature_prompt_attribution_calibration_spike_hardened_2026_05_30.md
    showed Haiku-based attribution failing catastrophically on mixed-cause
    regressions (HIGH-confidence-and-wrong, 14% rate). This package
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
    - Capture named constants used by name elsewhere — only literal
      kwarg values at recognized SDK call sites count.
"""
from __future__ import annotations

from .schema import PromptRecord, PromptDiff
from .fingerprint import normalize_text, fingerprint_text
from .ast_extractor import scan, scan_file
from .diff import diff_records
from .render import render_markdown

__all__ = [
    "PromptRecord",
    "PromptDiff",
    "normalize_text",
    "fingerprint_text",
    "scan",
    "scan_file",
    "diff_records",
    "render_markdown",
]
