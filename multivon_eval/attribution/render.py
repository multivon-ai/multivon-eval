"""Markdown rendering for the PR-comment "Prompt Changes" section.

Output is intentionally descriptive: it shows what changed without making
causal claims. No "Impact:" field, no attribution to a specific regression,
no confidence score. The eval-action layer can append a separate
"Regressed cases" table that does NOT cross-link to specific prompts.
"""
from __future__ import annotations

from .schema import PromptDiff


_MAX_TEXT_LINES = 6
_MAX_TEXT_CHARS = 400


def _truncate_for_display(text: str) -> str:
    """Truncate prompt text for the diff display.

    Keeps the first six lines and at most 400 chars. Adds an explicit
    truncation marker so a reader knows there's more in the file.
    """
    lines = text.splitlines()
    if len(lines) > _MAX_TEXT_LINES:
        lines = lines[:_MAX_TEXT_LINES]
        truncated_lines = True
    else:
        truncated_lines = False
    out = "\n".join(lines)
    truncated_chars = False
    if len(out) > _MAX_TEXT_CHARS:
        out = out[: _MAX_TEXT_CHARS]
        truncated_chars = True
    if truncated_lines or truncated_chars:
        out += "\n… (truncated)"
    return out


def _render_one(diff: PromptDiff) -> str:
    if diff.change_type == "added":
        rec = diff.after
        body = (
            f"**`{diff.call_site_id}` — added** ({rec.sdk}, role={rec.role})\n\n"
            f"```\n{_truncate_for_display(rec.text)}\n```"
        )
        return body

    if diff.change_type == "removed":
        rec = diff.before
        body = (
            f"**`{diff.call_site_id}` — removed** ({rec.sdk}, role={rec.role})\n\n"
            f"```\n{_truncate_for_display(rec.text)}\n```"
        )
        return body

    if diff.change_type == "dynamic":
        return (
            f"**`{diff.call_site_id}` — dynamic prompt changed**\n\n"
            f"The expression at this call site is built at runtime "
            f"(template, variable, or non-literal). "
            f"multivon attribution v1 does not analyze dynamic prompts; "
            f"this entry is informational only."
        )

    # "modified"
    return (
        f"**`{diff.call_site_id}` — modified** ({diff.after.sdk}, role={diff.after.role})\n\n"
        f"<sub>before</sub>\n```\n{_truncate_for_display(diff.before.text)}\n```\n\n"
        f"<sub>after</sub>\n```\n{_truncate_for_display(diff.after.text)}\n```"
    )


def render_markdown(
    diffs: list[PromptDiff],
    *,
    dynamic_unscanned_count: int = 0,
) -> str:
    """Render the structured-diff section as GitHub-flavored Markdown.

    dynamic_unscanned_count lets the caller report e.g. "3 prompts in
    .j2 templates were not analyzed" — the AST extractor itself doesn't
    know about Jinja files, but the caller might count them separately.
    """
    if not diffs and dynamic_unscanned_count == 0:
        return "## Prompt changes\n\nNo prompt changes detected in this PR."

    n_modified = sum(1 for d in diffs if d.change_type == "modified")
    n_added = sum(1 for d in diffs if d.change_type == "added")
    n_removed = sum(1 for d in diffs if d.change_type == "removed")
    n_dynamic = sum(1 for d in diffs if d.change_type == "dynamic")

    header_parts: list[str] = []
    if n_modified:
        header_parts.append(f"{n_modified} modified")
    if n_added:
        header_parts.append(f"{n_added} added")
    if n_removed:
        header_parts.append(f"{n_removed} removed")
    if n_dynamic:
        header_parts.append(f"{n_dynamic} dynamic")
    summary = " · ".join(header_parts) if header_parts else "none"

    sections = [_render_one(d) for d in diffs]

    out = [
        "## Prompt changes",
        "",
        f"_{summary}. AST-aware, descriptive only — no causal attribution. "
        f"Python SDK literals (`anthropic.messages.create`, "
        f"`openai.chat.completions.create`, `litellm.completion`) only._",
        "",
    ]
    if dynamic_unscanned_count > 0:
        out.append(
            f"_Plus {dynamic_unscanned_count} prompt source(s) outside the v1 "
            f"AST scope (templates, runtime-loaded). Not analyzed._"
        )
        out.append("")
    out.extend(sections)
    return "\n".join(out) + "\n"
