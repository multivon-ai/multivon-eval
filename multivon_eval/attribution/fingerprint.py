"""Prompt-text normalization + content-addressed fingerprinting.

Two prompts are "the same prompt" iff their fingerprints match. The
normalization rules are intentionally conservative: we strip leading/trailing
whitespace and collapse trailing whitespace on each line, but we DO NOT
case-fold or strip internal punctuation. Prompts are sensitive to subtle
phrasing, and an over-normalized fingerprint hides real changes.
"""
from __future__ import annotations

import hashlib


def normalize_text(text: str) -> str:
    """Stable normalization of prompt text.

    Rules:
      - Strip trailing whitespace from each line (matches editor "trim on save").
      - Strip surrounding whitespace from the whole string.
      - Preserve internal blank-line structure (don't collapse paragraphs).
      - Preserve case, punctuation, codepoints exactly.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def fingerprint_text(text: str) -> str:
    """SHA-256 hex digest of normalize_text(text).

    Hex output rather than truncated b64 so two records with the same prompt
    text always produce a string-identical fingerprint regardless of platform.
    """
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
