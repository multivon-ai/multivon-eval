"""Prompt-text normalization + content-addressed fingerprinting.

Two prompts are "the same prompt" iff their fingerprints match. The
normalization rules are intentionally conservative: we strip leading/trailing
whitespace and collapse trailing whitespace on each line, but we DO NOT
case-fold or strip internal punctuation. Prompts are sensitive to subtle
phrasing, and an over-normalized fingerprint hides real changes.
"""
from __future__ import annotations

import hashlib
import unicodedata


def normalize_text(text: str) -> str:
    """Stable normalization of prompt text.

    Rules:
      - NFC-normalize codepoints first: composed vs decomposed Unicode
        (e.g. "é" as U+00E9 vs "e"+U+0301) is an editor/OS artifact, not a
        prompt change — without it the same visible prompt fingerprints
        differently across machines (scanner v4).
      - Strip trailing whitespace from each line (matches editor "trim on save").
      - Strip surrounding whitespace from the whole string.
      - Preserve internal blank-line structure (don't collapse paragraphs).
      - Preserve case and punctuation exactly.
    """
    text = unicodedata.normalize("NFC", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def fingerprint_text(text: str) -> str:
    """SHA-256 hex digest of normalize_text(text).

    Hex output rather than truncated b64 so two records with the same prompt
    text always produce a string-identical fingerprint regardless of platform.
    """
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def loose_normalize_text(text: str) -> str:
    """Aggressive whitespace normalization for the *loose* fingerprint.

    Collapses every run of whitespace (spaces, tabs, newlines) to a single
    space and strips the ends. Two prompts that differ only in indentation /
    line wrapping loose-normalize to the same string. NFC-normalized first,
    same as :func:`normalize_text` — the loose fingerprint must never be
    *stricter* than the strict one.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def loose_fingerprint_text(text: str) -> str:
    """SHA-256 hex digest of loose_normalize_text(text).

    Label-only fingerprint: it exists so a staleness report can TAG a change
    as formatting-only (re-indented triple-quoted prompt, re-wrapped lines).
    It must never be used to *suppress* a change — normalize_text preserves
    leading indentation deliberately, so the strict fingerprint stays the
    source of truth for "did the prompt change".
    """
    return hashlib.sha256(loose_normalize_text(text).encode("utf-8")).hexdigest()
