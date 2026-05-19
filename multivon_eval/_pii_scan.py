"""Minimal local PII / secret scanner for the bootstrap command.

This is a vendored subset of multivon-guard's detector module — just the
high-confidence rules we need to scrub user traces before sending them
to a remote LLM judge. We intentionally keep the surface tiny:

  - One module, ~150 lines.
  - Pure regex + Luhn check for credit cards.
  - No dependency on multivon-guard (multivon-guard is private; this
    package is public).

If the user needs the full PII detector set (Aadhaar/Verhoeff, PAN,
multi-jurisdiction etc.) they can use ``PIIEvaluator`` at runtime; this
module's job is just the pre-flight redaction before a trace is shipped
to a judge during bootstrap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


Kind = Literal["secret", "pii"]


@dataclass(frozen=True, slots=True)
class Detection:
    kind: Kind
    label: str
    start: int
    end: int
    match: str


# Secret patterns — high-confidence shapes only.
_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_ANTHROPIC_KEY = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{80,200}\b")
_OPENAI_LEGACY = re.compile(r"\bsk-[A-Za-z0-9]{48}\b")
_OPENAI_PROJECT = re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{40,}\b")
_GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,255}\b")
_GOOGLE_KEY = re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")
_STRIPE_KEY = re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[0-9a-zA-Z]{24,}\b")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_PRIVATE_KEY_HEADER = re.compile(
    r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED |)PRIVATE KEY-----"
)

# PII patterns.
_EMAIL = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,24}\b")
_US_SSN = re.compile(
    r"\b(?!000|666|9[0-9]{2})[0-9]{3}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b"
)
_CC_CANDIDATE = re.compile(
    r"(?<![0-9])(?:[0-9]{4}[\s\-]?){3,4}[0-9]{1,4}(?![0-9])"
)


def _luhn_ok(digits: str) -> bool:
    if not digits:
        return False
    total = 0
    parity = len(digits) % 2
    for i, c in enumerate(digits):
        if not c.isdigit():
            return False
        n = int(c)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _find_credit_cards(text: str) -> list[tuple[int, int, str]]:
    hits = []
    for m in _CC_CANDIDATE.finditer(text):
        digits = re.sub(r"[\s\-]", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            hits.append((m.start(), m.end(), m.group(0)))
    return hits


_RULES: list[tuple[Kind, str, "re.Pattern[str]"]] = [
    ("secret", "aws_access_key", _AWS_KEY),
    ("secret", "anthropic_api_key", _ANTHROPIC_KEY),
    ("secret", "openai_project_key", _OPENAI_PROJECT),
    ("secret", "openai_legacy_key", _OPENAI_LEGACY),
    ("secret", "github_token", _GITHUB_TOKEN),
    ("secret", "google_api_key", _GOOGLE_KEY),
    ("secret", "stripe_key", _STRIPE_KEY),
    ("secret", "jwt", _JWT),
    ("secret", "private_key_pem", _PRIVATE_KEY_HEADER),
    ("pii", "us_ssn", _US_SSN),
    ("pii", "email", _EMAIL),
]


def scan(text: str) -> list[Detection]:
    """Return non-overlapping detections sorted by start index."""
    if not text:
        return []
    raw: list[Detection] = []
    for kind, label, pattern in _RULES:
        for m in pattern.finditer(text):
            raw.append(Detection(kind, label, m.start(), m.end(), m.group(0)))
    for start, end, match in _find_credit_cards(text):
        raw.append(Detection("pii", "credit_card", start, end, match))

    # De-overlap: secrets win over PII when spans collide.
    raw.sort(key=lambda d: (d.start, 0 if d.kind == "secret" else 1, -(d.end - d.start)))
    kept: list[Detection] = []
    for d in raw:
        if any(not (d.end <= k.start or k.end <= d.start) for k in kept):
            continue
        kept.append(d)
    kept.sort(key=lambda d: d.start)
    return kept


def redact(text: str, detections: list[Detection] | None = None) -> tuple[str, list[Detection]]:
    """Return ``(redacted_text, detections)``.

    If ``detections`` is None, this calls ``scan(text)`` first.
    Replaces each span with ``[REDACTED:<label>]``; spans are walked in
    reverse so earlier offsets stay valid.
    """
    if detections is None:
        detections = scan(text)
    if not detections:
        return text, []
    pieces = list(text)
    for d in sorted(detections, key=lambda d: d.start, reverse=True):
        pieces[d.start:d.end] = list(f"[REDACTED:{d.label}]")
    return "".join(pieces), detections


def summarize(detections_list: list[list[Detection]]) -> dict[str, int]:
    """Counts by label across a batch of trace scans."""
    counts: dict[str, int] = {}
    for dets in detections_list:
        for d in dets:
            counts[d.label] = counts.get(d.label, 0) + 1
    return counts
