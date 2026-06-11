"""Acceptance gates for generated eval cases — pdfhell-style gate discipline.

Every case produced by scaled seed-case generation (``multivon-eval
bootstrap --n-seed-cases N``) has to earn its place through a gate
pipeline before it lands in ``seed_cases.jsonl``:

1. :func:`gate_well_formed` — structural, free. Non-empty input and an
   expected behavior (``expected_output`` or expected-behavior text in
   metadata).
2. :func:`gate_duplicate` — deterministic, free. Loose-normalized
   identity OR token-overlap Jaccard ≥ 0.85 against every case accepted
   so far (across batches).
3. :func:`gate_hardness` — costs judge calls. Thin wrapper over
   ``multivon_eval.auto.validate_adversarial_cases``; only runs when the
   caller explicitly opts in with a baseline model.

The counterpart contract is :class:`GenerationReport` — no silent caps:
every generated case is accounted for as accepted, malformed, duplicate,
or outside the hardness band, and the discovery report prints exactly
those counts.

This module is standalone (no pdfhell dependency); the GateResult shape
is modeled on pdfhell's gate pattern conceptually.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .attribution.fingerprint import loose_normalize_text
from .case import EvalCase
from .judge import JudgeConfig

# Token-overlap similarity above which two inputs count as duplicates.
JACCARD_DUPLICATE_THRESHOLD = 0.85

# Metadata keys that can carry per-case expected-behavior prose when
# ``expected_output`` itself is absent.
_EXPECTED_BEHAVIOR_KEYS = (
    "expected_behavior",
    "expected_behaviour",
    "expected_response",
    "expected",
)


@dataclass(slots=True)
class GateResult:
    """Outcome of one gate applied to one case.

    ``passed=False`` means the case is dropped; ``reason`` says why.
    """

    gate: str
    passed: bool
    reason: str = ""


@dataclass(slots=True)
class GenerationReport:
    """Full accounting of a scaled case-generation run — no silent caps.

    Invariant: ``generated == accepted + dropped_malformed +
    dropped_duplicate + dropped_hardness``. Every generated case lands in
    exactly one bucket.
    """

    requested: int
    generated: int = 0
    accepted: int = 0
    dropped_malformed: int = 0
    dropped_duplicate: int = 0
    dropped_hardness: int = 0
    hardness_skipped: bool = True
    hardness_skip_reason: str = "no --validate-cases / baseline model"
    hardness_band: tuple[float, float] = (0.5, 1.0)
    n_batches: int = 0

    def summary_line(self) -> str:
        """One-line human summary, e.g. ``generated 500, accepted 431 —
        dropped 38 duplicates, 12 malformed, 19 outside hardness band
        [0.5, 1.0]``.
        """
        base = (
            f"generated {self.generated}, accepted {self.accepted} — "
            f"dropped {self.dropped_duplicate} duplicates, "
            f"{self.dropped_malformed} malformed"
        )
        if self.hardness_skipped:
            return base
        lo, hi = self.hardness_band
        return base + f", {self.dropped_hardness} outside hardness band [{lo}, {hi}]"


# ─── Gate 1: well-formed (structural, free) ───────────────────────────────


def gate_well_formed(case: EvalCase) -> GateResult:
    """A case is well-formed iff it has a non-empty input AND an expected
    behavior — either ``expected_output`` or expected-behavior text in
    metadata (``expected_behavior`` / ``expected_response`` / …).
    """
    if not isinstance(case.input, str) or not case.input.strip():
        return GateResult("well_formed", False, "empty input")

    if case.expected_output is not None and str(case.expected_output).strip():
        return GateResult("well_formed", True)

    meta = case.metadata or {}
    for key in _EXPECTED_BEHAVIOR_KEYS:
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            return GateResult("well_formed", True)

    return GateResult(
        "well_formed", False,
        "no expected_output and no expected-behavior text in metadata",
    )


# ─── Gate 2: duplicate (deterministic, free, cross-batch) ─────────────────


def _input_tokens(normalized: str) -> frozenset[str]:
    """Alphanumeric tokens of a loose-normalized input, casefolded.

    Casefolding + punctuation-stripping make the Jaccard overlap robust
    to trivial rephrasings ("What's the refund window?" vs "what is the
    refund window") without an LLM call.
    """
    return frozenset(re.findall(r"[a-z0-9]+", normalized.casefold()))


def token_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets. Two empty sets are identical."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def gate_duplicate(case: EvalCase, accepted: Iterable[EvalCase]) -> GateResult:
    """Duplicate iff EITHER the loose-normalized input is identical to an
    already-accepted case's input, OR the token-overlap Jaccard on the
    normalized inputs is ≥ :data:`JACCARD_DUPLICATE_THRESHOLD`.

    Deterministic, no LLM. ``accepted`` is every case accepted SO FAR —
    including earlier batches, so dedupe works across batches.
    """
    norm = loose_normalize_text(case.input or "")
    toks = _input_tokens(norm)
    for prior in accepted:
        prior_norm = loose_normalize_text(prior.input or "")
        if norm == prior_norm:
            return GateResult(
                "duplicate", False,
                "input identical to an accepted case after loose normalization",
            )
        sim = token_jaccard(toks, _input_tokens(prior_norm))
        if sim >= JACCARD_DUPLICATE_THRESHOLD:
            return GateResult(
                "duplicate", False,
                f"token-overlap Jaccard {sim:.2f} ≥ {JACCARD_DUPLICATE_THRESHOLD} "
                f"vs an accepted case",
            )
    return GateResult("duplicate", True)


# ─── Gate 3: hardness (costs judge calls — opt-in) ────────────────────────


def gate_hardness(
    cases: list[EvalCase],
    baseline_model: Callable[[str], str],
    *,
    n_shots: int = 3,
    hardness_band: tuple[float, float] = (0.5, 1.0),
    judge: JudgeConfig | None = None,
):
    """Thin wrapper over ``multivon_eval.auto.validate_adversarial_cases``.

    Returns ``(kept_cases, list[HardnessReport])`` — the existing N-shot
    failure-rate aggregation + band filter, unchanged. Callers should
    only invoke this when the user opted in (it costs baseline + judge
    calls per case per shot).
    """
    from .auto import validate_adversarial_cases  # lazy to avoid circular

    return validate_adversarial_cases(
        cases, baseline_model,
        n_shots=n_shots, hardness_band=hardness_band, judge=judge,
    )


# ─── Diversity-steering digest ────────────────────────────────────────────


def digest_inputs(
    cases: Sequence[EvalCase],
    *,
    max_words: int = 8,
    max_entries: int = 60,
) -> str:
    """Compact digest of accepted case inputs for batch-prompt steering.

    First ``max_words`` words of each input, one per line, capped at the
    ``max_entries`` most recent cases so the prompt stays bounded even at
    N=500. Cheap diversity steering: later batches see what already
    exists and are told not to duplicate it.
    """
    lines: list[str] = []
    for case in list(cases)[-max_entries:]:
        words = (case.input or "").split()
        snippet = " ".join(words[:max_words])
        if len(words) > max_words:
            snippet += " …"
        lines.append(f"- {snippet}")
    return "\n".join(lines)
