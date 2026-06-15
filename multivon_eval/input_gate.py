"""Input-quality preflight gate — free, deterministic, WARN-only (issue #14).

"Honest UNKNOWN over confident wrong" applied to the INPUT side. Today
bootstrap/generate silently consume whatever traces, documents, or
descriptions you hand them and produce a confident-looking suite even when
the inputs can't support one. This module makes garbage-in **loud and
auditable** BEFORE any paid LLM call, and composes with — never duplicates
— the post-generation OUTPUT gates in ``case_gates.py``.

Phase 1 is **WARN-only**. There is NO hard REFUSE and NO exit-2-block on
any signal. A WARN never stops generation; it prints and proceeds. (A WARN
can't break a CI; a default-REFUSE could surprise an existing user.
REFUSE promotion is a deliberate deferred follow-up.) The standalone
``assess`` command exits 0 on PROCEED, 1 on WARN so scripts can detect it;
the in-line preflight on bootstrap/generate NEVER changes exit behavior —
it only prints.

The four signals all reuse trusted machinery (zero new deps):

1. **trace_count** — imports :data:`discover.CALIBRATION_MIN_TRACES` (the
   single anti-drift constant) so the gate and the calibration warning
   can't drift.
2. **field_completeness** — per-field non-empty rate for input/output/
   context (reported per-field, never averaged). Names the currently-silent
   zero-output early-return at discover.py where calibration quietly
   produces uncalibrated thresholds.
3. **near_duplicate_ratio** — corpus unique fraction via
   ``case_gates.token_jaccard`` + ``loose_normalize_text``, reservoir-capped
   at the ``digest_inputs`` pattern (no O(n²) hang on big dumps).
4. **pii_secret_density** — mean detections/item via ``_pii_scan.scan`` /
   ``summarize``, reported as a NEUTRAL fact, never a moral block.

Verdict contract: two tiers in phase 1 (PROCEED / WARN), no scalar score
(the vanity metric this exists to prevent).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import _pii_scan
from .attribution.fingerprint import loose_normalize_text
from .case_gates import token_jaccard
from .discover import CALIBRATION_MIN_TRACES

# Near-duplicate Jaccard threshold (matches case_gates dedupe semantics).
_NEAR_DUP_JACCARD = 0.85
# Reservoir cap — same bounded-work pattern as case_gates.digest_inputs so a
# multi-million-row dump can't trigger O(n²) pairwise comparison.
_NEAR_DUP_SAMPLE_CAP = 400
_NEAR_DUP_SEED = 1729

# Field-completeness WARN thresholds.
_OUTPUT_PRESENT_MIN = 0.8
_CONTEXT_PRESENT_MIN = 0.5
# Unique-fraction WARN threshold.
_UNIQUE_FRACTION_MIN = 0.7
# Empty/too-short source-document WARN floor (characters).
_DOC_MIN_CHARS = 200

# Which signals are DEFINED for each kind. The headline denominator counts
# ALL defined signals for the kind (so it can never read M-of-M by dropping
# signals); signals not run for a kind are emitted as state="SKIPPED".
_SIGNALS_BY_KIND: dict[str, tuple[str, ...]] = {
    "bootstrap": (
        "trace_count",
        "field_completeness",
        "near_duplicate_ratio",
        "pii_secret_density",
    ),
    "generate": (
        "document_length",
        "near_duplicate_ratio",
        "pii_secret_density",
    ),
    "cases": (
        "case_count",
        "well_formed_rate",
    ),
}

_BLIND_SPOTS = [
    "semantic correctness of labels",
    "factual accuracy of source",
]


@dataclass(slots=True)
class SignalFinding:
    """One signal's outcome. ``state`` distinguishes a measured result from
    a signal that was not run for this ``kind`` (SKIPPED) or could not be
    measured (UNKNOWN) — a SKIPPED signal is NEVER a phantom green."""

    signal: str
    state: str          # "MEASURED" | "UNKNOWN" | "SKIPPED"
    tier: str           # "ok" | "warn"   (no "refuse" tier in phase 1)
    measured: str       # human value, e.g. "0.52" or "12 rows"
    threshold: str      # e.g. "< 0.70"
    message: str        # one-line plain consequence


@dataclass(slots=True)
class InputQualityReport:
    verdict: str                          # "PROCEED" | "WARN" (never REFUSE)
    kind: str                             # "bootstrap" | "generate" | "cases"
    findings: list[SignalFinding]
    measurable_total: int                 # denominator for the headline
    blind_spots: list[str] = field(default_factory=lambda: list(_BLIND_SPOTS))

    @property
    def flagged(self) -> list[SignalFinding]:
        return [f for f in self.findings if f.tier == "warn"]

    def render_text(self) -> str:
        """Empty string on PROCEED (be invisible when input is fine).

        On WARN: a determinacy headline in the staleness style whose
        denominator is ``measurable_total`` (ALL defined signals for the
        kind), one line per flagged finding, then a blind-spots footer so
        PROCEED is never read as a correctness guarantee.
        """
        if self.verdict == "PROCEED":
            return ""
        flagged = self.flagged
        lines = [
            f"input quality: WARN — {len(flagged)} of {self.measurable_total} "
            f"signals flagged"
        ]
        for f in flagged:
            lines.append(
                f"  {f.signal}={f.measured} ({f.threshold}) — {f.message}"
            )
        lines.append(
            "  not checked: " + ", ".join(self.blind_spots)
        )
        return "\n".join(lines)


# ─── per-signal measurement ───────────────────────────────────────────────


def _skipped(signal: str) -> SignalFinding:
    return SignalFinding(
        signal=signal, state="SKIPPED", tier="ok",
        measured="—", threshold="—",
        message="not measured for this input kind",
    )


def _measure_trace_count(traces: list) -> SignalFinding:
    n = len(traces)
    threshold = f"in [1, {CALIBRATION_MIN_TRACES})"
    if n == 0:
        return SignalFinding(
            "trace_count", "MEASURED", "warn", "0 rows", "> 0",
            "0 usable traces — nothing to bootstrap",
        )
    if n < CALIBRATION_MIN_TRACES:
        return SignalFinding(
            "trace_count", "MEASURED", "warn", f"{n} rows", threshold,
            f"n_traces={n} below {CALIBRATION_MIN_TRACES}; p25-based "
            f"calibration has wide CIs and thresholds can swing on noise",
        )
    return SignalFinding(
        "trace_count", "MEASURED", "ok", f"{n} rows",
        f">= {CALIBRATION_MIN_TRACES}", "enough traces to calibrate",
    )


def _present_rate(traces: list, key: str) -> float:
    if not traces:
        return 0.0
    present = sum(1 for t in traces if str(t.get(key) or "").strip())
    return present / len(traces)


def _measure_field_completeness(traces: list) -> SignalFinding:
    """Per-field non-empty rate (reported per-field, NEVER averaged)."""
    if not traces:
        return SignalFinding(
            "field_completeness", "MEASURED", "warn",
            "input=0.00 output=0.00 context=0.00", "—",
            "no traces — nothing to measure completeness over",
        )
    in_rate = _present_rate(traces, "input")
    out_rate = _present_rate(traces, "output")
    ctx_rate = _present_rate(traces, "context")
    # context-requiring iff ANY trace carries a context field at all.
    context_required = any("context" in t for t in traces)
    measured = (
        f"input={in_rate:.2f} output={out_rate:.2f} context={ctx_rate:.2f}"
    )

    if out_rate == 0:
        return SignalFinding(
            "field_completeness", "MEASURED", "warn", measured,
            f"output >= {_OUTPUT_PRESENT_MIN}",
            "0% of traces have an output — calibration will SILENTLY "
            "early-return and emit UNCALIBRATED (proposer-default) thresholds",
        )
    if out_rate < _OUTPUT_PRESENT_MIN:
        return SignalFinding(
            "field_completeness", "MEASURED", "warn", measured,
            f"output >= {_OUTPUT_PRESENT_MIN}",
            f"only {out_rate:.0%} of traces have an output — calibration "
            f"runs on a thin slice",
        )
    if context_required and ctx_rate < _CONTEXT_PRESENT_MIN:
        return SignalFinding(
            "field_completeness", "MEASURED", "warn", measured,
            f"context >= {_CONTEXT_PRESENT_MIN} (context-requiring)",
            f"only {ctx_rate:.0%} of traces carry context — context-using "
            f"evaluators (faithfulness, recall) will be under-grounded",
        )
    return SignalFinding(
        "field_completeness", "MEASURED", "ok", measured,
        f"output >= {_OUTPUT_PRESENT_MIN}", "fields adequately populated",
    )


def _unique_fraction(texts: list[str]) -> tuple[float, int]:
    """Unique fraction after loose-normalize + token-Jaccard at
    :data:`_NEAR_DUP_JACCARD`, reservoir-sampled at the cap so a huge dump
    can't O(n²) hang. Returns ``(fraction, n_compared)``."""
    sample = texts
    if len(texts) > _NEAR_DUP_SAMPLE_CAP:
        rng = random.Random(_NEAR_DUP_SEED)
        sample = rng.sample(texts, _NEAR_DUP_SAMPLE_CAP)
    n = len(sample)
    if n == 0:
        return 1.0, 0
    token_sets = [
        frozenset(loose_normalize_text(t or "").casefold().split())
        for t in sample
    ]
    unique = 0
    kept: list[frozenset[str]] = []
    for toks in token_sets:
        is_dup = False
        for prior in kept:
            if token_jaccard(toks, prior) >= _NEAR_DUP_JACCARD:
                is_dup = True
                break
        if not is_dup:
            kept.append(toks)
            unique += 1
    return unique / n, n


def _measure_near_duplicate_ratio(texts: list[str]) -> SignalFinding:
    frac, n = _unique_fraction(texts)
    measured = f"{frac:.2f} unique"
    if n == 0:
        return SignalFinding(
            "near_duplicate_ratio", "MEASURED", "ok", "1.00 unique",
            f">= {_UNIQUE_FRACTION_MIN}", "nothing to compare",
        )
    cap_note = f" (sampled {n})" if len(texts) > _NEAR_DUP_SAMPLE_CAP else ""
    if frac < _UNIQUE_FRACTION_MIN:
        return SignalFinding(
            "near_duplicate_ratio", "MEASURED", "warn", measured + cap_note,
            f">= {_UNIQUE_FRACTION_MIN}",
            f"only {frac:.0%} of inputs are distinct — a repetitive corpus "
            f"yields a narrow eval (legitimate for FAQ/narrow domains)",
        )
    return SignalFinding(
        "near_duplicate_ratio", "MEASURED", "ok", measured + cap_note,
        f">= {_UNIQUE_FRACTION_MIN}", "inputs are diverse",
    )


def _measure_pii_secret_density(texts: list[str]) -> SignalFinding:
    """Mean detections/item — a NEUTRAL warn-level fact, never a block."""
    if not texts:
        return SignalFinding(
            "pii_secret_density", "MEASURED", "ok", "0.00/item", "neutral",
            "no text to scan",
        )
    per_item = [_pii_scan.scan(t or "") for t in texts]
    counts = _pii_scan.summarize(per_item)
    total = sum(counts.values())
    mean = total / len(texts)
    measured = f"{mean:.2f}/item"
    if total == 0:
        return SignalFinding(
            "pii_secret_density", "MEASURED", "ok", measured, "neutral",
            "no PII/secrets detected",
        )
    labels = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return SignalFinding(
        "pii_secret_density", "MEASURED", "warn", measured, "neutral",
        f"PII/secret detections present ({labels}) — if this is a "
        f"PII-handling eval, this is expected; otherwise scrub the source",
    )


def _measure_document_length(document: str | None) -> SignalFinding:
    text = (document or "").strip()
    n = len(text)
    if n == 0:
        return SignalFinding(
            "document_length", "MEASURED", "warn", "0 chars", "> 0",
            "empty source document — nothing to generate cases from",
        )
    if n < _DOC_MIN_CHARS:
        return SignalFinding(
            "document_length", "MEASURED", "warn", f"{n} chars",
            f">= {_DOC_MIN_CHARS}",
            f"source is only {n} chars — too short to ground a meaningful "
            f"case set",
        )
    return SignalFinding(
        "document_length", "MEASURED", "ok", f"{n} chars",
        f">= {_DOC_MIN_CHARS}", "source has substance",
    )


def _measure_case_count(cases: list) -> SignalFinding:
    n = len(cases)
    if n == 0:
        return SignalFinding(
            "case_count", "MEASURED", "warn", "0 cases", "> 0",
            "no cases loaded — nothing to evaluate",
        )
    return SignalFinding(
        "case_count", "MEASURED", "ok", f"{n} cases", "> 0",
        "cases present",
    )


def _case_input(c) -> str:
    if isinstance(c, dict):
        return str(c.get("input") or "")
    return str(getattr(c, "input", "") or "")


def _case_well_formed(c) -> bool:
    """Non-empty input AND some expected behavior (output or behavior text)."""
    if not _case_input(c).strip():
        return False
    if isinstance(c, dict):
        exp = c.get("expected_output")
        if isinstance(exp, str) and exp.strip():
            return True
        meta = c.get("metadata") or {}
    else:
        exp = getattr(c, "expected_output", None)
        if isinstance(exp, str) and exp.strip():
            return True
        meta = getattr(c, "metadata", None) or {}
    for key in ("expected_behavior", "expected_behaviour",
                "expected_response", "expected"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            return True
    return False


def _measure_well_formed_rate(cases: list) -> SignalFinding:
    if not cases:
        return SignalFinding(
            "well_formed_rate", "MEASURED", "warn", "0.00", "> 0.8",
            "no cases — nothing to check well-formedness over",
        )
    well = sum(1 for c in cases if _case_well_formed(c))
    rate = well / len(cases)
    measured = f"{rate:.2f}"
    if rate < 0.8:
        return SignalFinding(
            "well_formed_rate", "MEASURED", "warn", measured, ">= 0.80",
            f"only {rate:.0%} of cases have a non-empty input AND an "
            f"expected behavior — the rest can't be scored",
        )
    return SignalFinding(
        "well_formed_rate", "MEASURED", "ok", measured, ">= 0.80",
        "cases are well-formed",
    )


# ─── public entry point ────────────────────────────────────────────────────


def assess_input(
    source=None,
    kind: str = "bootstrap",
    *,
    traces=None,
    document=None,
    cases=None,
    pii_policy: str = "redact",
) -> InputQualityReport:
    """Run the free preflight gate over already-loaded input.

    The caller passes whatever it already has — DON'T re-read or re-parse
    files the caller already loaded:

      - bootstrap passes ``traces=`` (loaded trace dicts) and ``document=``
        (the product description text).
      - generate passes ``document=`` (the source text).
      - cases-mode passes ``cases=`` (loaded EvalCase / dicts).

    ``source`` is a positional convenience: if the typed kwargs are absent,
    it is routed by ``kind`` (traces for bootstrap, document text for
    generate, cases for cases). ``kind`` routes which signals run; signals
    not defined for a kind are emitted as state="SKIPPED" (never a phantom
    green). The verdict is WARN if ANY signal flagged, else PROCEED.
    """
    if kind not in _SIGNALS_BY_KIND:
        raise ValueError(
            f"unknown kind {kind!r}; expected one of "
            f"{sorted(_SIGNALS_BY_KIND)}"
        )

    # Route the positional source if typed kwargs weren't supplied.
    if traces is None and kind == "bootstrap" and source is not None:
        traces = source
    if document is None and kind == "generate" and source is not None:
        document = source
    if cases is None and kind == "cases" and source is not None:
        cases = source

    traces = list(traces) if traces else []
    cases = list(cases) if cases else []

    defined = _SIGNALS_BY_KIND[kind]
    measured: dict[str, SignalFinding] = {}

    if kind == "bootstrap":
        measured["trace_count"] = _measure_trace_count(traces)
        measured["field_completeness"] = _measure_field_completeness(traces)
        trace_texts = [str(t.get("input") or "") for t in traces]
        measured["near_duplicate_ratio"] = _measure_near_duplicate_ratio(
            trace_texts
        )
        pii_texts = [
            " ".join(
                str(t.get(k) or "") for k in ("input", "output", "context")
            )
            for t in traces
        ]
        measured["pii_secret_density"] = _measure_pii_secret_density(pii_texts)
    elif kind == "generate":
        measured["document_length"] = _measure_document_length(document)
        paragraphs = [
            p for p in (document or "").split("\n\n") if p.strip()
        ]
        measured["near_duplicate_ratio"] = _measure_near_duplicate_ratio(
            paragraphs
        )
        measured["pii_secret_density"] = _measure_pii_secret_density(
            [document or ""]
        )
    elif kind == "cases":
        measured["case_count"] = _measure_case_count(cases)
        measured["well_formed_rate"] = _measure_well_formed_rate(cases)

    # Assemble findings in the defined order; un-run signals are SKIPPED.
    findings = [measured.get(name) or _skipped(name) for name in defined]
    verdict = "WARN" if any(f.tier == "warn" for f in findings) else "PROCEED"

    return InputQualityReport(
        verdict=verdict,
        kind=kind,
        findings=findings,
        measurable_total=len(defined),
        blind_spots=list(_BLIND_SPOTS),
    )
