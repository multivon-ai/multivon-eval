"""``multivon-eval bootstrap`` — cold-start eval bootstrap pipeline.

Takes a product description + sample traces, emits a ready-to-run
``EvalSuite`` + seed adversarial cases + calibrated thresholds + a
human-readable design report.

Public entry points:
  - :func:`bootstrap` — programmatic API used by the CLI
  - :func:`infer_product_shape` — heuristic over traces, surfaced so
    downstream tools can hook in without re-running the whole pipeline

The differentiator (per the v0.1 design): metric *selection* is grounded
in the user's product shape + the user's actual traces, not a generic
template. The single Claude Haiku call sees a structured summary of the
trace shape + the product description and proposes a metric set
constrained to the existing :mod:`multivon_eval` evaluator surface;
``auto_evaluators`` runs first as a deterministic anchor so the LLM has
to argue for its picks against a credible default.

Cost target: ≈$0.12 per bootstrap on default settings (one Haiku call
for metric recommendations + 15-trace LLM-judge calibration + one Haiku
call for 30 seed adversarial cases). Hard ceiling: $0.15.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import _pii_scan
from .case import EvalCase
from .judge import JudgeConfig, resolve_judge


# ─── Public types ─────────────────────────────────────────────────────────


ProductShape = Literal["rag", "qa", "agent", "trajectory", "conversation",
                       "multimodal", "structured_output", "bare"]
Tier = Literal["primary", "secondary", "guardrail"]
PIIPolicy = Literal["redact", "strict", "allow"]


@dataclass(slots=True)
class TraceSummary:
    """Compact stats over the input traces — what the LLM sees in the prompt."""
    count: int
    has_input_pct: float
    has_output_pct: float
    has_context_pct: float
    has_expected_output_pct: float
    has_expected_tool_calls_pct: float
    has_conversation_pct: float
    has_images_pct: float
    has_schema_pct: float
    avg_input_chars: int
    avg_output_chars: int
    pii_label_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class RecommendedEvaluator:
    """One evaluator the bootstrap pipeline ended up recommending."""
    name: str
    tier: Tier
    threshold: float
    rationale: str
    source: Literal["heuristic", "llm", "merged"]


@dataclass(slots=True)
class BootstrapResult:
    """Everything the bootstrap pipeline produced."""
    shape: ProductShape
    summary: TraceSummary
    evaluators: list[RecommendedEvaluator]
    seed_cases: list[EvalCase]
    cost_usd: float
    output_dir: Path
    artifacts: dict[str, Path]


# ─── Evaluator catalogue ──────────────────────────────────────────────────


# Names the LLM is allowed to propose. Everything outside this set is
# rejected and falls back to the heuristic anchor. This is the safety net
# the design doc commits to: the LLM cannot invent evaluators.
_ALLOWED_EVALUATORS = frozenset({
    # RAG primaries
    "Faithfulness", "Hallucination", "ContextPrecision", "ContextRecall",
    # QA primaries
    "AnswerAccuracy",
    # Agent primaries
    "ToolCallAccuracy", "ToolCallNecessity", "PlanQuality", "TaskCompletion",
    # Conversation primaries
    "ConversationRelevance", "KnowledgeRetention", "TurnConsistency",
    # Multimodal primaries
    "VQAFaithfulness", "DocumentGrounding",
    # Safety guardrails
    "Toxicity", "Bias", "PIIEvaluator",
    # Generic guardrails
    "NotEmpty", "SchemaEvaluator", "RegexMatch", "Contains",
    # Secondary
    "Relevance", "Coherence", "BLEU", "ROUGE",
})

# Per-evaluator default threshold when the LLM gives a bogus one.
_DEFAULT_THRESHOLD = 0.7

# Evaluators that need a judge configured (and therefore cost money to
# calibrate). Used to gate calibration onto a small subset.
_LLM_JUDGE_EVALUATORS = frozenset({
    "Faithfulness", "Hallucination", "ContextPrecision", "ContextRecall",
    "AnswerAccuracy", "PlanQuality", "TaskCompletion",
    "ConversationRelevance", "KnowledgeRetention", "TurnConsistency",
    "VQAFaithfulness", "DocumentGrounding",
    "Toxicity", "Bias", "Relevance", "Coherence",
})


# ─── Trace loading + shape inference ──────────────────────────────────────


def load_traces(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL file of traces. Skips empty lines + records a warning
    for malformed lines (returned as the ``_skipped`` key on the row dict
    only when CLI verbose is set; not surfaced by default).
    """
    p = Path(path)
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{i}: malformed JSONL ({e})") from e
        if not isinstance(obj, dict):
            raise ValueError(f"{p}:{i}: expected JSON object, got {type(obj).__name__}")
        if "input" not in obj or not obj["input"]:
            # Skipping silently — design doc says warn but don't crash.
            continue
        rows.append(obj)
    return rows


def summarize_traces(
    traces: list[dict[str, Any]],
    *,
    pii_policy: PIIPolicy = "redact",
) -> tuple[TraceSummary, list[dict[str, Any]]]:
    """Compute summary stats + (optionally) redact PII in the traces.

    Returns ``(summary, sanitized_traces)``. When ``pii_policy='redact'``,
    every text field in each trace is scanned + redacted; when
    ``'strict'``, this function raises ``ValueError`` on the first
    high-confidence detection; when ``'allow'``, scanning still runs (so
    counts populate the summary) but no redaction is applied.
    """
    if not traces:
        return TraceSummary(
            count=0, has_input_pct=0, has_output_pct=0, has_context_pct=0,
            has_expected_output_pct=0, has_expected_tool_calls_pct=0,
            has_conversation_pct=0, has_images_pct=0, has_schema_pct=0,
            avg_input_chars=0, avg_output_chars=0,
        ), []

    n = len(traces)
    has_input = sum(1 for t in traces if t.get("input"))
    has_output = sum(1 for t in traces if t.get("output"))
    has_context = sum(1 for t in traces if t.get("context"))
    has_expected = sum(1 for t in traces if t.get("expected_output"))
    has_tools = sum(1 for t in traces if t.get("expected_tool_calls"))
    has_conversation = sum(1 for t in traces if t.get("conversation"))

    input_chars = sum(len(str(t.get("input", ""))) for t in traces)
    output_chars = sum(len(str(t.get("output", ""))) for t in traces)

    has_images = sum(
        1 for t in traces
        if (t.get("metadata") or {}).get("image_url") or (t.get("metadata") or {}).get("images")
    )
    has_schema = sum(1 for t in traces if (t.get("metadata") or {}).get("schema"))

    # PII scan + maybe-redact
    all_detections: list[list[_pii_scan.Detection]] = []
    sanitized: list[dict[str, Any]] = []
    for trace in traces:
        new_trace = dict(trace)
        scan_targets = []
        for field_name in ("input", "output", "expected_output"):
            v = new_trace.get(field_name)
            if isinstance(v, str):
                scan_targets.append((field_name, v))
        ctx = new_trace.get("context")
        if isinstance(ctx, str):
            scan_targets.append(("context", ctx))
        elif isinstance(ctx, list):
            for j, chunk in enumerate(ctx):
                if isinstance(chunk, str):
                    scan_targets.append((f"context[{j}]", chunk))

        per_trace_dets: list[_pii_scan.Detection] = []
        for field_name, text in scan_targets:
            redacted, dets = _pii_scan.redact(text)
            per_trace_dets.extend(dets)
            if pii_policy == "strict" and dets:
                raise ValueError(
                    f"--strict-pii: trace had {len(dets)} high-confidence "
                    f"detection(s) (labels={[d.label for d in dets]}). "
                    f"Use --allow-pii to override or scrub your traces."
                )
            if pii_policy == "redact":
                # Apply redaction back to the trace
                if "[" in field_name:
                    # context[j]
                    idx = int(field_name.split("[")[1].rstrip("]"))
                    new_ctx = list(new_trace["context"])
                    new_ctx[idx] = redacted
                    new_trace["context"] = new_ctx
                else:
                    new_trace[field_name] = redacted
        all_detections.append(per_trace_dets)
        sanitized.append(new_trace)

    summary = TraceSummary(
        count=n,
        has_input_pct=100 * has_input / n,
        has_output_pct=100 * has_output / n,
        has_context_pct=100 * has_context / n,
        has_expected_output_pct=100 * has_expected / n,
        has_expected_tool_calls_pct=100 * has_tools / n,
        has_conversation_pct=100 * has_conversation / n,
        has_images_pct=100 * has_images / n,
        has_schema_pct=100 * has_schema / n,
        avg_input_chars=input_chars // n,
        avg_output_chars=output_chars // n if has_output else 0,
        pii_label_counts=_pii_scan.summarize(all_detections),
    )
    return summary, sanitized


def infer_product_shape(summary: TraceSummary) -> ProductShape:
    """Heuristic over the trace summary. Matches the multivon_eval.auto
    auto_evaluators heuristic so the LLM proposal layered on top stays
    on the same path.
    """
    # Priority order: most specific shape wins.
    if summary.has_conversation_pct > 30:
        return "conversation"
    if summary.has_expected_tool_calls_pct > 20:
        return "agent"
    if summary.has_images_pct > 20:
        return "multimodal"
    if summary.has_schema_pct > 20:
        return "structured_output"
    if summary.has_context_pct > 40:
        return "rag"
    if summary.has_expected_output_pct > 30:
        return "qa"
    if summary.has_input_pct > 0:
        return "qa"  # bare-input degenerate path
    return "bare"


# ─── LLM proposal ─────────────────────────────────────────────────────────


_PROPOSAL_SYSTEM = """You are an expert LLM evaluation strategist helping a software team \
bootstrap an eval suite for their product.

You will recommend evaluators from a FIXED set. You CANNOT invent evaluator names \
or recommend evaluators outside the allowed list. Your job is to:

1. Pick 4-6 evaluators total across primary/secondary/guardrail tiers, matched to \
the product's shape and the team's stated risks.
2. Suggest a threshold (0.0-1.0) for each, with one-sentence rationale.
3. Be opinionated — picking too many is worse than picking too few.

Allowed evaluators (use these names EXACTLY):
- RAG primaries: Faithfulness, Hallucination, ContextPrecision, ContextRecall
- QA primary: AnswerAccuracy
- Agent primaries: ToolCallAccuracy, ToolCallNecessity, PlanQuality, TaskCompletion
- Conversation primaries: ConversationRelevance, KnowledgeRetention, TurnConsistency
- Multimodal primaries: VQAFaithfulness, DocumentGrounding
- Safety guardrails: Toxicity, Bias, PIIEvaluator
- Generic guardrails: NotEmpty, SchemaEvaluator, RegexMatch, Contains
- Secondary: Relevance, Coherence, BLEU, ROUGE

Output ONLY valid JSON, no prose, no markdown fences. Schema:
{
  "evaluators": [
    {"name": "Faithfulness", "tier": "primary", "threshold": 0.72, "rationale": "..."},
    ...
  ],
  "discussion": "2-3 sentences explaining the overall metric mix and why."
}
"""


def propose_evaluators_via_llm(
    description: str,
    shape: ProductShape,
    summary: TraceSummary,
    sample_traces: list[dict[str, Any]],
    *,
    judge: JudgeConfig | None = None,
) -> tuple[list[RecommendedEvaluator], str, float]:
    """Single LLM call returning a constrained list of recommended evaluators.

    Returns ``(evaluators, discussion, cost_usd)``. ``cost_usd`` is a
    rough estimate based on input + output tokens (we don't have exact
    pricing without the response header).
    """
    resolved = resolve_judge(judge)

    # Trim sample traces to keep the prompt bounded
    sample = sample_traces[:5]
    sample_repr = "\n".join(
        f"  trace{i+1}: {json.dumps({k: v for k, v in t.items() if k in ('input', 'output', 'context', 'expected_output')}, default=str)[:400]}"
        for i, t in enumerate(sample)
    )

    user_prompt = f"""PRODUCT DESCRIPTION:
{description.strip()}

INFERRED PRODUCT SHAPE: {shape}

TRACE STATISTICS (n={summary.count}):
  has input: {summary.has_input_pct:.0f}%
  has output: {summary.has_output_pct:.0f}%
  has context (RAG signal): {summary.has_context_pct:.0f}%
  has expected_output: {summary.has_expected_output_pct:.0f}%
  has expected_tool_calls (agent signal): {summary.has_expected_tool_calls_pct:.0f}%
  has conversation (multi-turn): {summary.has_conversation_pct:.0f}%
  has images (multimodal): {summary.has_images_pct:.0f}%
  has schema (structured output): {summary.has_schema_pct:.0f}%
  avg input chars: {summary.avg_input_chars}
  avg output chars: {summary.avg_output_chars}
  pii detected pre-redaction: {summary.pii_label_counts or 'none'}

SAMPLE TRACES (PII-redacted):
{sample_repr if sample_repr else '  (no traces provided)'}

Pick 4-6 evaluators matched to this product. Return JSON only.
"""

    text = _call_judge(resolved, _PROPOSAL_SYSTEM, user_prompt)
    cost = _estimate_cost(resolved, _PROPOSAL_SYSTEM + user_prompt, text)

    evaluators, discussion = _parse_proposal(text)
    return evaluators, discussion, cost


def _parse_proposal(text: str) -> tuple[list[RecommendedEvaluator], str]:
    """Parse the LLM's JSON response into ``RecommendedEvaluator`` objects.

    Falls back to an empty list + the raw text as discussion if the
    response isn't valid JSON or doesn't match the schema. The caller
    layers in the heuristic anchor when this happens.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.lstrip("`").lstrip("json").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return [], text[:500]

    raw_evals = obj.get("evaluators", [])
    discussion = obj.get("discussion", "")
    if not isinstance(raw_evals, list):
        return [], discussion

    evaluators: list[RecommendedEvaluator] = []
    for raw in raw_evals:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if name not in _ALLOWED_EVALUATORS:
            # Silently drop — the safety net.
            continue
        tier = raw.get("tier") or "primary"
        if tier not in ("primary", "secondary", "guardrail"):
            tier = "primary"
        try:
            threshold = float(raw.get("threshold", _DEFAULT_THRESHOLD))
        except (TypeError, ValueError):
            threshold = _DEFAULT_THRESHOLD
        threshold = max(0.0, min(1.0, threshold))
        rationale = str(raw.get("rationale", "")).strip()[:300]
        evaluators.append(RecommendedEvaluator(
            name=name, tier=tier, threshold=threshold,
            rationale=rationale, source="llm",
        ))
    return evaluators, discussion


# ─── Heuristic anchor (uses auto_evaluators) ──────────────────────────────


def heuristic_recommendations(shape: ProductShape, summary: TraceSummary) -> list[RecommendedEvaluator]:
    """Map the shape inference onto a deterministic evaluator set.

    Mirrors ``multivon_eval.auto.auto_evaluators`` but skips the EvalCase
    machinery since the bootstrap pipeline has a TraceSummary, not a
    single case. Used as a safety net when the LLM proposal is empty or
    invalid; also used to seed the prompt with the default the LLM has
    to argue against.
    """
    recs: list[RecommendedEvaluator] = []

    if shape == "rag":
        recs.extend([
            RecommendedEvaluator("Faithfulness", "primary", 0.7,
                                 "RAG shape detected (input + context). Primary gate against ungrounded claims.",
                                 "heuristic"),
            RecommendedEvaluator("Hallucination", "primary", 0.7,
                                 "RAG → flag claims not in retrieved context.", "heuristic"),
            RecommendedEvaluator("ContextPrecision", "secondary", 0.6,
                                 "Check retrieved context isn't padded with noise.", "heuristic"),
        ])
        if summary.has_expected_output_pct > 30:
            recs.append(RecommendedEvaluator(
                "ContextRecall", "secondary", 0.6,
                "RAG + expected_output present → check context covers the answer.",
                "heuristic",
            ))
    elif shape == "qa":
        if summary.has_expected_output_pct > 30:
            recs.append(RecommendedEvaluator(
                "AnswerAccuracy", "primary", 0.7,
                "QA shape (input + expected_output) → semantic-match.", "heuristic",
            ))
        recs.append(RecommendedEvaluator(
            "Relevance", "secondary", 0.7,
            "QA → check output addresses the question.", "heuristic",
        ))
    elif shape == "agent":
        recs.extend([
            RecommendedEvaluator("ToolCallAccuracy", "primary", 0.7,
                                 "expected_tool_calls present → deterministic tool match.",
                                 "heuristic"),
            RecommendedEvaluator("ToolCallNecessity", "secondary", 0.7,
                                 "Agent → penalise redundant tool calls.", "heuristic"),
        ])
    elif shape == "conversation":
        recs.extend([
            RecommendedEvaluator("ConversationRelevance", "primary", 0.7,
                                 "Multi-turn dialog → relevance across turns.", "heuristic"),
            RecommendedEvaluator("KnowledgeRetention", "primary", 0.7,
                                 "Multi-turn → check the model carries earlier context.",
                                 "heuristic"),
        ])
    elif shape == "multimodal":
        recs.append(RecommendedEvaluator(
            "VQAFaithfulness", "primary", 0.7,
            "metadata.image_url present → image-grounded VQA.", "heuristic",
        ))
    elif shape == "structured_output":
        recs.append(RecommendedEvaluator(
            "SchemaEvaluator", "primary", 1.0,
            "metadata.schema present → structured-output validation.", "heuristic",
        ))

    # Always-on guardrails
    recs.append(RecommendedEvaluator(
        "NotEmpty", "guardrail", 1.0,
        "Trivial sanity check — catches empty/whitespace outputs.", "heuristic",
    ))
    if summary.pii_label_counts:
        recs.append(RecommendedEvaluator(
            "PIIEvaluator", "guardrail", 1.0,
            f"PII detected in traces ({sum(summary.pii_label_counts.values())} hits) → enable PII guardrail.",
            "heuristic",
        ))

    return recs


def merge_recommendations(
    heuristic: list[RecommendedEvaluator],
    llm: list[RecommendedEvaluator],
    *,
    max_count: int = 6,
) -> list[RecommendedEvaluator]:
    """Merge heuristic anchor + LLM proposal into a single list.

    Strategy: LLM picks take priority for the same evaluator name (their
    thresholds + rationales are more contextual). Heuristic picks fill
    in missing tiers — there should always be at least one primary and
    one guardrail. Hard cap at ``max_count``.
    """
    by_name: dict[str, RecommendedEvaluator] = {}
    for rec in llm:
        by_name[rec.name] = RecommendedEvaluator(
            name=rec.name, tier=rec.tier, threshold=rec.threshold,
            rationale=rec.rationale, source="merged" if rec.name in {r.name for r in heuristic} else "llm",
        )
    for rec in heuristic:
        if rec.name not in by_name:
            by_name[rec.name] = rec

    merged = list(by_name.values())

    # Ensure at least one guardrail (NotEmpty is the safe default)
    if not any(r.tier == "guardrail" for r in merged):
        merged.append(RecommendedEvaluator(
            "NotEmpty", "guardrail", 1.0,
            "Default guardrail added by safety net.", "heuristic",
        ))

    # Sort: primary → secondary → guardrail
    tier_order = {"primary": 0, "secondary": 1, "guardrail": 2}
    merged.sort(key=lambda r: (tier_order[r.tier], r.name))

    # Hard cap
    return merged[:max_count]


# ─── Threshold calibration ────────────────────────────────────────────────


def calibrate_thresholds(
    evaluators: list[RecommendedEvaluator],
    traces: list[dict[str, Any]],
    *,
    judge: JudgeConfig | None = None,
    deterministic_sample_size: int = 50,
    llm_judge_sample_size: int = 15,
    seed: int = 1729,
) -> list[RecommendedEvaluator]:
    """Tune each evaluator's threshold against the user's actual traces.

    Strategy: for each evaluator, score the threshold sample, then set
    the suggested threshold to the 25th percentile of observed scores
    (the "baseline" — most user traces should pass; the failures are
    where the eval is actually doing work).

    LLM-judge evaluators use a smaller sample (cost). Deterministic ones
    use the full sample.

    Warns when n_traces < 20: p25 over a small sample has wide CIs and the
    resulting thresholds can swing wildly between bootstraps on the same
    product. Field data showed a Faithfulness threshold dropping from
    the default 0.85 down to 0.50 from p25 of just 8 traces — almost
    certainly noise. Users see the warning and can choose --skip-calibration
    or gather more traces before treating thresholds as authoritative.
    """
    if not traces:
        return evaluators

    if len(traces) < 20:
        import sys
        sys.stderr.write(
            f"\n  ⚠ calibration warning: n_traces={len(traces)} is below 20\n"
            f"    p25-based thresholds have wide CIs at this sample size.\n"
            f"    The bootstrap will still emit calibrated thresholds, but\n"
            f"    don't treat them as authoritative until you've validated\n"
            f"    them against ≥30 representative traces, or use\n"
            f"    --skip-calibration to keep the proposer's defaults.\n\n"
        )
        sys.stderr.flush()

    import multivon_eval as m

    rng = random.Random(seed)
    traces_with_output = [t for t in traces if t.get("output")]
    if not traces_with_output:
        # No outputs to score → no calibration possible
        return evaluators

    deterministic_pool = traces_with_output[:deterministic_sample_size]
    llm_pool = traces_with_output[:llm_judge_sample_size]
    if len(traces_with_output) > llm_judge_sample_size:
        llm_pool = rng.sample(traces_with_output, llm_judge_sample_size)

    out: list[RecommendedEvaluator] = []
    for rec in evaluators:
        if not hasattr(m, rec.name):
            # Unknown name (shouldn't happen post-merge); pass through unchanged
            out.append(rec)
            continue

        is_llm_judge = rec.name in _LLM_JUDGE_EVALUATORS
        pool = llm_pool if is_llm_judge else deterministic_pool
        if not pool:
            out.append(rec)
            continue

        evaluator_cls = getattr(m, rec.name)
        try:
            evaluator = evaluator_cls(judge=judge) if is_llm_judge else evaluator_cls()
        except TypeError:
            try:
                evaluator = evaluator_cls()
            except Exception:
                # Can't instantiate (e.g. needs an arg we don't have) → skip
                out.append(rec)
                continue

        scores: list[float] = []
        for trace in pool:
            case = _trace_to_case(trace)
            output = trace.get("output", "")
            try:
                result = evaluator.evaluate(case, output)
                scores.append(float(result.score))
            except Exception:
                # Calibration error per-trace is non-fatal; skip the trace.
                continue

        if len(scores) >= 3:
            scores.sort()
            # 25th percentile
            idx = max(0, int(0.25 * (len(scores) - 1)))
            calibrated = scores[idx]
            # Clamp to a reasonable range — never go below 0.5 for primary
            # evaluators (otherwise pass-rate becomes meaningless).
            if rec.tier == "primary":
                calibrated = max(0.5, calibrated)
            calibrated = round(calibrated, 2)
            out.append(RecommendedEvaluator(
                name=rec.name, tier=rec.tier, threshold=calibrated,
                rationale=rec.rationale + f" Threshold calibrated to p25 of {len(scores)} traces (was {rec.threshold:.2f}).",
                source="merged",
            ))
        else:
            # Not enough successful scores to calibrate; keep proposed
            out.append(rec)

    return out


def _trace_to_case(trace: dict[str, Any]) -> EvalCase:
    """Project a JSONL trace row into an EvalCase for evaluator scoring."""
    return EvalCase(
        input=str(trace.get("input", "")),
        expected_output=trace.get("expected_output"),
        context=trace.get("context"),
        expected_tool_calls=trace.get("expected_tool_calls"),
        conversation=trace.get("conversation"),
        metadata=trace.get("metadata") or {},
    )


# ─── Seed adversarial cases ──────────────────────────────────────────────


_FAILURE_MODE_FOR_SHAPE: dict[ProductShape, str] = {
    "rag": "ungrounded_claim",
    "qa": "ungrounded_claim",
    "agent": "tool_misuse",
    "conversation": "off_topic",
    "multimodal": "ungrounded_claim",
    "structured_output": "format_violation",
    "trajectory": "tool_misuse",
    "bare": "off_topic",
}


def generate_seed_cases(
    description: str,
    shape: ProductShape,
    *,
    n: int = 30,
    judge: JudgeConfig | None = None,
) -> tuple[list[EvalCase], float]:
    """Wrap ``multivon_eval.auto.generate_adversarial_cases`` for the
    shape's primary failure mode. Returns ``(cases, cost_estimate_usd)``.
    """
    from .auto import generate_adversarial_cases  # lazy to avoid circular

    mode = _FAILURE_MODE_FOR_SHAPE.get(shape, "off_topic")
    try:
        cases = generate_adversarial_cases(description, mode, n=n, judge=judge)
    except Exception:
        return [], 0.0
    # Rough cost estimate: 1 Haiku call returning ~150 tokens × n
    resolved = resolve_judge(judge)
    estimated_output_tokens = 150 * len(cases)
    cost = _estimate_cost_from_tokens(resolved, input_tokens=2000, output_tokens=estimated_output_tokens)
    return cases, cost


# ─── Orchestrator ─────────────────────────────────────────────────────────


def bootstrap(
    description_path: Path | str,
    traces_path: Path | str,
    output_dir: Path | str,
    *,
    judge: JudgeConfig | None = None,
    pii_policy: PIIPolicy = "redact",
    skip_seed_cases: bool = False,
    skip_calibration: bool = False,
    n_seed_cases: int = 30,
    seed: int = 1729,
) -> BootstrapResult:
    """End-to-end bootstrap pipeline.

    The output directory is created (parents=True) and gets four files:
    ``eval_suite.py``, ``seed_cases.jsonl``, ``thresholds.yaml``,
    ``DISCOVERY_REPORT.md``. The function does not import or execute the
    emitted ``eval_suite.py``; that's the user's job.
    """
    description = Path(description_path).read_text(encoding="utf-8")
    raw_traces = load_traces(traces_path)
    summary, sanitized_traces = summarize_traces(raw_traces, pii_policy=pii_policy)
    shape = infer_product_shape(summary)

    heuristic = heuristic_recommendations(shape, summary)
    llm_recs, discussion, llm_cost = propose_evaluators_via_llm(
        description, shape, summary, sanitized_traces, judge=judge,
    )
    merged = merge_recommendations(heuristic, llm_recs)

    if not skip_calibration:
        calibrated = calibrate_thresholds(
            merged, sanitized_traces, judge=judge, seed=seed,
        )
    else:
        calibrated = merged

    if skip_seed_cases:
        seed_cases, seed_cost = [], 0.0
    else:
        seed_cases, seed_cost = generate_seed_cases(
            description, shape, n=n_seed_cases, judge=judge,
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _emit_artifacts(
        out_dir=out_dir,
        description=description,
        shape=shape,
        summary=summary,
        evaluators=calibrated,
        discussion=discussion,
        seed_cases=seed_cases,
    )

    return BootstrapResult(
        shape=shape,
        summary=summary,
        evaluators=calibrated,
        seed_cases=seed_cases,
        cost_usd=round(llm_cost + seed_cost, 4),
        output_dir=out_dir,
        artifacts=artifacts,
    )


# ─── Artifact emitters ────────────────────────────────────────────────────


def _emit_artifacts(
    *,
    out_dir: Path,
    description: str,
    shape: ProductShape,
    summary: TraceSummary,
    evaluators: list[RecommendedEvaluator],
    discussion: str,
    seed_cases: list[EvalCase],
) -> dict[str, Path]:
    """Write the four artifacts and return their paths keyed by short name."""
    paths = {
        "eval_suite": out_dir / "eval_suite.py",
        "seed_cases": out_dir / "seed_cases.jsonl",
        "thresholds": out_dir / "thresholds.yaml",
        "report": out_dir / "DISCOVERY_REPORT.md",
    }

    paths["eval_suite"].write_text(_render_eval_suite_py(shape, evaluators), encoding="utf-8")
    paths["thresholds"].write_text(_render_thresholds_yaml(evaluators), encoding="utf-8")
    paths["report"].write_text(
        _render_report_md(description, shape, summary, evaluators, discussion),
        encoding="utf-8",
    )

    if seed_cases:
        with paths["seed_cases"].open("w", encoding="utf-8") as f:
            for case in seed_cases:
                f.write(json.dumps(_case_to_jsonl(case), default=str))
                f.write("\n")
    else:
        paths["seed_cases"].write_text("", encoding="utf-8")

    return paths


def _render_eval_suite_py(shape: ProductShape, evaluators: list[RecommendedEvaluator]) -> str:
    imports = sorted({r.name for r in evaluators})
    import_line = "from multivon_eval import EvalSuite, EvalCase, " + ", ".join(imports)

    # Build the rationale block as a docstring above the suite-construction
    # so we never have to truncate a multi-sentence rationale into an inline
    # comment. Inline comments get a one-word tier tag only.
    rationale_lines = []
    for r in evaluators:
        rationale_lines.append(f"  - {r.name} ({r.tier}, threshold={r.threshold}):")
        rationale_lines.append(f"      {r.rationale.strip()}")
    rationale_block = "\n".join(rationale_lines)

    eval_lines = []
    for r in evaluators:
        # Most evaluators accept threshold= as a kwarg; SchemaEvaluator and a
        # few take other shapes. We emit threshold= and let a runtime
        # TypeError surface a mismatch if one exists.
        eval_lines.append(f"        {r.name}(threshold={r.threshold}),  # {r.tier}")

    body = f'''"""Eval suite generated by `multivon-eval bootstrap`.

Inferred product shape: {shape}
Edit this file before shipping — these are starting points, not the final answer.

Evaluator rationale (full text — see DISCOVERY_REPORT.md for context):
{rationale_block}
"""
from __future__ import annotations

{import_line}


def make_suite() -> EvalSuite:
    suite = EvalSuite("Bootstrap Suite")
    suite.add_evaluators(
{chr(10).join(eval_lines)}
    )
    # TODO: add your cases here — see seed_cases.jsonl for adversarial starters.
    return suite


if __name__ == "__main__":
    suite = make_suite()
    print(f"Suite ready with {{len(suite.evaluators)}} evaluators.")
    print("Next: load seed_cases.jsonl + your real cases, then suite.run(my_model).")
'''
    return body


def _render_thresholds_yaml(evaluators: list[RecommendedEvaluator]) -> str:
    lines = ["# Thresholds calibrated from your traces (p25 of baseline scores)",
             "# Edit freely — these are starting suggestions.",
             ""]
    for r in evaluators:
        lines.append(f"{r.name}:")
        lines.append(f"  threshold: {r.threshold}")
        lines.append(f"  tier: {r.tier}")
        lines.append(f"  source: {r.source}")
        lines.append(f"  rationale: {json.dumps(r.rationale)}")
        lines.append("")
    return "\n".join(lines)


def _render_why_this_mix(
    shape: ProductShape,
    summary: TraceSummary,
    evaluators: list[RecommendedEvaluator],
    proposer_notes: str,
) -> str:
    """Build the 'Why this mix' prose deterministically from the final list.

    The LLM-proposed `discussion` is unconstrained and routinely mentions
    evaluators not actually in the suite ("we skip Hallucination", "add
    PIIEvaluator as a hard guardrail") even when the suite contradicts
    that. To avoid that drift, this function generates the prose from
    the *committed* evaluator list — single source of truth. The LLM's
    notes are appended at the bottom for context, clearly labeled, so
    users can see the proposer's reasoning without it overriding ground
    truth.
    """
    names = [e.name for e in evaluators]
    primary = [e.name for e in evaluators if e.tier == "primary"]
    secondary = [e.name for e in evaluators if e.tier == "secondary"]
    guardrail = [e.name for e in evaluators if e.tier == "guardrail"]

    lines: list[str] = []
    lines.append(
        f"Your traces look like a **{shape}** product (n={summary.count}). "
        f"This mix targets that shape with {len(evaluators)} evaluator(s) — "
        f"{len(primary)} primary, {len(secondary)} secondary, "
        f"{len(guardrail)} guardrail."
    )

    if primary:
        lines.append(
            "**Primary** ("
            + ", ".join(f"`{n}`" for n in primary)
            + ") are the headline pass/fail gates — what you'd quote in a release-readiness review."
        )
    if secondary:
        lines.append(
            "**Secondary** ("
            + ", ".join(f"`{n}`" for n in secondary)
            + ") are quality signals — useful for diagnosis but not for shipping."
        )
    if guardrail:
        lines.append(
            "**Guardrail** ("
            + ", ".join(f"`{n}`" for n in guardrail)
            + ") catch obvious shape/safety failures that should never reach production."
        )

    # PII guardrail honesty check — flag the gap between detection in
    # traces and inclusion in the suite. The bootstrap previously *said*
    # "add PIIEvaluator as a hard guardrail" while *omitting* it from
    # the suite; this surfaces the mismatch instead of papering over it.
    pii_seen = sum(summary.pii_label_counts.values()) if summary.pii_label_counts else 0
    has_pii_eval = "PIIEvaluator" in names
    if pii_seen and not has_pii_eval:
        lines.append(
            f"> ⚠ **PII gap**: bootstrap detected {pii_seen} PII match(es) in your traces "
            "but the suite above doesn't include `PIIEvaluator`. If PII safety is a real "
            "concern for your product, add it explicitly: "
            "`suite.add_evaluators(PIIEvaluator(jurisdiction=\"hipaa\"))` "
            "(or `\"gdpr\"`, `\"dpdp\"`, `\"all\"`)."
        )

    if proposer_notes.strip():
        lines.append("")
        lines.append("---")
        lines.append("**Proposer notes (LLM reasoning, advisory):**")
        lines.append("")
        lines.append("> " + proposer_notes.strip().replace("\n", "\n> "))

    return "\n\n".join(lines)


def _render_report_md(
    description: str,
    shape: ProductShape,
    summary: TraceSummary,
    evaluators: list[RecommendedEvaluator],
    discussion: str,
) -> str:
    pii_line = (
        "PII redacted before LLM call: "
        + (", ".join(f"{k}={v}" for k, v in summary.pii_label_counts.items()) or "none")
    )

    eval_rows = []
    for r in evaluators:
        eval_rows.append(
            f"| `{r.name}` | {r.tier} | {r.threshold:.2f} | {r.source} | {r.rationale} |"
        )

    summary_rows = [
        f"- traces analyzed: **{summary.count}**",
        f"- inferred product shape: **{shape}**",
        f"- has output: {summary.has_output_pct:.0f}%",
        f"- has context: {summary.has_context_pct:.0f}%",
        f"- has expected_output: {summary.has_expected_output_pct:.0f}%",
        f"- has expected_tool_calls: {summary.has_expected_tool_calls_pct:.0f}%",
        f"- has conversation: {summary.has_conversation_pct:.0f}%",
        f"- avg input chars: {summary.avg_input_chars}",
        f"- avg output chars: {summary.avg_output_chars}",
        f"- {pii_line}",
    ]

    return f"""# Discovery Report

> Generated by `multivon-eval bootstrap`. This is an *eval design review* —
> share with your team before shipping.

## Your product (summary you provided)

{description.strip()[:600]}{"..." if len(description) > 600 else ""}

## Trace evidence

{chr(10).join(summary_rows)}

## Recommended evaluator mix

| Evaluator | Tier | Threshold | Source | Rationale |
|-----------|------|-----------|--------|-----------|
{chr(10).join(eval_rows)}

## Why this mix

{_render_why_this_mix(shape, summary, evaluators, discussion)}

## What's NOT here (and why)

This suite intentionally picks 4-6 evaluators. Picking 12 is worse — every
evaluator added is another threshold to tune, another false-positive risk, and
another noise source. If you find a gap, add ONE evaluator at a time and
measure whether your pass-rate signal improves.

## How to use this

1. Open `eval_suite.py` and add your real cases (or load `seed_cases.jsonl`).
2. Run it: `python eval_suite.py` — verify the suite loads.
3. Wire it into your existing model: `suite.run(your_model_fn)` returns an
   `EvalReport` you can save to JSON / HTML / CSV.
4. Tune thresholds in `thresholds.yaml` if the calibrated p25 values cause
   too many false failures.

## Seed adversarial cases

See `seed_cases.jsonl`. These are designed to STRESS-TEST the primary
evaluators above — a model that passes your seed cases is one that handles
the most common failure mode for `{shape}` shape.
"""


def _case_to_jsonl(case: EvalCase) -> dict[str, Any]:
    """Project an EvalCase into a JSONL-friendly dict."""
    return {
        "input": case.input,
        "expected_output": case.expected_output,
        "context": case.context,
        "expected_tool_calls": case.expected_tool_calls,
        "tags": list(case.tags),
        "metadata": dict(case.metadata),
    }


# ─── Judge call + cost estimation ─────────────────────────────────────────


def _call_judge(judge_cfg: JudgeConfig, system: str, user: str) -> str:
    """Synchronous judge call returning the raw text response."""
    if judge_cfg.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=judge_cfg.model,
            max_tokens=2048,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    if judge_cfg.provider == "openai":
        import openai
        client = openai.OpenAI()
        kw: dict[str, Any] = {}
        is_reasoning = any(judge_cfg.model.startswith(p) for p in ("gpt-5", "o1", "o3"))
        if is_reasoning:
            kw["max_completion_tokens"] = 2048
        else:
            kw["max_tokens"] = 2048
            kw["temperature"] = 0.2
        resp = client.chat.completions.create(
            model=judge_cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kw,
        )
        return resp.choices[0].message.content or ""

    if judge_cfg.provider == "google":
        from google import genai
        client = genai.Client()
        resp = client.models.generate_content(
            model=judge_cfg.model,
            contents=f"{system}\n\n---\n\n{user}",
        )
        return resp.text or ""

    raise ValueError(f"unsupported judge provider for bootstrap: {judge_cfg.provider}")


# Rough per-1M-token cost for Claude Haiku (input/output). Used only for
# the user-facing cost estimate; not authoritative.
_HAIKU_INPUT_USD_PER_1M = 1.0
_HAIKU_OUTPUT_USD_PER_1M = 5.0


def _estimate_cost(judge_cfg: JudgeConfig, prompt: str, response: str) -> float:
    # Rough heuristic: ~4 chars per token
    input_tokens = len(prompt) // 4
    output_tokens = len(response) // 4
    return _estimate_cost_from_tokens(judge_cfg, input_tokens, output_tokens)


def _estimate_cost_from_tokens(judge_cfg: JudgeConfig, input_tokens: int, output_tokens: int) -> float:
    if "haiku" in (judge_cfg.model or "").lower():
        in_cost = input_tokens * _HAIKU_INPUT_USD_PER_1M / 1_000_000
        out_cost = output_tokens * _HAIKU_OUTPUT_USD_PER_1M / 1_000_000
        return round(in_cost + out_cost, 4)
    # Unknown model: surface zero rather than guess
    return 0.0
