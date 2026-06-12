"""
Synthetic dataset generation for multivon-eval.

Eliminates the cold-start problem: point at your docs or text and get
eval cases ready to run. No manually labeled data required to get started.

Usage:
    from multivon_eval import generate_from_text, generate_from_file

    # From raw text
    cases = generate_from_text(my_docs, n=20, task="qa")

    # From a file
    cases = generate_from_file("docs/faq.txt", n=15)

    # Hallucination pairs (faithful + hallucinated)
    pairs = generate_hallucination_pairs(my_docs, n=10)

    # Contrast pairs: judge-verified unfaithful twins of labeled cases
    twins, report = generate_contrast_pairs(cases, budget_usd=1.0)

    # Use immediately
    suite.add_cases(cases)
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from .case import EvalCase
from .evaluators.llm_judge import _judge_call


TaskType = Literal["qa", "summarization", "hallucination"]


def generate_from_text(
    text: str,
    n: int = 10,
    task: TaskType = "qa",
    context_window: int = 3000,
    record_spans: bool = True,
    unanswerable_fraction: float = 0.0,
    return_report: bool = False,
):
    """
    Generate eval cases from raw text.

    Args:
        text:           Source text (docs, knowledge base, FAQ, etc.)
        n:              Number of cases to generate.
        task:           "qa" — question/answer pairs with context.
                        "summarization" — doc chunk + expected summary.
                        "hallucination" — faithful answer + expected_output="faithful".
        context_window: Max characters of source to include in each prompt.
        record_spans:   (task="qa") record each case's grounding span —
                        ``metadata["source_span"] = {"start", "end",
                        "sha256_chunk"}`` offsets into ``text`` where the
                        LLM-reported quote was located. When the quote
                        cannot be found, the span is recorded as None
                        (honestly) and counted in the report.
        unanswerable_fraction:
                        (task="qa") fraction of the n cases deliberately
                        NOT answerable from the text. Their expected
                        behavior is refusal (``expected_output=None``,
                        ``metadata["expected_behavior"]="refusal"``).
        return_report:  Also return the GenerationReport (gate
                        accounting). Default False keeps the historical
                        ``list[EvalCase]`` return shape.

    Returns:
        List of EvalCase objects ready to add to a suite — or
        ``(cases, GenerationReport)`` when ``return_report=True``.
    """
    text = text.strip()
    if not 0.0 <= unanswerable_fraction <= 1.0:
        raise ValueError("unanswerable_fraction must be between 0.0 and 1.0")

    if task == "qa":
        cases, report = _generate_qa(
            _chunk_text_with_offsets(text, context_window), n,
            record_spans=record_spans,
            unanswerable_fraction=unanswerable_fraction,
        )
        return (cases, report) if return_report else cases

    chunks = _chunk_text(text, context_window)
    if task == "summarization":
        cases = _generate_summarization(chunks, n)
    elif task == "hallucination":
        cases = _generate_hallucination(chunks, n)
    else:
        raise ValueError(f"Unknown task: {task!r}. Use 'qa', 'summarization', or 'hallucination'.")
    if unanswerable_fraction:
        raise ValueError(f"unanswerable_fraction applies to task='qa' only (got task={task!r})")
    if return_report:
        from .case_gates import GenerationReport
        # Legacy tasks keep their historical (ungated) behavior — the
        # report only does the accounting. See _generate_qa for gates.
        return cases, GenerationReport(
            requested=n, generated=len(cases), accepted=len(cases),
            kind=f"doc_{task}",
        )
    return cases


def generate_from_file(
    path: str,
    n: int = 10,
    task: TaskType = "qa",
    record_spans: bool = True,
    unanswerable_fraction: float = 0.0,
    return_report: bool = False,
):
    """
    Generate eval cases from a text file (.txt, .md, .rst, .py, etc.).

    Args:
        path:   Path to the source file.
        n:      Number of cases to generate.
        task:   See generate_from_text (as do the remaining params).
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return generate_from_text(
        text, n=n, task=task, record_spans=record_spans,
        unanswerable_fraction=unanswerable_fraction,
        return_report=return_report,
    )


def generate_hallucination_pairs(
    text: str,
    n: int = 10,
) -> list[dict]:
    """
    Generate faithful + hallucinated answer pairs for hallucination benchmarking.

    Returns a list of dicts: {question, context, faithful_answer, hallucinated_answer}.
    These can be used to build hallucination detection benchmarks like HaluEval.
    """
    chunks = _chunk_text(text, 3000)
    prompt = f"""You are building a hallucination detection benchmark dataset.

Source text:
\"\"\"
{chunks[0]}
\"\"\"

Generate {n} question-answer pairs. For each, provide:
1. A specific factual question answerable from the text
2. A faithful answer (directly grounded in the text)
3. A hallucinated answer (plausible-sounding but containing at least one false claim)

Return a JSON array. Each element:
{{
  "question": "...",
  "context": "the relevant excerpt from the source text",
  "faithful_answer": "...",
  "hallucinated_answer": "..."
}}

Return ONLY the JSON array, no commentary."""

    try:
        raw = _judge_call(prompt, max_tokens=3000)
        data = _extract_json_array(raw)
        return data[:n]
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}\nRaw response: {raw[:500] if 'raw' in dir() else 'none'}")


# ── Private helpers ────────────────────────────────────────────────────────

def _chunk_text_with_offsets(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split text into overlapping chunks, keeping each chunk's start
    offset into ``text`` (so QA spans can be anchored globally)."""
    if len(text) <= max_chars:
        return [(0, text)]
    chunks: list[tuple[int, str]] = []
    step = max_chars - 200  # 200-char overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + max_chars]
        if chunk.strip():
            chunks.append((i, chunk))
    return chunks


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into overlapping chunks."""
    return [chunk for _, chunk in _chunk_text_with_offsets(text, max_chars)]


def _locate_span(chunk_offset: int, chunk: str, quote: str) -> dict[str, Any] | None:
    """Anchor an LLM-reported quote inside its source chunk.

    Returns ``{"start", "end", "sha256_chunk"}`` with offsets into the
    ORIGINAL source text, or None when the quote can't be located —
    callers record that honestly instead of fabricating offsets.
    """
    if not quote:
        return None
    pos = chunk.find(quote)
    if pos < 0:
        return None
    return {
        "start": chunk_offset + pos,
        "end": chunk_offset + pos + len(quote),
        "sha256_chunk": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
    }


_UNANSWERABLE_NOTE = (
    "deliberately unanswerable from the source text — correct behavior is refusal"
)


def _generate_qa(
    chunks_off: list[tuple[int, str]],
    n: int,
    *,
    record_spans: bool = True,
    unanswerable_fraction: float = 0.0,
):
    """Span-grounded doc-QA (+ optional unanswerable bait questions).

    Returns ``(accepted_cases, GenerationReport)`` — accepted cases passed
    ``gate_well_formed`` + ``gate_duplicate`` (vs this batch).
    """
    from .case_gates import GenerationReport, gate_duplicate, gate_well_formed
    from .provenance import git_info, stamp_metadata_inplace

    n_unanswerable = int(round(n * unanswerable_fraction))
    n_answerable = n - n_unanswerable
    report = GenerationReport(requested=n, kind="doc_qa")
    git = git_info(".")
    accepted: list[EvalCase] = []

    def _admit(case: EvalCase) -> bool:
        report.generated += 1
        if not gate_well_formed(case).passed:
            report.dropped_malformed += 1
            return False
        if not gate_duplicate(case, accepted).passed:
            report.dropped_duplicate += 1
            return False
        accepted.append(case)
        return True

    remaining = n_answerable
    for offset, chunk in chunks_off:
        if remaining <= 0:
            break
        batch = min(max(1, n_answerable // len(chunks_off)), remaining)
        prompt = f"""Generate {batch} question-answer pairs from this text. Questions should require understanding the text, not just keyword lookup.

Text:
\"\"\"
{chunk}
\"\"\"

Return a JSON array. Each element:
{{
  "question": "...",
  "answer": "...",
  "context_excerpt": "the 1-3 sentences COPIED VERBATIM from the text that contain the answer"
}}

Return ONLY the JSON array."""
        try:
            raw = _judge_call(prompt, max_tokens=2000)
            data = _extract_json_array(raw)
        except Exception:
            continue
        for item in data[:batch]:
            excerpt = item.get("context_excerpt", "")
            metadata: dict[str, Any] = {
                "generated": True, "task": "qa",
                "generation": {"kind": "doc_qa", "unanswerable": False},
            }
            span = _locate_span(offset, chunk, excerpt) if record_spans else None
            if record_spans:
                metadata["source_span"] = span
            stamp_metadata_inplace(
                metadata, authored_by="generator:doc_qa", git=git, targets=[],
            )
            case = EvalCase(
                input=item.get("question", ""),
                expected_output=item.get("answer", ""),
                context=excerpt or chunk[:500],
                metadata=metadata,
            )
            if _admit(case) and record_spans and span is None:
                report.spans_missing += 1
        remaining -= len(data[:batch])

    remaining = n_unanswerable
    for offset, chunk in chunks_off:
        if remaining <= 0:
            break
        batch = min(max(1, n_unanswerable // len(chunks_off)), remaining)
        prompt = f"""Generate {batch} questions that are topically related to this text but CANNOT be answered from it (they ask for facts the text does not contain). A well-behaved assistant given only this text should refuse or say it does not know.

Text:
\"\"\"
{chunk}
\"\"\"

Return a JSON array. Each element:
{{
  "question": "..."
}}

Return ONLY the JSON array."""
        try:
            raw = _judge_call(prompt, max_tokens=1500)
            data = _extract_json_array(raw)
        except Exception:
            continue
        for item in data[:batch]:
            metadata = {
                "generated": True, "task": "qa",
                "expected_behavior": "refusal",
                "note": _UNANSWERABLE_NOTE,
                "generation": {"kind": "doc_qa", "unanswerable": True},
            }
            if record_spans:
                metadata["source_span"] = None  # nothing to ground, by design
            stamp_metadata_inplace(
                metadata, authored_by="generator:doc_qa", git=git, targets=[],
            )
            _admit(EvalCase(
                input=item.get("question", ""),
                expected_output=None,
                context=chunk,
                metadata=metadata,
            ))
        remaining -= len(data[:batch])

    report.accepted = len(accepted)
    return accepted, report


def _generate_summarization(chunks: list[str], n: int) -> list[EvalCase]:
    cases = []
    for chunk in chunks[:n]:
        prompt = f"""Write a 2-3 sentence faithful summary of this text. Include only information present in the text.

Text:
\"\"\"
{chunk[:2000]}
\"\"\"

Return a JSON object:
{{
  "summary": "..."
}}

Return ONLY the JSON object."""

        try:
            raw = _judge_call(prompt, max_tokens=300)
            data = _extract_json_object(raw)
            cases.append(EvalCase(
                input="Summarize the following text.",
                expected_output=data.get("summary", ""),
                context=chunk[:2000],
                metadata={"generated": True, "task": "summarization"},
            ))
        except Exception:
            continue

        if len(cases) >= n:
            break

    return cases[:n]


def _generate_hallucination(chunks: list[str], n: int) -> list[EvalCase]:
    cases = []
    per_chunk = max(1, n // len(chunks))
    remaining = n

    for chunk in chunks:
        if remaining <= 0:
            break
        batch = min(per_chunk, remaining)
        prompt = f"""Generate {batch} QA pairs from this text. Each should have a faithful answer grounded in the text.

Text:
\"\"\"
{chunk}
\"\"\"

Return a JSON array. Each element:
{{
  "question": "...",
  "faithful_answer": "answer grounded in the text",
  "context_excerpt": "the 1-3 relevant sentences from the text"
}}

Return ONLY the JSON array."""

        try:
            raw = _judge_call(prompt, max_tokens=2000)
            data = _extract_json_array(raw)
            for item in data[:batch]:
                cases.append(EvalCase(
                    input=item.get("question", ""),
                    expected_output="faithful",
                    context=item.get("context_excerpt", chunk[:500]),
                    metadata={
                        "generated": True,
                        "task": "hallucination",
                        "faithful_answer": item.get("faithful_answer", ""),
                    },
                ))
            remaining -= len(data[:batch])
        except Exception:
            continue

    return cases[:n]


def _extract_json_array(text: str) -> list:
    text = text.strip()
    # Try to find a JSON array in the response
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


# ── Contrast pairs (implementation in _contrast.py; re-exported here so the
#    public import path stays multivon_eval.generate.generate_contrast_pairs;
#    the split keeps this file under the ~500-line house cap) ───────────────
from ._contrast import generate_contrast_pairs  # noqa: E402
