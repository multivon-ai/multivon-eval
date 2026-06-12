"""Contrast-pair generation — implementation behind
``multivon_eval.generate.generate_contrast_pairs`` (split out to keep
``generate.py`` under the house line cap; import it from there).

Generalizes ``generate_hallucination_pairs``: instead of asking one LLM
call to invent question + faithful + hallucinated answers from raw text,
this starts from EXISTING labeled cases and proposes a minimally-edited
unfaithful twin of each expected answer, then (optionally) verifies with
the Faithfulness judge that the flip is real before accepting it.
"""
from __future__ import annotations

import sys
import uuid
from collections.abc import Sequence
from typing import Any

from .case import EvalCase

_CONTRAST_SYSTEM = (
    "You construct contrast pairs for hallucination evals. Given a context "
    "and a faithful answer, produce a minimally-edited UNFAITHFUL twin: "
    "change exactly one fact so the answer is no longer supported by the "
    "context, keeping the original style and length. Return ONLY a JSON "
    'object: {"unfaithful_answer": "...", "changed_fact": "what you changed"}'
)

_CONTRAST_EXPECTED_BEHAVIOR = (
    "fail: this twin's answer contains a minimally-edited false fact — "
    "Faithfulness on (context, unfaithful_answer) must score below threshold"
)


def generate_contrast_pairs(
    cases: Sequence[EvalCase],
    judge=None,
    verify: bool = True,
    budget_usd: float = 1.0,
):
    """Judge-verified minimally-different failing twins of labeled cases.

    For each case with ``context`` + ``expected_output``, ONE LLM call
    proposes a minimally-edited UNFAITHFUL twin of the expected answer
    (one changed fact, same style/length). With ``verify=True`` the twin
    is only accepted if Faithfulness on (context, twin-answer) actually
    scores below its threshold — i.e. the flip is real; unconfirmed twins
    are counted in ``report.dropped_unverified``, never kept.

    Accepted twins share a ``pair_id`` (uuid) with their source case —
    written into BOTH cases' metadata — so downstream comparisons are
    genuinely paired (McNemar). The twin carries ``expected_output=None``
    (it has no gold answer; its expectation is *failing* Faithfulness,
    spelled out in ``metadata["expected_behavior"]``).

    ``budget_usd`` is a HARD ceiling on estimated judge spend: when
    exceeded, remaining cases are skipped and the partial twin list is
    returned (the ``simulate`` _Spend pattern).

    Returns ``(twins, GenerationReport)``.
    """
    # Lazy imports: generate.py imports this module at its bottom, and
    # discover/simulate are heavyweight — resolve them at call time.
    from .case_gates import GenerationReport, gate_duplicate, gate_well_formed
    from .discover import _call_judge, _estimate_cost, _estimate_cost_from_tokens
    from .evaluators.llm_judge import Faithfulness
    from .generate import _extract_json_object
    from .judge import resolve_judge
    from .provenance import git_info, read_provenance, stamp_metadata_inplace
    from .simulate import _Spend

    resolved = resolve_judge(judge)
    eligible = [c for c in cases if c.context and c.expected_output]
    report = GenerationReport(requested=len(eligible), kind="contrast")
    # Up-front estimate: proposal call (+ claim extraction + claim checks
    # when verifying). The ceiling below is what actually stops spend.
    per_case_est = _estimate_cost_from_tokens(resolved, 1500, 300) * (4 if verify else 1)
    print(
        f"[contrast] {len(eligible)} eligible case(s) "
        f"(skipped {len(cases) - len(eligible)} without context+expected_output) — "
        f"estimated spend ≈${per_case_est * len(eligible):.4f} · "
        f"hard ceiling ${budget_usd:.2f}",
        file=sys.stderr,
    )

    spend = _Spend(budget_usd)
    git = git_info(".")
    twins: list[EvalCase] = []
    budget_stopped = 0
    for case in eligible:
        if spend.exceeded():
            budget_stopped += 1  # hard stop — partials preserved, never lost
            continue
        user = (
            f'CONTEXT:\n"""\n{case.context_str()}\n"""\n\n'
            f'FAITHFUL ANSWER:\n"""\n{case.expected_output}\n"""\n\n'
            f"Return ONLY the JSON object."
        )
        try:
            raw = _call_judge(resolved, _CONTRAST_SYSTEM, user)
        except Exception:
            continue  # proposal call failed — nothing generated for this case
        spend.add(_estimate_cost(resolved, _CONTRAST_SYSTEM + user, raw))
        try:
            obj = _extract_json_object(raw)
        except Exception:
            obj = {}
        twin_answer = obj.get("unfaithful_answer")
        if not isinstance(twin_answer, str) or not twin_answer.strip():
            continue  # malformed proposal — visible as requested - generated

        report.generated += 1
        verified = False
        judge_score: float | None = None
        if verify:
            ev = Faithfulness(judge=judge)
            try:
                res = ev.evaluate(
                    EvalCase(input=case.input, context=case.context), twin_answer,
                )
            except Exception:
                report.dropped_unverified += 1
                continue
            spend.add(_estimate_cost_from_tokens(resolved, 2000, 300))
            if res.metadata.get("skipped") or res.score >= ev.threshold:
                # The flip is NOT real (judge still finds the twin faithful,
                # or could not score it) — dropped and counted, never kept.
                report.dropped_unverified += 1
                continue
            verified, judge_score = True, float(res.score)

        status, prov = read_provenance(case.metadata)
        src_uid = prov.get("case_uid") if (status == "ok" and prov) else None
        pair_id = uuid.uuid4().hex
        metadata: dict[str, Any] = {
            "pair_id": pair_id,
            "unfaithful_answer": twin_answer,
            "changed_fact": obj.get("changed_fact"),
            "expected_behavior": _CONTRAST_EXPECTED_BEHAVIOR,
            "generation": {
                "kind": "contrast",
                "expectation": "fail",
                "verified": verified,
                "judge_score": judge_score,
                "source_case_uid": src_uid,
            },
        }
        stamp_metadata_inplace(
            metadata, authored_by="generator:contrast", git=git, targets=[],
        )
        twin = EvalCase(
            input=case.input, expected_output=None,
            context=case.context, metadata=metadata,
        )
        if not gate_well_formed(twin).passed:
            report.dropped_malformed += 1
            continue
        if not gate_duplicate(twin, twins).passed:
            report.dropped_duplicate += 1
            continue
        case.metadata["pair_id"] = pair_id  # BOTH sides carry the pair id
        twins.append(twin)

    if budget_stopped:
        print(
            f"[contrast] budget ceiling ${budget_usd:.2f} hit — "
            f"{budget_stopped} case(s) not attempted; partial twins preserved",
            file=sys.stderr,
        )
    report.accepted = len(twins)
    return twins, report


__all__ = ["generate_contrast_pairs"]
