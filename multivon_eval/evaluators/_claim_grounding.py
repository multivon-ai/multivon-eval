"""Complete verification of a bounded set of extracted text claims."""
from __future__ import annotations

from ..exceptions import JudgeUnavailable
from ..result import EvalResult


def validate_claim_limit(value):
    if type(value) is not int or value < 1:
        raise ValueError("max_claims must be a positive integer")


def evaluate_claims(*, name, context, output, judge, threshold, max_claims, call, extract, parse):
    validate_claim_limit(max_claims)
    raw = call(
        "Extract every factual claim from this response as a JSON list of strings.\n"
        "Include only verifiable statements. Return ONLY a JSON array.\n\n"
        f"Response:\n{output}\n\nJSON array:", judge, max_tokens=512,
    )
    extracted = extract(raw)
    if extracted is None:
        raise ValueError(f"{name}: could not extract a claims JSON array from judge reply")
    if any(not isinstance(claim, str) or not claim.strip() for claim in extracted):
        raise ValueError(f"{name}: claims must be non-empty strings")
    claims = list(dict.fromkeys(claim.strip() for claim in extracted))
    metadata = {
        "protocol": "multivon.text-claim-grounding/v2",
        "threshold": threshold, "max_claims": max_claims,
        "extracted_claims": len(extracted), "unique_claims": len(claims),
        "verified_claims": 0, "claims": claims, "claims_response": raw,
        "verdict_responses": [],
        "coverage_scope": "Unique extracted claims only; extraction completeness is unverified",
    }
    if not claims or len(claims) > max_claims:
        reason = ("No verifiable claims extracted; faithfulness was not measured" if not claims else
                  f"Extracted {len(claims)} unique claims, exceeding max_claims={max_claims}; "
                  "faithfulness was not measured. Increase max_claims to verify the complete extracted set")
        return EvalResult(name, 0.0, False, "[skipped] " + reason, {**metadata, "skipped": True})
    verdicts, reasons = [], []
    for claim in claims:
        answer = call(
            f"Context:\n{context}\n\nClaim: {claim}\n\n"
            'Is this claim fully supported by the context? Answer with only "Yes" or "No".',
            judge, max_tokens=100,
        )
        verdict = parse(answer)
        verdicts.append(verdict)
        metadata["verdict_responses"].append(answer)
        reasons.append(f"{'?' if verdict is None else '✓' if verdict else '✗'} {claim[:80]}")
    unknown = sum(v is None for v in verdicts)
    if unknown:
        raise JudgeUnavailable(
            f"judge verdicts parseable for only {len(claims) - unknown} of {len(claims)} "
            f"claim(s) ({unknown} UNKNOWN) — insufficient verdict coverage to score; "
            "every extracted claim requires a verdict",
            provider=judge.provider, model=judge.model,
        )
    supported = sum(verdicts)
    score = supported / len(claims)
    metadata.update(verified_claims=len(claims), supported_claims=supported)
    return EvalResult(name, score, score >= threshold,
                      f"{supported}/{len(claims)} claims grounded\n" + "\n".join(reasons), metadata)
