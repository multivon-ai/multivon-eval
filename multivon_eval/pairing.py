"""Evidence-aware report pairing; prompt equality is not case identity."""
from __future__ import annotations

from collections import Counter

from .result import CaseResult
from .trials import trial_integrity_issues


def pair_cases(baseline: list[CaseResult], proposal: list[CaseResult]):
    issues = [issue for row in baseline + proposal for issue in trial_integrity_issues(row)]
    if any((row.case_id or row.case_digest) and not row.trials for row in baseline + proposal):
        issues.append("Identified cases are missing retained trial evidence")
    if any(not c.case_id or not c.case_digest or c.evidence_error
           for c in baseline + proposal):
        issues.append("Missing case identity or incomplete evidence; legacy prompt pairs are diagnostic only")
        # Keep legacy reports inspectable, but callers must not use these pairs
        # for significance or a successful gate without an explicit override.
        if all(not c.case_id and not c.case_digest for c in baseline + proposal):
            from .compare import _pair_by_input
            paired, added, removed = _pair_by_input(baseline, proposal)
            return paired, added, removed, issues
    counts_b = Counter(c.case_id for c in baseline)
    counts_p = Counter(c.case_id for c in proposal)
    ambiguous = {key for key, n in counts_b.items() if n > 1}
    ambiguous.update(key for key, n in counts_p.items() if n > 1)
    if ambiguous:
        issues.append("Duplicate case IDs cannot be paired: " + ", ".join(sorted(str(k) for k in ambiguous)))
    by_id = {c.case_id: (i, c) for i, c in enumerate(baseline)
             if c.case_id and c.case_id not in ambiguous}
    paired, added, consumed = [], [], set()
    for candidate in proposal:
        match = by_id.get(candidate.case_id)
        if (match is None or candidate.case_id in ambiguous or not candidate.case_digest
                or candidate.evidence_error):
            added.append(candidate)
            continue
        i, original = match
        if original.case_digest != candidate.case_digest or original.evidence_error:
            issues.append(f"Case definition changed or evidence incomplete: {candidate.case_id}")
            added.append(candidate)
            continue
        consumed.add(i)
        paired.append((original, candidate))
    removed = [c for i, c in enumerate(baseline) if i not in consumed]
    return paired, added, removed, issues
