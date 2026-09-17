"""
Compare two :class:`EvalReport` snapshots.

The everyday prompt-engineering question: *did this change help?*
Answer it concretely::

    multivon-eval compare baseline.json proposal.json

Returns a structured diff: pass-rate delta, per-case regressions and
improvements, and a McNemar p-value so the reader can tell a real
shift from noise on a small dataset.

Pairing uses stable case IDs and matching content digests. Legacy prompt-only
pairs remain diagnostic and cannot pass a regression gate by default.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

from .experiments import mcnemar_test
from .result import CaseResult, EvalReport, EvalStatus, EVALUATION_STATUSES

if TYPE_CHECKING:
    from .passk import PassKResult


_IMPROVED = "improved"
_REGRESSED = "regressed"
_UNCHANGED = "unchanged"


@dataclass
class CaseDiff:
    """One paired-case comparison row."""

    case_input: str
    baseline_status: EvalStatus
    proposal_status: EvalStatus
    baseline_score: float
    proposal_score: float
    case_id: str | None = None

    @property
    def direction(self) -> str:
        """Pass-state direction: improved / regressed / unchanged.

        Only completed quality verdicts can improve or regress. Error and
        skipped pairs stay in the diff for inspection but do not establish
        a quality direction or enter the significance test.
        """
        if (self.baseline_status not in EVALUATION_STATUSES
                or self.proposal_status not in EVALUATION_STATUSES):
            return _UNCHANGED
        b_pass = self.baseline_status == EvalStatus.PASSED
        p_pass = self.proposal_status == EvalStatus.PASSED
        if b_pass == p_pass:
            return _UNCHANGED
        return _IMPROVED if p_pass else _REGRESSED

    @property
    def score_delta(self) -> float:
        return self.proposal_score - self.baseline_score


@dataclass
class ReportDiff:
    """Structured diff between two :class:`EvalReport` snapshots."""

    baseline_name: str
    proposal_name: str
    baseline_pass_rate: float
    proposal_pass_rate: float
    baseline_avg_score: float
    proposal_avg_score: float
    baseline_errors: int
    proposal_errors: int
    baseline_flaky: int
    proposal_flaky: int
    paired: list[CaseDiff] = field(default_factory=list)
    added: list[CaseResult] = field(default_factory=list)
    removed: list[CaseResult] = field(default_factory=list)
    mcnemar_p: Optional[float] = None
    # Populated only when BOTH reports were multi-run. Displayed values-only:
    # McNemar stays the sole significance test (over paired pass/fail).
    baseline_pass_hat_k: "Optional[PassKResult]" = None
    proposal_pass_hat_k: "Optional[PassKResult]" = None
    identity_issues: list[str] = field(default_factory=list)

    @property
    def pass_rate_delta(self) -> float:
        return self.proposal_pass_rate - self.baseline_pass_rate

    @property
    def avg_score_delta(self) -> float:
        return self.proposal_avg_score - self.baseline_avg_score

    @property
    def errors_delta(self) -> int:
        return self.proposal_errors - self.baseline_errors

    @property
    def flaky_delta(self) -> int:
        return self.proposal_flaky - self.baseline_flaky

    @property
    def regressions(self) -> list[CaseDiff]:
        return [c for c in self.paired if c.direction == _REGRESSED]

    @property
    def improvements(self) -> list[CaseDiff]:
        return [c for c in self.paired if c.direction == _IMPROVED]

    @property
    def unchanged(self) -> list[CaseDiff]:
        return [c for c in self.paired if c.direction == _UNCHANGED]

    def to_dict(self) -> dict:
        return {
            "baseline": {
                "name": self.baseline_name,
                "pass_rate": self.baseline_pass_rate,
                "avg_score": self.baseline_avg_score,
                "errors": self.baseline_errors,
                "flaky": self.baseline_flaky,
            },
            "proposal": {
                "name": self.proposal_name,
                "pass_rate": self.proposal_pass_rate,
                "avg_score": self.proposal_avg_score,
                "errors": self.proposal_errors,
                "flaky": self.proposal_flaky,
            },
            "deltas": {
                "pass_rate": self.pass_rate_delta,
                "avg_score": self.avg_score_delta,
                "errors": self.errors_delta,
                "flaky": self.flaky_delta,
            },
            "paired_count": len(self.paired),
            "identity_verified": not self.identity_issues,
            "identity_issues": self.identity_issues,
            "regressions": [
                {
                    "input": c.case_input,
                    "case_id": c.case_id,
                    "baseline_status": c.baseline_status.value,
                    "proposal_status": c.proposal_status.value,
                    "baseline_score": c.baseline_score,
                    "proposal_score": c.proposal_score,
                }
                for c in self.regressions
            ],
            "improvements": [
                {
                    "input": c.case_input,
                    "case_id": c.case_id,
                    "baseline_status": c.baseline_status.value,
                    "proposal_status": c.proposal_status.value,
                    "baseline_score": c.baseline_score,
                    "proposal_score": c.proposal_score,
                }
                for c in self.improvements
            ],
            "added_count": len(self.added),
            "removed_count": len(self.removed),
            "mcnemar_p": self.mcnemar_p,
        }

    def to_text(self, *, regressions_only: bool = False) -> str:
        """Render a terse terminal diff. ASCII-only — pipes to logs cleanly."""
        lines: list[str] = []
        lines.extend(f"Identity warning: {issue}" for issue in self.identity_issues)
        lines.append(f"Comparing:")
        lines.append(f"  baseline: {self.baseline_name}")
        lines.append(f"  proposal: {self.proposal_name}")
        lines.append("")
        lines.append(
            f"Pass rate:    {self.baseline_pass_rate:.3f} -> "
            f"{self.proposal_pass_rate:.3f}  ({_sign_pp(self.pass_rate_delta)})"
        )
        lines.append(
            f"Avg score:    {self.baseline_avg_score:.3f} -> "
            f"{self.proposal_avg_score:.3f}  ({_sign(self.avg_score_delta)})"
        )
        lines.append(
            f"Errors:       {self.baseline_errors} -> {self.proposal_errors}  "
            f"({_sign_i(self.errors_delta)})"
        )
        lines.append(
            f"Flaky:        {self.baseline_flaky} -> {self.proposal_flaky}  "
            f"({_sign_i(self.flaky_delta)})"
        )
        b_phk, p_phk = self.baseline_pass_hat_k, self.proposal_pass_hat_k
        if (b_phk is not None and p_phk is not None
                and b_phk.value is not None and p_phk.value is not None):
            lines.append(
                f"pass^{b_phk.k}:       "
                f"{b_phk.value:.3f} [{b_phk.ci_low:.3f}, {b_phk.ci_high:.3f}] -> "
                f"{p_phk.value:.3f} [{p_phk.ci_low:.3f}, {p_phk.ci_high:.3f}]"
            )
        if self.added or self.removed:
            lines.append("")
            if self.added:
                lines.append(f"Cases added in proposal: {len(self.added)}")
            if self.removed:
                lines.append(f"Cases removed from baseline: {len(self.removed)}")

        regressions = self.regressions
        if regressions:
            lines.append("")
            lines.append(f"Regressions ({len(regressions)}):")
            for c in regressions:
                lines.append(
                    f"  - {_short(c.case_input)}  "
                    f"{c.baseline_status.value} -> {c.proposal_status.value}  "
                    f"({c.baseline_score:.2f} -> {c.proposal_score:.2f})"
                )

        if not regressions_only:
            improvements = self.improvements
            if improvements:
                lines.append("")
                lines.append(f"Improvements ({len(improvements)}):")
                for c in improvements:
                    lines.append(
                        f"  + {_short(c.case_input)}  "
                        f"{c.baseline_status.value} -> {c.proposal_status.value}  "
                        f"({c.baseline_score:.2f} -> {c.proposal_score:.2f})"
                    )

        if self.mcnemar_p is not None:
            lines.append("")
            lines.append(
                f"Statistical significance: McNemar p = {self.mcnemar_p:.4f}"
                + (
                    "  (significant at p<0.05)" if self.mcnemar_p < 0.05
                    else "  (cannot distinguish from noise on this dataset)"
                )
            )

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render a GitHub-flavored Markdown summary suitable for PR comments."""
        lines: list[str] = []
        lines.append("## Eval comparison")
        lines.append("")
        lines.extend(f"Identity warning: {issue}\n" for issue in self.identity_issues)
        lines.append("| Metric | Baseline | Proposal | Δ |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            f"| Pass rate | {self.baseline_pass_rate:.3f} | "
            f"{self.proposal_pass_rate:.3f} | {_sign_pp(self.pass_rate_delta)} |"
        )
        lines.append(
            f"| Avg score | {self.baseline_avg_score:.3f} | "
            f"{self.proposal_avg_score:.3f} | {_sign(self.avg_score_delta)} |"
        )
        lines.append(
            f"| Errors | {self.baseline_errors} | {self.proposal_errors} | "
            f"{_sign_i(self.errors_delta)} |"
        )
        lines.append(
            f"| Flaky | {self.baseline_flaky} | {self.proposal_flaky} | "
            f"{_sign_i(self.flaky_delta)} |"
        )

        if self.regressions:
            lines.append("")
            lines.append(f"### Regressions ({len(self.regressions)})")
            lines.append("")
            for c in self.regressions:
                lines.append(
                    f"- `{_short(c.case_input)}`: "
                    f"`{c.baseline_status.value}` → `{c.proposal_status.value}`"
                    f" (score {c.baseline_score:.2f} → {c.proposal_score:.2f})"
                )

        if self.improvements:
            lines.append("")
            lines.append(f"### Improvements ({len(self.improvements)})")
            lines.append("")
            for c in self.improvements:
                lines.append(
                    f"- `{_short(c.case_input)}`: "
                    f"`{c.baseline_status.value}` → `{c.proposal_status.value}`"
                    f" (score {c.baseline_score:.2f} → {c.proposal_score:.2f})"
                )

        if self.mcnemar_p is not None:
            lines.append("")
            sig = "**significant** (p<0.05)" if self.mcnemar_p < 0.05 else "not significant"
            lines.append(f"McNemar p = {self.mcnemar_p:.4f} — {sig}")

        return "\n".join(lines)


def _short(text: str, n: int = 64) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "..."


def _sign(x: float) -> str:
    return ("+" if x >= 0 else "") + f"{x:.3f}"


def _sign_i(x: int) -> str:
    return ("+" if x >= 0 else "") + str(x)


def _sign_pp(x: float) -> str:
    return ("+" if x >= 0 else "") + f"{x*100:.1f}pp"


def _pair_by_input(
    baseline_cases: list[CaseResult], proposal_cases: list[CaseResult],
) -> tuple[list[tuple[CaseResult, CaseResult]], list[CaseResult], list[CaseResult]]:
    """Pair cases by ``case_input``, sequential within duplicates.

    Returns (paired, added, removed). Cases in proposal only → added;
    cases in baseline only → removed. When both sides have N copies of
    the same input they get paired 1-1-1 in occurrence order — this
    matches what an operator means when they rerun the same prompt
    twice in a suite.

    Tracks consumption by POSITIONAL INDEX (not ``id()``) so a list
    that happens to contain the same ``CaseResult`` object twice (e.g.
    ``cases = [cr] * 3``) still pairs all three occurrences.
    """
    by_input_b: dict[str, list[int]] = defaultdict(list)
    for idx, cr in enumerate(baseline_cases):
        by_input_b[cr.case_input].append(idx)

    paired: list[tuple[CaseResult, CaseResult]] = []
    added: list[CaseResult] = []
    consumed_b: set[int] = set()  # baseline indices already paired
    for p_cr in proposal_cases:
        bucket = by_input_b.get(p_cr.case_input, [])
        match_idx = None
        for b_idx in bucket:
            if b_idx in consumed_b:
                continue
            match_idx = b_idx
            consumed_b.add(b_idx)
            break
        if match_idx is None:
            added.append(p_cr)
        else:
            paired.append((baseline_cases[match_idx], p_cr))

    removed: list[CaseResult] = []
    for idx, b_cr in enumerate(baseline_cases):
        if idx not in consumed_b:
            removed.append(b_cr)
    return paired, added, removed


def compare_reports(baseline: EvalReport, proposal: EvalReport, *,
                    allow_legacy_identity: bool = False) -> ReportDiff:
    """Compute a structured diff between two :class:`EvalReport` snapshots.

    Pairs cases by stable ID and matching case digest. McNemar
    p-value is computed only over PAIRED cases — added / removed cases
    can't enter a paired test.

    A McNemar test on zero paired cases is undefined; in that case
    ``mcnemar_p`` is set to ``None`` rather than 1.0 (which would mean
    "tested and found no difference" — misleading).
    """
    from .pairing import pair_cases
    paired_pairs, added, removed, identity_issues = pair_cases(
        baseline.case_results, proposal.case_results
    )
    identity_issues.extend(baseline.evidence_issues + proposal.evidence_issues)
    if baseline.runs_per_case != proposal.runs_per_case:
        identity_issues.append("Runs per case changed; repeated-run decisions are not comparable")
    from .dependencies import comparison_issues
    dependency_issues = comparison_issues(baseline.suite_lock, proposal.suite_lock)
    identity_issues.extend(dependency_issues)

    paired_diffs: list[CaseDiff] = []
    for b_cr, p_cr in paired_pairs:
        paired_diffs.append(CaseDiff(
            case_input=b_cr.case_input,
            baseline_status=b_cr.status,
            proposal_status=p_cr.status,
            baseline_score=b_cr.score,
            proposal_score=p_cr.score,
            case_id=b_cr.case_id,
        ))

    # Errors and skips on either side are missing measurements, not failures.
    # Counting them as False would falsely inflate the discordant-pair
    # count toward "regression" or "improvement" depending on the
    # other side.
    mcnemar_pairs = [
        d for d in paired_diffs
        if d.baseline_status in EVALUATION_STATUSES
        and d.proposal_status in EVALUATION_STATUSES
    ]
    legacy_override = (allow_legacy_identity and not baseline.evidence_issues
                       and not proposal.evidence_issues
                       and baseline.suite_lock is None and proposal.suite_lock is None) and all(
        not c.case_id and not c.case_digest and not c.evidence_error and not c.trials
        for c in baseline.case_results + proposal.case_results)
    if mcnemar_pairs and (not identity_issues or legacy_override):
        mcnemar_p = mcnemar_test(
            [d.baseline_status == EvalStatus.PASSED for d in mcnemar_pairs],
            [d.proposal_status == EvalStatus.PASSED for d in mcnemar_pairs],
        )
    else:
        mcnemar_p = None

    # pass^k side-by-side (values only) when both runs were multi-run.
    # k = the largest k both reports can support without extrapolating.
    baseline_phk = proposal_phk = None
    if baseline.runs_per_case > 1 and proposal.runs_per_case > 1:
        k = min(baseline.runs_per_case, proposal.runs_per_case)
        baseline_phk = baseline.pass_hat_k(k)
        proposal_phk = proposal.pass_hat_k(k)

    return ReportDiff(
        baseline_name=baseline.suite_name or "baseline",
        proposal_name=proposal.suite_name or "proposal",
        baseline_pass_rate=baseline.pass_rate,
        proposal_pass_rate=proposal.pass_rate,
        baseline_avg_score=baseline.avg_score,
        proposal_avg_score=proposal.avg_score,
        baseline_errors=baseline.errors,
        proposal_errors=proposal.errors,
        baseline_flaky=baseline.flaky_count,
        proposal_flaky=proposal.flaky_count,
        paired=paired_diffs,
        added=added,
        removed=removed,
        mcnemar_p=mcnemar_p,
        baseline_pass_hat_k=baseline_phk,
        proposal_pass_hat_k=proposal_phk,
        identity_issues=identity_issues,
    )


def _load_report(path: Union[str, Path]) -> EvalReport:
    """Load a JSON-serialized :class:`EvalReport` from disk."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalReport.from_dict(raw)


# ── CLI subcommand (wired by multivon_eval.cli) ────────────────────────────


def _cli(argv: list[str]) -> int:
    from .compare_cli import _cli as compare_cli
    return compare_cli(argv)


__all__ = [
    "CaseDiff", "ReportDiff",
    "compare_reports", "_cli",
]
