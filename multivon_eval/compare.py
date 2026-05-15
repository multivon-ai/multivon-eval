"""
Compare two :class:`EvalReport` snapshots.

The everyday prompt-engineering question: *did this change help?*
Answer it concretely::

    multivon-eval compare baseline.json proposal.json

Returns a structured diff: pass-rate delta, per-case regressions and
improvements, and a McNemar p-value so the reader can tell a real
shift from noise on a small dataset.

Pairing convention: cases are paired by ``case_input`` (sequential
within duplicates). Cases present in only one side are reported as
``added`` / ``removed`` rather than silently dropped.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from .experiments import mcnemar_test
from .result import CaseResult, EvalReport, EvalStatus


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

    @property
    def direction(self) -> str:
        """Pass-state direction: improved / regressed / unchanged.

        Improvement = quality-fail or error → pass.
        Regression  = pass → quality-fail or error.
        Status-state changes within "error space" (e.g. judge_error
        → evaluator_error) count as ``unchanged`` for direction, but
        callers can still inspect the underlying statuses.

        SKIPPED on either side is treated as ``unchanged`` for the
        direction, because the case was deliberately not evaluated on
        at least one side — we don't have signal about whether the
        model behavior changed. Codex round-1 finding (otherwise
        skipped→pass would be reported as an improvement, which is
        misleading).
        """
        if self.baseline_status == EvalStatus.SKIPPED or self.proposal_status == EvalStatus.SKIPPED:
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
            "regressions": [
                {
                    "input": c.case_input,
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
        sign = lambda x: ("+" if x >= 0 else "") + f"{x:.3f}"
        sign_i = lambda x: ("+" if x >= 0 else "") + str(x)
        sign_pp = lambda x: ("+" if x >= 0 else "") + f"{x*100:.1f}pp"
        lines.append(f"Comparing:")
        lines.append(f"  baseline: {self.baseline_name}")
        lines.append(f"  proposal: {self.proposal_name}")
        lines.append("")
        lines.append(
            f"Pass rate:    {self.baseline_pass_rate:.3f} -> "
            f"{self.proposal_pass_rate:.3f}  ({sign_pp(self.pass_rate_delta)})"
        )
        lines.append(
            f"Avg score:    {self.baseline_avg_score:.3f} -> "
            f"{self.proposal_avg_score:.3f}  ({sign(self.avg_score_delta)})"
        )
        lines.append(
            f"Errors:       {self.baseline_errors} -> {self.proposal_errors}  "
            f"({sign_i(self.errors_delta)})"
        )
        lines.append(
            f"Flaky:        {self.baseline_flaky} -> {self.proposal_flaky}  "
            f"({sign_i(self.flaky_delta)})"
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
        sign_pp = lambda x: ("+" if x >= 0 else "") + f"{x*100:.1f}pp"
        sign = lambda x: ("+" if x >= 0 else "") + f"{x:.3f}"
        sign_i = lambda x: ("+" if x >= 0 else "") + str(x)

        lines: list[str] = []
        lines.append("## Eval comparison")
        lines.append("")
        lines.append("| Metric | Baseline | Proposal | Δ |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            f"| Pass rate | {self.baseline_pass_rate:.3f} | "
            f"{self.proposal_pass_rate:.3f} | {sign_pp(self.pass_rate_delta)} |"
        )
        lines.append(
            f"| Avg score | {self.baseline_avg_score:.3f} | "
            f"{self.proposal_avg_score:.3f} | {sign(self.avg_score_delta)} |"
        )
        lines.append(
            f"| Errors | {self.baseline_errors} | {self.proposal_errors} | "
            f"{sign_i(self.errors_delta)} |"
        )
        lines.append(
            f"| Flaky | {self.baseline_flaky} | {self.proposal_flaky} | "
            f"{sign_i(self.flaky_delta)} |"
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
    ``cases = [cr] * 3``) still pairs all three occurrences. Codex
    round-1 finding.
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


def compare_reports(baseline: EvalReport, proposal: EvalReport) -> ReportDiff:
    """Compute a structured diff between two :class:`EvalReport` snapshots.

    Pairs cases by ``case_input`` (sequential within duplicates). McNemar
    p-value is computed only over PAIRED cases — added / removed cases
    can't enter a paired test.

    A McNemar test on zero paired cases is undefined; in that case
    ``mcnemar_p`` is set to ``None`` rather than 1.0 (which would mean
    "tested and found no difference" — misleading).
    """
    paired_pairs, added, removed = _pair_by_input(
        baseline.case_results, proposal.case_results
    )

    paired_diffs: list[CaseDiff] = []
    for b_cr, p_cr in paired_pairs:
        paired_diffs.append(CaseDiff(
            case_input=b_cr.case_input,
            baseline_status=b_cr.status,
            proposal_status=p_cr.status,
            baseline_score=b_cr.score,
            proposal_score=p_cr.score,
        ))

    # Exclude paired-but-skipped cases from McNemar — a skipped case
    # on either side is a deliberate "not evaluated," not a failure.
    # Counting them as False would falsely inflate the discordant-pair
    # count toward "regression" or "improvement" depending on the
    # other side. Codex round-1 finding.
    mcnemar_pairs = [
        d for d in paired_diffs
        if d.baseline_status != EvalStatus.SKIPPED
        and d.proposal_status != EvalStatus.SKIPPED
    ]
    if mcnemar_pairs:
        mcnemar_p = mcnemar_test(
            [d.baseline_status == EvalStatus.PASSED for d in mcnemar_pairs],
            [d.proposal_status == EvalStatus.PASSED for d in mcnemar_pairs],
        )
    else:
        mcnemar_p = None

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
    )


def _load_report(path: Union[str, Path]) -> EvalReport:
    """Load a JSON-serialized :class:`EvalReport` from disk."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalReport.from_dict(raw)


# ── CLI subcommand (wired by multivon_eval.cli) ────────────────────────────


def _cli(argv: list[str]) -> int:
    """Argparse subcommand: ``multivon-eval compare …``"""
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="multivon-eval compare",
        description=(
            "Compare two eval report JSON files. Reports pass-rate delta, "
            "per-case regressions and improvements, and a McNemar p-value."
        ),
    )
    p.add_argument("baseline", help="Baseline report JSON")
    p.add_argument("proposal", help="Proposal report JSON to compare")
    p.add_argument(
        "--regressions-only", action="store_true",
        help="Show only regressions in the per-case section (good for CI gates)",
    )
    p.add_argument(
        "--markdown", action="store_true",
        help="Emit GitHub-flavored Markdown (suitable for PR comments)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit the diff as JSON",
    )
    p.add_argument(
        "--fail-on-regression", action="store_true",
        help="Exit 1 if any regressions are detected (CI gate)",
    )

    args = p.parse_args(argv)

    try:
        baseline = _load_report(args.baseline)
        proposal = _load_report(args.proposal)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    diff = compare_reports(baseline, proposal)

    if args.json:
        print(json.dumps(diff.to_dict(), indent=2, default=str))
    elif args.markdown:
        print(diff.to_markdown())
    else:
        print(diff.to_text(regressions_only=args.regressions_only))

    if args.fail_on_regression and diff.regressions:
        return 1
    return 0


__all__ = [
    "CaseDiff", "ReportDiff",
    "compare_reports", "_cli",
]
