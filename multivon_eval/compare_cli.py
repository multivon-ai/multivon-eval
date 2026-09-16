"""Command-line comparison of saved evaluation reports."""
from __future__ import annotations

import json

from .compare import _load_report, compare_reports
from .result import EVALUATION_STATUSES


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
    p.add_argument("--allow-legacy-identity", action="store_true",
                   help="Explicitly trust prompt-only pairing for reports without case IDs")
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
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    diff = compare_reports(baseline, proposal, allow_legacy_identity=args.allow_legacy_identity)

    if args.json:
        print(json.dumps(diff.to_dict(), indent=2, default=str))
    elif args.markdown:
        print(diff.to_markdown())
    else:
        print(diff.to_text(regressions_only=args.regressions_only))

    if args.fail_on_regression:
        legacy_override = (args.allow_legacy_identity and not baseline.evidence_issues
                           and not proposal.evidence_issues) and all(
            not c.case_id and not c.case_digest and not c.evidence_error and not c.trials
            for c in baseline.case_results + proposal.case_results)
        if (diff.identity_issues and not legacy_override
                or not diff.paired or diff.added or diff.removed
                or any(d.baseline_status not in EVALUATION_STATUSES
                       or d.proposal_status not in EVALUATION_STATUSES for d in diff.paired)):
            print("Comparison INDETERMINATE: incomplete paired quality measurements.", file=sys.stderr)
            return 2
        if diff.regressions:
            return 1
    return 0

