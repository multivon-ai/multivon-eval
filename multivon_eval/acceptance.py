"""Explicit release decisions from required checks, coverage and task slices."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal

from .case_manifest import digest
from .result import CaseResult, EvalGateFailure, EvalReport
from .trials import trial_integrity_issues


def _probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")


def _count(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class CheckRequirement:
    """Acceptance rate over cases; repeated trials are grouped within a case.

    A case passes a check only if every selected trial passes it. Coverage is
    the fraction of cases with that check measured on every selected trial.
    A critical check rejects on any observed failure, even with missing data.
    """
    evaluator: str
    min_pass_rate: float = 1.0
    min_cases: int = 1
    min_coverage: float = 1.0
    critical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, str) or not self.evaluator.strip():
            raise ValueError("evaluator must be nonempty")
        _probability("min_pass_rate", self.min_pass_rate)
        _probability("min_coverage", self.min_coverage)
        _count("min_cases", self.min_cases)
        if type(self.critical) is not bool:
            raise ValueError("critical must be boolean")


@dataclass(frozen=True)
class SliceRequirement:
    """A required tag slice. All policy checks also apply within this slice."""
    tag: str
    min_cases: int = 1
    min_pass_rate: float = 1.0
    min_coverage: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.tag, str) or not self.tag.strip():
            raise ValueError("tag must be nonempty")
        _count("min_cases", self.min_cases)
        _probability("min_pass_rate", self.min_pass_rate)
        _probability("min_coverage", self.min_coverage)


@dataclass(frozen=True)
class AcceptanceFinding:
    kind: Literal["quality", "evidence"]
    code: str
    scope: str
    message: str


@dataclass(frozen=True)
class AcceptanceResult:
    decision: Literal["accept", "reject", "indeterminate"]
    policy_digest: str
    findings: tuple[AcceptanceFinding, ...]
    measurements: tuple[dict, ...]

    @property
    def exit_code(self) -> int:
        return {"accept": 0, "reject": 1, "indeterminate": 2}[self.decision]

    def to_dict(self) -> dict:
        return {"schema": "multivon.acceptance/v1", "decision": self.decision,
                "exit_code": self.exit_code, "policy_digest": self.policy_digest,
                "findings": [asdict(f) for f in self.findings],
                "measurements": list(self.measurements)}

    def assert_accepted(self) -> None:
        if self.decision == "accept":
            return
        exc = EvalGateFailure(f"Acceptance {self.decision}: " + "; ".join(f.message for f in self.findings))
        exc.code = self.exit_code
        exc.acceptance = self
        raise exc


@dataclass(frozen=True)
class AcceptancePolicy:
    """A preregistered deterministic release contract, not a statistical test.

    By default every recorded attempt counts, so successful retries cannot hide
    an unsafe execution. Use final_attempt only for explicitly recoverable
    workflows. Unknown data never counts as a successful measurement.
    """
    checks: tuple[CheckRequirement, ...]
    slices: tuple[SliceRequirement, ...] = ()
    min_cases: int = 1
    min_source_groups: int = 0
    max_error_rate: float = 0.0
    trial_scope: Literal["all_attempts", "final_attempt"] = "all_attempts"
    require_trials: bool = True
    require_identity: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "slices", tuple(self.slices))
        if not self.checks or any(not isinstance(c, CheckRequirement) for c in self.checks):
            raise ValueError("At least one CheckRequirement is required")
        if any(not isinstance(s, SliceRequirement) for s in self.slices):
            raise ValueError("slices must contain SliceRequirement values")
        if len({c.evaluator for c in self.checks}) != len(self.checks):
            raise ValueError("Duplicate check requirement")
        if len({s.tag for s in self.slices}) != len(self.slices):
            raise ValueError("Duplicate slice requirement")
        _count("min_cases", self.min_cases)
        _count("min_source_groups", self.min_source_groups, 0)
        _probability("max_error_rate", self.max_error_rate)
        if self.trial_scope not in {"all_attempts", "final_attempt"}:
            raise ValueError("trial_scope must be all_attempts or final_attempt")
        if type(self.require_trials) is not bool or type(self.require_identity) is not bool:
            raise ValueError("require_trials and require_identity must be boolean")

    @property
    def digest(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict:
        return {"schema": "multivon.policy/v1", **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> AcceptancePolicy:
        values = dict(data)
        if values.pop("schema", None) != "multivon.policy/v1":
            raise ValueError("Unsupported acceptance policy schema")
        values["checks"] = tuple(CheckRequirement(**c) for c in values["checks"])
        values["slices"] = tuple(SliceRequirement(**s) for s in values.get("slices", []))
        return cls(**values)

    def evaluate(self, report: EvalReport) -> AcceptanceResult:
        findings: list[AcceptanceFinding] = []
        measurements: list[dict] = []
        rows = report.case_results

        def evidence(code: str, message: str, scope: str = "all") -> None:
            findings.append(AcceptanceFinding("evidence", code, scope, message))

        if len(rows) < self.min_cases:
            evidence("insufficient_cases", f"Need {self.min_cases} cases; observed {len(rows)}")
        for issue in report.evidence_issues:
            evidence("report_evidence", issue)
        if self.require_identity:
            if any(not r.case_id or not r.case_digest for r in rows):
                evidence("missing_identity", "Some cases have no verified identity")
            if any(n > 1 for key, n in Counter(r.case_id for r in rows).items() if key):
                evidence("duplicate_identity", "Duplicate case IDs cannot supply independent case counts")
        if any(r.evidence_error for r in rows):
            evidence("incomplete_evidence", "Some case evidence could not be captured")
        for row in rows:
            for issue in trial_integrity_issues(row):
                evidence("inconsistent_trial_evidence", f"{row.case_id}: {issue}")
        if self.require_trials and any(not r.trials for r in rows):
            evidence("missing_trials", "Some cases have no saved trials")

        selected = {id(r): self._trials(r) for r in rows}
        errored = sum(any(t.get("model_error") is not None or t.get("judge_error") is not None
                         or t.get("evaluator_error") is not None for t in selected[id(r)]) for r in rows)
        error_rate = errored / len(rows) if rows else None
        measurements.append({"scope": "all", "metric": "case_error_rate", "cases": len(rows),
                             "errored_cases": errored, "value": error_rate})
        if error_rate is not None and error_rate > self.max_error_rate:
            evidence("error_budget", f"Case error rate {error_rate:.3f} exceeds {self.max_error_rate:.3f}")

        if self.min_source_groups:
            sources = set()
            missing_source = False
            for row in rows:
                groups = {t.get("case", {}).get("source_id") for t in selected[id(row)]}
                if not groups or None in groups or len(groups) != 1:
                    missing_source = True
                sources.update(g for g in groups if g is not None)
            if missing_source:
                evidence("missing_source_groups", "Some cases have missing or inconsistent source groups")
            if len(sources) < self.min_source_groups:
                evidence("insufficient_source_groups", f"Need {self.min_source_groups} source groups; observed {len(sources)}")
            measurements.append({"scope": "all", "metric": "source_groups", "value": len(sources)})

        scopes = [("all", rows, None)] + [
            (f"tag:{s.tag}", [r for r in rows if s.tag in r.tags], s) for s in self.slices]
        for scope, members, slice_rule in scopes:
            if slice_rule and len(members) < slice_rule.min_cases:
                evidence("insufficient_slice", f"{scope}: need {slice_rule.min_cases} cases; observed {len(members)}", scope)
            for requirement in self.checks:
                complete = passed = 0
                critical_failure = False
                for row in members:
                    verdicts = []
                    for trial in selected[id(row)]:
                        matches = [e for e in trial["evaluators"] if e["name"] == requirement.evaluator]
                        usable = len(matches) == 1 and trial.get("model_error") is None
                        match = matches[0] if len(matches) == 1 else None
                        usable = usable and not match.get("metadata", {}).get("skipped")
                        usable = usable and not match.get("metadata", {}).get("error_kind")
                        usable = usable and type(match.get("passed")) is bool
                        if not usable:
                            verdicts.append(None)
                        else:
                            verdicts.append(match["passed"])
                            critical_failure |= requirement.critical and not match["passed"]
                    if verdicts and all(v is not None for v in verdicts):
                        complete += 1
                        passed += int(all(verdicts))
                coverage = complete / len(members) if members else 0.0
                pass_rate = passed / complete if complete else None
                min_cases = max(requirement.min_cases, slice_rule.min_cases if slice_rule else 1)
                min_coverage = max(requirement.min_coverage, slice_rule.min_coverage if slice_rule else 0)
                min_rate = max(requirement.min_pass_rate, slice_rule.min_pass_rate if slice_rule else 0)
                measurements.append({"scope": scope, "metric": requirement.evaluator,
                                     "cases": len(members), "measured_cases": complete,
                                     "passed_cases": passed, "coverage": coverage, "pass_rate": pass_rate})
                if complete < min_cases:
                    evidence("insufficient_measurements", f"{scope}/{requirement.evaluator}: need {min_cases} measured cases; observed {complete}", scope)
                if coverage < min_coverage:
                    evidence("check_coverage", f"{scope}/{requirement.evaluator}: coverage {coverage:.3f} below {min_coverage:.3f}", scope)
                if critical_failure:
                    findings.append(AcceptanceFinding("quality", "critical_failure", scope,
                                                     f"{scope}/{requirement.evaluator}: critical invariant failed"))
                elif pass_rate is not None and pass_rate < min_rate:
                    findings.append(AcceptanceFinding("quality", "quality_threshold", scope,
                                                     f"{scope}/{requirement.evaluator}: pass rate {pass_rate:.3f} below {min_rate:.3f}"))
        decision = ("reject" if any(f.kind == "quality" for f in findings) else
                    "indeterminate" if findings else "accept")
        return AcceptanceResult(decision, self.digest, tuple(findings), tuple(measurements))

    def _trials(self, row: CaseResult) -> list[dict]:
        if row.trials:
            trials = [t.data for t in row.trials]
            if self.trial_scope == "final_attempt":
                last = max(t["attempt"] for t in trials)
                trials = [t for t in trials if t["attempt"] == last]
            return trials
        if self.require_trials:
            return []
        return [{"evaluators": [{"name": r.evaluator, "passed": r.passed, "metadata": r.metadata}
                                for r in row.results], "model_error": row.model_error,
                 "judge_error": row.judge_error, "evaluator_error": row.evaluator_error}]


def _cli(argv: list[str]) -> int:
    import argparse
    import json
    import sys
    from pathlib import Path
    parser = argparse.ArgumentParser(prog="multivon-eval gate")
    parser.add_argument("report")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        policy = AcceptancePolicy.from_dict(json.loads(Path(args.policy).read_text(encoding="utf-8")))
        report = EvalReport.from_dict(json.loads(Path(args.report).read_text(encoding="utf-8")))
        result = policy.evaluate(report)
        payload = json.dumps(result.to_dict(), indent=2) + "\n"
        if args.output:
            output = Path(args.output)
            if output.resolve() in {Path(args.policy).resolve(), Path(args.report).resolve()}:
                raise ValueError("Decision output cannot overwrite the report or policy")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return result.exit_code
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Acceptance INDETERMINATE: {exc}", file=sys.stderr)
        return 2
