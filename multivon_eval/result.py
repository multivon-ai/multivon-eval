from __future__ import annotations
import json
import csv
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Result of a single evaluator on a single case."""
    evaluator: str
    score: float          # 0.0 – 1.0
    passed: bool
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    """All evaluator results for a single test case."""
    case_input: str
    actual_output: str
    results: list[EvalResult]
    latency_ms: float = 0.0
    tags: list[str] = field(default_factory=list)

    # Multi-run fields — populated when suite.run(runs > 1)
    runs: int = 1
    all_scores: list[float] = field(default_factory=list)   # one score per run
    pass_count: int = -1  # -1 = single run (not tracked)

    @property
    def passed(self) -> bool:
        if self.pass_count >= 0:
            return self.pass_count == self.runs
        return all(r.passed for r in self.results)

    @property
    def score(self) -> float:
        if self.all_scores:
            return sum(self.all_scores) / len(self.all_scores)
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def score_std(self) -> float:
        if len(self.all_scores) < 2:
            return 0.0
        mean = self.score
        return math.sqrt(sum((s - mean) ** 2 for s in self.all_scores) / len(self.all_scores))

    @property
    def run_pass_rate(self) -> float:
        """Fraction of runs that passed. 1.0 when runs=1."""
        if self.pass_count < 0:
            return 1.0 if self.passed else 0.0
        return self.pass_count / self.runs if self.runs > 0 else 0.0

    @property
    def is_flaky(self) -> bool:
        """True if the case sometimes passes and sometimes fails across runs."""
        if self.runs <= 1 or self.pass_count < 0:
            return False
        return 0 < self.pass_count < self.runs


@dataclass
class EvalReport:
    """Aggregated results for an entire eval suite run."""
    suite_name: str
    case_results: list[CaseResult]
    model_id: str = ""

    @property
    def total(self) -> int:
        return len(self.case_results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.case_results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_score(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.score for r in self.case_results) / len(self.case_results)

    @property
    def flaky_count(self) -> int:
        return sum(1 for r in self.case_results if r.is_flaky)

    @property
    def stability_score(self) -> float:
        """Fraction of cases that behave consistently (always pass or always fail)."""
        if not self.case_results:
            return 1.0
        consistent = sum(1 for r in self.case_results if not r.is_flaky)
        return consistent / self.total

    @property
    def runs_per_case(self) -> int:
        if self.case_results:
            return self.case_results[0].runs
        return 1

    def scores_by_evaluator(self) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for cr in self.case_results:
            for r in cr.results:
                totals.setdefault(r.evaluator, []).append(r.score)
        return {k: sum(v) / len(v) for k, v in totals.items()}

    def passed_by_evaluator(self) -> dict[str, float]:
        totals: dict[str, list[bool]] = {}
        for cr in self.case_results:
            for r in cr.results:
                totals.setdefault(r.evaluator, []).append(r.passed)
        return {k: sum(v) / len(v) for k, v in totals.items()}

    def to_json(self) -> str:
        def _ser(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return obj.__dict__
            return str(obj)
        return json.dumps(
            {
                "suite": self.suite_name,
                "model": self.model_id,
                "summary": {
                    "total": self.total,
                    "passed": self.passed,
                    "failed": self.failed,
                    "pass_rate": round(self.pass_rate, 4),
                    "avg_score": round(self.avg_score, 4),
                    "runs_per_case": self.runs_per_case,
                    "flaky_count": self.flaky_count,
                    "stability_score": round(self.stability_score, 4),
                    "by_evaluator": {k: round(v, 4) for k, v in self.scores_by_evaluator().items()},
                },
                "cases": [
                    {
                        "input": cr.case_input,
                        "output": cr.actual_output,
                        "passed": cr.passed,
                        "score": round(cr.score, 4),
                        "score_std": round(cr.score_std, 4),
                        "run_pass_rate": round(cr.run_pass_rate, 4),
                        "is_flaky": cr.is_flaky,
                        "runs": cr.runs,
                        "latency_ms": round(cr.latency_ms, 1),
                        "tags": cr.tags,
                        "evaluators": [
                            {
                                "name": r.evaluator,
                                "score": round(r.score, 4),
                                "passed": r.passed,
                                "reason": r.reason,
                            }
                            for r in cr.results
                        ],
                    }
                    for cr in self.case_results
                ],
            },
            default=_ser,
            indent=2,
        )

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    def save_csv(self, path: str) -> None:
        rows = []
        for cr in self.case_results:
            for r in cr.results:
                rows.append({
                    "input": cr.case_input[:200],
                    "output": cr.actual_output[:200],
                    "evaluator": r.evaluator,
                    "score": round(r.score, 4),
                    "score_std": round(cr.score_std, 4),
                    "run_pass_rate": round(cr.run_pass_rate, 4),
                    "is_flaky": cr.is_flaky,
                    "passed": r.passed,
                    "reason": r.reason[:300],
                    "latency_ms": round(cr.latency_ms, 1),
                    "tags": ",".join(cr.tags),
                })
        with open(path, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
