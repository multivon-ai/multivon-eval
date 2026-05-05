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


class EvalGateFailure(SystemExit):
    """
    Raised by EvalSuite.run() when pass_rate < fail_threshold.

    Subclasses SystemExit so CI scripts see exit code 1. Can be caught
    explicitly by callers that want to inspect the report before exiting.
    """
    def __init__(self, message: str, pass_rate: float, threshold: float) -> None:
        super().__init__(message)
        self.pass_rate = pass_rate
        self.threshold = threshold


@dataclass
class CaseResult:
    """All evaluator results for a single test case."""
    case_input: str
    actual_output: str
    results: list[EvalResult]
    latency_ms: float = 0.0
    tags: list[str] = field(default_factory=list)
    model_error: str | None = None  # set when the model call raised an exception

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
class CalibrationResult:
    """Judge accuracy against human-labeled ground truth."""
    n: int
    agreement: float
    precision: float
    recall: float
    f1: float
    by_evaluator: dict[str, dict[str, float]] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Judge Calibration — {self.n} labeled cases",
            f"  Agreement:  {self.agreement:.1%}",
            f"  Precision:  {self.precision:.1%}",
            f"  Recall:     {self.recall:.1%}",
            f"  F1 Score:   {self.f1:.1%}",
        ]
        if self.by_evaluator:
            lines.append("  By evaluator:")
            for ev, stats in self.by_evaluator.items():
                lines.append(f"    {ev}: agreement={stats['agreement']:.1%}  F1={stats['f1']:.1%}")
        return "\n".join(lines)


@dataclass
class PairwiseResult:
    """Pairwise judge verdict for one case."""
    case_input: str
    output_a: str
    output_b: str
    winner: str   # "A", "B", or "Tie"
    reason: str = ""


@dataclass
class PairwiseReport:
    """Aggregated results from suite.run_pairwise()."""
    suite_name: str
    model_a_id: str
    model_b_id: str
    results: "list[PairwiseResult]"

    @property
    def wins_a(self) -> int:
        return sum(1 for r in self.results if r.winner == "A")

    @property
    def wins_b(self) -> int:
        return sum(1 for r in self.results if r.winner == "B")

    @property
    def ties(self) -> int:
        return sum(1 for r in self.results if r.winner == "Tie")

    @property
    def total(self) -> int:
        return len(self.results)

    def p_value(self) -> float:
        """Sign test p-value (H0: wins_a == wins_b, ties excluded)."""
        import math as _math
        n = self.wins_a + self.wins_b
        if n == 0:
            return 1.0
        stat = (abs(self.wins_a - self.wins_b) - 1) ** 2 / n
        from .experiments import _norm_cdf
        return 2 * (1 - _norm_cdf(_math.sqrt(stat)))

    def __str__(self) -> str:
        p = self.p_value()
        sig = "significant" if p < 0.05 else "not significant"
        label_a = self.model_a_id or "Model A"
        label_b = self.model_b_id or "Model B"
        if self.wins_a > self.wins_b:
            verdict = label_a
        elif self.wins_b > self.wins_a:
            verdict = label_b
        else:
            verdict = "Tie"
        return (
            f"Pairwise: {label_a} vs {label_b}  ({self.total} cases)\n"
            f"  {label_a} wins: {self.wins_a}  "
            f"{label_b} wins: {self.wins_b}  Ties: {self.ties}\n"
            f"  Verdict: {verdict}  (p={p:.3f}, {sig})"
        )


@dataclass
class EvalReport:
    """Aggregated results for an entire eval suite run."""
    suite_name: str
    case_results: list[CaseResult]
    model_id: str = ""
    judge_reliability: float | None = None

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

    @property
    def failed_cases(self) -> list["CaseResult"]:
        """Cases where at least one evaluator failed."""
        return [cr for cr in self.case_results if not cr.passed]

    @property
    def passed_cases(self) -> list["CaseResult"]:
        """Cases where all evaluators passed."""
        return [cr for cr in self.case_results if cr.passed]

    def filter_by_evaluator(self, name: str) -> list["CaseResult"]:
        """Cases where the named evaluator failed. Useful for drilling into a specific check."""
        return [
            cr for cr in self.case_results
            if any(r.evaluator == name and not r.passed for r in cr.results)
        ]

    def sample(self, n: int, *, failed_only: bool = False) -> list["CaseResult"]:
        """
        Random sample of n cases. Pass failed_only=True to sample from failures only.
        Useful for spot-checking a large eval run without reading every result.
        """
        import random
        pool = self.failed_cases if failed_only else self.case_results
        return random.sample(pool, min(n, len(pool)))

    def pass_rate_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """Wilson score 95% CI on the pass rate. More reliable than normal approx for small n."""
        from .experiments import wilson_interval
        return wilson_interval(self.passed, self.total, confidence)

    def avg_score_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """Bootstrap CI on the mean score. Use when n < 30 or score distribution is skewed."""
        from .experiments import bootstrap_interval
        scores = [cr.score for cr in self.case_results]
        return bootstrap_interval(scores, confidence)

    def score_percentiles(self, percentiles: list[int] | None = None) -> dict[str, float]:
        """
        Score distribution percentiles. Reveals bimodal patterns avg_score hides.

        A model that scores 0.95 or 0.40 (never in between) has the same avg_score
        as one that scores 0.67 consistently — but they behave very differently.

        Args:
            percentiles: Percentile ranks to compute (default [10, 50, 90]).

        Returns:
            Dict like {"p10": 0.41, "p50": 0.82, "p90": 0.96}.
        """
        if percentiles is None:
            percentiles = [10, 50, 90]
        scores = sorted(cr.score for cr in self.case_results)
        if not scores:
            return {}
        n = len(scores)
        result: dict[str, float] = {}
        for p in percentiles:
            idx = (p / 100) * (n - 1)
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            frac = idx - lo
            val = scores[lo] * (1 - frac) + scores[hi] * frac
            result[f"p{p}"] = round(val, 4)
        return result

    def scores_by_tag(self) -> dict[str, float]:
        """Average score per tag across all tagged cases."""
        totals: dict[str, list[float]] = {}
        for cr in self.case_results:
            for tag in cr.tags:
                totals.setdefault(tag, []).append(cr.score)
        return {k: round(sum(v) / len(v), 4) for k, v in totals.items()}

    def passed_by_tag(self) -> dict[str, float]:
        """Pass rate per tag across all tagged cases."""
        totals: dict[str, list[bool]] = {}
        for cr in self.case_results:
            for tag in cr.tags:
                totals.setdefault(tag, []).append(cr.passed)
        return {k: round(sum(v) / len(v), 4) for k, v in totals.items()}

    def count_by_tag(self) -> dict[str, int]:
        """Number of cases per tag."""
        counts: dict[str, int] = {}
        for cr in self.case_results:
            for tag in cr.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts

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

    @classmethod
    def from_dict(cls, data: dict) -> "EvalReport":
        """Reconstruct an EvalReport from the dict produced by to_json()."""
        case_results = []
        for c in data.get("cases", []):
            results = [
                EvalResult(
                    evaluator=e["name"],
                    score=e["score"],
                    passed=e["passed"],
                    reason=e.get("reason", ""),
                )
                for e in c.get("evaluators", [])
            ]
            runs = c.get("runs", 1)
            all_scores = c.get("all_scores") or []
            # Legacy round-trip: reconstruct if all_scores not stored (pre-fix JSON)
            if runs > 1 and not all_scores:
                score = c.get("score", 0.0)
                std = c.get("score_std", 0.0)
                all_scores = [score] * runs if std == 0 else [score + std, score - std] + [score] * max(0, runs - 2)
            cr = CaseResult(
                case_input=c["input"],
                actual_output=c["output"],
                results=results,
                latency_ms=c.get("latency_ms", 0.0),
                tags=c.get("tags", []),
                model_error=c.get("model_error"),
                runs=runs,
                all_scores=all_scores,
                pass_count=-1,
            )
            if runs > 1:
                rpr = c.get("run_pass_rate", 1.0)
                cr.pass_count = round(rpr * runs)
            case_results.append(cr)
        return cls(
            suite_name=data.get("suite", ""),
            model_id=data.get("model", ""),
            case_results=case_results,
        )

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
                    "pass_rate_ci_95": list(self.pass_rate_ci()),
                    "avg_score": round(self.avg_score, 4),
                    "avg_score_ci_95": list(self.avg_score_ci()),
                    "score_percentiles": self.score_percentiles(),
                    "runs_per_case": self.runs_per_case,
                    "flaky_count": self.flaky_count,
                    "stability_score": round(self.stability_score, 4),
                    "judge_reliability": self.judge_reliability,
                    "by_evaluator": {k: round(v, 4) for k, v in self.scores_by_evaluator().items()},
                },
                "cases": [
                    {
                        "input": cr.case_input,
                        "output": cr.actual_output,
                        "model_error": cr.model_error,
                        "passed": cr.passed,
                        "score": round(cr.score, 4),
                        "score_std": round(cr.score_std, 4),
                        "all_scores": cr.all_scores,
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

    def to_html(self) -> str:
        """Return a self-contained HTML report string."""
        from .reporters.html import to_html as _to_html
        return _to_html(self)

    def save_html(self, path: str) -> None:
        """Write the HTML report to path."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())

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
