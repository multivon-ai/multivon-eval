from __future__ import annotations
import json
import csv
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvalStatus(str, Enum):
    """Terminal status of a case after running the suite.

    Subclasses :class:`str` so the value is JSON-serializable as-is, and so
    ``status == "passed"`` works without unwrapping.

    The split between quality-level outcomes (``PASSED`` / ``FAILED_QUALITY``)
    and infrastructure-level outcomes (``MODEL_ERROR`` / ``JUDGE_ERROR`` /
    ``EVALUATOR_ERROR`` / ``TIMEOUT`` / ``SKIPPED``) is load-bearing — error
    cases are excluded from ``pass_rate`` and ``avg_score`` so a transient
    judge outage doesn't masquerade as a model regression.
    """
    PASSED = "passed"
    FAILED_QUALITY = "failed_quality"
    MODEL_ERROR = "model_error"
    JUDGE_ERROR = "judge_error"
    EVALUATOR_ERROR = "evaluator_error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


# Statuses that count as a real, completed quality measurement. Anything
# else is plumbing — excluded from pass_rate / avg_score denominators.
EVALUATION_STATUSES = frozenset({EvalStatus.PASSED, EvalStatus.FAILED_QUALITY})

# Statuses that indicate something went wrong below the model/quality layer.
ERROR_STATUSES = frozenset({
    EvalStatus.MODEL_ERROR, EvalStatus.JUDGE_ERROR,
    EvalStatus.EVALUATOR_ERROR, EvalStatus.TIMEOUT,
})


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
    Raised when a gate on the run fails (pass_rate, budget, etc.).

    Subclasses SystemExit so CI scripts see exit code 1. Can be caught
    explicitly by callers that want to inspect the report before exiting.

    pass_rate and threshold are set for pass-rate gate failures and may be
    None for other gate types (e.g., budget violations).
    """
    def __init__(self, message: str, pass_rate: float | None = None,
                 threshold: float | None = None) -> None:
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
    judge_error: str | None = None  # set when a judge call raised (transient/auth/etc)
    evaluator_error: str | None = None  # set when an evaluator itself raised (bug)
    skipped: bool = False  # set when the case was deliberately skipped (e.g. tag-filter)
    # ``agent_trace`` is populated when the suite was run with a tracer.
    # Exposed on CaseResult (not just EvalCase) so notebooks can iterate the
    # captured steps from the report without reaching back into the suite.
    agent_trace: list[Any] | None = None  # list[AgentStep] when set; Any to avoid circular import

    # Multi-run fields — populated when suite.run(runs > 1)
    runs: int = 1
    all_scores: list[float] = field(default_factory=list)   # one score per run
    pass_count: int = -1  # -1 = single run (not tracked)

    @property
    def status(self) -> "EvalStatus":
        """High-level outcome of the case.

        Order of precedence (highest → lowest):
          1. Plumbing failures (skipped → model_error → judge_error → evaluator_error)
          2. Quality outcome (passed if all evaluators passed, else failed_quality)

        Used by :attr:`EvalReport.pass_rate` to exclude error cases from the
        denominator — a transient judge outage shouldn't drag pass_rate down
        as if the model regressed.
        """
        if self.skipped:
            return EvalStatus.SKIPPED
        if self.model_error is not None:
            return EvalStatus.MODEL_ERROR
        if self.judge_error is not None:
            return EvalStatus.JUDGE_ERROR
        if self.evaluator_error is not None:
            return EvalStatus.EVALUATOR_ERROR
        # No infrastructure failure → fall through to quality outcome.
        if self.pass_count >= 0:
            return EvalStatus.PASSED if self.pass_count == self.runs else EvalStatus.FAILED_QUALITY
        all_passed = all(r.passed for r in self.results) if self.results else False
        return EvalStatus.PASSED if all_passed else EvalStatus.FAILED_QUALITY

    @property
    def passed(self) -> bool:
        """Whether this case is a clean pass.

        Defined as ``status == EvalStatus.PASSED`` so the two views always
        agree. A case with NO evaluator results is not a pass — there's
        nothing to prove it succeeded, and ``status`` classifies it as
        ``FAILED_QUALITY``. A case in any error state (model/judge/
        evaluator/timeout) is not a pass even if individual evaluator
        results were collected before the error fired.
        """
        return self.status == EvalStatus.PASSED

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
    costs: Any = None  # multivon_eval.costs.Costs or None; populated by suite.run()

    @property
    def total(self) -> int:
        return len(self.case_results)

    @property
    def evaluated(self) -> int:
        """Cases where evaluation actually completed (no error/skip)."""
        return sum(1 for r in self.case_results if r.status in EVALUATION_STATUSES)

    @property
    def errors(self) -> int:
        """Cases where evaluation could not complete (model/judge/evaluator/timeout)."""
        return sum(1 for r in self.case_results if r.status in ERROR_STATUSES)

    @property
    def skipped(self) -> int:
        """Cases deliberately skipped (e.g., via tag filter)."""
        return sum(1 for r in self.case_results if r.status == EvalStatus.SKIPPED)

    @property
    def errors_by_kind(self) -> dict[str, int]:
        """Breakdown of error cases by status — model_error vs judge_error etc."""
        counts: dict[str, int] = {}
        for r in self.case_results:
            if r.status in ERROR_STATUSES:
                counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return counts

    @property
    def passed(self) -> int:
        return sum(1 for r in self.case_results if r.status == EvalStatus.PASSED)

    @property
    def failed(self) -> int:
        """Cases that completed evaluation and failed on quality (NOT errors)."""
        return sum(1 for r in self.case_results if r.status == EvalStatus.FAILED_QUALITY)

    @property
    def pass_rate(self) -> float:
        """Fraction of EVALUATED cases that passed.

        Error and skipped cases are excluded from the denominator — a judge
        outage or a crashed model_fn shouldn't be mistaken for a quality
        regression. Use :attr:`errors` to surface infrastructure problems
        independently.
        """
        denom = self.evaluated
        return self.passed / denom if denom else 0.0

    @property
    def avg_score(self) -> float:
        """Average evaluator score across cases that actually evaluated."""
        evaluated = [r for r in self.case_results if r.status in EVALUATION_STATUSES]
        if not evaluated:
            return 0.0
        return sum(r.score for r in evaluated) / len(evaluated)

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

    def assert_budget(
        self,
        *,
        max_total_cost_usd: float | None = None,
        max_avg_cost_per_case_usd: float | None = None,
        max_total_tokens: int | None = None,
        max_p95_latency_ms: float | None = None,
        max_avg_latency_ms: float | None = None,
    ) -> None:
        """Enforce cost / token / latency budgets on the run.

        Raises :class:`EvalGateFailure` (subclass of SystemExit) if any
        provided budget is exceeded — same exit semantics as the existing
        ``suite.run(fail_threshold=...)`` gate, so it works in CI without
        additional plumbing.

        All thresholds are opt-in: pass only the dimensions you want to
        enforce. None == no limit. Built so CFO-level constraints and
        infra-level SLOs (p95 latency) can both be gated in the same call.

        Inspired by Promptfoo's ``cost`` and ``latency`` assertions, but
        scoped to the whole run rather than per-case — cost is a
        cross-call aggregate concern, not a per-evaluator one.
        """
        violations: list[str] = []

        # Cost gates — only enforceable if pricing data is present.
        if (max_total_cost_usd is not None or max_avg_cost_per_case_usd is not None) and self.costs is not None:
            total = self.costs.total_cost_usd
            if total is None:
                violations.append(
                    "Cost budget requested but at least one model lacks pricing data — "
                    "register pricing via multivon_eval.register_pricing() to enable gating."
                )
            else:
                if max_total_cost_usd is not None and total > max_total_cost_usd:
                    violations.append(
                        f"Total cost ${total:.4f} exceeds budget ${max_total_cost_usd:.4f}"
                    )
                if max_avg_cost_per_case_usd is not None and self.total > 0:
                    avg = total / self.total
                    if avg > max_avg_cost_per_case_usd:
                        violations.append(
                            f"Avg cost/case ${avg:.4f} exceeds budget ${max_avg_cost_per_case_usd:.4f}"
                        )

        # Token gates.
        if max_total_tokens is not None and self.costs is not None:
            if self.costs.total_tokens > max_total_tokens:
                violations.append(
                    f"Total tokens {self.costs.total_tokens:,} exceeds budget {max_total_tokens:,}"
                )

        # Latency gates use the per-case CaseResult.latency_ms timing.
        if max_avg_latency_ms is not None or max_p95_latency_ms is not None:
            latencies = sorted(
                cr.latency_ms for cr in self.case_results if cr.latency_ms is not None
            )
            if latencies:
                if max_avg_latency_ms is not None:
                    avg_ms = sum(latencies) / len(latencies)
                    if avg_ms > max_avg_latency_ms:
                        violations.append(
                            f"Avg latency {avg_ms:.0f}ms exceeds budget {max_avg_latency_ms:.0f}ms"
                        )
                if max_p95_latency_ms is not None:
                    # Linear-interpolation p95 (good enough for budget gating).
                    idx = max(0, int(round(0.95 * (len(latencies) - 1))))
                    p95_ms = latencies[idx]
                    if p95_ms > max_p95_latency_ms:
                        violations.append(
                            f"p95 latency {p95_ms:.0f}ms exceeds budget {max_p95_latency_ms:.0f}ms"
                        )

        if violations:
            raise EvalGateFailure(
                "Budget gate FAILED:\n  • " + "\n  • ".join(violations)
            )

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
        """Wilson score 95% CI on the pass rate.

        Denominator matches :attr:`pass_rate` (``evaluated``) so the CI
        describes the same metric being reported. More reliable than the
        normal approximation for small n.
        """
        from .experiments import wilson_interval
        return wilson_interval(self.passed, self.evaluated, confidence)

    def avg_score_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """Bootstrap CI on the mean score over evaluated cases (no errors)."""
        from .experiments import bootstrap_interval
        scores = [cr.score for cr in self.case_results if cr.status in EVALUATION_STATUSES]
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
                judge_error=c.get("judge_error"),
                evaluator_error=c.get("evaluator_error"),
                skipped=c.get("skipped", False),
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
                    "evaluated": self.evaluated,
                    "passed": self.passed,
                    "failed": self.failed,
                    "errors": self.errors,
                    "errors_by_kind": self.errors_by_kind,
                    "skipped": self.skipped,
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
                    "costs": self.costs.to_dict() if self.costs is not None else None,
                },
                "cases": [
                    {
                        "input": cr.case_input,
                        "output": cr.actual_output,
                        "status": cr.status.value,
                        "model_error": cr.model_error,
                        "judge_error": cr.judge_error,
                        "evaluator_error": cr.evaluator_error,
                        "skipped": cr.skipped,
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
