"""
Experiment tracking for multivon-eval.

Records every suite run locally so you can compare results across model
versions, prompt changes, or time. No cloud required — stored as JSONL
in ~/.multivon/experiments/.

Usage:
    from multivon_eval import Experiment

    # Wrap a suite run in an experiment
    exp = Experiment("rag-pipeline")
    report = suite.run(model_fn)
    run_id = exp.record(report, tags={"model": "gpt-4o", "prompt_v": "2"})

    # Compare two runs
    exp.compare(run_id_a, run_id_b)

    # List all runs
    exp.history()

CLI:
    multivon-eval experiments list
    multivon-eval experiments compare <run_a> <run_b>
"""
from __future__ import annotations
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .result import EvalReport

__all__ = [
    "Experiment", "list_experiments", "compare_experiments",
    "wilson_interval", "bootstrap_interval", "runs_needed",
    "min_detectable_effect", "cohens_h", "benjamini_hochberg",
    "mcnemar_test", "bayesian_interval",
]


def _experiments_dir() -> Path:
    base = Path(os.environ.get("MULTIVON_HOME", Path.home() / ".multivon"))
    d = base / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class RunRecord:
    run_id: str
    suite_name: str
    model_id: str
    timestamp: str
    pass_rate: float
    avg_score: float
    total: int
    passed: int
    failed: int
    scores_by_evaluator: dict[str, float]
    tags: dict[str, str] = field(default_factory=dict)
    runs_per_case: int = 1
    flaky_count: int = 0
    stability_score: float = 1.0
    pass_by_evaluator: dict[str, float] = field(default_factory=dict)
    # 0.7.0: the denominator behind pass_rate (excludes error/skipped cases).
    # Defaulted from ``total`` for legacy records that didn't store it.
    evaluated: int = -1
    errors: int = 0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "suite_name": self.suite_name,
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "total": self.total,
            "evaluated": self.evaluated,
            "errors": self.errors,
            "passed": self.passed,
            "failed": self.failed,
            "scores_by_evaluator": self.scores_by_evaluator,
            "pass_by_evaluator": self.pass_by_evaluator,
            "tags": self.tags,
            "runs_per_case": self.runs_per_case,
            "flaky_count": self.flaky_count,
            "stability_score": self.stability_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(
            run_id=d["run_id"],
            suite_name=d["suite_name"],
            model_id=d.get("model_id", ""),
            timestamp=d["timestamp"],
            pass_rate=d["pass_rate"],
            avg_score=d["avg_score"],
            total=d["total"],
            # Legacy records (pre-0.7.0) didn't store evaluated; default to total.
            evaluated=d.get("evaluated", d["total"]),
            errors=d.get("errors", 0),
            passed=d["passed"],
            failed=d["failed"],
            scores_by_evaluator=d.get("scores_by_evaluator", {}),
            pass_by_evaluator=d.get("pass_by_evaluator", {}),
            tags=d.get("tags", {}),
            runs_per_case=d.get("runs_per_case", 1),
            flaky_count=d.get("flaky_count", 0),
            stability_score=d.get("stability_score", 1.0),
        )


class Experiment:
    """
    Track and compare suite runs over time.

    Each Experiment has a name (usually your pipeline name) and stores
    run records in ~/.multivon/experiments/<name>.jsonl.
    """

    def __init__(self, name: str):
        self.name = name
        self._path = _experiments_dir() / f"{name}.jsonl"

    def record(
        self,
        report: EvalReport,
        tags: dict[str, str] | None = None,
        run_id: str | None = None,
    ) -> str:
        """
        Save a run to the experiment history.

        Args:
            report:  EvalReport from suite.run().
            tags:    Optional metadata (model name, prompt version, etc.)
            run_id:  Optional explicit run ID. Auto-generated if not provided.

        Returns:
            The run_id (use it later for compare()).
        """
        run_id = run_id or _short_id()
        record = RunRecord(
            run_id=run_id,
            suite_name=report.suite_name,
            model_id=report.model_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            pass_rate=round(report.pass_rate, 4),
            avg_score=round(report.avg_score, 4),
            total=report.total,
            evaluated=report.evaluated,
            errors=report.errors,
            passed=report.passed,
            failed=report.failed,
            scores_by_evaluator={k: round(v, 4) for k, v in report.scores_by_evaluator().items()},
            pass_by_evaluator={k: round(v, 4) for k, v in report.passed_by_evaluator().items()},
            tags=tags or {},
            runs_per_case=report.runs_per_case,
            flaky_count=report.flaky_count,
            stability_score=round(report.stability_score, 4),
        )
        with open(self._path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        print(f"  [experiment] run saved → {run_id} ({self.name})")
        return run_id

    def history(self, n: int = 20) -> list[RunRecord]:
        """Return the last n runs, newest first."""
        if not self._path.exists():
            return []
        lines = self._path.read_text().strip().splitlines()
        records = [RunRecord.from_dict(json.loads(l)) for l in lines if l.strip()]
        return list(reversed(records))[:n]

    def compare(self, run_id_a: str, run_id_b: str) -> None:
        """
        Print a side-by-side comparison of two runs.

        The second run is treated as "new" — changes are shown as deltas.
        """
        all_runs = {r.run_id: r for r in self.history(n=1000)}

        if run_id_a not in all_runs:
            raise ValueError(f"Run '{run_id_a}' not found in experiment '{self.name}'")
        if run_id_b not in all_runs:
            raise ValueError(f"Run '{run_id_b}' not found in experiment '{self.name}'")

        a = all_runs[run_id_a]
        b = all_runs[run_id_b]

        _print_comparison(a, b)

    def print_history(self, n: int = 10) -> None:
        """Print the last n runs as a table."""
        runs = self.history(n)
        if not runs:
            print(f"No runs recorded for experiment '{self.name}'.")
            return

        print(f"\n  Experiment: {self.name}")
        print(f"  {'Run ID':<12} {'Timestamp':<22} {'Model':<20} {'Pass rate':>10} {'Avg score':>10} Tags")
        print(f"  {'-'*90}")
        for r in runs:
            ts = r.timestamp[:19].replace("T", " ")
            model = (r.model_id or "-")[:18]
            tags = " ".join(f"{k}={v}" for k, v in r.tags.items())
            print(f"  {r.run_id:<12} {ts:<22} {model:<20} {r.pass_rate:>9.1%} {r.avg_score:>10.4f} {tags}")
        print()


def compare_experiments(exp_name: str, run_id_a: str, run_id_b: str) -> None:
    """Convenience function for CLI use."""
    Experiment(exp_name).compare(run_id_a, run_id_b)


def list_experiments() -> list[str]:
    """Return names of all experiments that have recorded runs."""
    d = _experiments_dir()
    return [f.stem for f in sorted(d.glob("*.jsonl"))]


# ── Internal ───────────────────────────────────────────────────────────────

def _short_id() -> str:
    return uuid.uuid4().hex[:8]


def _delta(a: float, b: float) -> str:
    diff = b - a
    if abs(diff) < 0.0001:
        return "  (no change)"
    return f"  {diff:+.4f}"


def _norm_cdf(x: float) -> float:
    return (1 + math.erf(x / math.sqrt(2))) / 2


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF via Newton-Raphson (for p in (0.0001, 0.9999))."""
    p = max(1e-9, min(1 - 1e-9, p))
    x = 0.0
    for _ in range(50):
        fx = _norm_cdf(x) - p
        dfx = math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
        if dfx == 0:
            break
        x -= fx / dfx
    return x


def wilson_interval(pass_count: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds on the true pass rate.
    More reliable than normal approximation for small n or extreme pass rates.

    Args:
        pass_count: Number of passing cases.
        n:          Total cases.
        confidence: Confidence level (default 0.95 → 95% CI).

    Returns:
        (lower_bound, upper_bound) both in [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    z = _norm_ppf(1 - (1 - confidence) / 2)
    p_hat = pass_count / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def runs_needed(
    delta: float,
    alpha: float = 0.05,
    power: float = 0.80,
    baseline: float = 0.70,
) -> int:
    """
    Minimum number of test cases needed to detect an improvement of `delta`
    in pass rate with the given statistical power.

    Uses the standard two-proportion z-test sample size formula.

    Args:
        delta:    Minimum detectable effect (e.g., 0.05 = 5 percentage points).
        alpha:    Significance level (default 0.05 → p < 0.05).
        power:    Desired power (default 0.80 → 80% chance of detecting true effect).
        baseline: Expected baseline pass rate (default 0.70).

    Returns:
        Minimum n (same for both groups in an A/B comparison).

    Example:
        runs_needed(delta=0.05)  # → 620
        runs_needed(delta=0.10)  # → 160
    """
    p1 = baseline
    p2 = min(1.0, baseline + delta)
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_beta = _norm_ppf(power)
    n = ((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))) / (delta ** 2)
    return math.ceil(n)


def min_detectable_effect(
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
    baseline: float = 0.70,
) -> float:
    """
    Minimum effect size detectable with n test cases at the given power.

    The inverse of runs_needed(): given your current dataset size, returns
    the smallest pass-rate improvement you can reliably detect.

    Args:
        n:        Number of test cases (same for both groups).
        alpha:    Significance level (default 0.05).
        power:    Desired power (default 0.80).
        baseline: Expected baseline pass rate (default 0.70).

    Returns:
        Minimum detectable delta as a fraction (e.g., 0.08 = 8pp).

    Example:
        min_detectable_effect(50)   # → ~0.19 (need 19pp shift to see it)
        min_detectable_effect(200)  # → ~0.10
        min_detectable_effect(500)  # → ~0.06
    """
    if n <= 0:
        return 1.0
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_beta = _norm_ppf(power)
    # Solve: n = (z_a + z_b)^2 * (p1*(1-p1) + p2*(1-p2)) / delta^2
    # Approximate p2 ≈ p1 for the variance term → p*(1-p)*2
    p = baseline
    var = p * (1 - p) * 2
    delta = math.sqrt((z_alpha + z_beta) ** 2 * var / n)
    return round(min(delta, 1.0), 4)


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for two proportions.

    |h| < 0.2  → small effect
    |h| < 0.5  → medium effect
    |h| >= 0.5 → large effect

    Args:
        p1: Baseline pass rate.
        p2: New pass rate.

    Returns:
        Cohen's h (signed, positive means improvement).
    """
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return round(phi2 - phi1, 4)


def _cohens_h_label(h: float) -> str:
    ah = abs(h)
    if ah < 0.2:
        return "small"
    if ah < 0.5:
        return "medium"
    return "large"


def bootstrap_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the mean of a list of scores.

    Preferred over Wilson for continuous scores or when N < 30.
    Uses the percentile method.

    Args:
        scores:     List of float scores (0.0–1.0).
        confidence: Confidence level (default 0.95 → 95% CI).
        n_samples:  Bootstrap resamples (default 2000, enough for most uses).
        seed:       Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound).

    Example:
        lo, hi = bootstrap_interval([0.8, 0.6, 0.9, 0.7, 0.85])
        print(f"95% CI: [{lo:.2f}, {hi:.2f}]")
    """
    import random
    if not scores:
        return (0.0, 1.0)
    if len(scores) == 1:
        return (scores[0], scores[0])
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_samples):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - confidence) / 2 * n_samples)
    hi_idx = int((1 + confidence) / 2 * n_samples) - 1
    return (round(means[lo_idx], 4), round(means[hi_idx], 4))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """
    Benjamini-Hochberg procedure for controlling the False Discovery Rate.

    Use when testing multiple hypotheses simultaneously — e.g., comparing N
    evaluators in an experiment. Raw p-values inflate the false positive rate:
    with 10 evaluators at α=0.05 you expect ~0.5 spurious "significant" results.
    BH corrects for this while being less conservative than Bonferroni.

    Args:
        p_values: Raw p-values from multiple simultaneous tests.
        alpha:    FDR level (default 0.05 → 5% false discovery rate).

    Returns:
        BH-adjusted p-values in the same order as input. A result is
        significant if its adjusted p-value < alpha.

    Example:
        raw = [0.001, 0.040, 0.030, 0.200, 0.800]
        adj = benjamini_hochberg(raw)
        # adj ≈ [0.005, 0.100, 0.075, 0.250, 0.800]
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        orig_idx, p = indexed[rank - 1]
        adj = min(p * n / rank, 1.0)
        adj = min(adj, prev)
        adjusted[orig_idx] = adj
        prev = adj
    return adjusted


def mcnemar_test(results_a: list[bool], results_b: list[bool]) -> float:
    """
    McNemar's test for paired binary model comparison.

    More powerful than two-proportion z-test when comparing two models on the
    same test cases — it only uses discordant pairs (cases where A and B
    disagree), which is all the statistical information in a paired comparison.

    Uses Edwards' continuity correction for small samples.

    Args:
        results_a: List of bool pass/fail for model A (one per case).
        results_b: List of bool pass/fail for model B (same cases, same order).

    Returns:
        Two-tailed p-value. p < 0.05 means the models differ significantly.
        p ≥ 0.05 means you cannot distinguish them on this dataset.

    Example:
        from multivon_eval import mcnemar_test
        p = mcnemar_test(
            [r.passed for cr in report_a.case_results for r in cr.results],
            [r.passed for cr in report_b.case_results for r in cr.results],
        )
    """
    if len(results_a) != len(results_b):
        raise ValueError("results_a and results_b must have the same length")
    b = sum(1 for a, r in zip(results_a, results_b) if a and not r)      # A-pass, B-fail
    c = sum(1 for a, r in zip(results_a, results_b) if not a and r)      # A-fail, B-pass
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # Edwards' continuity correction
    return 2 * (1 - _norm_cdf(math.sqrt(stat)))


def bayesian_interval(
    pass_count: int,
    n: int,
    confidence: float = 0.95,
    n_samples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bayesian credible interval for a pass rate.

    Uses a uniform Beta(1, 1) prior (Bayes-Laplace). The posterior is
    Beta(pass_count + 1, n - pass_count + 1). Returns the central credible
    interval at the given confidence level.

    For n > 100 uses a normal approximation to the Beta posterior.
    For n ≤ 100 uses exact Gamma-ratio sampling (no scipy required).

    Bayesian CIs have a direct interpretation: "there is a 95% probability the
    true pass rate is in [lo, hi]", conditional on the data and prior. Use
    alongside wilson_interval() when you want both perspectives.

    Args:
        pass_count: Number of passing cases.
        n:          Total cases.
        confidence: Credible interval width (default 0.95 → 95% CI).
        n_samples:  Gamma samples for exact path (n ≤ 100, default 10 000).
        seed:       Random seed for reproducibility (exact path only).

    Returns:
        (lower, upper) credible interval bounds in [0, 1].

    Example:
        lo, hi = bayesian_interval(80, 100)   # → approximately (0.71, 0.87)
        lo, hi = bayesian_interval(8, 10)     # → approximately (0.52, 0.97)
    """
    import random as _random
    if n == 0:
        return (0.0, 1.0)
    a = pass_count + 1              # posterior alpha (uniform prior)
    b_param = n - pass_count + 1   # posterior beta
    half = (1 - confidence) / 2

    if n > 100:
        # Normal approximation to Beta(a, b_param)
        mean = a / (a + b_param)
        var = a * b_param / ((a + b_param) ** 2 * (a + b_param + 1))
        z = _norm_ppf(1 - half)
        sd = math.sqrt(var)
        return (max(0.0, round(mean - z * sd, 4)), min(1.0, round(mean + z * sd, 4)))

    # Exact: Beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b))
    # Gamma(k, 1) = -sum_{i=1}^{k} log(U_i) for k iid Uniform samples
    rng = _random.Random(seed)
    samples: list[float] = []
    for _ in range(n_samples):
        g_a = -sum(math.log(rng.random() or 1e-300) for _ in range(a))
        g_b = -sum(math.log(rng.random() or 1e-300) for _ in range(b_param))
        samples.append(g_a / (g_a + g_b))
    samples.sort()
    lo_idx = int(half * n_samples)
    hi_idx = min(int((1 - half) * n_samples), n_samples - 1)
    return (round(samples[lo_idx], 4), round(samples[hi_idx], 4))


def _two_proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-proportion z-test. Returns p-value (two-tailed)."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return 1.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs((p2 - p1) / se)
    return 2 * (1 - _norm_cdf(z))


def _significance_label(p_value: float) -> str:
    if p_value < 0.01:
        return "p<0.01 ✦✦ highly significant"
    if p_value < 0.05:
        return f"p={p_value:.2f} ✦ significant"
    if p_value < 0.10:
        return f"p={p_value:.2f} marginal"
    return f"p={p_value:.2f} not significant (likely noise)"


def _print_comparison(a: RunRecord, b: RunRecord) -> None:
    print(f"\n  {'='*60}")
    print(f"  Experiment comparison: {a.run_id} → {b.run_id}")
    print(f"  {'='*60}\n")

    def _row(label: str, va: Any, vb: Any, fmt: str = "", suffix: str = "") -> None:
        if fmt == "%":
            sa, sb = f"{va:.1%}", f"{vb:.1%}"
            delta = _delta(va, vb)
        elif fmt == "f":
            sa, sb = f"{va:.4f}", f"{vb:.4f}"
            delta = _delta(va, vb)
        else:
            sa, sb = str(va), str(vb)
            delta = ""
        change = "↑" if (isinstance(vb, float) and vb > va) else ("↓" if (isinstance(vb, float) and vb < va) else "")
        print(f"  {label:<24} {sa:>12}  →  {sb:<12} {change} {delta}{suffix}")

    print(f"  {'Metric':<24} {'Before':>12}     {'After':<12}")
    print(f"  {'-'*60}")
    _row("Model", a.model_id or "-", b.model_id or "-")
    _row("Timestamp", a.timestamp[:19], b.timestamp[:19])
    _row("Pass rate", a.pass_rate, b.pass_rate, "%")
    _row("Avg score", a.avg_score, b.avg_score, "f")
    _row("Total cases", a.total, b.total)
    _row("Passed", a.passed, b.passed)
    _row("Failed", a.failed, b.failed)

    if a.runs_per_case > 1 or b.runs_per_case > 1:
        _row("Runs/case", a.runs_per_case, b.runs_per_case)
        _row("Flaky cases", a.flaky_count, b.flaky_count)
        _row("Stability", a.stability_score, b.stability_score, "f")

    all_evals = sorted(set(a.scores_by_evaluator) | set(b.scores_by_evaluator))
    if all_evals:
        # Compute per-evaluator p-values if pass rates are stored
        ev_pvals: list[float] = []
        has_pvals = bool(a.pass_by_evaluator and b.pass_by_evaluator)
        if has_pvals:
            for ev in all_evals:
                pa = a.pass_by_evaluator.get(ev, 0.0)
                pb = b.pass_by_evaluator.get(ev, 0.0)
                ev_pvals.append(_two_proportion_z_test(pa, a.total, pb, b.total))
            adj_pvals = benjamini_hochberg(ev_pvals) if len(ev_pvals) > 1 else ev_pvals

        print(f"\n  {'Evaluator scores':<24} {'Before':>12}     {'After':<12}", end="")
        if has_pvals:
            print(f"  {'BH-adj p':>10}", end="")
        print()
        print(f"  {'-'*60}")
        for i, ev in enumerate(all_evals):
            va = a.scores_by_evaluator.get(ev, 0.0)
            vb = b.scores_by_evaluator.get(ev, 0.0)
            if has_pvals:
                ap = adj_pvals[i]
                sig = " *" if ap < 0.05 else ""
                _row(f"  {ev}"[:24], va, vb, "f", suffix=f"  {ap:.3f}{sig}")
            else:
                _row(f"  {ev}"[:24], va, vb, "f")
        if has_pvals and len(all_evals) > 1:
            print(f"  (* significant after Benjamini-Hochberg correction, FDR 5%)")

    if a.tags or b.tags:
        print(f"\n  Tags A: {a.tags}")
        print(f"  Tags B: {b.tags}")

    # Confidence intervals on pass rates — use the same denominator as
    # pass_rate (evaluated cases, excluding errors). Falls back to total
    # for legacy records that pre-date the 0.7.0 status enum.
    denom_a = a.evaluated if a.evaluated >= 0 else a.total
    denom_b = b.evaluated if b.evaluated >= 0 else b.total
    ci_a = wilson_interval(a.passed, denom_a)
    ci_b = wilson_interval(b.passed, denom_b)
    print(f"\n  95% CI (before): [{ci_a[0]:.1%}, {ci_a[1]:.1%}]")
    print(f"  95% CI (after):  [{ci_b[0]:.1%}, {ci_b[1]:.1%}]")

    # Statistical significance + effect size — same denominator.
    delta_pass = b.pass_rate - a.pass_rate
    p_value = _two_proportion_z_test(a.pass_rate, denom_a, b.pass_rate, denom_b)
    h = cohens_h(a.pass_rate, b.pass_rate)
    print(f"  Statistical significance: {_significance_label(p_value)}")
    if abs(delta_pass) >= 0.001:
        print(f"  Effect size (Cohen's h):  {h:+.3f} ({_cohens_h_label(h)})")

    # Min-detectable-effect warning when dataset is small
    mde = min_detectable_effect(max(denom_a, denom_b), baseline=min(a.pass_rate, b.pass_rate))
    if mde > 0.05:
        print(f"  Min detectable effect at n={max(denom_a, denom_b)}: ~{mde:.0%}  "
              f"(changes smaller than this are not reliably detectable)")

    # Power hint: if not significant, suggest how many more cases are needed
    if p_value >= 0.05 and abs(delta_pass) >= 0.01:
        needed = runs_needed(abs(delta_pass), baseline=min(a.pass_rate, b.pass_rate))
        if needed > max(denom_a, denom_b):
            print(f"  Hint: need ≥{needed} test cases to detect this {abs(delta_pass):.0%} delta at 80% power.")

    # Verdict
    print(f"  Verdict: ", end="")
    if abs(delta_pass) < 0.01:
        print("No meaningful change in pass rate.")
    elif delta_pass > 0:
        print(f"IMPROVED — pass rate up {delta_pass:+.1%}")
    else:
        print(f"REGRESSION — pass rate down {delta_pass:+.1%}")
    print()
