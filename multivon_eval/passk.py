"""
pass@k and pass^k reliability metrics from multi-run eval data.

Two different questions about the same ``--runs N`` data:

  - **pass@k** — "can the model do it in k tries?" (capability).
    Unbiased HumanEval combinatorial estimator.
  - **pass^k** — "does it succeed all k tries?" (reliability).
    Exact hypergeometric estimator — what a user hitting the feature
    k times actually experiences.

Both are computed per case from the n recorded trials (c of which
passed), then averaged over cases. Suite-level confidence intervals use
a CLUSTER bootstrap that resamples CASES with replacement — trials
within a case are correlated, so resampling raw trials would fake
precision.

When k exceeds the recorded trials per case the answer is honestly
UNKNOWN (:attr:`PassKResult.value` is ``None``) — multivon-eval does
not extrapolate beyond the data.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from math import comb

METRIC_PASS_AT_K = "pass@k"
METRIC_PASS_HAT_K = "pass^k"

ESTIMATOR_NAMES = {
    METRIC_PASS_AT_K: "combinatorial-unbiased",
    METRIC_PASS_HAT_K: "hypergeometric-exact",
}


def _validate(n: int, c: int, k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, {n}], got {c}")
    if k > n:
        raise ValueError(
            f"k={k} exceeds the {n} recorded trials — cannot estimate without extrapolating"
        )


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k samples passes) — unbiased HumanEval estimator.

    ``1 - comb(n-c, k) / comb(n, k)``: one minus the probability that a
    uniformly drawn size-k subset of the n recorded trials contains only
    failures. Raises ValueError when ``k > n`` (callers translate that
    to an honest-UNKNOWN result rather than extrapolating).
    """
    _validate(n, c, k)
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """P(all k samples pass) — exact hypergeometric estimator.

    ``comb(c, k) / comb(n, k)``: the probability that a uniformly drawn
    size-k subset of the n recorded trials is all passes. NEVER the
    plug-in ``(c/n)**k``, which samples with replacement and is
    upward-biased for finite n — a vanity metric this function
    deliberately refuses to compute.
    """
    _validate(n, c, k)
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


@dataclass
class PassKResult:
    """Suite-level pass@k / pass^k estimate with a cluster-bootstrap CI.

    ``value is None`` means honestly UNKNOWN (see ``unknown_reason``) —
    typically because k exceeds the recorded runs per case.
    """
    k: int
    metric: str                 # 'pass@k' | 'pass^k'
    value: float | None
    ci_low: float | None
    ci_high: float | None
    estimator: str              # 'combinatorial-unbiased' | 'hypergeometric-exact'
    n_cases: int
    runs: int
    unknown_reason: str = ""


def _quantile(sorted_vals: list[float], q: float) -> float:
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _cluster_bootstrap_ci(
    values: list[float], confidence: float, n_boot: int, seed: int,
) -> tuple[float, float]:
    """Percentile CI on the mean of per-case estimates, resampling CASES.

    Degenerate samples (all per-case estimates identical — including the
    all-pass ceiling and single-case suites) collapse a percentile
    bootstrap to a point, which overstates certainty. Fall back to a
    Wilson interval on the mean so a 10/10 suite still reports
    ``ci_low < 1.0``, matching :meth:`EvalReport.pass_rate_ci` honesty.
    """
    if min(values) == max(values):
        from .experiments import wilson_interval
        m = len(values)
        return wilson_interval(values[0] * m, m, confidence)
    rng = random.Random(seed)
    m = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(m)) / m
        for _ in range(n_boot)
    )
    alpha = 1 - confidence
    return _quantile(means, alpha / 2), _quantile(means, 1 - alpha / 2)


def suite_pass_k(
    case_stats: list[tuple[int, int]],
    k: int,
    *,
    metric: str,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> PassKResult:
    """Average the per-case estimator over ``case_stats`` = [(n, c), ...].

    CI is a seeded cluster bootstrap over cases (see module docstring).
    Raises ValueError if k exceeds any case's recorded trials — callers
    holding an :class:`EvalReport` should check ``runs_per_case`` first
    and return an UNKNOWN result instead.
    """
    if metric not in ESTIMATOR_NAMES:
        raise ValueError(f"metric must be one of {sorted(ESTIMATOR_NAMES)}, got {metric!r}")
    estimator_fn = pass_at_k if metric == METRIC_PASS_AT_K else pass_hat_k
    estimator_name = ESTIMATOR_NAMES[metric]
    if not case_stats:
        return PassKResult(
            k=k, metric=metric, value=None, ci_low=None, ci_high=None,
            estimator=estimator_name, n_cases=0, runs=0,
            unknown_reason="UNKNOWN — no evaluated cases to estimate from.",
        )
    values = [estimator_fn(n, c, k) for n, c in case_stats]
    point = sum(values) / len(values)
    ci_low, ci_high = _cluster_bootstrap_ci(values, confidence, n_boot, seed)
    return PassKResult(
        k=k, metric=metric, value=point,
        ci_low=min(ci_low, point), ci_high=max(ci_high, point),
        estimator=estimator_name,
        n_cases=len(values), runs=min(n for n, _ in case_stats),
    )


__all__ = [
    "pass_at_k", "pass_hat_k", "suite_pass_k",
    "PassKResult",
    "METRIC_PASS_AT_K", "METRIC_PASS_HAT_K", "ESTIMATOR_NAMES",
]
