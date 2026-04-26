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

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "suite_name": self.suite_name,
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "scores_by_evaluator": self.scores_by_evaluator,
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
            passed=d["passed"],
            failed=d["failed"],
            scores_by_evaluator=d.get("scores_by_evaluator", {}),
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
            passed=report.passed,
            failed=report.failed,
            scores_by_evaluator={k: round(v, 4) for k, v in report.scores_by_evaluator().items()},
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

    def _row(label: str, va: Any, vb: Any, fmt: str = "") -> None:
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
        print(f"  {label:<24} {sa:>12}  →  {sb:<12} {change} {delta}")

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
        print(f"\n  {'Evaluator scores':<24} {'Before':>12}     {'After':<12}")
        print(f"  {'-'*60}")
        for ev in all_evals:
            va = a.scores_by_evaluator.get(ev, 0.0)
            vb = b.scores_by_evaluator.get(ev, 0.0)
            _row(f"  {ev}"[:24], va, vb, "f")

    if a.tags or b.tags:
        print(f"\n  Tags A: {a.tags}")
        print(f"  Tags B: {b.tags}")

    # Statistical significance
    p_value = _two_proportion_z_test(a.pass_rate, a.total, b.pass_rate, b.total)
    print(f"\n  Statistical significance: {_significance_label(p_value)}")

    # Verdict
    delta_pass = b.pass_rate - a.pass_rate
    print(f"  Verdict: ", end="")
    if abs(delta_pass) < 0.01:
        print("No meaningful change in pass rate.")
    elif delta_pass > 0:
        print(f"IMPROVED — pass rate up {delta_pass:+.1%}")
    else:
        print(f"REGRESSION — pass rate down {delta_pass:+.1%}")
    print()
