"""Empirical hardness filtering; oracle validity must be established separately."""
from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from numbers import Real

from .case import EvalCase
from .case_manifest import canonical_json
from .evaluators.base import Evaluator
from .judge import JudgeConfig


@dataclass(slots=True)
class HardnessReport:
    """One record per submitted case, including unavailable measurements.

    ``n_shots`` is the requested count. Scores/outputs retain shot positions;
    null means no valid measurement/output. ``failure_rate`` and the legacy
    mean/majority properties are null unless every requested shot is measured.
    Repeated shots of one source are not independent dataset examples.
    """

    case: EvalCase
    evaluator_name: str
    n_shots: int
    scores: list[float | None]
    baseline_outputs: list[str | None]
    failure_rate: float | None
    in_hardness_band: bool
    shots: list[dict] = field(default_factory=list)

    @property
    def measured_shots(self) -> int:
        return sum(score is not None for score in self.scores)

    @property
    def baseline_failed(self) -> bool | None:
        """At least half of requested shots fail, or unknown if incomplete."""
        return None if self.failure_rate is None else self.failure_rate >= 0.5

    @property
    def baseline_score(self) -> float | None:
        if self.measured_shots != self.n_shots or not self.n_shots:
            return None
        return sum(score for score in self.scores if score is not None) / self.n_shots


def _evaluator(case: EvalCase, judge: JudgeConfig | None) -> tuple[str, Evaluator]:
    import multivon_eval as m

    names = (case.metadata or {}).get('stress_tests', [])
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError('stress_tests must be a list of evaluator names')
    for name in names:
        cls = getattr(m, name, None)
        if not isinstance(cls, type) or not issubclass(cls, Evaluator):
            continue
        kwargs = {}
        if judge is not None:
            parameters = inspect.signature(cls).parameters
            if 'judge' in parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
            ):
                kwargs['judge'] = judge
        return name, cls(**kwargs)
    raise ValueError('No supported stress-test evaluator')


def validate_adversarial_cases(
    cases: list[EvalCase],
    baseline_model: Callable[[str], str],
    *,
    n_shots: int = 3,
    hardness_band: tuple[float, float] = (0.5, 1.0),
    judge: JudgeConfig | None = None,
) -> tuple[list[EvalCase], list[HardnessReport]]:
    """Keep fully measured cases whose empirical failure rate is in the band.

    Baseline errors, skipped/error verdicts, invalid scores and grader errors
    stay visible and make the case ineligible. All requested shots are attempted
    after per-shot errors; evaluator setup failures make no baseline calls.
    This filter does not validate labels or establish statistical reliability.
    """
    if type(n_shots) is not int or n_shots < 1:
        raise ValueError('n_shots must be a positive integer')
    if (not isinstance(hardness_band, (tuple, list)) or len(hardness_band) != 2
            or any(isinstance(x, bool) or not isinstance(x, Real) for x in hardness_band)):
        raise ValueError('hardness_band must be (lo, hi) in [0, 1]')
    lo, hi = hardness_band
    if not 0 <= lo <= hi <= 1:
        raise ValueError('hardness_band must be (lo, hi) in [0, 1]')

    reports, kept = [], []
    for case in cases:
        report = HardnessReport(case, '', n_shots, [], [], None, False)
        reports.append(report)
        try:
            name, evaluator = _evaluator(case, judge)
            report.evaluator_name = name
        except Exception as exc:  # noqa: BLE001 — retain failures from caller-supplied code
            report.scores = [None] * n_shots
            report.baseline_outputs = [None] * n_shots
            report.shots = [{'status': 'not_run', 'reason': str(exc),
                             'error_type': type(exc).__name__} for _ in range(n_shots)]
            continue

        failures = 0
        for _ in range(n_shots):
            shot = {}
            report.shots.append(shot)
            report.scores.append(None)
            report.baseline_outputs.append(None)
            try:
                output = baseline_model(case.input)
                if not isinstance(output, str):
                    raise TypeError('baseline_model must return a string')
                report.baseline_outputs[-1] = output
            except Exception as exc:  # noqa: BLE001 — retain failures from caller-supplied code
                shot.update(status='model_error', reason=str(exc), error_type=type(exc).__name__)
                continue
            try:
                result = evaluator.evaluate(case, output)
                raw = asdict(result)
                try:
                    canonical_json(raw)
                except (TypeError, ValueError):
                    shot['result_repr'] = repr(raw)
                    raise ValueError('Verdict is not portable JSON') from None
                shot['result'] = raw
                missing = result.metadata.get('skipped') or result.metadata.get('error_kind')
                if missing:
                    shot.update(status='unmeasured', reason=result.reason)
                    continue
                score = result.score
                if (type(result.passed) is not bool or isinstance(score, bool)
                        or not isinstance(score, Real) or not math.isfinite(score)
                        or not 0 <= score <= 1):
                    raise ValueError('Verdict requires a boolean passed and finite score in [0, 1]')
                report.scores[-1] = float(score)
                failures += not result.passed
                shot['status'] = 'passed' if result.passed else 'failed_quality'
            except Exception as exc:  # noqa: BLE001 — retain failures from caller-supplied code
                shot.update(status='evaluator_error', reason=str(exc), error_type=type(exc).__name__)

        if report.measured_shots == n_shots:
            report.failure_rate = failures / n_shots
            report.in_hardness_band = lo <= report.failure_rate <= hi
        if report.in_hardness_band:
            kept.append(case)
    return kept, reports
