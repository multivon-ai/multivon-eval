"""multivon-eval validate — grade your graders before blaming the model.

Audits the EVAL, not the agent: for every case that carries a reference
(known-good) output, run the suite's evaluators against that reference.
A reference that fails its own graders means the task is unsolvable or
the grader is miscalibrated — either way the agent is innocent, and any
0% pass rate blamed on the model was a lie.

Honest-output rules baked in:

* A validate run NEVER calls the model under test — reference-only.
* LLM-judge evaluators are skipped by default (free/offline audit);
  ``include_judges=True`` opts in and reports the judge spend.
* Cases without any reference are listed as UNVALIDATABLE with a nudge,
  never silently dropped.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .case import EvalCase
from .exceptions import JudgeUnavailable
from .result import EvalResult

if TYPE_CHECKING:
    from .suite import EvalSuite

STATUS_OK = "OK"
STATUS_BROKEN = "BROKEN_TASK_OR_GRADER"
STATUS_NO_DISCRIMINATION = "NO_DISCRIMINATION"
STATUS_UNVALIDATABLE = "UNVALIDATABLE"
# Report-level only: zero graders executed across the entire run — the
# validate pass produced no information and must not read as green.
STATUS_NOTHING_VALIDATED = "NOTHING_VALIDATED"

_NUDGE = "add expected_output or reference_output to validate this task"


@dataclass
class CaseValidation:
    """Validation verdict for a single case."""
    case_index: int
    case_input: str
    status: str  # one of the STATUS_* constants
    failed_graders: list[EvalResult] = field(default_factory=list)
    skipped_graders: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ValidationReport:
    """Aggregate verdict from :func:`validate_suite`."""
    suite_name: str
    results: list[CaseValidation]
    # Costs snapshot when include_judges=True ran judge-backed graders.
    costs: Any = None

    @property
    def broken(self) -> list[CaseValidation]:
        return [r for r in self.results if r.status == STATUS_BROKEN]

    @property
    def no_discrimination(self) -> list[CaseValidation]:
        return [r for r in self.results if r.status == STATUS_NO_DISCRIMINATION]

    @property
    def unvalidatable(self) -> list[CaseValidation]:
        return [r for r in self.results if r.status == STATUS_UNVALIDATABLE]

    @property
    def ok(self) -> list[CaseValidation]:
        return [r for r in self.results if r.status == STATUS_OK]

    @property
    def nothing_validated(self) -> bool:
        """True when zero graders executed across the whole run (every
        case is UNVALIDATABLE) — e.g. a judge-only suite audited in the
        default offline mode. Such a run validated nothing and must not
        report green."""
        return bool(self.results) and len(self.unvalidatable) == len(self.results)

    @property
    def status(self) -> str:
        """Report-level verdict: OK / BROKEN_TASK_OR_GRADER /
        NOTHING_VALIDATED (zero graders executed — never OK)."""
        if self.nothing_validated:
            return STATUS_NOTHING_VALIDATED
        return STATUS_OK if self.passed else STATUS_BROKEN

    @property
    def passed(self) -> bool:
        """No broken tasks/graders AND at least one grader actually
        executed. NO_DISCRIMINATION and (some) UNVALIDATABLE cases are
        warnings, not failures — but a run where EVERY case is
        unvalidatable produced zero information and is not a pass."""
        return not self.broken and not self.nothing_validated

    @property
    def effective_informative_cases(self) -> tuple[int, int]:
        """(ok_count, validated_count) — validated excludes UNVALIDATABLE."""
        validated = len(self.results) - len(self.unvalidatable)
        return (len(self.ok), validated)

    def to_dict(self) -> dict:
        ok_count, validated = self.effective_informative_cases
        skipped_all = sorted({g for r in self.results for g in r.skipped_graders})
        return {
            "suite": self.suite_name,
            "passed": self.passed,
            "status": self.status,
            "costs": self.costs.to_dict() if self.costs is not None else None,
            "skipped_judge_graders": skipped_all,
            "summary": {
                "cases": len(self.results),
                "ok": len(self.ok),
                "broken": len(self.broken),
                "no_discrimination": len(self.no_discrimination),
                "unvalidatable": len(self.unvalidatable),
                "effective_informative_cases": [ok_count, validated],
            },
            "results": [
                {
                    "case_index": r.case_index,
                    "case_input": r.case_input,
                    "status": r.status,
                    "reason": r.reason,
                    "failed_graders": [
                        {
                            "evaluator": g.evaluator,
                            "score": round(g.score, 4),
                            "passed": g.passed,
                            "reason": g.reason,
                        }
                        for g in r.failed_graders
                    ],
                    "skipped_graders": list(r.skipped_graders),
                }
                for r in self.results
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        ok_count, validated = self.effective_informative_cases
        if self.passed:
            verdict = "PASSED"
        elif self.nothing_validated:
            verdict = "NOTHING_VALIDATED"
        else:
            verdict = "FAILED"
        lines = [
            f"Validate: {self.suite_name} — "
            f"{verdict} "
            f"({len(self.broken)} broken, {len(self.no_discrimination)} "
            f"non-discriminating, {len(self.unvalidatable)} unvalidatable, "
            f"{len(self.ok)} OK)",
            f"  effective informative cases: {ok_count}/{validated} validated",
        ]
        if self.nothing_validated:
            lines.append(
                "  NOTHING_VALIDATED: zero graders executed — this run "
                "validated nothing; it is NOT a green result."
            )
        for r in self.broken:
            who = r.failed_graders[0].evaluator if r.failed_graders else "?"
            why = r.failed_graders[0].reason if r.failed_graders else r.reason
            lines.append(
                f"  BROKEN_TASK_OR_GRADER: {r.case_input[:60]!r} — {who}: {why}"
            )
        for r in self.no_discrimination:
            lines.append(f"  NO_DISCRIMINATION: {r.case_input[:60]!r} — {r.reason}")
        for r in self.unvalidatable:
            lines.append(f"  UNVALIDATABLE: {r.case_input[:60]!r} — {r.reason}")
        skipped_all = sorted({g for r in self.results for g in r.skipped_graders})
        if skipped_all:
            lines.append(
                f"  judge-backed grader(s) not run (offline default): "
                f"{', '.join(skipped_all)} — pass --judges to include them"
            )
        if self.costs is not None and getattr(self.costs, "total_tokens", 0):
            spend = self.costs.total_cost_usd
            spend_s = f"${spend:.4f}" if spend is not None else "unknown (no pricing data)"
            lines.append(
                f"  judge spend: {spend_s} · {self.costs.total_tokens:,} tokens"
            )
        return "\n".join(lines)


def _resolve_reference(case: EvalCase) -> tuple[str | None, str | None]:
    """Return (reference, error). Callable references are invoked here;
    an exception is a BROKEN verdict (with traceback), never a silent skip."""
    ref = case.reference_output
    if callable(ref):
        try:
            ref = ref(case)
        except Exception:
            return None, traceback.format_exc()
    if ref is None:
        ref = case.expected_output
    if not isinstance(ref, str):
        return None, None
    return ref, None


def _run_grader(ev: Any, case: EvalCase, output: str) -> EvalResult:
    """Run one grader against a reference; a raising grader is itself a
    finding (BROKEN), consistent with the honest error-accounting rule.

    :class:`JudgeUnavailable` is the one exception to that rule — a judge
    outage is infrastructure, not a grader verdict, so it's tagged
    ``error_kind='judge_unavailable'`` and routed to UNVALIDATABLE by the
    caller instead of BROKEN."""
    from .evaluators.deterministic import Latency, MaxLatency

    ev_name = getattr(ev, "name", type(ev).__name__)
    try:
        if isinstance(ev, (Latency, MaxLatency)):
            # A reference has no runtime; 0ms keeps latency graders inert.
            return ev.evaluate(case, output, latency_ms=0.0)
        return ev.evaluate(case, output)
    except JudgeUnavailable as ex:
        return EvalResult(
            evaluator=ev_name, score=0.0, passed=False,
            reason=f"judge unavailable while grading reference: {ex}",
            metadata={"error_kind": "judge_unavailable"},
        )
    except Exception as ex:
        return EvalResult(
            evaluator=ev_name, score=0.0, passed=False,
            reason=f"grader raised on reference: {type(ex).__name__}: {ex}",
            metadata={"error_kind": "evaluator_error"},
        )


def _find_contrast_twin(case: EvalCase, cases: list[EvalCase]) -> str | None:
    """Known-bad output from this case's contrast twin, if one is in the
    suite. Twins are linked by metadata['pair_id'] (the _contrast.py
    machinery writes it into both cases) and carry the bad output in
    metadata['unfaithful_answer']."""
    pair_id = (case.metadata or {}).get("pair_id")
    if not pair_id:
        return None
    for other in cases:
        if other is case:
            continue
        md = other.metadata or {}
        if md.get("pair_id") != pair_id:
            continue
        bad = md.get("unfaithful_answer")
        if isinstance(bad, str) and bad.strip():
            return bad
    return None


def _validate_case(
    idx: int,
    case: EvalCase,
    all_cases: list[EvalCase],
    evaluators: list[Any],
    *,
    include_judges: bool,
    contrast: bool,
) -> CaseValidation:
    if not evaluators:
        return CaseValidation(
            case_index=idx, case_input=case.input,
            status=STATUS_UNVALIDATABLE, reason="no evaluators",
        )

    reference, ref_error = _resolve_reference(case)
    if ref_error is not None:
        return CaseValidation(
            case_index=idx, case_input=case.input, status=STATUS_BROKEN,
            reason=f"reference_output callable raised:\n{ref_error}",
        )
    if reference is None:
        return CaseValidation(
            case_index=idx, case_input=case.input,
            status=STATUS_UNVALIDATABLE, reason=_NUDGE,
        )

    failed: list[EvalResult] = []
    skipped: list[str] = []
    judge_outages: list[EvalResult] = []
    executed = 0  # graders that produced a REAL verdict on the reference
    # Graders that scored the reference for real (not skipped) and passed —
    # the pool the contrast check probes for discrimination.
    ref_passing: list[Any] = []
    for ev in evaluators:
        ev_name = getattr(ev, "name", type(ev).__name__)
        if getattr(ev, "uses_llm_judge", False) and not include_judges:
            skipped.append(ev_name)
            continue
        result = _run_grader(ev, case, reference)
        if result.metadata.get("skipped"):
            continue  # case shape doesn't fit this grader — counts neither way
        if result.metadata.get("error_kind") == "judge_unavailable":
            # Infrastructure, not a grader verdict — never BROKEN.
            judge_outages.append(result)
            continue
        executed += 1
        if result.passed:
            ref_passing.append(ev)
        else:
            failed.append(result)

    if failed:
        # Only genuine grader verdicts may say BROKEN.
        return CaseValidation(
            case_index=idx, case_input=case.input, status=STATUS_BROKEN,
            failed_graders=failed, skipped_graders=skipped,
            reason="reference output fails its own grader(s) — the task is "
                   "unsolvable or the grader is miscalibrated",
        )

    if judge_outages:
        return CaseValidation(
            case_index=idx, case_input=case.input,
            status=STATUS_UNVALIDATABLE, skipped_graders=skipped,
            reason=(
                f"judge unavailable (infrastructure, not a grader verdict): "
                f"{judge_outages[0].reason} — rerun when the judge endpoint "
                f"is reachable"
            ),
        )

    if executed == 0:
        # Zero graders ran — this case was NOT validated and must not be OK.
        if skipped and len(skipped) == len(evaluators):
            reason = "all graders are judge-backed; rerun with --judges"
        elif skipped:
            reason = (
                f"no grader produced a verdict — judge-backed grader(s) "
                f"{', '.join(skipped)} skipped (offline default; rerun with "
                f"--judges) and the remaining grader(s) did not apply to "
                f"this case's shape"
            )
        else:
            reason = "no grader applies to this case's shape — nothing validated"
        return CaseValidation(
            case_index=idx, case_input=case.input,
            status=STATUS_UNVALIDATABLE, skipped_graders=skipped,
            reason=reason,
        )

    if contrast:
        bad_output = _find_contrast_twin(case, all_cases)
        if bad_output is not None:
            lenient = []
            for ev in ref_passing:
                # Judge-backed graders never run in the offline default —
                # the ref-grading loop above can't put them in ref_passing
                # when include_judges is False, but the zero-LLM-calls
                # guarantee must not depend on that invariant at a distance.
                if getattr(ev, "uses_llm_judge", False) and not include_judges:
                    continue
                r = _run_grader(ev, case, bad_output)
                if r.passed and not r.metadata.get("skipped"):
                    lenient.append(getattr(ev, "name", type(ev).__name__))
            if lenient:
                return CaseValidation(
                    case_index=idx, case_input=case.input,
                    status=STATUS_NO_DISCRIMINATION, skipped_graders=skipped,
                    reason=f"grader(s) {', '.join(lenient)} pass both the "
                           f"reference and the known-bad contrast twin — "
                           f"zero information on this case",
                )

    return CaseValidation(
        case_index=idx, case_input=case.input, status=STATUS_OK,
        skipped_graders=skipped,
    )


def validate_suite(
    suite: "EvalSuite",
    *,
    include_judges: bool = False,
    contrast: bool = True,
) -> ValidationReport:
    """Run every case's graders against its reference output.

    Never calls the model under test. Deterministic graders run always;
    judge-backed graders only with ``include_judges=True`` (their spend is
    tracked and reported). With ``contrast=True``, cases linked to a
    contrast twin also get a discrimination check against the twin's
    known-bad output — zero LLM calls in the default offline mode (the
    contrast rerun excludes judge-backed graders unless
    ``include_judges=True``, in which case its judge spend is included in
    the tracked costs).
    """
    evaluators = list(suite.evaluators)
    cases = list(suite.cases)

    costs = None
    if include_judges:
        from .costs import CostTracker, set_active_tracker, reset_token
        tracker = CostTracker()
        token = set_active_tracker(tracker)
        try:
            # Same warmup contract as suite.run() — e.g. CheckEvaluator
            # generates its questions here, inside the cost tracker.
            for ev in evaluators:
                if hasattr(ev, "prepare"):
                    ev.prepare()
            results = [
                _validate_case(i, c, cases, evaluators,
                               include_judges=True, contrast=contrast)
                for i, c in enumerate(cases)
            ]
        finally:
            reset_token(token)
        costs = tracker.snapshot()
    else:
        results = [
            _validate_case(i, c, cases, evaluators,
                           include_judges=False, contrast=contrast)
            for i, c in enumerate(cases)
        ]

    return ValidationReport(suite_name=suite.name, results=results, costs=costs)


__all__ = [
    "CaseValidation",
    "ValidationReport",
    "validate_suite",
    "STATUS_OK",
    "STATUS_BROKEN",
    "STATUS_NO_DISCRIMINATION",
    "STATUS_UNVALIDATABLE",
    "STATUS_NOTHING_VALIDATED",
]
