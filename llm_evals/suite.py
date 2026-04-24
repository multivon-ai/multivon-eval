from __future__ import annotations
import time
from typing import Callable

from .case import EvalCase
from .result import CaseResult, EvalReport
from .evaluators.base import Evaluator
from .evaluators.deterministic import Latency
from .reporters.terminal import print_report


class EvalSuite:
    """
    Orchestrates running a set of evaluators over a set of test cases.

    Usage:
        suite = EvalSuite("My Suite")
        suite.add_cases(cases)
        suite.add_evaluators(Relevance(), Faithfulness())
        report = suite.run(my_model_fn)
        report.save_json("results.json")
    """

    def __init__(self, name: str, model_id: str = ""):
        self.name = name
        self.model_id = model_id
        self._cases: list[EvalCase] = []
        self._evaluators: list[Evaluator] = []

    def add_case(self, case: EvalCase) -> "EvalSuite":
        self._cases.append(case)
        return self

    def add_cases(self, cases: list[EvalCase]) -> "EvalSuite":
        self._cases.extend(cases)
        return self

    def add_evaluator(self, evaluator: Evaluator) -> "EvalSuite":
        self._evaluators.append(evaluator)
        return self

    def add_evaluators(self, *evaluators: Evaluator) -> "EvalSuite":
        self._evaluators.extend(evaluators)
        return self

    def run(
        self,
        model_fn: Callable[[str], str],
        verbose: bool = True,
        fail_threshold: float | None = None,
    ) -> EvalReport:
        """
        Run all evaluators over all cases.

        Args:
            model_fn:        A callable that takes a string input and returns a string output.
            verbose:         Print a live report to the terminal.
            fail_threshold:  If set, raises SystemExit(1) if pass_rate < threshold. Use in CI.
        """
        case_results = []

        for i, case in enumerate(self._cases):
            t0 = time.time()
            try:
                output = model_fn(case.input)
            except Exception as e:
                output = f"[MODEL ERROR: {e}]"
            latency_ms = (time.time() - t0) * 1000

            results = []
            for ev in self._evaluators:
                if isinstance(ev, Latency):
                    result = ev.evaluate(case, output, latency_ms=latency_ms)
                else:
                    result = ev.evaluate(case, output)
                results.append(result)

            case_results.append(CaseResult(
                case_input=case.input,
                actual_output=output,
                results=results,
                latency_ms=latency_ms,
                tags=case.tags,
            ))

        report = EvalReport(
            suite_name=self.name,
            case_results=case_results,
            model_id=self.model_id,
        )

        if verbose:
            print_report(report)

        if fail_threshold is not None and report.pass_rate < fail_threshold:
            raise SystemExit(
                f"\nEval failed: pass rate {report.pass_rate:.1%} < threshold {fail_threshold:.1%}"
            )

        return report
