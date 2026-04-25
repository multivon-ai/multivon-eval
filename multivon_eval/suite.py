from __future__ import annotations
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Awaitable

from .case import EvalCase
from .result import CaseResult, EvalReport
from .evaluators.base import Evaluator
from .evaluators.deterministic import Latency, MaxLatency
from .reporters.terminal import print_report


class EvalSuite:
    """
    Orchestrates running evaluators over test cases.

    Supports both synchronous and async (parallel) execution.

    Usage:
        suite = EvalSuite("My Suite")
        suite.add_cases(cases)
        suite.add_evaluators(Relevance(), Faithfulness())
        report = suite.run(my_model_fn)           # serial
        report = suite.run(my_model_fn, workers=4)  # parallel
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

    def _run_case(self, case: EvalCase, model_fn: Callable[[str], str]) -> CaseResult:
        t0 = time.time()
        try:
            output = model_fn(case.input)
        except Exception as e:
            output = f"[MODEL ERROR: {e}]"
        latency_ms = (time.time() - t0) * 1000

        results = []
        for ev in self._evaluators:
            if isinstance(ev, (Latency, MaxLatency)):
                result = ev.evaluate(case, output, latency_ms=latency_ms)
            else:
                result = ev.evaluate(case, output)
            results.append(result)

        return CaseResult(
            case_input=case.input,
            actual_output=output,
            results=results,
            latency_ms=latency_ms,
            tags=case.tags,
        )

    def run(
        self,
        model_fn: Callable[[str], str],
        verbose: bool = True,
        fail_threshold: float | None = None,
        workers: int = 1,
    ) -> EvalReport:
        """
        Run all evaluators over all cases.

        Args:
            model_fn:        Callable str → str.
            verbose:         Print terminal report.
            fail_threshold:  Exit(1) in CI if pass_rate < threshold.
            workers:         Number of parallel threads (default 1 = serial).
                             Set > 1 to run cases concurrently.
        """
        if workers > 1:
            case_results = self._run_parallel(model_fn, workers)
        else:
            case_results = [self._run_case(case, model_fn) for case in self._cases]

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

    def _run_parallel(
        self, model_fn: Callable[[str], str], workers: int
    ) -> list[CaseResult]:
        results: dict[int, CaseResult] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._run_case, case, model_fn): i
                for i, case in enumerate(self._cases)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = CaseResult(
                        case_input=self._cases[idx].input,
                        actual_output=f"[ERROR: {e}]",
                        results=[],
                        latency_ms=0.0,
                    )
        return [results[i] for i in range(len(self._cases))]

    async def run_async(
        self,
        model_fn: Callable[[str], Awaitable[str]],
        verbose: bool = True,
        fail_threshold: float | None = None,
        concurrency: int = 5,
    ) -> EvalReport:
        """
        Run evals with an async model function.

        Args:
            model_fn:    Async callable str → str.
            concurrency: Max concurrent model calls (default 5).
        """
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(case: EvalCase) -> CaseResult:
            async with sem:
                t0 = time.time()
                try:
                    output = await model_fn(case.input)
                except Exception as e:
                    output = f"[MODEL ERROR: {e}]"
                latency_ms = (time.time() - t0) * 1000

                results = []
                for ev in self._evaluators:
                    if isinstance(ev, (Latency, MaxLatency)):
                        result = ev.evaluate(case, output, latency_ms=latency_ms)
                    else:
                        result = ev.evaluate(case, output)
                    results.append(result)

                return CaseResult(
                    case_input=case.input,
                    actual_output=output,
                    results=results,
                    latency_ms=latency_ms,
                    tags=case.tags,
                )

        case_results = await asyncio.gather(*[_run_one(c) for c in self._cases])

        report = EvalReport(
            suite_name=self.name,
            case_results=list(case_results),
            model_id=self.model_id,
        )

        if verbose:
            print_report(report)

        if fail_threshold is not None and report.pass_rate < fail_threshold:
            raise SystemExit(
                f"\nEval failed: pass rate {report.pass_rate:.1%} < threshold {fail_threshold:.1%}"
            )

        return report
