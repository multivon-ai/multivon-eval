from __future__ import annotations
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Awaitable

from .case import EvalCase
from .result import CaseResult, EvalReport, EvalResult
from .evaluators.base import Evaluator
from .evaluators.deterministic import Latency, MaxLatency
from .reporters.terminal import print_report


class EvalSuite:
    """
    Orchestrates running evaluators over test cases.

    Supports single-run and multi-run (for flakiness detection) modes,
    plus serial and parallel execution.

    Usage:
        suite = EvalSuite("My Suite")
        suite.add_cases(cases)
        suite.add_evaluators(Relevance(), Faithfulness())

        report = suite.run(my_model_fn)              # single run, serial
        report = suite.run(my_model_fn, workers=4)   # single run, parallel
        report = suite.run(my_model_fn, runs=5)      # multi-run, flakiness detection
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

    def _run_case_once(self, case: EvalCase, model_fn: Callable[[str], str]) -> CaseResult:
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

    def _run_case(
        self, case: EvalCase, model_fn: Callable[[str], str], runs: int = 1
    ) -> CaseResult:
        if runs == 1:
            return self._run_case_once(case, model_fn)

        single_runs = [self._run_case_once(case, model_fn) for _ in range(runs)]
        return _aggregate_runs(case, single_runs)

    def run(
        self,
        model_fn: Callable[[str], str],
        verbose: bool = True,
        fail_threshold: float | None = None,
        workers: int = 1,
        runs: int = 1,
    ) -> EvalReport:
        """
        Run all evaluators over all cases.

        Args:
            model_fn:        Callable str → str.
            verbose:         Print terminal report.
            fail_threshold:  Exit(1) in CI if pass_rate < threshold.
            workers:         Parallel threads for cases (default 1 = serial).
            runs:            Times to run each case (default 1). Use > 1 to
                             detect flaky cases and get score confidence intervals.
        """
        if workers > 1:
            case_results = self._run_parallel(model_fn, workers, runs)
        else:
            case_results = [self._run_case(case, model_fn, runs) for case in self._cases]

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
        self, model_fn: Callable[[str], str], workers: int, runs: int
    ) -> list[CaseResult]:
        results: dict[int, CaseResult] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._run_case, case, model_fn, runs): i
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
        runs: int = 1,
    ) -> EvalReport:
        """
        Run evals with an async model function.

        Args:
            model_fn:    Async callable str → str.
            concurrency: Max concurrent model calls (default 5).
            runs:        Times to run each case (default 1).
        """
        sem = asyncio.Semaphore(concurrency)

        async def _run_one_async(case: EvalCase) -> CaseResult:
            async with sem:
                single_runs = []
                for _ in range(runs):
                    t0 = time.time()
                    try:
                        output = await model_fn(case.input)
                    except Exception as e:
                        output = f"[MODEL ERROR: {e}]"
                    latency_ms = (time.time() - t0) * 1000

                    ev_results = []
                    for ev in self._evaluators:
                        if isinstance(ev, (Latency, MaxLatency)):
                            result = ev.evaluate(case, output, latency_ms=latency_ms)
                        else:
                            result = ev.evaluate(case, output)
                        ev_results.append(result)

                    single_runs.append(CaseResult(
                        case_input=case.input,
                        actual_output=output,
                        results=ev_results,
                        latency_ms=latency_ms,
                        tags=case.tags,
                    ))

                if runs == 1:
                    return single_runs[0]
                return _aggregate_runs(case, single_runs)

        case_results = await asyncio.gather(*[_run_one_async(c) for c in self._cases])

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


def _aggregate_runs(case: EvalCase, single_runs: list[CaseResult]) -> CaseResult:
    """Merge N single-run CaseResults into one aggregated CaseResult."""
    n = len(single_runs)
    all_scores = [cr.score for cr in single_runs]
    pass_count = sum(1 for cr in single_runs if all(r.passed for r in cr.results))
    avg_latency = sum(cr.latency_ms for cr in single_runs) / n

    # Aggregate per-evaluator scores (mean score, majority-vote passed)
    ev_data: dict[str, list] = {}
    for cr in single_runs:
        for r in cr.results:
            ev_data.setdefault(r.evaluator, []).append(r)

    agg_results = []
    for ev_name, ev_results in ev_data.items():
        avg_score = sum(r.score for r in ev_results) / len(ev_results)
        pass_votes = sum(1 for r in ev_results if r.passed)
        agg_results.append(EvalResult(
            evaluator=ev_name,
            score=avg_score,
            passed=pass_votes > len(ev_results) / 2,  # majority vote
            reason=f"avg over {n} runs (passed {pass_votes}/{len(ev_results)})",
        ))

    return CaseResult(
        case_input=case.input,
        actual_output=single_runs[-1].actual_output,  # last run's output
        results=agg_results,
        latency_ms=avg_latency,
        tags=case.tags,
        runs=n,
        all_scores=all_scores,
        pass_count=pass_count,
    )
