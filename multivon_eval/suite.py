from __future__ import annotations
import asyncio
import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from .case import EvalCase
from .exceptions import JudgeUnavailable
from .result import CalibrationResult, CaseResult, EvalGateFailure, EvalReport, EvalResult
from .evaluators.base import Evaluator
from .evaluators.deterministic import Latency, MaxLatency
from .reporters.terminal import print_report

if TYPE_CHECKING:
    from pathlib import Path
    from .integrations.base import AgentTracer
    from .lockfile import SuiteLock
    from .retry import JudgeRetry


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

    @property
    def evaluators(self) -> list[Evaluator]:
        """Public read-only view of the suite's evaluators.

        Use this to introspect what's configured. Returns a copy so
        callers can iterate without risk of mutating internal state.
        Add evaluators via :meth:`add_evaluator` / :meth:`add_evaluators` /
        :meth:`add_check`, not by appending to this list.
        """
        return list(self._evaluators)

    @property
    def cases(self) -> list[EvalCase]:
        """Public read-only view of the suite's test cases."""
        return list(self._cases)

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

    def lock(self) -> "SuiteLock":
        """Return a content-addressed fingerprint of this suite.

        See :mod:`multivon_eval.lockfile` for the schema. The lock is
        cheap to compute and useful for CI:

            saved = SuiteLock.from_json(Path("suite.lock").read_text())
            suite.verify_lock(saved)  # raises LockMismatch if anything drifted

        Use this to catch silent prompt updates, judge model swaps, or
        dataset changes that would otherwise invalidate historical
        comparisons.
        """
        from .lockfile import build_suite_lock
        return build_suite_lock(self)

    def write_lock(self, path: "str | Path") -> "Path":
        """Save the current lock to ``path``. Returns the resolved Path."""
        from pathlib import Path as _P
        out = _P(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.lock().to_json(), encoding="utf-8")
        return out

    def verify_lock(self, saved: "SuiteLock | str | Path") -> None:
        """Raise :class:`LockMismatch` if the current suite has drifted.

        ``saved`` can be a :class:`SuiteLock` instance, a JSON string,
        or a path to a saved lock file.
        """
        from pathlib import Path as _P
        from .lockfile import SuiteLock, verify_suite_against_lock
        if isinstance(saved, SuiteLock):
            lock = saved
        elif isinstance(saved, (str, _P)) and _P(str(saved)).exists():
            lock = SuiteLock.from_json(_P(str(saved)).read_text(encoding="utf-8"))
        elif isinstance(saved, str):
            lock = SuiteLock.from_json(saved)
        else:
            raise TypeError(f"verify_lock: unsupported argument {type(saved).__name__}")
        verify_suite_against_lock(self, lock)

    def add_check(
        self,
        criterion: str,
        threshold: float = 0.7,
        num_questions: int = 3,
        questions: "list[str] | None" = None,
        name: str = "",
        judge: "Any | None" = None,
    ) -> "EvalSuite":
        """
        Add a natural-language quality check.

        The judge auto-generates yes/no questions from the criterion before the
        evaluation loop runs. Questions are cached and reused across all cases.
        Pass ``questions=`` to bypass generation for reproducible/CI evals.

        Args:
            criterion:     Plain-English description of what to check.
            threshold:     Minimum score to pass (default 0.7).
            num_questions: Number of questions to generate (1–10, default 3).
            questions:     Skip generation; provide explicit yes/no questions.
            name:          Override the evaluator name shown in reports.
            judge:         Override the judge model for this check only.

        Returns self for chaining::

            (suite
                .add_check("Response mentions the return policy")
                .add_check("Tone is professional", threshold=0.8)
                .add_case(EvalCase(input="What is your return policy?"))
            )
        """
        from .evaluators.llm_judge import CheckEvaluator
        self._evaluators.append(CheckEvaluator(
            criterion=criterion,
            threshold=threshold,
            num_questions=num_questions,
            questions=questions,
            name=name,
            judge=judge,
        ))
        return self

    def _run_case_once(
        self,
        case: EvalCase,
        model_fn: Callable[[str], str],
        tracer: "AgentTracer | None" = None,
    ) -> CaseResult:
        if tracer is not None:
            tracer.reset()

        t0 = time.time()
        model_error: str | None = None
        try:
            output = model_fn(case.input)
        except Exception as e:
            model_error = str(e)
            output = f"[MODEL ERROR: {e}]"
        latency_ms = (time.time() - t0) * 1000

        if tracer is not None:
            trace = tracer.get_trace()
            if trace:
                case = dataclasses.replace(case, agent_trace=trace)

        results = []
        # judge_error / evaluator_error are populated below if any evaluator
        # raises. Latching the FIRST one of each kind is enough for status
        # classification — surfacing more detail (per-evaluator failures) is
        # already in the EvalResult.reason strings.
        judge_error: str | None = None
        evaluator_error: str | None = None
        for ev in self._evaluators:
            ev_name = getattr(ev, "name", type(ev).__name__)
            if model_error is not None and not isinstance(ev, (Latency, MaxLatency)):
                results.append(EvalResult(
                    evaluator=ev_name,
                    score=0.0,
                    passed=False,
                    reason=f"[skipped — model error: {model_error}]",
                ))
                continue
            try:
                if isinstance(ev, (Latency, MaxLatency)):
                    result = ev.evaluate(case, output, latency_ms=latency_ms)
                else:
                    result = ev.evaluate(case, output)
            except JudgeUnavailable as ju:
                if judge_error is None:
                    judge_error = str(ju)
                result = EvalResult(
                    evaluator=ev_name, score=0.0, passed=False,
                    reason=f"[judge unavailable: {ju}]",
                    metadata={"error_kind": "judge_error", "error_detail": str(ju)},
                )
            except Exception as ex:
                # An evaluator itself crashed — distinct from a judge outage.
                # Capture so downstream code can route it to evaluator_error.
                if evaluator_error is None:
                    evaluator_error = f"{type(ex).__name__}: {ex}"
                result = EvalResult(
                    evaluator=ev_name, score=0.0, passed=False,
                    reason=f"[evaluator error: {type(ex).__name__}: {ex}]",
                    metadata={"error_kind": "evaluator_error",
                              "error_detail": f"{type(ex).__name__}: {ex}"},
                )
            results.append(result)

        return CaseResult(
            case_input=case.input,
            actual_output=output,
            model_error=model_error,
            judge_error=judge_error,
            evaluator_error=evaluator_error,
            results=results,
            latency_ms=latency_ms,
            tags=case.tags,
            agent_trace=case.agent_trace,
        )

    def _run_case(
        self,
        case: EvalCase,
        model_fn: Callable[[str], str],
        runs: int = 1,
        tracer: "AgentTracer | None" = None,
        early_stop: bool = False,
    ) -> CaseResult:
        if runs == 1:
            return self._run_case_once(case, model_fn, tracer=tracer)

        single_runs: list[CaseResult] = []
        for i in range(runs):
            single_runs.append(self._run_case_once(case, model_fn, tracer=tracer))
            if early_stop and i >= 1:
                if _sprt_stop(single_runs):
                    break

        return _aggregate_runs(case, single_runs)

    def _run_case_with_retry(
        self,
        case: EvalCase,
        model_fn: Callable[[str], str],
        runs: int = 1,
        tracer: "AgentTracer | None" = None,
        early_stop: bool = False,
        judge_retry: "JudgeRetry | None" = None,
    ) -> CaseResult:
        """Wrap :meth:`_run_case` with the retry policy. No-op when
        ``judge_retry`` is None — the common path.

        Retry happens at the CASE level: a single retriable status
        re-runs all evaluators for the case (and all sub-runs when
        ``runs > 1``). This is wasteful when only one evaluator hit
        the outage, but it keeps the multi-run + early-stop semantics
        intact and matches what an operator typically wants: a clean
        end-to-end result, not a partial patchwork.
        """
        if judge_retry is None:
            return self._run_case(case, model_fn, runs, tracer=tracer, early_stop=early_stop)

        from .retry import should_retry, sleep_for_attempt

        retry_errors: list[str] = []
        cr: CaseResult | None = None
        for attempt in range(1, judge_retry.max_attempts + 1):
            # Sleep BEFORE the attempt for attempt >= 2 — attempt 1 runs
            # immediately. This keeps the no-retry happy path zero-cost.
            if attempt > 1:
                sleep_for_attempt(judge_retry, attempt)
            cr = self._run_case(case, model_fn, runs, tracer=tracer, early_stop=early_stop)
            if not should_retry(cr.status, judge_retry):
                break
            # ``retry_errors`` records "errors that *prompted* a retry" —
            # not "errors observed" — so we don't append on the last
            # attempt. Otherwise an exhausted 3-attempt chain would
            # report retry_attempts=3 even though only 2 retries
            # actually happened.
            if attempt >= judge_retry.max_attempts:
                break
            retry_errors.append(
                cr.judge_error or cr.evaluator_error or cr.model_error or f"<{cr.status.value}>"
            )
        assert cr is not None  # max_attempts >= 1 guarantees one pass

        # Record retry history on the case result. ``retry_attempts``
        # is the number of FAILED prior attempts — for a case that
        # passed first try it's 0. For an exhausted retry chain on
        # max_attempts=3 it's 2 (two retries failed before giving up).
        if retry_errors:
            cr = dataclasses.replace(
                cr,
                retry_attempts=len(retry_errors),
                retry_errors=list(retry_errors),
            )
        return cr

    def run(
        self,
        model_fn: Callable[[str], str],
        verbose: bool = True,
        fail_threshold: float | None = None,
        workers: int = 1,
        runs: int = 1,
        tracer: "AgentTracer | None" = None,
        early_stop: bool = False,
        judge_retry: "JudgeRetry | None" = None,
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
            tracer:          AgentTracer instance. Tracers are stateful, so
                             workers > 1 is not allowed when a tracer is provided.
            early_stop:      Stop each case early once the result is statistically
                             clear (SPRT). Only applies when runs > 1. Reduces LLM
                             spend on easy cases without sacrificing accuracy.
            judge_retry:     :class:`JudgeRetry` policy for transient judge /
                             timeout errors. Default ``None`` = no retry.
                             Cases whose status is in ``policy.retry_on`` are
                             re-evaluated up to ``policy.max_attempts`` times
                             with exponential backoff. The retry history lands
                             on ``CaseResult.retry_attempts`` /
                             ``CaseResult.retry_errors``.
        """
        if tracer is not None and workers > 1:
            raise ValueError(
                "tracer and workers > 1 are incompatible — tracers are stateful. "
                "Run with workers=1 (the default) when using a tracer."
            )

        # Warmup: prepare evaluators that need pre-run setup (e.g. CheckEvaluator
        # generates yes/no questions here so errors surface before the eval loop
        # and no individual case pays the generation latency cost).
        for ev in self._evaluators:
            if hasattr(ev, "prepare"):
                ev.prepare()

        if tracer is not None:
            instrumented_fn = tracer.instrument(model_fn)
        else:
            instrumented_fn = model_fn

        # Cost tracker is attached to the contextvar for the duration of
        # this run. Every judge call inside `judge.make_judge_call` reports
        # token counts back to it. Lazy import keeps the costs module out
        # of the import graph for callers that never invoke run().
        from .costs import CostTracker, set_active_tracker, reset_token
        cost_tracker = CostTracker()
        cost_token = set_active_tracker(cost_tracker)

        try:
            if workers > 1:
                case_results = self._run_parallel(
                    instrumented_fn, workers, runs, judge_retry=judge_retry,
                )
            else:
                case_results = [
                    self._run_case_with_retry(
                        case, instrumented_fn, runs,
                        tracer=tracer, early_stop=early_stop,
                        judge_retry=judge_retry,
                    )
                    for case in self._cases
                ]
        finally:
            reset_token(cost_token)

        judge_reliability = _measure_judge_reliability(
            self._evaluators, case_results, self._cases
        )

        report = EvalReport(
            suite_name=self.name,
            case_results=case_results,
            model_id=self.model_id,
            judge_reliability=judge_reliability,
            costs=cost_tracker.snapshot(),
            suite_lock=_safe_lock(self),
        )

        if verbose:
            print_report(report)

        # Auto-save outputs when CLI flags are injected via env vars
        import os as _os
        if html_path := _os.environ.get("MULTIVON_HTML_OUTPUT"):
            report.save_html(html_path)
            print(f"  HTML report saved → {html_path}")
        if json_path := _os.environ.get("MULTIVON_JSON_OUTPUT"):
            report.save_json(json_path)
            print(f"  JSON report saved → {json_path}")

        if fail_threshold is not None and report.pass_rate < fail_threshold:
            raise EvalGateFailure(
                f"\nEval failed: pass rate {report.pass_rate:.1%} < threshold {fail_threshold:.1%}",
                pass_rate=report.pass_rate,
                threshold=fail_threshold,
            )

        return report

    def run_with_openai(
        self,
        model: str = "gpt-4o",
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        client: "Any | None" = None,
        **run_kwargs: "Any",
    ) -> EvalReport:
        """
        Run evals against an OpenAI model.

        Convenience wrapper around suite.run(OpenAIAdapter(...)). For custom
        behavior — retry logic, prompt templating, structured outputs — subclass
        OpenAIAdapter and pass an instance to suite.run() directly.

        Args:
            model:         OpenAI model ID (default "gpt-4o").
            system_prompt: System message prepended to every call.
            temperature:   Sampling temperature (default 0.0).
            max_tokens:    Max output tokens (default 1024).
            client:        openai.OpenAI instance. Created from OPENAI_API_KEY if None.
            **run_kwargs:  Forwarded to suite.run() (verbose, workers, runs, etc.).
        """
        from .adapters import OpenAIAdapter
        return self.run(
            OpenAIAdapter(
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                client=client,
            ),
            **run_kwargs,
        )

    def run_with_anthropic(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        client: "Any | None" = None,
        **run_kwargs: "Any",
    ) -> EvalReport:
        """
        Run evals against an Anthropic model.

        Convenience wrapper around suite.run(AnthropicAdapter(...)). For custom
        behavior subclass AnthropicAdapter and pass an instance to suite.run().

        Args:
            model:         Anthropic model ID (default "claude-haiku-4-5-20251001").
            system_prompt: System message.
            temperature:   Sampling temperature (default 0.0).
            max_tokens:    Max output tokens (default 1024).
            client:        anthropic.Anthropic instance. Created from ANTHROPIC_API_KEY if None.
            **run_kwargs:  Forwarded to suite.run() (verbose, workers, runs, etc.).
        """
        from .adapters import AnthropicAdapter
        return self.run(
            AnthropicAdapter(
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                client=client,
            ),
            **run_kwargs,
        )

    def run_with_litellm(
        self,
        model: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **run_kwargs: "Any",
    ) -> EvalReport:
        """
        Run evals against any LiteLLM-supported provider.

        Covers 100+ providers (Azure, Bedrock, Vertex, Ollama, Groq, …) without
        writing a custom adapter. Requires: pip install 'multivon-eval[litellm]'

        Args:
            model:         LiteLLM model string, e.g. "azure/gpt-4o",
                           "bedrock/anthropic.claude-3-sonnet-…", "ollama/llama3.2".
            system_prompt: System message prepended to every call.
            temperature:   Sampling temperature (default 0.0).
            max_tokens:    Max output tokens (default 1024).
            **run_kwargs:  Forwarded to suite.run() (verbose, workers, runs, …).
                           Provider-specific kwargs (api_base, api_key, …) are
                           also forwarded to litellm.completion().

        Examples:

            # Azure OpenAI
            report = suite.run_with_litellm(
                "azure/gpt-4o",
                api_base="https://my-deployment.openai.azure.com",
                api_key=os.environ["AZURE_API_KEY"],
                api_version="2024-02-01",
            )

            # AWS Bedrock
            report = suite.run_with_litellm(
                "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
            )

            # Local Ollama
            report = suite.run_with_litellm(
                "ollama/llama3.2", api_base="http://localhost:11434"
            )
        """
        from .adapters import LiteLLMAdapter

        # Split run_kwargs from litellm-specific kwargs (api_base, api_key, etc.)
        _run_keys = {"verbose", "fail_threshold", "workers", "runs", "tracer", "early_stop"}
        adapter_kwargs = {k: v for k, v in run_kwargs.items() if k not in _run_keys}
        suite_kwargs = {k: v for k, v in run_kwargs.items() if k in _run_keys}

        return self.run(
            LiteLLMAdapter(
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **adapter_kwargs,
            ),
            **suite_kwargs,
        )

    def run_pairwise(
        self,
        model_a: "Callable[[str], str]",
        model_b: "Callable[[str], str]",
        *,
        model_a_id: str = "Model A",
        model_b_id: str = "Model B",
        judge: "Any | None" = None,
        verbose: bool = True,
    ) -> "Any":
        """
        Head-to-head comparison: run both models on every case, ask an LLM
        judge which response is better, return win/loss/tie counts with a
        sign-test p-value.

        Unlike pass/fail evals, pairwise comparison produces a preference
        signal even when neither model clearly passes or fails.

        Args:
            model_a:     First model callable (str → str).
            model_b:     Second model callable (str → str).
            model_a_id:  Label for model A in the report.
            model_b_id:  Label for model B in the report.
            judge:       JudgeConfig override. Uses global judge if None.
            verbose:     Print summary (default True).

        Returns:
            PairwiseReport with per-case winners and aggregate statistics.

        Example:
            report = suite.run_pairwise(
                gpt4o_fn, claude_fn,
                model_a_id="GPT-4o", model_b_id="Claude Haiku",
            )
            print(report)
        """
        from .result import PairwiseReport, PairwiseResult
        from .judge import resolve_judge
        from .evaluators.llm_judge import _call

        resolved = resolve_judge(judge)
        results: list[PairwiseResult] = []

        for case in self._cases:
            try:
                out_a = model_a(case.input)
            except Exception as e:
                out_a = f"[ERROR: {e}]"
            try:
                out_b = model_b(case.input)
            except Exception as e:
                out_b = f"[ERROR: {e}]"

            prompt = (
                "You are an impartial judge evaluating two AI responses to the same input.\n\n"
                f"Input: {case.input}\n\n"
                f"Response A:\n{out_a}\n\n"
                f"Response B:\n{out_b}\n\n"
                "Which response is better? Reply with ONLY 'A', 'B', or 'Tie' on the "
                "first line, then a brief explanation on the next line."
            )
            try:
                raw = _call(prompt, resolved, max_tokens=200)
                first_line = raw.strip().split("\n")[0].strip().upper()
                if first_line.startswith("A"):
                    winner = "A"
                elif first_line.startswith("B"):
                    winner = "B"
                else:
                    winner = "Tie"
                reason = raw.strip()
            except Exception as e:
                winner = "Tie"
                reason = f"[Judge error: {e}]"

            results.append(PairwiseResult(
                case_input=case.input,
                output_a=out_a,
                output_b=out_b,
                winner=winner,
                reason=reason,
            ))

        report = PairwiseReport(
            suite_name=self.name,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            results=results,
        )

        if verbose:
            print(report)

        return report

    def run_on_cases(
        self,
        traced_outputs: list[tuple[EvalCase, str]],
        verbose: bool = True,
        fail_threshold: float | None = None,
    ) -> EvalReport:
        """
        Run evaluators on pre-evaluated (case, output) pairs.

        Use this with CaseImporter to evaluate imported traces without
        re-running the agent:

            cases = importer.load()
            pairs = [(c, c.metadata["_output"]) for c in cases]
            report = suite.run_on_cases(pairs)

        Args:
            traced_outputs:  List of (EvalCase, output_str) pairs.
            verbose:         Print terminal report.
            fail_threshold:  Exit(1) in CI if pass_rate < threshold.
        """
        case_results = []
        for case, output in traced_outputs:
            results = []
            # Apply the same per-evaluator isolation as the live run path —
            # an imported trace shouldn't crash the whole suite if one
            # evaluator's judge is unavailable.
            judge_err: str | None = None
            evaluator_err: str | None = None
            for ev in self._evaluators:
                ev_name = getattr(ev, "name", type(ev).__name__)
                try:
                    result = ev.evaluate(case, output)
                except JudgeUnavailable as ju:
                    if judge_err is None:
                        judge_err = str(ju)
                    result = EvalResult(
                        evaluator=ev_name, score=0.0, passed=False,
                        reason=f"[judge unavailable: {ju}]",
                    )
                except Exception as ex:
                    if evaluator_err is None:
                        evaluator_err = f"{type(ex).__name__}: {ex}"
                    result = EvalResult(
                        evaluator=ev_name, score=0.0, passed=False,
                        reason=f"[evaluator error: {type(ex).__name__}: {ex}]",
                    )
                results.append(result)
            case_results.append(CaseResult(
                case_input=case.input,
                actual_output=output,
                results=results,
                latency_ms=0.0,
                tags=case.tags,
                judge_error=judge_err,
                evaluator_error=evaluator_err,
                agent_trace=case.agent_trace,
            ))

        report = EvalReport(
            suite_name=self.name,
            case_results=case_results,
            model_id=self.model_id,
            suite_lock=_safe_lock(self),
        )

        if verbose:
            print_report(report)

        if fail_threshold is not None and report.pass_rate < fail_threshold:
            raise EvalGateFailure(
                f"\nEval failed: pass rate {report.pass_rate:.1%} < threshold {fail_threshold:.1%}",
                pass_rate=report.pass_rate,
                threshold=fail_threshold,
            )

        return report

    def calibrate(
        self,
        labeled_pairs: "list[tuple[EvalCase, str, bool]]",
    ) -> "CalibrationResult":
        """
        Measure judge accuracy against human-labeled ground truth.

        Runs all evaluators on each (case, output) pair and compares pass/fail
        decisions against your human labels. Reports agreement, precision, recall,
        and F1 per evaluator — revealing which judges are calibrated and which drift.

        Args:
            labeled_pairs: List of (case, model_output, human_pass_label) tuples.
                           human_pass_label=True means a human expert marked this
                           case as passing.

        Returns:
            CalibrationResult with per-evaluator and overall accuracy metrics.

        Example:
            result = suite.calibrate([
                (EvalCase(input="What is 2+2?"), "4", True),
                (EvalCase(input="What is 2+2?"), "purple", False),
            ])
            print(result)
        """
        from .result import CalibrationResult

        tp = fp = fn = tn = 0
        by_ev: dict[str, dict[str, int]] = {}

        for case, output, human_pass in labeled_pairs:
            for ev in self._evaluators:
                r = ev.evaluate(case, output)
                ev_name = r.evaluator
                counts = by_ev.setdefault(ev_name, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
                if human_pass and r.passed:
                    tp += 1; counts["tp"] += 1
                elif not human_pass and r.passed:
                    fp += 1; counts["fp"] += 1
                elif human_pass and not r.passed:
                    fn += 1; counts["fn"] += 1
                else:
                    tn += 1; counts["tn"] += 1

        total = tp + fp + fn + tn
        agreement = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        ev_stats: dict[str, dict[str, float]] = {}
        for ev_name, c in by_ev.items():
            _total = c["tp"] + c["fp"] + c["fn"] + c["tn"]
            _agr = (c["tp"] + c["tn"]) / _total if _total > 0 else 0.0
            _prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
            _rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
            _f1 = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0.0
            ev_stats[ev_name] = {
                "agreement": round(_agr, 4),
                "precision": round(_prec, 4),
                "recall": round(_rec, 4),
                "f1": round(_f1, 4),
            }

        return CalibrationResult(
            n=len(labeled_pairs),
            agreement=round(agreement, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            by_evaluator=ev_stats,
        )

    def _run_parallel(
        self,
        model_fn: Callable[[str], str],
        workers: int,
        runs: int,
        judge_retry: "JudgeRetry | None" = None,
    ) -> list[CaseResult]:
        results: dict[int, CaseResult] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._run_case_with_retry, case, model_fn, runs,
                    None, False, judge_retry,
                ): i
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

    # ── Factory suites ────────────────────────────────────────────────────────

    @classmethod
    def for_rag(cls, name: str = "RAG Eval", *, threshold: float = 0.85) -> "EvalSuite":
        """Faithfulness, hallucination, context precision/recall, relevance.

        Best for: RAG pipelines, question-answering systems, retrieval-augmented chatbots.

        Usage:
            suite = EvalSuite.for_rag()
            suite.add_cases(cases)
            report = suite.run(my_rag_fn, runs=5)
        """
        from .evaluators.deterministic import NotEmpty
        from .evaluators.llm_judge import (
            Faithfulness, Hallucination, Relevance,
            ContextPrecision, ContextRecall,
        )
        return (
            cls(name)
            .add_evaluators(
                NotEmpty(),
                Faithfulness(threshold=threshold),
                Hallucination(threshold=threshold),
                ContextPrecision(threshold=threshold),
                ContextRecall(threshold=threshold),
                Relevance(threshold=threshold),
            )
        )

    @classmethod
    def for_agents(cls, name: str = "Agent Eval", *, require_order: bool = False) -> "EvalSuite":
        """Tool call accuracy, necessity, trajectory efficiency, plan quality, task completion.

        Best for: LLM agents, tool-augmented systems, multi-step pipelines.
        Run with runs=5 to detect flaky cases: suite.run(fn, runs=5)

        Usage:
            suite = EvalSuite.for_agents()
            suite.add_cases(cases)
            report = suite.run(my_agent_fn, runs=5)
        """
        from .evaluators.agent import (
            ToolCallAccuracy, ToolCallNecessity, TrajectoryEfficiency,
            PlanQuality, TaskCompletion,
        )
        return (
            cls(name)
            .add_evaluators(
                ToolCallAccuracy(require_order=require_order),
                ToolCallNecessity(),
                TrajectoryEfficiency(),
                PlanQuality(),
                TaskCompletion(),
            )
        )

    @classmethod
    def for_support_bot(cls, name: str = "Support Bot Eval") -> "EvalSuite":
        """Faithfulness, relevance, coherence, toxicity, not-empty.

        Best for: Customer support bots, help desks, FAQ systems.

        Usage:
            suite = EvalSuite.for_support_bot()
            suite.add_cases(cases)
            report = suite.run(my_bot_fn)
        """
        from .evaluators.deterministic import NotEmpty
        from .evaluators.llm_judge import Faithfulness, Relevance, Coherence, Toxicity
        return (
            cls(name)
            .add_evaluators(
                NotEmpty(),
                Faithfulness(),
                Relevance(),
                Coherence(),
                Toxicity(),
            )
        )

    @classmethod
    def for_summarization(cls, name: str = "Summarization Eval") -> "EvalSuite":
        """Faithfulness, coherence, relevance, summarization quality.

        Best for: Document summarizers, meeting note takers, digest generators.

        Usage:
            suite = EvalSuite.for_summarization()
            suite.add_cases(cases)
            report = suite.run(my_summarizer_fn)
        """
        from .evaluators.deterministic import NotEmpty
        from .evaluators.llm_judge import Faithfulness, Coherence, Relevance, Summarization
        return (
            cls(name)
            .add_evaluators(
                NotEmpty(),
                Faithfulness(),
                Coherence(),
                Relevance(),
                Summarization(),
            )
        )

    @classmethod
    def for_document_intelligence(
        cls,
        name: str = "Document Intelligence Eval",
        schema=None,
    ) -> "EvalSuite":
        """Schema validation, faithfulness, answer accuracy.

        Best for: Data extraction, document parsing, structured output pipelines.
        Pass a Pydantic model or JSON Schema dict as `schema` to validate output structure.

        Usage:
            from pydantic import BaseModel
            class Invoice(BaseModel):
                vendor: str
                amount: float
                date: str

            suite = EvalSuite.for_document_intelligence(schema=Invoice)
            suite.add_cases(cases)
            report = suite.run(my_extractor_fn)
        """
        from .evaluators.deterministic import NotEmpty
        from .evaluators.llm_judge import Faithfulness, AnswerAccuracy
        from .evaluators.compliance import SchemaEvaluator
        suite = cls(name).add_evaluators(NotEmpty(), Faithfulness(), AnswerAccuracy())
        if schema is not None:
            suite.add_evaluator(SchemaEvaluator(schema))
        return suite

    @classmethod
    def for_regulated(
        cls,
        name: str = "Regulated AI Eval",
        *,
        jurisdiction: str = "hipaa",
        schema=None,
    ) -> "EvalSuite":
        """PII detection, schema validation, faithfulness — zero data egress.

        Best for: Healthcare, finance, legal, and public sector AI systems
        subject to HIPAA, GDPR, CCPA, EU AI Act, or NIST AI RMF requirements.

        Pair with ComplianceReporter to produce tamper-evident audit trails:
            reporter = ComplianceReporter("/audit/evals", framework="eu_ai_act")
            reporter.record(report, tags={"system": "triage-bot", "version": "1.0"})

        Usage:
            suite = EvalSuite.for_regulated(jurisdiction="hipaa")
            suite.add_cases(cases)
            report = suite.run(my_fn)
        """
        from .evaluators.deterministic import NotEmpty
        from .evaluators.llm_judge import Faithfulness, Relevance
        from .evaluators.compliance import PIIEvaluator, SchemaEvaluator
        suite = (
            cls(name)
            .add_evaluators(
                PIIEvaluator(jurisdiction=jurisdiction, redact=True),
                NotEmpty(),
                Faithfulness(),
                Relevance(),
            )
        )
        if schema is not None:
            suite.add_evaluator(SchemaEvaluator(schema, strict=True))
        return suite

    @classmethod
    def eu_ai_act_high_risk(
        cls,
        name: str = "EU AI Act High-Risk Eval",
        *,
        jurisdiction: str = "gdpr",
        schema=None,
    ) -> "EvalSuite":
        """Auditor-ready evaluator set for high-risk AI systems under the EU AI Act.

        Wires the standard measurable controls of the Act's high-risk obligations:

            Art. 9(2)(b)    Foreseeable misuse              → Toxicity
            Art. 10(2)(f-g) Bias examination & mitigation   → Bias
            Art. 10(5)      Personal data processing        → PIIEvaluator
            Art. 15(1)      Accuracy                        → Faithfulness, Hallucination, Relevance
            Art. 15(2)      Robustness                      → NotEmpty, SchemaEvaluator (if schema),
                                                              and (recommended) SelfConsistency via runs>1

        Pair with ``ComplianceReporter`` to satisfy Art. 12 (record-keeping) and
        to print a coverage report flagging any remaining gaps:

            suite = EvalSuite.eu_ai_act_high_risk()
            suite.add_cases(cases)
            reporter = ComplianceReporter("./audit-logs", framework="eu-ai-act")
            print(reporter.coverage(suite))
            report = suite.run(model_fn, runs=5)
            reporter.record(report, tags={"system": "triage-bot"})

        Art. 13 (transparency), Art. 14 (human oversight), and Art. 15(4-5)
        (cybersecurity) are process controls and require organizational
        measures beyond model evaluation.
        """
        from .evaluators.compliance import PIIEvaluator, SchemaEvaluator
        from .evaluators.deterministic import NotEmpty
        from .evaluators.llm_judge import (
            Bias, Faithfulness, Hallucination, Relevance, Toxicity,
        )
        suite = (
            cls(name)
            .add_evaluators(
                NotEmpty(),
                Faithfulness(),
                Hallucination(),
                Relevance(),
                Toxicity(),
                Bias(),
                PIIEvaluator(jurisdiction=jurisdiction, redact=True),
            )
        )
        if schema is not None:
            suite.add_evaluator(SchemaEvaluator(schema, strict=True))
        return suite

    @classmethod
    def hipaa_safe_harbor(
        cls,
        name: str = "HIPAA Safe Harbor Eval",
        *,
        schema=None,
    ) -> "EvalSuite":
        """Evaluator set for healthcare AI under HIPAA Security Rule + Safe Harbor.

        Wires the technical-safeguard controls that operate on AI output. Use
        ``ComplianceReporter(framework="hipaa")`` to surface the matching
        45 CFR §164.312 / §164.514(b)(2) annotations and to detect
        administrative + physical safeguards that still require organizational
        measures (BAA, access logs, training).

            §164.312(a)  Access control       → PIIEvaluator (PHI mediation)
            §164.312(b)  Audit controls       → Faithfulness, Hallucination, AnswerAccuracy
            §164.312(c)  Integrity of ePHI    → NotEmpty, SchemaEvaluator (if schema)
            §164.514(b)(2) De-identification  → PIIEvaluator(jurisdiction="hipaa")

        Usage:
            suite = EvalSuite.hipaa_safe_harbor()
            suite.add_cases(cases)
            reporter = ComplianceReporter("./audit-logs", framework="hipaa")
            print(reporter.coverage(suite))
            report = suite.run(model_fn, runs=5)
            reporter.record(report, tags={"system": "clinical-triage"})

        HIPAA PII coverage caveat: ``PIIEvaluator(jurisdiction="hipaa")``
        detects 13 of 18 Safe Harbor PHI identifiers via regex. The 5 remaining
        (names, geographic subdivisions below state, photographs, biometric
        identifiers, and any unique non-pattern identifier) require an upstream
        de-identification step. The library is honest about this.
        """
        from .evaluators.compliance import PIIEvaluator, SchemaEvaluator
        from .evaluators.deterministic import NotEmpty
        from .evaluators.llm_judge import AnswerAccuracy, Faithfulness, Hallucination
        suite = (
            cls(name)
            .add_evaluators(
                PIIEvaluator(jurisdiction="hipaa", redact=True),
                NotEmpty(),
                Faithfulness(),
                Hallucination(),
                AnswerAccuracy(),
            )
        )
        if schema is not None:
            suite.add_evaluator(SchemaEvaluator(schema, strict=True))
        return suite

    @classmethod
    def for_chatbot(cls, name: str = "Chatbot Eval") -> "EvalSuite":
        """Conversation relevance, knowledge retention, turn consistency, completeness.

        Best for: Multi-turn chatbots, conversational assistants, dialogue systems.

        Usage:
            suite = EvalSuite.for_chatbot()
            suite.add_cases(cases)
            report = suite.run(my_chatbot_fn)
        """
        from .evaluators.conversation import (
            ConversationRelevance, KnowledgeRetention,
            ConversationCompleteness, TurnConsistency,
        )
        return (
            cls(name)
            .add_evaluators(
                ConversationRelevance(),
                KnowledgeRetention(),
                TurnConsistency(),
                ConversationCompleteness(),
            )
        )

    @classmethod
    def for_classification(cls, name: str = "Classification Eval") -> "EvalSuite":
        """Exact match and answer accuracy for label prediction tasks.

        Best for: Intent classification, sentiment analysis, routing, tagging.

        Usage:
            suite = EvalSuite.for_classification()
            suite.add_cases(cases)
            report = suite.run(my_classifier_fn)
        """
        from .evaluators.deterministic import NotEmpty, ExactMatch
        from .evaluators.llm_judge import AnswerAccuracy
        return (
            cls(name)
            .add_evaluators(
                NotEmpty(),
                ExactMatch(),
                AnswerAccuracy(),
            )
        )

    @classmethod
    def for_coding(cls, name: str = "Coding Agent Eval", *, language: str = "python") -> "EvalSuite":
        """Exact match, answer accuracy, and ROUGE for code generation tasks.

        Best for: code generation, function completion, test generation.
        Note: ``language`` is reserved for future language-specific evaluators.

        Usage:
            suite = EvalSuite.for_coding()
            suite = EvalSuite.for_coding("TypeScript Eval", language="typescript")
            suite.add_cases(cases)
            report = suite.run(my_codegen_fn)
        """
        from .evaluators.deterministic import NotEmpty, ExactMatch, ROUGE
        from .evaluators.llm_judge import AnswerAccuracy
        return (
            cls(name)
            .add_evaluators(
                NotEmpty(),
                ExactMatch(),
                AnswerAccuracy(),
                ROUGE(),
            )
        )

    @classmethod
    def for_medical(cls, name: str = "Medical AI Eval", *, jurisdiction: str = "hipaa") -> "EvalSuite":
        """PII detection, faithfulness, hallucination, and answer accuracy for clinical AI.

        Best for: clinical decision support, medical Q&A, patient-facing chatbots.
        Always pair with ComplianceReporter to maintain tamper-evident audit trails.

        Usage:
            suite = EvalSuite.for_medical()
            suite = EvalSuite.for_medical("Clinical QA", jurisdiction="gdpr")
            suite.add_cases(cases)
            report = suite.run(my_clinical_fn)
        """
        from .evaluators.compliance import PIIEvaluator
        from .evaluators.deterministic import NotEmpty
        from .evaluators.llm_judge import Faithfulness, Hallucination, AnswerAccuracy
        return (
            cls(name)
            .add_evaluators(
                PIIEvaluator(jurisdiction=jurisdiction, redact=True),
                NotEmpty(),
                Faithfulness(),
                AnswerAccuracy(),
                Hallucination(),
            )
        )

    @classmethod
    def for_legal(cls, name: str = "Legal AI Eval") -> "EvalSuite":
        """Faithfulness, hallucination, answer accuracy, and bias for legal AI.

        Best for: contract review, legal Q&A, regulatory guidance systems.
        Hallucination threshold matters most — fabricated citations are a critical failure mode.

        Usage:
            suite = EvalSuite.for_legal()
            suite.add_cases(cases)
            report = suite.run(my_legal_fn)
        """
        from .evaluators.deterministic import NotEmpty
        from .evaluators.llm_judge import Faithfulness, Hallucination, AnswerAccuracy, Bias
        return (
            cls(name)
            .add_evaluators(
                NotEmpty(),
                Faithfulness(),
                Hallucination(),
                AnswerAccuracy(),
                Bias(),
            )
        )

    @classmethod
    def for_financial(cls, name: str = "Financial AI Eval") -> "EvalSuite":
        """Faithfulness, hallucination, answer accuracy, and PII detection for financial AI.

        Best for: financial advice bots, earnings summarizers, trading signal generators.
        Pair with ComplianceReporter for regulatory audit trails (SEC, FINRA, MiFID II).

        Usage:
            suite = EvalSuite.for_financial()
            suite.add_cases(cases)
            report = suite.run(my_financial_fn)
        """
        from .evaluators.compliance import PIIEvaluator
        from .evaluators.deterministic import NotEmpty
        from .evaluators.llm_judge import Faithfulness, Hallucination, AnswerAccuracy
        return (
            cls(name)
            .add_evaluators(
                NotEmpty(),
                Faithfulness(),
                Hallucination(),
                AnswerAccuracy(),
                PIIEvaluator(jurisdiction="all"),
            )
        )

    async def run_async(
        self,
        model_fn: Callable[[str], Awaitable[str]],
        verbose: bool = True,
        fail_threshold: float | None = None,
        concurrency: int = 5,
        runs: int = 1,
        evaluator_concurrency: int | None = None,
        judge_retry: "JudgeRetry | None" = None,
    ) -> EvalReport:
        """Run evals with an async model function.

        Args:
            model_fn:               Async callable str → str.
            concurrency:            Max concurrent cases in flight (default 5).
            runs:                   Times to run each case (default 1).
            evaluator_concurrency:  Max concurrent evaluators *per case*.
                                    Defaults to running all evaluators in
                                    parallel. Set to 1 for strictly sequential
                                    evaluation within a case.
            judge_retry:            :class:`JudgeRetry` policy for transient
                                    judge / timeout errors. See
                                    :meth:`run` for semantics. Async path uses
                                    ``asyncio.sleep`` so concurrent retries
                                    don't block the event loop.

        Returns an :class:`EvalReport`. Each evaluator's ``aevaluate`` is
        awaited, so LLM-judge calls overlap I/O rather than serialising.
        """
        for ev in self._evaluators:
            if hasattr(ev, "prepare"):
                ev.prepare()

        sem = asyncio.Semaphore(concurrency)
        ev_sem = asyncio.Semaphore(evaluator_concurrency) if evaluator_concurrency else None

        async def _eval_one(ev, case: EvalCase, output: str, latency_ms: float, model_error: str | None):
            ev_name = getattr(ev, "name", type(ev).__name__)
            if model_error is not None and not isinstance(ev, (Latency, MaxLatency)):
                return EvalResult(
                    evaluator=ev_name,
                    score=0.0,
                    passed=False,
                    reason=f"[skipped — model error: {model_error}]",
                    metadata={"error_kind": "model_error_skip"},
                )
            # Catch judge + evaluator exceptions so one outage doesn't crash
            # the whole case. The error TYPE is tagged via
            # ``EvalResult.metadata['error_kind']`` so downstream code can
            # classify without parsing reason strings.
            try:
                if isinstance(ev, (Latency, MaxLatency)):
                    return await ev.aevaluate(case, output, latency_ms=latency_ms)
                return await ev.aevaluate(case, output)
            except JudgeUnavailable as ju:
                return EvalResult(
                    evaluator=ev_name, score=0.0, passed=False,
                    reason=f"[judge unavailable: {ju}]",
                    metadata={"error_kind": "judge_error", "error_detail": str(ju)},
                )
            except Exception as ex:
                return EvalResult(
                    evaluator=ev_name, score=0.0, passed=False,
                    reason=f"[evaluator error: {type(ex).__name__}: {ex}]",
                    metadata={"error_kind": "evaluator_error",
                              "error_detail": f"{type(ex).__name__}: {ex}"},
                )

        async def _gated_eval(ev, case, output, latency_ms, model_error):
            if ev_sem is None:
                return await _eval_one(ev, case, output, latency_ms, model_error)
            async with ev_sem:
                return await _eval_one(ev, case, output, latency_ms, model_error)

        async def _run_one_async(case: EvalCase) -> CaseResult:
            async with sem:
                single_runs = []
                for _ in range(runs):
                    t0 = time.time()
                    async_model_error: str | None = None
                    try:
                        output = await model_fn(case.input)
                    except Exception as e:
                        async_model_error = str(e)
                        output = f"[MODEL ERROR: {e}]"
                    latency_ms = (time.time() - t0) * 1000

                    ev_results = await asyncio.gather(*[
                        _gated_eval(ev, case, output, latency_ms, async_model_error)
                        for ev in self._evaluators
                    ])

                    # Surface judge/evaluator errors via the metadata sentinel
                    # set by _eval_one. Avoids brittle string parsing of
                    # human-readable reason strings.
                    async_judge_error: str | None = None
                    async_evaluator_error: str | None = None
                    for r in ev_results:
                        kind = r.metadata.get("error_kind") if r.metadata else None
                        if async_judge_error is None and kind == "judge_error":
                            async_judge_error = r.metadata.get("error_detail", r.reason)
                        elif async_evaluator_error is None and kind == "evaluator_error":
                            async_evaluator_error = r.metadata.get("error_detail", r.reason)

                    single_runs.append(CaseResult(
                        case_input=case.input,
                        actual_output=output,
                        model_error=async_model_error,
                        judge_error=async_judge_error,
                        evaluator_error=async_evaluator_error,
                        results=list(ev_results),
                        latency_ms=latency_ms,
                        tags=case.tags,
                        agent_trace=case.agent_trace,
                    ))

                if runs == 1:
                    return single_runs[0]
                return _aggregate_runs(case, single_runs)

        async def _run_one_async_with_retry(case: EvalCase) -> CaseResult:
            """Wrap ``_run_one_async`` with the retry policy. Mirrors the
            sync helper but uses ``asyncio.sleep`` so concurrent retries
            don't block the event loop."""
            if judge_retry is None:
                return await _run_one_async(case)
            from .retry import should_retry, async_sleep_for_attempt
            retry_errors: list[str] = []
            cr: CaseResult | None = None
            for attempt in range(1, judge_retry.max_attempts + 1):
                if attempt > 1:
                    await async_sleep_for_attempt(judge_retry, attempt)
                cr = await _run_one_async(case)
                if not should_retry(cr.status, judge_retry):
                    break
                if attempt >= judge_retry.max_attempts:
                    break  # exhausted — don't tally the final failure as a retry trigger
                retry_errors.append(
                    cr.judge_error or cr.evaluator_error or cr.model_error
                    or f"<{cr.status.value}>"
                )
            assert cr is not None
            if retry_errors:
                cr = dataclasses.replace(
                    cr,
                    retry_attempts=len(retry_errors),
                    retry_errors=list(retry_errors),
                )
            return cr

        # Cost tracker for the duration of run_async. Same mechanism as
        # the sync .run(); contextvars are correct under asyncio.
        from .costs import CostTracker, set_active_tracker, reset_token
        cost_tracker = CostTracker()
        cost_token = set_active_tracker(cost_tracker)

        try:
            case_results = await asyncio.gather(
                *[_run_one_async_with_retry(c) for c in self._cases]
            )
        finally:
            reset_token(cost_token)

        judge_reliability = _measure_judge_reliability(
            self._evaluators, list(case_results), self._cases
        )

        report = EvalReport(
            suite_name=self.name,
            case_results=list(case_results),
            model_id=self.model_id,
            judge_reliability=judge_reliability,
            costs=cost_tracker.snapshot(),
            suite_lock=_safe_lock(self),
        )

        if verbose:
            print_report(report)

        if fail_threshold is not None and report.pass_rate < fail_threshold:
            raise EvalGateFailure(
                f"\nEval failed: pass rate {report.pass_rate:.1%} < threshold {fail_threshold:.1%}",
                pass_rate=report.pass_rate,
                threshold=fail_threshold,
            )

        return report


def _safe_lock(suite: "EvalSuite") -> "Any":
    """Compute the suite's SuiteLock; swallow most failures.

    Used inside the suite.run hot path to attach reproducibility metadata
    to the report without ever letting the lockfile machinery crash a
    successful eval run. If fingerprinting fails for any reason, we'd
    rather ship a report without a lock than no report at all.

    Exception: ``FileNotFoundError`` from an unshipped calibration
    version (``MULTIVON_CALIBRATION_VERSION=v_unshipped``) is re-raised.
    Silently degrading there would defeat the whole point of pinning —
    the user thinks they're pinned to a known version, but the audit
    log would record no version and audit-package would fall back to
    the shipped default. Better to fail loudly at run time so the user
    fixes the config. Codex D12 round-2 finding.
    """
    try:
        return suite.lock()
    except FileNotFoundError:
        raise
    except Exception:
        return None


def _aggregate_runs(case: EvalCase, single_runs: list[CaseResult]) -> CaseResult:
    """Merge N single-run CaseResults into one aggregated CaseResult."""
    n = len(single_runs)
    all_scores = [cr.score for cr in single_runs]
    # Count a run as passing ONLY if its CaseResult.passed agrees — that
    # property already encodes the EvalStatus rule (error/skipped states
    # are NOT passes even if individual evaluator results show passed=True).
    pass_count = sum(1 for cr in single_runs if cr.passed)
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

    # Propagate error fields from any single run that hit one — otherwise a
    # judge outage on run 3-of-5 would be lost in the aggregate and the
    # case would silently downgrade to FAILED_QUALITY. Latch the FIRST
    # error of each kind across runs.
    agg_model_err: str | None = None
    agg_judge_err: str | None = None
    agg_eval_err: str | None = None
    agg_skipped = False
    for r in single_runs:
        if agg_model_err is None and r.model_error is not None:
            agg_model_err = r.model_error
        if agg_judge_err is None and r.judge_error is not None:
            agg_judge_err = r.judge_error
        if agg_eval_err is None and r.evaluator_error is not None:
            agg_eval_err = r.evaluator_error
        if r.skipped:
            agg_skipped = True
    return CaseResult(
        case_input=case.input,
        actual_output=single_runs[-1].actual_output,  # last run's output
        results=agg_results,
        latency_ms=avg_latency,
        tags=case.tags,
        runs=n,
        all_scores=all_scores,
        pass_count=pass_count,
        model_error=agg_model_err,
        judge_error=agg_judge_err,
        evaluator_error=agg_eval_err,
        skipped=agg_skipped,
        agent_trace=case.agent_trace,
    )


def _sprt_stop(runs_so_far: list[CaseResult], alpha: float = 0.05, beta: float = 0.20) -> bool:
    """
    Wald's Sequential Probability Ratio Test.

    Returns True when there is enough evidence to stop early — either the
    case is clearly passing (LR for H1_pass exceeds threshold) or clearly
    failing (LR for H1_fail exceeds threshold).

    Runs two one-sided SPRTs:
      Test 1: H0=p≤0.5  vs H1=p≥0.8  (clearly passing)
      Test 2: H0=p≥0.5  vs H1=p≤0.2  (clearly failing)
    Stop when either LR >= (1-beta)/alpha.

    alpha: false-positive rate (default 0.05)
    beta:  false-negative rate (default 0.20 → 80% power)
    """
    import math as _math
    # Use CaseResult.passed (status-aware) — error/skipped runs are not
    # "successes" for SPRT termination even if evaluators happened to record
    # passed=True before the error fired.
    passes = sum(1 for cr in runs_so_far if cr.passed)
    n = len(runs_so_far)
    if n < 2:
        return False

    fails = n - passes
    p0, p1_pass, p1_fail = 0.5, 0.8, 0.2
    threshold = (1 - beta) / alpha  # e.g. 16.0 at default settings

    # LR for "clearly passing": how much more likely is p=0.8 vs p=0.5?
    lr_pass = (p1_pass / p0) ** passes * ((1 - p1_pass) / (1 - p0)) ** fails
    if lr_pass >= threshold:
        return True

    # LR for "clearly failing": how much more likely is p=0.2 vs p=0.5?
    lr_fail = (p1_fail / p0) ** passes * ((1 - p1_fail) / (1 - p0)) ** fails
    if lr_fail >= threshold:
        return True

    return False


def _measure_judge_reliability(
    evaluators: "list[Evaluator]",
    case_results: "list[CaseResult]",
    original_cases: "list[EvalCase]",
) -> "float | None":
    """
    Re-run LLM evaluators on a sample of (case, output) pairs and measure
    agreement between first and second judge calls. Returns None if reliability
    check is disabled or there are no LLM evaluators to check.

    Agreement is % of (case, evaluator) pairs where both calls return the same
    pass/fail decision. High variance in the judge (< 80% agreement) means
    your eval scores contain substantial noise from the judge itself.
    """
    from .judge import get_global_judge
    judge_cfg = get_global_judge()
    if not judge_cfg.reliability_check or not case_results:
        return None

    import random as _rand
    sample_size = min(judge_cfg.reliability_sample, len(case_results))
    sample_indices = _rand.sample(range(len(case_results)), sample_size)

    agreements: list[bool] = []
    for idx in sample_indices:
        cr = case_results[idx]
        orig_case = original_cases[idx] if idx < len(original_cases) else EvalCase(input=cr.case_input)
        for ev in evaluators:
            if not hasattr(ev, "evaluate"):
                continue
            ev_name = getattr(ev, "name", type(ev).__name__.lower())
            first = next((r for r in cr.results if r.evaluator == ev_name), None)
            if first is None:
                continue
            try:
                second = ev.evaluate(orig_case, cr.actual_output)
                agreements.append(first.passed == second.passed)
            except Exception:
                pass

    if not agreements:
        return None
    return round(sum(agreements) / len(agreements), 4)
