from __future__ import annotations
import asyncio
import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Awaitable

from .case import EvalCase
from .result import CaseResult, EvalReport, EvalResult
from .evaluators.base import Evaluator
from .evaluators.deterministic import Latency, MaxLatency
from .reporters.terminal import print_report

if TYPE_CHECKING:
    from .integrations.base import AgentTracer


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

    def _run_case_once(
        self,
        case: EvalCase,
        model_fn: Callable[[str], str],
        tracer: "AgentTracer | None" = None,
    ) -> CaseResult:
        if tracer is not None:
            tracer.reset()

        t0 = time.time()
        try:
            output = model_fn(case.input)
        except Exception as e:
            output = f"[MODEL ERROR: {e}]"
        latency_ms = (time.time() - t0) * 1000

        if tracer is not None:
            trace = tracer.get_trace()
            if trace:
                case = dataclasses.replace(case, agent_trace=trace)

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

    def run(
        self,
        model_fn: Callable[[str], str],
        verbose: bool = True,
        fail_threshold: float | None = None,
        workers: int = 1,
        runs: int = 1,
        tracer: "AgentTracer | None" = None,
        early_stop: bool = False,
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
        """
        if tracer is not None and workers > 1:
            raise ValueError(
                "tracer and workers > 1 are incompatible — tracers are stateful. "
                "Run with workers=1 (the default) when using a tracer."
            )

        if tracer is not None:
            instrumented_fn = tracer.instrument(model_fn)
        else:
            instrumented_fn = model_fn

        if workers > 1:
            case_results = self._run_parallel(instrumented_fn, workers, runs)
        else:
            case_results = [
                self._run_case(case, instrumented_fn, runs, tracer=tracer, early_stop=early_stop)
                for case in self._cases
            ]

        report = EvalReport(
            suite_name=self.name,
            case_results=case_results,
            model_id=self.model_id,
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
            raise SystemExit(
                f"\nEval failed: pass rate {report.pass_rate:.1%} < threshold {fail_threshold:.1%}"
            )

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
            for ev in self._evaluators:
                result = ev.evaluate(case, output)
                results.append(result)
            case_results.append(CaseResult(
                case_input=case.input,
                actual_output=output,
                results=results,
                latency_ms=0.0,
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
    passes = sum(1 for cr in runs_so_far if all(r.passed for r in cr.results))
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
