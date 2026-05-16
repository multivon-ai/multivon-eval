"""
SelfConsistency evaluator — zero-resource hallucination detection via repeated sampling.

Based on SelfCheckGPT (Manakul et al., EMNLP 2023) with production improvements:

1. Adaptive N: starts at n_init samples, escalates only when the score is
   borderline — most cases resolve at n=5, expensive cases get n=20.

2. Free-sample integration: pass samples= to reuse outputs already generated
   by suite.run(runs=N), so consistency checking costs zero extra API calls.

3. Two backends:
   - "nli"  — local DeBERTa cross-encoder, no API calls, ~150ms/check.
              Requires: pip install transformers torch
   - "llm"  — LLM judge via JudgeConfig, highest accuracy, uses API quota.

4. Sentence-level breakdown in reason string for debuggability.

Key limitation (inherited from the paper): consistency ≠ factuality.
If a model consistently hallucinates the same wrong fact (common for popular
misconceptions), this evaluator will score it as consistent. Pair with
Faithfulness when a reference document is available.
"""
from __future__ import annotations

import re
from typing import Callable

from .base import Evaluator
from ..case import EvalCase
from ..exceptions import JudgeUnavailable
from ..result import EvalResult
from ..judge import JudgeConfig, resolve_judge, make_judge_call

__all__ = ["SelfConsistency"]

_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# Module-level cache — pipeline is expensive to load, reuse across evaluations
_nli_cache: dict[str, object] = {}


def _load_nli_pipeline(model: str):
    if model in _nli_cache:
        return _nli_cache[model]
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "text-classification",
            model=model,
            top_k=None,
            device=-1,  # CPU; set CUDA_VISIBLE_DEVICES to use GPU
        )
        _nli_cache[model] = pipe
        return pipe
    except ImportError:
        return None


def _contradiction_prob(premise: str, hypothesis: str, pipe) -> float:
    """Return P(contradiction) for a premise/hypothesis pair."""
    try:
        result = pipe({"text": premise, "text_pair": hypothesis})
        items = result[0] if isinstance(result[0], list) else result
        for item in items:
            if "contradict" in str(item.get("label", "")).lower():
                return float(item["score"])
    except Exception:
        pass
    return 0.0


def _split_sentences(text: str) -> list[str]:
    """Split text into evaluable sentences (min 20 chars to skip fragments)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 20]


class SelfConsistency(Evaluator):
    """
    Zero-resource hallucination detection via repeated sampling.

    Core idea: if a model truly knows something, stochastic samples of the same
    prompt converge. If it hallucinated, samples contradict each other.

    Args:
        model_fn:   The same callable passed to suite.run(). Used to generate
                    consistency samples. Not required if you pass samples=
                    directly to evaluate().
        n:          Starting sample count (default 5). With adaptive=True this
                    is the floor; up to max_n samples may be used.
        max_n:      Sample ceiling when adaptive=True (default 20).
        adaptive:   Escalate sample count when score is borderline (default True).
        backend:    "nli" (local, no API cost), "llm" (best accuracy), or
                    "auto" (nli if transformers is installed, else llm).
        nli_model:  HuggingFace model name for the nli backend.
        judge:      JudgeConfig for the llm backend. Falls back to global config.
        threshold:  Pass/fail threshold (default 0.7).

    Basic usage:
        suite.add_evaluators(SelfConsistency(model_fn=my_pipeline))
        report = suite.run(my_pipeline)

    Free-sample usage (zero extra API calls):
        # Generate samples manually and reuse across multiple evaluators
        samples = [my_pipeline(case.input) for _ in range(5)]
        evaluator.evaluate(case, output, samples=samples)

    With explicit judge:
        SelfConsistency(
            model_fn=my_pipeline,
            backend="llm",
            judge=JudgeConfig(provider="openai", model="gpt-4o-mini"),
        )
    """

    name = "self_consistency"

    # Score range considered borderline — triggers adaptive escalation
    _BORDERLINE_LO = 0.3
    _BORDERLINE_HI = 0.7

    def __init__(
        self,
        model_fn: Callable[[str], str] | None = None,
        n: int = 5,
        max_n: int = 20,
        adaptive: bool = True,
        backend: str = "auto",
        nli_model: str = _DEFAULT_NLI_MODEL,
        judge: JudgeConfig | None = None,
        threshold: float = 0.7,
    ):
        super().__init__(threshold)
        self._model_fn = model_fn
        self._n = n
        self._max_n = max_n
        self._adaptive = adaptive
        self._backend = backend
        self._nli_model = nli_model
        self._judge_cfg = judge

    def _active_backend(self) -> str:
        if self._backend != "auto":
            return self._backend
        pipe = _load_nli_pipeline(self._nli_model)
        return "nli" if pipe is not None else "llm"

    def _generate_samples(self, input_text: str, count: int) -> list[str]:
        if self._model_fn is None:
            raise ValueError(
                "SelfConsistency needs model_fn= at construction time, "
                "or pass pre-generated samples= to evaluate()."
            )
        results = []
        for _ in range(count):
            try:
                results.append(self._model_fn(input_text))
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------
    # NLI backend
    # ------------------------------------------------------------------

    def _score_nli(self, output: str, samples: list[str]) -> tuple[float, list[str]]:
        pipe = _load_nli_pipeline(self._nli_model)
        if pipe is None:
            # Graceful fallback if transformers wasn't importable at score time
            return self._score_llm(output, samples)

        sentences = _split_sentences(output) or [output]
        sentence_scores: list[float] = []
        reasons: list[str] = []

        for sent in sentences[:8]:
            contradiction_probs = []
            for sample in samples:
                prob = _contradiction_prob(sample, sent, pipe)
                contradiction_probs.append(prob)

            if not contradiction_probs:
                continue

            avg_contradiction = sum(contradiction_probs) / len(contradiction_probs)
            sentence_scores.append(avg_contradiction)
            flag = "✗" if avg_contradiction > 0.5 else "✓"
            reasons.append(f"{flag} [{avg_contradiction:.2f}] {sent[:80]}")

        if not sentence_scores:
            return 0.5, ["NLI scoring produced no results"]

        hallucination_rate = sum(sentence_scores) / len(sentence_scores)
        return 1.0 - hallucination_rate, reasons

    # ------------------------------------------------------------------
    # LLM backend
    # ------------------------------------------------------------------

    def _score_llm(self, output: str, samples: list[str]) -> tuple[float, list[str]]:
        judge = resolve_judge(self._judge_cfg)
        sentences = _split_sentences(output) or [output]

        consistent, total = 0, 0
        reasons: list[str] = []

        for sent in sentences[:6]:
            for sample in samples[:5]:  # cap LLM calls: 6 × 5 = 30 max
                prompt = (
                    f"Reference passage:\n{sample}\n\n"
                    f"Statement: {sent}\n\n"
                    f"Does the reference passage support or contradict this statement? "
                    f'Answer with only "Consistent" or "Contradicts".'
                )
                try:
                    answer = make_judge_call(prompt, judge)
                    is_consistent = "consistent" in answer.strip().lower()
                    consistent += int(is_consistent)
                    total += 1
                except JudgeUnavailable:
                    raise
                except Exception:
                    total += 1

        if total == 0:
            return 0.5, ["No comparisons completed"]

        score = consistent / total
        reasons.append(f"{consistent}/{total} sentence-sample pairs consistent")
        return score, reasons

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(  # type: ignore[override]
        self,
        case: EvalCase,
        output: str,
        samples: list[str] | None = None,
    ) -> EvalResult:
        """
        Args:
            case:    The eval case.
            output:  The model output to evaluate.
            samples: Pre-generated samples from the same model. If provided,
                     no additional model calls are made (free consistency check).
                     If None, model_fn is called n times to generate samples.
        """
        backend = self._active_backend()

        # Seed from pre-generated samples, then top up if model_fn is available
        all_samples = list(samples or [])
        needed = max(0, self._n - len(all_samples))
        if needed > 0 and self._model_fn is not None:
            all_samples += self._generate_samples(case.input, needed)

        if not all_samples:
            return self._result(
                0.5,
                "No samples available — pass model_fn= or samples= to SelfConsistency",
            )

        # First pass
        score, reasons = (
            self._score_nli(output, all_samples)
            if backend == "nli"
            else self._score_llm(output, all_samples)
        )

        # Adaptive escalation: generate more samples if score is borderline
        if (
            self._adaptive
            and self._BORDERLINE_LO < score < self._BORDERLINE_HI
            and len(all_samples) < self._max_n
            and self._model_fn is not None
        ):
            extra = min(self._max_n - len(all_samples), 10)
            all_samples += self._generate_samples(case.input, extra)
            score, reasons = (
                self._score_nli(output, all_samples)
                if backend == "nli"
                else self._score_llm(output, all_samples)
            )

        n_used = len(all_samples)
        header = f"{backend} backend · {n_used} sample{'s' if n_used != 1 else ''}"
        detail = "\n".join(reasons)
        return self._result(score, f"{header}\n{detail}")
