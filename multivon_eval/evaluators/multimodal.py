"""Multimodal QAG evaluators (experimental, 0.7.3).

QAG-based evaluators for outputs grounded in images or PDF pages. These
are the seed evaluators for the Document Agent Acceptance Protocol —
the first multimodal capabilities shipped in the library.

The judge model must support vision input. Currently supported:

- ``anthropic`` provider with Claude 3.5+ (Haiku 4.5, Sonnet 4.6, Opus 4.7).
- ``openai`` provider with GPT-4o+ (gpt-4o, gpt-4o-mini, gpt-5.x).
- ``google`` provider with Gemini 1.5+ (default judge for cost reasons).

Status: experimental. Calibrated thresholds will ship in a follow-up
release once the Document Agent Acceptance Protocol v1 dataset is
finalised. Until then, the standard calibration-fallback policy applies
(see :func:`multivon_eval.set_calibration_fallback_policy`).

Each evaluator reads images from ``case.metadata``:

- ``case.metadata["image_url"]`` — single URL (http/https or data: URI).
- ``case.metadata["image_path"]`` — single local file path.
- ``case.metadata["images"]`` — list of URLs or paths (multi-page documents).

For :class:`DocumentGrounding`, ``case.metadata["images"]`` is the
expected key (one entry per page).
"""
from __future__ import annotations

import base64
import json
import mimetypes
import pathlib
import re
from typing import Iterable

from .base import Evaluator
from ..calibration import calibrated_threshold as _calibrated_threshold
from ..case import EvalCase
from ..exceptions import JudgeUnavailable
from ..judge import JudgeConfig, resolve_judge
from ..result import EvalResult


# Vision-capable models per provider. Used to gate friendlier error
# messages when a user wires a text-only judge into a vision evaluator.
_VISION_CAPABLE = {
    "anthropic": {"claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7",
                  "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"},
    "openai": {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5", "gpt-5-mini",
               "gpt-5.5", "gpt-5.5-mini"},
    "google": {"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
               "gemini-1.5-pro", "gemini-1.5-flash"},
}


def _is_vision_capable(judge: JudgeConfig) -> bool:
    """Return True if ``judge.model`` plausibly accepts image input.

    We match on family prefix (e.g. ``gemini-2.5-pro`` matches the literal
    name and also any ``gemini-2.5-pro-preview-…`` snapshot). Conservative:
    when in doubt we return True and let the provider API surface the
    real error. The point of this check is to give a nicer hint when the
    user obviously wired a text-only judge like ``gpt-3.5-turbo``.
    """
    model = (judge.model or "").lower()
    if not model:
        return True  # let the provider decide
    for known in _VISION_CAPABLE.get(judge.provider, set()):
        if model.startswith(known.lower()):
            return True
    # If we don't know the provider's capability map, don't block.
    return judge.provider not in _VISION_CAPABLE


def _image_to_data_uri(src: str) -> tuple[str, str, str]:
    """Return ``(data_uri, mime_type, base64_data)`` for an image source.

    ``src`` may be:
    - an ``http(s)://`` URL — returned as-is with mime guessed from suffix
      (the provider will fetch it server-side);
    - a ``data:`` URI — returned as-is;
    - a local filesystem path — read and inlined as a data URI.

    For provider APIs that prefer a URL (OpenAI) we still emit the data
    URI; OpenAI accepts both forms.
    """
    if src.startswith("data:"):
        # data:<mime>;base64,<...>
        match = re.match(r"data:([^;]+);base64,(.+)$", src)
        if not match:
            raise ValueError(f"unrecognised data URI: {src[:60]}")
        return src, match.group(1), match.group(2)
    if src.startswith("http://") or src.startswith("https://"):
        mime = mimetypes.guess_type(src)[0] or "image/jpeg"
        return src, mime, ""
    path = pathlib.Path(src).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {src}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}", mime, data


def _call_vision_judge(
    prompt: str,
    images: list[str],
    judge: JudgeConfig,
    max_tokens: int = 200,
) -> str:
    """Call a vision-capable judge with a text prompt + one or more images.

    Provider dispatch:
    - ``anthropic``: messages API with content blocks (text + base64 image).
    - ``openai``: chat.completions with ``image_url`` content parts.
    - ``google``: generateContent with inline image parts.

    Raises :class:`JudgeUnavailable` if the SDK isn't installed or no API
    key is set. Keeps the surface narrow so the existing
    ``JudgeUnavailable`` retry/cost-tracking infrastructure still applies.
    """
    if not _is_vision_capable(judge):
        raise JudgeUnavailable(
            f"multimodal evaluator requires a vision-capable judge; "
            f"{judge.provider}/{judge.model} is text-only. Try Gemini "
            "2.5 Flash (cheap), Claude Haiku 4.5, or GPT-4o-mini."
        )
    provider = judge.provider
    if provider == "anthropic":
        return _anthropic_vision_call(prompt, images, judge, max_tokens)
    if provider == "openai":
        return _openai_vision_call(prompt, images, judge, max_tokens)
    if provider == "google":
        return _google_vision_call(prompt, images, judge, max_tokens)
    raise JudgeUnavailable(
        f"provider {provider!r} is not yet wired for vision input; "
        "use anthropic, openai, or google."
    )


def _anthropic_vision_call(
    prompt: str, images: list[str], judge: JudgeConfig, max_tokens: int
) -> str:
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as exc:
        raise JudgeUnavailable("anthropic SDK not installed") from exc
    content: list[dict] = []
    for img in images:
        _, mime, b64 = _image_to_data_uri(img)
        if b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
        else:
            content.append({"type": "image", "source": {"type": "url", "url": img}})
    content.append({"type": "text", "text": prompt})
    client = anthropic.Anthropic(api_key=judge.api_key) if getattr(judge, "api_key", None) else anthropic.Anthropic()
    msg = client.messages.create(
        model=judge.model,
        max_tokens=max_tokens,
        temperature=judge.temperature,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _openai_vision_call(
    prompt: str, images: list[str], judge: JudgeConfig, max_tokens: int
) -> str:
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise JudgeUnavailable("openai SDK not installed") from exc
    parts: list[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        data_uri, _, _ = _image_to_data_uri(img)
        parts.append({"type": "image_url", "image_url": {"url": data_uri}})
    client = openai.OpenAI(
        api_key=judge.api_key if getattr(judge, "api_key", None) else None,
        base_url=judge.base_url if judge.base_url else None,
    )
    resp = client.chat.completions.create(
        model=judge.model,
        max_tokens=max_tokens,
        temperature=judge.temperature,
        messages=[{"role": "user", "content": parts}],
    )
    return resp.choices[0].message.content or ""


def _google_vision_call(
    prompt: str, images: list[str], judge: JudgeConfig, max_tokens: int
) -> str:
    try:
        from google import genai  # type: ignore[import-not-found]
        from google.genai import types as genai_types  # type: ignore[import-not-found]
    except ImportError as exc:
        raise JudgeUnavailable("google-genai SDK not installed") from exc
    contents: list = []
    for img in images:
        _, mime, b64 = _image_to_data_uri(img)
        if b64:
            contents.append(
                genai_types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime)
            )
        else:
            # Gemini doesn't fetch remote URLs server-side; we'd have to
            # download first. Keep this minimal — strongly suggest local
            # paths or data URIs for Gemini.
            raise JudgeUnavailable(
                "google-genai requires local files or data URIs for image input; "
                f"got remote URL: {img}"
            )
    contents.append(prompt)
    client = genai.Client(api_key=judge.api_key) if getattr(judge, "api_key", None) else genai.Client()
    resp = client.models.generate_content(
        model=judge.model,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            temperature=judge.temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text or ""


_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _parse_yes_no(text: str) -> bool:
    """Return True if the first yes/no token is 'yes'."""
    m = _YES_NO_RE.search(text or "")
    if not m:
        return False
    return m.group(1).lower() == "yes"


def _get_images(case: EvalCase) -> list[str]:
    """Extract image sources from case metadata in the documented order."""
    md = case.metadata or {}
    if "images" in md and md["images"]:
        items = md["images"]
        return list(items) if isinstance(items, Iterable) and not isinstance(items, (str, bytes)) else [items]
    if "image_url" in md and md["image_url"]:
        return [md["image_url"]]
    if "image_path" in md and md["image_path"]:
        return [md["image_path"]]
    return []


class VQAFaithfulness(Evaluator):
    """Image-grounded faithfulness: does the response describe what is
    actually visible in the image?

    Experimental — first shipped 2026-05-16 as part of the Document Agent
    Acceptance Protocol seed work. The QAG prompt asks the vision judge
    to confirm/deny three claims grounded in the image, then scores as
    the fraction confirmed. Default threshold falls through the standard
    calibration-fallback policy.

    Usage::

        from multivon_eval import EvalCase, EvalSuite, VQAFaithfulness
        from multivon_eval import JudgeConfig

        case = EvalCase(
            input="What is the patient's diagnosis on this scan?",
            metadata={"image_path": "scans/chest-xray-001.png"},
        )
        suite = EvalSuite()
        suite.add_evaluators(VQAFaithfulness(judge=JudgeConfig(
            provider="google", model="gemini-2.5-flash", temperature=0.0,
        )))
    """

    name = "vqa_faithfulness"

    def __init__(self, threshold: float | None = None, judge: JudgeConfig | None = None):
        self._explicit_threshold = threshold
        self._judge_cfg = judge
        super().__init__(threshold if threshold is not None else 0.7)

    def _resolve_threshold(self, judge: JudgeConfig) -> float:
        if self._explicit_threshold is not None:
            return self._explicit_threshold
        return _calibrated_threshold(self.name, judge)

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        images = _get_images(case)
        if not images:
            return self._result(
                0.0,
                "No image provided — VQAFaithfulness requires case.metadata['image_url'], "
                "['image_path'], or ['images'].",
            )
        judge = resolve_judge(self._judge_cfg)
        self.threshold = self._resolve_threshold(judge)

        # Decompose the response into 3 image-grounded claims. We ask the
        # judge to generate the claims itself, then verify each one
        # against the image. Two-stage QAG keeps the prompt short and
        # auditable.
        try:
            claims_raw = _call_vision_judge(
                "Below is an answer about an image. Extract up to 3 specific factual "
                "claims that this answer makes about what is visible in the image. "
                "Return a JSON array of short strings.\n\n"
                f"Answer:\n{output}\n\nJSON array:",
                images,
                judge,
                max_tokens=400,
            )
            match = re.search(r"\[.*?\]", claims_raw, re.DOTALL)
            claims = json.loads(match.group()) if match else []
        except JudgeUnavailable:
            raise
        except Exception as exc:
            return self._result(0.0, f"Failed to extract claims: {exc}")
        if not claims:
            return self._result(
                1.0, "No image-grounded claims found in answer; trivially faithful."
            )

        verified: list[bool] = []
        reasons: list[str] = []
        for claim in claims[:5]:
            try:
                ans = _call_vision_judge(
                    f"Is the following claim about the image accurate?\n\n"
                    f"Claim: {claim}\n\nAnswer with only \"Yes\" or \"No\".",
                    images,
                    judge,
                    max_tokens=20,
                )
                ok = _parse_yes_no(ans)
                verified.append(ok)
                reasons.append(f"{'✓' if ok else '✗'} {claim[:90]}")
            except JudgeUnavailable:
                raise
            except Exception as exc:  # pragma: no cover
                verified.append(False)
                reasons.append(f"✗ {claim[:90]} (eval error: {exc})")
        score = sum(verified) / len(verified) if verified else 0.0
        return self._result(
            score,
            f"{sum(verified)}/{len(verified)} image-grounded claims verified\n"
            + "\n".join(reasons),
        )


class DocumentGrounding(Evaluator):
    """Document-page-grounded faithfulness for multi-page document agents.

    Experimental. Seed evaluator for the Document Agent Acceptance Protocol
    v0.1. ``case.metadata['images']`` is a list of page images (one per
    page) and the response must reference what is visible across those
    pages. Score is the fraction of QAG questions answered positively
    against the assembled document.

    For now this is essentially :class:`VQAFaithfulness` extended to
    multi-image input — the protocol-specific questions (page citation
    accuracy, table extraction fidelity, exception handling) will land in
    a follow-up release once design partners surface the precise failure
    modes that matter. Track progress in
    `multivon-strategy/plans/strategy-2026-05-16.md`.
    """

    name = "document_grounding"

    def __init__(self, threshold: float | None = None, judge: JudgeConfig | None = None):
        self._explicit_threshold = threshold
        self._judge_cfg = judge
        super().__init__(threshold if threshold is not None else 0.7)

    def _resolve_threshold(self, judge: JudgeConfig) -> float:
        if self._explicit_threshold is not None:
            return self._explicit_threshold
        return _calibrated_threshold(self.name, judge)

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        images = _get_images(case)
        if not images:
            return self._result(
                0.0,
                "No document pages provided — DocumentGrounding requires "
                "case.metadata['images'] (list of page image paths/URLs/data-URIs).",
            )
        judge = resolve_judge(self._judge_cfg)
        self.threshold = self._resolve_threshold(judge)

        prompt = (
            "You are evaluating an answer produced about a multi-page document. "
            "All pages are shown above as images. Answer the following questions "
            "with only \"Yes\" or \"No\" — be strict and grounded only in what is "
            f"visible in the document images.\n\n"
            f"Answer being evaluated:\n{output}\n\n"
            "Q1: Is every factual claim in the answer supported by content visible "
            "in at least one of the document pages?\n"
            "Q2: Does the answer avoid inventing any entity (name, date, number, "
            "amount, clause) that does not appear in the pages?\n"
            "Q3: Does the answer correctly handle the most important exception, "
            "caveat, or carve-out visible on the pages?\n\n"
            "Reply as three lines, each starting Q<n>: <Yes|No>."
        )
        try:
            raw = _call_vision_judge(prompt, images, judge, max_tokens=200)
        except JudgeUnavailable:
            raise
        except Exception as exc:
            return self._result(0.0, f"Vision judge error: {exc}")
        # Parse Q1/Q2/Q3 lines tolerantly.
        per_q: list[bool] = []
        reasons: list[str] = []
        for q in ("Q1", "Q2", "Q3"):
            m = re.search(rf"{q}\s*:\s*(yes|no)", raw, re.IGNORECASE)
            if not m:
                per_q.append(False)
                reasons.append(f"✗ {q} not answered (raw: {raw[:60]!r})")
                continue
            ok = m.group(1).lower() == "yes"
            per_q.append(ok)
            reasons.append(f"{'✓' if ok else '✗'} {q}")
        score = sum(per_q) / len(per_q) if per_q else 0.0
        return self._result(score, "\n".join(reasons))


__all__ = ["VQAFaithfulness", "DocumentGrounding"]
