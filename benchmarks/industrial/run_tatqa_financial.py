"""Run the frozen paired TAT-QA financial-answer experiment.

The runner reads only the label-free projection produced by
``prepare_tatqa_financial.py``. Pass ``ANTHROPIC_API_KEY`` through the environment.
Raw provider evidence is written to ``--out`` and should remain outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anthropic
from tatqa_financial import (
    MODEL,
    TREATMENTS,
    output_schema,
    parse_response,
    prompt_for,
    sha256,
    write_json,
)

from multivon_eval import ProviderJournal, capture_provider_events

SEED = 17092026
SYSTEM = """You are a financial-document question answering system. Return the
requested structured record directly. Use only the supplied document, keep
answers concise, and do not include private reasoning."""
PRICING = {
    "input_per_million_usd": 2.0,
    "output_per_million_usd": 10.0,
    "source": "https://docs.anthropic.com/en/docs/about-claude/pricing",
    "retrieved": "2026-09-17",
}
_LOCAL = threading.local()


def client() -> anthropic.Anthropic:
    if not hasattr(_LOCAL, "client"):
        _LOCAL.client = anthropic.Anthropic(max_retries=2, timeout=180)
    return _LOCAL.client


def text_content(message: Any) -> str:
    blocks = [block.text for block in message.content if getattr(block, "type", None) == "text"]
    if len(blocks) != 1:
        raise ValueError(f"Expected one text output block, got {len(blocks)}")
    return blocks[0]


def execute(
    context: dict[str, Any],
    treatment: str,
    model: str,
    max_tokens: int,
    journal: ProviderJournal,
) -> dict[str, Any]:
    identity = {"context_id": context["context_id"], "treatment": treatment}
    result: dict[str, Any] = dict(identity)
    started = time.monotonic()
    with capture_provider_events(journal=journal, labels={"experiment": "tatqa-financial-v1", **identity}) as capture:
        try:
            message = client().messages.create(
                model=model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": output_schema(context, treatment)},
                },
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt_for(context, treatment)}],
            )
            raw = text_content(message)
            parsed = parse_response(raw, context, treatment)
            result.update(
                status="scored",
                responses=parsed,
                model=message.model,
                stop_reason=message.stop_reason,
                usage=message.usage.model_dump(mode="json"),
                request_id=getattr(message, "_request_id", None),
            )
        except Exception as error:  # noqa: BLE001 - failures stay in the denominator
            result.update(status="error", error_type=type(error).__name__)
    evidence = capture.snapshot()
    result.update(
        seconds=time.monotonic() - started,
        capture_id=capture.capture_id,
        coverage_gaps=evidence["coverage_gaps"],
        requests_without_response=evidence["requests_without_complete_response"],
    )
    return result


def validate_inputs(inputs_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("inputs_sha256") != sha256(inputs_path):
        raise ValueError("Input projection hash does not match its manifest")
    contexts = json.loads(inputs_path.read_text())
    if len(contexts) != 277 or sum(len(context["questions"]) for context in contexts) != 1663:
        raise ValueError("Expected the complete 277-context, 1663-question projection")
    forbidden = {
        "answer",
        "answer_type",
        "answer_from",
        "scale",
        "derivation",
        "facts",
        "mappings",
        "tree_derivation",
    }
    for context in contexts:
        for question in context["questions"]:
            if forbidden & question.keys():
                raise ValueError("Gold field found in model input projection")
    return contexts, manifest


def protocol_for(
    root: Path,
    inputs_path: Path,
    manifest_path: Path,
    model: str,
    workers: int,
    max_tokens: int,
) -> dict[str, Any]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {
        "study": "TAT-QA financial answer and evidence-record comparison",
        "revision": revision,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "common_sha256": hashlib.sha256(Path(__file__).with_name("tatqa_financial.py").read_bytes()).hexdigest(),
        "inputs_sha256": sha256(inputs_path),
        "manifest_sha256": sha256(manifest_path),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("anthropic", "multivon-eval")
        },
        "model": model,
        "model_snapshot": "Provider response model retained per request; API lists no dated Sonnet 5 ID",
        "thinking": {"type": "adaptive", "effort": "low"},
        "temperature": "omitted",
        "max_tokens_per_context": max_tokens,
        "structured_outputs": "Anthropic JSON schema output_config",
        "treatments": list(TREATMENTS),
        "contexts": 277,
        "questions": 1663,
        "planned_logical_requests": 554,
        "workers": workers,
        "shuffle_seed": SEED,
        "retry_policy": "Anthropic Python SDK max_retries=2; observed attempts retained by ProviderJournal",
        "primary": "Official TAT-QA exact match and F1 on all 1663 released test-gold questions",
        "paired_uncertainty": "5000 context-bootstrap resamples; paired treatments retained together",
        "evidence_secondary": "Agreement with released annotation locations; paragraph agreement is order-level",
        "missing_or_malformed": "Retained as zero official score and zero workflow acceptance",
        "tuning": "No labels supplied to the model; no prompt/model selection after test outcomes",
        "pricing": PRICING,
        "limits": [
            "Public released test labels may be contaminated in model training",
            "Annotation-location agreement is not proof of semantic citation validity",
            "The model alias is mutable and does not identify immutable weights",
            "Public data and simulated review records do not establish customer usefulness",
        ],
    }


def load_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    completed = {(row["context_id"], row["treatment"]): row for row in rows}
    if len(completed) != len(rows):
        raise ValueError("Duplicate completed jobs in predictions.jsonl")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise RuntimeError("Commit the frozen runner before inference")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    contexts, _ = validate_inputs(args.inputs.resolve(), args.manifest.resolve())
    protocol = protocol_for(
        root, args.inputs.resolve(), args.manifest.resolve(), args.model, args.workers, args.max_tokens
    )
    predictions_path = args.out / "predictions.jsonl"
    if args.resume:
        if not args.out.is_dir():
            raise ValueError("--resume requires an existing output directory")
        if json.loads((args.out / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol mismatch")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "protocol.json", protocol)
    completed = load_completed(predictions_path)
    jobs = [
        (context, treatment)
        for context in contexts
        for treatment in TREATMENTS
        if (context["context_id"], treatment) not in completed
    ]
    random.Random(SEED).shuffle(jobs)
    mode = "a" if args.resume else "w"
    start = time.monotonic()
    with (
        ProviderJournal(args.out / "events.sqlite") as journal,
        predictions_path.open(mode) as stream,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = [
            pool.submit(execute, context, treatment, args.model, args.max_tokens, journal)
            for context, treatment in jobs
        ]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if count % 20 == 0 or count == len(jobs):
                print(
                    json.dumps(
                        {
                            "completed_this_run": count,
                            "remaining_this_run": len(jobs) - count,
                            "total_completed": len(completed) + count,
                            "elapsed_seconds": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
    all_rows = load_completed(predictions_path)
    write_json(
        args.out / "execution.json",
        {
            "jobs_expected": 554,
            "jobs_completed": len(all_rows),
            "status_counts": {
                status: sum(row["status"] == status for row in all_rows.values())
                for status in sorted({row["status"] for row in all_rows.values()})
            },
            "seconds_this_run": time.monotonic() - start,
        },
    )
    secret = os.environ["ANTHROPIC_API_KEY"].encode()
    for path in args.out.iterdir():
        if path.is_file() and secret in path.read_bytes():
            raise RuntimeError("Credential found in evidence artifact")


if __name__ == "__main__":
    main()
