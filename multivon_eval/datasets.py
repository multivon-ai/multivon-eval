"""Portable case snapshots and versioned, source-separated datasets."""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from .case import AgentStep, EvalCase, ToolCall

DATASET_SCHEMA = "multivon.dataset/v1"


def canonical_json(value: Any) -> str:
    """Strict portable JSON: reject lossy keys, objects, NaN and infinity."""
    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError("Portable JSON requires string object keys")
            for v in item.values():
                check(v)
        elif isinstance(item, (list, tuple)):
            for v in item:
                check(v)
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def trace_to_data(trace: list[AgentStep] | None) -> list[dict] | None:
    if trace is None:
        return None
    return [{"thought": step.thought, "output": step.output,
             "tool_calls": [{"name": t.name, "arguments": t.arguments,
                             "result": t.result} for t in step.tool_calls]}
            for step in trace]


def trace_from_data(data: list[dict] | None) -> list[AgentStep] | None:
    if data is None:
        return None
    if not isinstance(data, list):
        raise ValueError("agent_trace must be a list")
    result = []
    for step in data:
        if not isinstance(step, dict) or set(step) - {"thought", "output", "tool_calls"}:
            raise ValueError("Invalid agent trace step")
        calls = step.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        tools = []
        for call in calls:
            if (not isinstance(call, dict) or set(call) - {"name", "arguments", "result"}
                    or not isinstance(call.get("name"), str)
                    or not isinstance(call.get("arguments", {}), dict)):
                raise ValueError("Invalid tool call")
            tools.append(ToolCall(**call))
        if any(not isinstance(step.get(k, ""), str) for k in ("thought", "output")):
            raise ValueError("Trace thought and output must be strings")
        result.append(AgentStep(step.get("thought", ""), tools, step.get("output", "")))
    return result


def case_to_dict(case: EvalCase, *, include_reference: bool = True) -> dict:
    """Serialize all case fields, rejecting nonportable values without coercion."""
    data = {f.name: getattr(case, f.name) for f in fields(EvalCase)}
    data["agent_trace"] = trace_to_data(case.agent_trace)
    if not include_reference:
        data.pop("reference_output")
    elif callable(case.reference_output):
        raise ValueError("Callable reference_output cannot be exported; materialize it first")
    # Round trip detaches nested containers from the mutable source case.
    data = json.loads(canonical_json(data))
    case_from_dict(data)
    return data


def case_from_dict(data: dict) -> EvalCase:
    """Parse a portable case without silently discarding unknown fields."""
    if not isinstance(data, dict):
        raise ValueError("Case must be an object")
    unknown = set(data) - {f.name for f in fields(EvalCase)}
    if unknown:
        raise ValueError(f"Unknown case fields: {', '.join(sorted(unknown))}")
    if not isinstance(data.get("input"), str):
        raise ValueError("Case input must be a string")
    for name in ("expected_output", "reference_output", "case_id", "revision", "source_id"):
        value = data.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string or null")
        if name in {"case_id", "revision", "source_id"} and value == "":
            raise ValueError(f"{name} cannot be empty")
    context = data.get("context")
    if context is not None and not (isinstance(context, str) or
            isinstance(context, list) and all(isinstance(x, str) for x in context)):
        raise ValueError("context must be text or a list of text")
    for name in ("tags", "expected_tool_calls"):
        value = data.get(name, [] if name == "tags" else None)
        if value is None and name == "expected_tool_calls":
            continue
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"{name} must be a list of strings")
    if not isinstance(data.get("metadata", {}), dict):
        raise ValueError("metadata must be an object")
    conversation = data.get("conversation")
    if conversation is not None and (not isinstance(conversation, list) or any(
            not isinstance(m, dict) or not isinstance(m.get("role"), str)
            or not isinstance(m.get("content"), str) for m in conversation)):
        raise ValueError("conversation must contain role/content messages")
    copy = json.loads(canonical_json(data))
    copy["agent_trace"] = trace_from_data(copy.get("agent_trace"))
    return EvalCase(**copy)


def case_identity(case: EvalCase) -> tuple[str, str]:
    data = case_to_dict(case, include_reference=False)
    explicit_id = data.pop("case_id")
    content_digest = digest(data)
    return explicit_id or f"sha256:{content_digest}", content_digest


class Dataset:
    """An immutable snapshot with named splits and source-group leakage checks.

    Explicit IDs are recommended for cases that evolve. Supply source_id for
    variants derived from the same document, customer session, or environment.
    Splits must assign every case exactly once. No inferred source grouping is
    claimed when source_id is absent.
    """

    def __init__(self, name: str, cases: Iterable[EvalCase], *,
                 splits: Mapping[str, Iterable[str]] | None = None):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be nonempty")
        rows = []
        ids: set[str] = set()
        for case in cases:
            data = case_to_dict(case)
            case_id, content_digest = case_identity(case)
            if case_id in ids:
                raise ValueError(f"Duplicate case ID: {case_id}; assign distinct explicit IDs")
            ids.add(case_id)
            rows.append({"id": case_id, "digest": content_digest, "case": data})
        split_map = {key: list(value) for key, value in (splits or {}).items()}
        assigned: dict[str, str] = {}
        sources: dict[str, str] = {}
        by_id = {row["id"]: row for row in rows}
        for split, members in split_map.items():
            if not isinstance(split, str) or not split:
                raise ValueError("Split names must be nonempty strings")
            for case_id in members:
                if case_id not in ids:
                    raise ValueError(f"Unknown case in split {split}: {case_id}")
                if case_id in assigned:
                    raise ValueError(f"Case assigned more than once: {case_id}")
                assigned[case_id] = split
                source = by_id[case_id]["case"].get("source_id")
                if source is not None:
                    if source in sources and sources[source] != split:
                        raise ValueError(f"Source leakage across splits: {source}")
                    sources[source] = split
        if split_map and set(assigned) != ids:
            raise ValueError("Splits must assign every case exactly once")
        body = {"schema": DATASET_SCHEMA, "name": name,
                "cases": sorted(rows, key=lambda row: row["id"]),
                "splits": {k: sorted(v) for k, v in split_map.items()}}
        self._snapshot = canonical_json({**body, "digest": digest(body)})

    @property
    def manifest(self) -> dict:
        return json.loads(self._snapshot)

    @property
    def digest(self) -> str:
        return self.manifest["digest"]

    @property
    def cases(self) -> list[EvalCase]:
        return [case_from_dict(r["case"]) for r in self.manifest["cases"]]

    def split(self, name: str) -> list[EvalCase]:
        manifest = self.manifest
        ids = set(manifest["splits"][name])
        return [case_from_dict(r["case"]) for r in manifest["cases"] if r["id"] in ids]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self._snapshot + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> Dataset:
        if data.get("schema") != DATASET_SCHEMA:
            raise ValueError("Unsupported dataset schema")
        result = cls(data["name"], [case_from_dict(r["case"]) for r in data["cases"]],
                     splits=data["splits"])
        if canonical_json(result.manifest) != canonical_json(data):
            raise ValueError("Dataset manifest digest or contents do not match")
        return result

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
