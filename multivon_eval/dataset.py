from __future__ import annotations
import csv
import json
from pathlib import Path
from .case import EvalCase


def load_jsonl(path: str) -> list[EvalCase]:
    """Load test cases from a JSONL file. Each line is a JSON object."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(EvalCase(
                input=data["input"],
                expected_output=data.get("expected_output"),
                context=data.get("context"),
                metadata=data.get("metadata", {}),
                tags=data.get("tags", []),
            ))
    return cases


def load_csv(path: str) -> list[EvalCase]:
    """Load test cases from a CSV file with columns: input, expected_output, context, tags."""
    cases = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags = [t.strip() for t in row.get("tags", "").split(",") if t.strip()]
            cases.append(EvalCase(
                input=row["input"],
                expected_output=row.get("expected_output") or None,
                context=row.get("context") or None,
                tags=tags,
            ))
    return cases


def load(path: str) -> list[EvalCase]:
    """Auto-detect format from file extension and load cases."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return load_jsonl(path)
    elif p.suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported format: {p.suffix}. Use .jsonl or .csv")
