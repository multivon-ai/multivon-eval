from __future__ import annotations
import csv
import json
from pathlib import Path
from .case import EvalCase
from .datasets import case_from_dict, case_to_dict, canonical_json


def load_jsonl(path: str) -> list[EvalCase]:
    """Load test cases from a JSONL file. Each line is a JSON object."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(case_from_dict(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return cases


def save_jsonl(cases: list[EvalCase], path: str) -> None:
    """Write portable cases, validating the full payload before opening the file."""
    payload = "".join(canonical_json(case_to_dict(c)) + "\n" for c in cases)
    Path(path).write_text(payload, encoding="utf-8")


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
