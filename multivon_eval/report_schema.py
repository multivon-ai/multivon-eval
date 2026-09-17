"""Portable report envelope validation using JSON Schema 2020-12.

Nested evidence validates its own schema and digests when EvalReport loads it.
This schema checks structure, not the authenticity or validity of measurements.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

from jsonschema import Draft202012Validator


def report_schema() -> dict:
    """Return a detached JSON Schema for the supported v1/v2 report envelope."""
    return json.loads(files("multivon_eval").joinpath("schemas/report.json").read_text("utf-8"))


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(report_schema())


def validate_report(data: dict) -> None:
    """Reject malformed envelopes; allow additive fields within known versions.

    Error messages omit report values because outputs can contain private data.
    This does not replace EvalReport.from_dict's nested evidence validation.
    """
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Report must contain finite, JSON-serializable values") from None
    error = next(_validator().iter_errors(data), None)
    if error is not None:
        # Use only numeric offsets and schema-owned property names in diagnostics.
        path = "/".join(str(part) for part in error.absolute_schema_path)
        raise ValueError(f"Invalid report structure (schema rule: {path})")
