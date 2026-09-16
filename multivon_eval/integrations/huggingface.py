"""Consume Hugging Face datasets without implementing a second dataset engine.

Loading, revisions, streaming, transforms, media decoding and storage stay in
the datasets library. This bridge validates a bounded evaluation selection.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..case import EvalCase
from ..case_manifest import CaseManifest, case_from_dict


def from_huggingface(
    dataset: Any,
    *,
    name: str,
    record_to_case: Callable[[dict], EvalCase] | None = None,
    provenance: dict | None = None,
    max_cases: int = 10_000,
) -> CaseManifest:
    """Convert an already selected Dataset/IterableDataset or split dictionary.

    Iterating a remote stream can fetch data through the upstream library.
    Pass a pinned Hub revision to load_dataset before calling this function;
    record its repository/commit in provenance. Upstream
    cache fingerprints are retained as provenance, not treated as a substitute
    for case content digests or stable semantic IDs. Infinite/oversized streams
    raise at max_cases + 1 rather than silently taking an incomplete sample.
    """
    try:
        from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, __version__
    except ImportError as exc:
        raise ImportError("Install multivon-eval[datasets] for the Hugging Face bridge") from exc
    if type(max_cases) is not int or max_cases < 1:
        raise ValueError("max_cases must be a positive integer")
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        partitions = dict(dataset.items())
    elif isinstance(dataset, (Dataset, IterableDataset)):
        partitions = {str(dataset.split or "evaluation"): dataset}
    else:
        raise TypeError("Expected a Hugging Face Dataset, IterableDataset, or split dictionary")
    rows, splits, fingerprints = [], {}, {}
    mapper = record_to_case or case_from_dict
    for split, upstream in partitions.items():
        if not isinstance(upstream, (Dataset, IterableDataset)):
            raise TypeError(f"Unsupported dataset in split {split}")
        fingerprints[split] = getattr(upstream, "_fingerprint", None)
        splits[split] = []
        for record in upstream:
            if len(rows) >= max_cases:
                raise ValueError(f"Selection exceeds max_cases={max_cases}; select a bounded evaluation set upstream")
            case = mapper(record)
            if not isinstance(case, EvalCase):
                raise TypeError("record_to_case must return EvalCase")
            rows.append(case)
            splits[split].append(case.identity()[0])
    return CaseManifest(name, rows, splits=splits, provenance={
        "source": dict(provenance or {}), "adapter": "huggingface",
        "datasets_version": __version__, "upstream_fingerprints": fingerprints,
    })
