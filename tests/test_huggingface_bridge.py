"""Use the real optional Hugging Face implementation, including Arrow/Parquet."""
import json

import pytest

datasets = pytest.importorskip("datasets")

from multivon_eval import CaseManifest, EvalCase
from multivon_eval.integrations.huggingface import from_huggingface


def mapping(row):
    return EvalCase(row["question"], row["answer"], case_id=row["id"], source_id=row["source"])


def records():
    return [
        {"id": "a", "question": "Amount?", "answer": "42", "source": "invoice-a"},
        {"id": "b", "question": "Amount?", "answer": "84", "source": "invoice-b"},
    ]


def test_parquet_huggingface_manifest_round_trip(tmp_path):
    upstream = datasets.Dataset.from_list(records())
    path = tmp_path / "cases.parquet"
    upstream.to_parquet(str(path))
    restored = datasets.Dataset.from_parquet(str(path))
    partitions = datasets.DatasetDict({"dev": restored.select([0]), "test": restored.select([1])})
    manifest = from_huggingface(partitions, name="invoices", record_to_case=mapping,
                               provenance={"kind": "local fixture", "file": "cases.parquet"})
    assert manifest.manifest["splits"] == {"dev": ["a"], "test": ["b"]}
    assert manifest.split("test")[0].expected_output == "84"
    assert manifest.manifest["provenance"]["upstream_fingerprints"]["dev"] == partitions["dev"]._fingerprint
    saved = json.loads(json.dumps(manifest.manifest))
    assert CaseManifest.from_dict(saved).digest == manifest.digest
    partitions["test"] = partitions["test"].map(lambda r: {"answer": "changed"})
    assert from_huggingface(partitions, name="invoices", record_to_case=mapping).digest != manifest.digest


def test_import_rejects_cross_split_source_leakage():
    a, b = records()
    b["source"] = a["source"]
    upstream = datasets.DatasetDict({"dev": datasets.Dataset.from_list([a]),
                                     "test": datasets.Dataset.from_list([b])})
    with pytest.raises(ValueError, match="Source leakage"):
        from_huggingface(upstream, name="leak", record_to_case=mapping)


def test_streaming_is_bounded_and_never_silently_truncated():
    upstream = datasets.IterableDataset.from_generator(lambda: iter(records()))
    with pytest.raises(ValueError, match="exceeds max_cases"):
        from_huggingface(upstream, name="stream", record_to_case=mapping, max_cases=1)
    manifest = from_huggingface(upstream, name="stream", record_to_case=mapping, max_cases=2)
    assert len(manifest.cases) == 2
    assert manifest.manifest["provenance"]["upstream_fingerprints"] == {str(upstream.split): None}


def test_native_case_rows_require_no_mapper():
    upstream = datasets.Dataset.from_list([{"input": "x", "expected_output": "y", "case_id": "one"}])
    assert from_huggingface(upstream, name="native").cases[0].expected_output == "y"


def test_bad_mapper_and_non_dataset_rejected():
    with pytest.raises(TypeError, match="Expected a Hugging Face"):
        from_huggingface(records(), name="wrong")
    with pytest.raises(TypeError, match="return EvalCase"):
        from_huggingface(datasets.Dataset.from_list(records()), name="wrong", record_to_case=lambda row: row)


def test_documented_huggingface_example(tmp_path, monkeypatch):
    import re
    from pathlib import Path
    source = (Path(__file__).parents[1] / "docs/guides/versioned-evidence.mdx").read_text()
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", source, re.S):
        exec(compile(block, "versioned-evidence.mdx", "exec"), namespace)
    assert namespace["manifest"].split("test")[0].source_id == "invoice-1"
