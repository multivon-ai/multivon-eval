"""Inspect pinned upstream TAT-QA files; no generation, relabeling or inference.

python benchmarks/industrial/audit_public_financial_data.py --cache /tmp/tatqa --download
Without --download, only the supplied local cache is read. Dataset license:
CC BY 4.0 per the upstream README; retain attribution when using the data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

REVISION = "870accc41953dcde885aabeb963d94aabdc0fbc3"
REPOSITORY = "https://github.com/NExTplusplus/TAT-QA"
FILES = (
    "tatqa_dataset_train.json", "tatqa_dataset_dev.json",
    "tatqa_dataset_test.json", "tatqa_dataset_test_gold.json",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def inspect_files(cache: Path) -> dict:
    summaries, context_ids, question_ids, inputs, contents = {}, {}, {}, {}, {}
    for name in FILES:
        raw = (cache / name).read_bytes()
        rows = json.loads(raw)
        questions = [q for row in rows for q in row["questions"]]
        context_ids[name] = {row["table"]["uid"] for row in rows}
        question_ids[name] = {q["uid"] for q in questions}
        inputs[name] = {
            q["uid"]: digest({"table": row["table"], "paragraphs": row["paragraphs"],
                               "question": q["question"]})
            for row in rows for q in row["questions"]
        }
        contents[name] = Counter(
            digest({"table": row["table"]["table"],
                    "paragraphs": [(p["order"], p["text"]) for p in row["paragraphs"]],
                    "question": q["question"]})
            for row in rows for q in row["questions"]
        )
        summaries[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "contexts": len(rows), "questions": len(questions),
            "context_fields": sorted({k for row in rows for k in row}),
            "table_fields": sorted({k for row in rows for k in row["table"]}),
            "question_fields": sorted({k for q in questions for k in q}),
            "answer_types": dict(Counter(q.get("answer_type", "unlabeled") for q in questions)),
            "missing_answers": sum("answer" not in q for q in questions),
            "unique_context_ids": len(context_ids[name]),
            "unique_question_ids": len(question_ids[name]),
        }
    splits = FILES[:3]
    original, gold = FILES[2:]
    common = question_ids[original] & question_ids[gold]
    return {
        "repository": REPOSITORY, "revision": REVISION,
        "files": summaries,
        "context_id_overlap": {a + " / " + b: len(context_ids[a] & context_ids[b])
                               for i, a in enumerate(splits) for b in splits[i + 1:]},
        "test_gold_coverage": {
            "common_question_ids": len(common),
            "original_ids_not_in_gold": len(question_ids[original] - question_ids[gold]),
            "gold_ids_not_in_original": len(question_ids[gold] - question_ids[original]),
            "changed_input_question_ids": sorted(q for q in common
                                                 if inputs[original][q] != inputs[gold][q]),
            "exact_content_overlap_ignoring_ids": len(contents[original].keys() & contents[gold].keys()),
            "original_contents_not_in_gold": len(contents[original].keys() - contents[gold].keys()),
            "gold_contents_not_in_original": len(contents[gold].keys() - contents[original].keys()),
            "ambiguous_original_contents": sum(n > 1 for n in contents[original].values()),
            "ambiguous_gold_contents": sum(n > 1 for n in contents[gold].values()),
            "content_matching_rule": "Exact table cells, ordered (paragraph order, text) pairs and question; identifiers excluded. No semantic normalization.",
        },
        "limitations": [
            "Context IDs do not establish source-report or company-disjoint splits.",
            "Presence of gold fields is not validation of their correctness.",
            "No model inference or benchmark scoring performed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.download:
        for name in FILES:
            url = f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{REVISION}/dataset_raw/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
            json.loads(raw)
            (args.cache / name).write_bytes(raw)
    result = inspect_files(args.cache)
    output = args.out or args.cache / "tatqa-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
