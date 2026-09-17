"""Reproduce the pinned RAGChecker scorer without modifying it or calling a model.

Requires numpy, scipy, pandas, plotly, kaleido and a Kaleido-compatible Chrome.
The upstream checkout is read only; data are staged in a temporary directory.
Only aggregate results, hashes, logs and native plots are retained in --out.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

REVISION = "6091f08c00e676e87a970f2aeb4a23a484746348"
REPOSITORY = "https://github.com/amazon-science/RAGChecker"
DIRECTORY = "data/meta_evaluation"
METRICS = {
    "trulens": ["groundedness", "answer_relevance"],
    "ares": ["Answer Relevance Scores", "Answer Faithfulness Scores"],
    "ragas": ["faithfulness", "answer_correctness", "answer_similarity", "answer_relevancy"],
    "crud": ["bleu-avg", "rouge-L", "bertScore", "QA_avg_F1", "QA_recall"],
    "ragchecker": ["correctness_label", "completeness_label", "overall_label"],
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_bytes(checkout: Path, name: str) -> bytes:
    """Read the Git object, rejecting local edits to any consumed source file."""
    relative = f"{DIRECTORY}/{name}"
    raw = subprocess.check_output(["git", "-C", str(checkout), "show", f"{REVISION}:{relative}"])
    if (checkout / relative).read_bytes() != raw:
        raise ValueError(f"Upstream working file differs from pinned revision: {name}")
    return raw


def audit(data: dict) -> dict:
    human = data["human"]
    if len(human) != 560:
        raise ValueError("Expected 560 annotation records")
    ids = [row["instance_id"] for row in human[::2]]
    questions = [(row["dataset"], row["query_id"]) for row in human[::2]]
    if len(set(ids)) != 280 or len(set(questions)) != 280:
        raise ValueError("Expected 280 distinct cases and question keys")
    header = ("instance_id", "dataset", "query_id", "query", "gt_answer")
    missing = {}
    for baseline, metrics in METRICS.items():
        rows = data[baseline]
        if len(rows) != 280:
            raise ValueError(f"Unexpected case count: {baseline}")
        for i, row in enumerate(rows):
            for annotation in human[2 * i:2 * i + 2]:
                aligned = all(row[key] == annotation[key] for key in header)
                aligned &= all(
                    row[model][key] == annotation[model][key]
                    for model in ("model1", "model2") for key in ("model_name", "response")
                )
                if not aligned:
                    raise ValueError(f"Mismatched baseline/human inputs: {baseline}, row {i}")
        missing[baseline] = {}
        for metric in metrics:
            scores = np.array([[r[model][baseline][metric] for model in ("model1", "model2")]
                               for r in rows], dtype=float)
            if np.isinf(scores).any():
                raise ValueError(f"Infinite baseline score: {baseline}/{metric}")
            delta = np.array([r["model2"][baseline][metric] - r["model1"][baseline][metric]
                              for r in rows], dtype=float)
            if np.isinf(delta).any() or np.isnan(delta).all():
                raise ValueError(f"Unsupported nonfinite population: {baseline}/{metric}")
            missing[baseline][metric] = {
                "case_count": len(delta), "nan_deltas": int(np.isnan(delta).sum()),
                "imputation_median": float(np.nanmedian(delta)),
            }
    agreement = {}
    for metric in METRICS["ragchecker"]:
        labels = np.array([r[metric] for r in human])
        if not np.isin(labels, [-2, -1, 0, 1, 2]).all():
            raise ValueError("Unexpected human preference label")
        agreement[metric] = {
            "case_count": 280,
            "exact_agreements": int((labels[::2] == labels[1::2]).sum()),
            "within_one_agreements": int((abs(labels[::2] - labels[1::2]) <= 1).sum()),
        }
    return {
        "cases": 280, "annotation_records": 560, "unique_question_keys": 280,
        "cases_by_domain": dict(Counter(r["dataset"] for r in human[::2])),
        "baseline_input_alignment_conflicts": 0,
        "missing_delta_audit": missing, "human_agreement": agreement,
        "interpretation": "Two annotations per case are not independent cases. Source-document independence is unverified.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkout, out = args.upstream.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError("Use a new output directory to preserve previous evidence")
    files = {"human": "human_labeled_data.json"}
    files.update({b: f"baseline_{b}{'' if b == 'ragchecker' else '_llama3'}.json" for b in METRICS})
    raw_files = {name: source_bytes(checkout, name) for name in ["meta_eval.py", *files.values()]}
    audit_result = audit({key: json.loads(raw_files[name]) for key, name in files.items()})
    out.mkdir(parents=True)
    manifest = {
        "repository": REPOSITORY, "revision": REVISION,
        "runner_sha256": sha256(Path(__file__).read_bytes()),
        "sources": {name: {"sha256": sha256(raw), "bytes": len(raw)} for name, raw in raw_files.items()},
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "plotly", "kaleido")},
        "protocol": {
            "scorer": "Unchanged upstream meta_eval.py; default use_llama3=True",
            "missing_values": "Official per-metric median imputation before normalization",
            "correlation_scale": "Pearson and Spearman multiplied by 100, rounded to 2 decimals",
            "population": "280 cases; each prediction duplicated for two human annotations",
            "inference": "None: released predictions rescored; not fresh baseline model execution",
            "network": "No model clients used; browser renderer not network-sandboxed",
        },
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "audit.json", audit_result)
    (out / "requirements-freeze.txt").write_bytes(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    )
    with tempfile.TemporaryDirectory(prefix="ragchecker-native-") as temporary:
        stage = Path(temporary)
        for name, raw in raw_files.items():
            (stage / name).write_bytes(raw)
        with (out / "native-scorer.log").open("wb") as log:
            completed = subprocess.run([sys.executable, "meta_eval.py"], cwd=stage,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False)
        write_json(out / "execution.json", {"returncode": completed.returncode})
        completed.check_returncode()
        for name in ("meta_eval_results.json", "human_correctness.png",
                     "human_completeness.png", "human_overall.png"):
            shutil.copyfile(stage / name, out / name)
    write_json(out / "artifact-hashes.json", {
        p.name: {"sha256": sha256(p.read_bytes()), "bytes": p.stat().st_size}
        for p in sorted(out.iterdir()) if p.is_file()
    })
    print(out)


if __name__ == "__main__":
    main()
