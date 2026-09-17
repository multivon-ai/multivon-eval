"""Score the completed frozen experiment; never invokes a model."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from reproduce_ragchecker_meta import source_bytes, write_json
from scipy import stats

METHODS = ("answer_accuracy", "direct_rating")
DIMENSIONS = ("overall_label", "correctness_label", "completeness_label")


def native_correlation(upstream):
    # Execute only the reviewed, pinned upstream function, not its plotting main.
    tree = ast.parse(source_bytes(upstream, "meta_eval.py"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "correlation")
    namespace = {"stats": stats}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned_meta_eval.py", "exec"), namespace)  # noqa: S102 — pinned upstream function only
    return namespace["correlation"]


def correlations(x, labels, scorer):
    if np.ptp(x) == 0 or np.ptp(labels) == 0:
        return {"pearson_x100": None, "spearman_x100": None}
    p, s = scorer(np.repeat(x, 2), labels.ravel())
    return {"pearson_x100": float(p), "spearman_x100": float(s)}


def bootstrap(deltas, labels, domains):
    rng = np.random.default_rng(17092026)
    groups = [np.flatnonzero(domains == domain) for domain in sorted(set(domains))]
    draws = {m: [] for m in METHODS}
    differences = []
    for _ in range(2000):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        y = labels[indices].ravel()
        values = {}
        for method in METHODS:
            x = np.repeat(deltas[method][indices], 2)
            values[method] = float(stats.pearsonr(x, y).statistic) if np.ptp(x) and np.ptp(y) else np.nan
            draws[method].append(values[method])
        differences.append(values["answer_accuracy"] - values["direct_rating"])

    def interval(values):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        return {"ci95": np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None,
                "defined_resamples": len(finite), "total_resamples": len(values)}

    return {**{m: interval(draws[m]) for m in METHODS},
            "answer_accuracy_minus_direct": interval(differences),
            "unit": "Case, stratified by domain; two annotations retained together",
            "scale": "Ordinary Pearson correlation; conditional on one set of model outputs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    execution = json.loads((args.run / "execution.json").read_text())
    if execution["jobs_completed"] != 1120:
        raise ValueError("Experiment is incomplete")
    protocol = json.loads((args.run / "protocol.json").read_text())
    if hashlib.sha256((args.run / "inputs.json").read_bytes()).hexdigest() != protocol["inputs_sha256"]:
        raise ValueError("Input hash mismatch")
    inputs = json.loads((args.run / "inputs.json").read_text())
    human = json.loads(source_bytes(args.upstream, "human_labeled_data.json"))
    baseline = json.loads(source_bytes(args.upstream, "baseline_ragchecker.json"))
    for i, row in enumerate(inputs):
        for annotation in human[2 * i:2 * i + 2]:
            if (row["id"] != annotation["instance_id"] or row["query"] != annotation["query"]
                    or row["reference"] != annotation["gt_answer"]
                    or row["responses"] != [annotation["model1"]["response"], annotation["model2"]["response"]]):
                raise ValueError("Human/judge input alignment mismatch")
    predictions = [json.loads(line) for line in (args.run / "predictions.jsonl").read_text().splitlines()]
    keyed = {(p["id"], p["side"], p["method"]): p for p in predictions}
    expected = {(r["id"], side, m) for r in inputs for side in (0, 1) for m in METHODS}
    if len(predictions) != 1120 or len(keyed) != 1120 or set(keyed) != expected:
        raise ValueError("Missing, duplicate or unexpected predictions")
    scorer = native_correlation(args.upstream)
    labels = {d: np.array([r[d] for r in human]).reshape(280, 2) for d in DIMENSIONS}
    domains = np.array([r["dataset"] for r in inputs])
    deltas, coverage = {}, {}
    numeric_rows = []
    for method in METHODS:
        scores = []
        for row in inputs:
            pair = [keyed[row["id"], side, method] for side in (0, 1)]
            for p in pair:
                value = p["score"]
                if value is not None and (type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError("Invalid score")
                numeric_rows.append({k: p[k] for k in ("id", "dataset", "side", "method", "score", "status", "seconds")}
                                      | {"partial_verdict_coverage": p.get("partial_verdict_coverage", False)})
            scores.append([np.nan if p["score"] is None else p["score"] for p in pair])
        scores = np.asarray(scores)
        delta = scores[:, 1] - scores[:, 0]
        if np.isnan(delta).all():
            raise ValueError("No complete pairs: correlation unavailable")
        median = float(np.nanmedian(delta))
        population = [p for p in predictions if p["method"] == method]
        coverage[method] = {
            "responses": 560, "scored_responses": int(np.isfinite(scores).sum()),
            "complete_pairs": int(np.isfinite(delta).sum()), "imputed_deltas": int(np.isnan(delta).sum()),
            "imputation_median": median, "status_counts": dict(Counter(p["status"] for p in population)),
            "partial_verdict_responses": sum(p.get("partial_verdict_coverage", False) for p in population),
            "responses_with_capture_gaps": sum(bool(p["coverage_gaps"] or p["requests_without_response"]) for p in population),
            "response_latency_seconds": {
                "median": float(np.median([p["seconds"] for p in population])),
                "p95": float(np.percentile([p["seconds"] for p in population], 95)),
            },
        }
        delta[np.isnan(delta)] = median
        # Native normalization is a positive affine transform: correlations are invariant.
        if np.ptp(delta):
            delta = ((delta - delta.min()) / np.ptp(delta) - .5) * 4
        deltas[method] = delta
    result = {
        "cases": 280, "annotation_records": 560, "primary": "overall_label Pearson",
        "coverage": coverage,
        "correlations": {d: {m: correlations(x, labels[d], scorer) for m, x in deltas.items()}
                         for d in DIMENSIONS},
        "historical_ragchecker": {
            d: correlations(np.array([r["model2"]["ragchecker"][d] - r["model1"]["ragchecker"][d]
                                      for r in baseline]), labels[d], scorer) for d in DIMENSIONS},
        "primary_paired_bootstrap": bootstrap(deltas, labels["overall_label"], domains),
        "per_domain_overall": {
            domain: {"cases": int(sum(domains == domain)),
                     **{m: correlations(x[domains == domain], labels["overall_label"][domains == domain], scorer)
                        for m, x in deltas.items()}}
            for domain in sorted(set(domains))},
        "run_protocol": protocol,
        "source_hashes": {name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
                          for name in ("protocol.json", "inputs.json", "predictions.jsonl", "events.sqlite", "execution.json")},
    }
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "results.json", result)
    write_json(args.out / "predictions.json", numeric_rows)
    print(json.dumps({"correlations": result["correlations"], "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
