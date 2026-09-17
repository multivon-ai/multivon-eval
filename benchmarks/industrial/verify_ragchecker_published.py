"""Verify public numeric results against upstream labels, without API credentials.

This verifies score arithmetic, not the unpublished wire journal or billing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from reproduce_ragchecker_meta import source_bytes
from score_ragchecker_judges import DIMENSIONS, METHODS, bootstrap, correlations, native_correlation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.bundle / "results.json").read_text())
    predictions = json.loads((args.bundle / "predictions.json").read_text())
    human = json.loads(source_bytes(args.upstream, "human_labeled_data.json"))
    rows = human[::2]
    keyed = {(p["id"], p["side"], p["method"]): p for p in predictions}
    expected = {(r["instance_id"], side, method) for r in rows for side in (0, 1) for method in METHODS}
    if len(predictions) != 1120 or len(keyed) != 1120 or set(keyed) != expected:
        raise ValueError("Missing or duplicate numeric predictions")
    deltas = {}
    for method in METHODS:
        values = np.array([[keyed[r["instance_id"], side, method]["score"] for side in (0, 1)]
                           for r in rows], dtype=float)
        if np.isinf(values).any() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid numeric score")
        delta = values[:, 1] - values[:, 0]
        if int(np.isnan(delta).sum()) != result["coverage"][method]["imputed_deltas"]:
            raise ValueError("Pair coverage mismatch")
        if int(np.isfinite(values).sum()) != result["coverage"][method]["scored_responses"]:
            raise ValueError("Response coverage mismatch")
        delta[np.isnan(delta)] = np.nanmedian(delta)
        if np.ptp(delta):
            delta = ((delta - delta.min()) / np.ptp(delta) - .5) * 4
        deltas[method] = delta
    scorer = native_correlation(args.upstream)
    labels = {d: np.array([r[d] for r in human]).reshape(280, 2) for d in DIMENSIONS}
    actual = {d: {m: correlations(x, labels[d], scorer) for m, x in deltas.items()} for d in DIMENSIONS}
    if actual != result["correlations"]:
        raise ValueError("Published correlations differ from numeric predictions")
    domains = np.array([r["dataset"] for r in rows])
    if bootstrap(deltas, labels["overall_label"], domains) != result["primary_paired_bootstrap"]:
        raise ValueError("Published bootstrap differs from numeric predictions")
    print("Verified full-population correlations, coverage and paired bootstrap from public numeric predictions")


if __name__ == "__main__":
    main()
