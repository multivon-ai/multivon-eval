"""Post-hoc formatting diagnostic; does not replace the preregistered result.

Accept a leading standalone integer followed by whitespace/end, analogously to
QAG's leading-verdict parsing. Do not search explanations for arbitrary numbers.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
from reproduce_ragchecker_meta import source_bytes, write_json
from score_ragchecker_judges import bootstrap, correlations, native_correlation


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / "predictions.jsonl").read_text().splitlines()]
    if len(rows) != 1120:
        raise ValueError("Complete experiment required")
    recovered = 0
    for row in rows:
        if row["method"] == "direct_rating" and row["status"] == "unparseable":
            match = re.match(r"\A(100|[1-9]?[0-9])(?=\s|$)", row["reply"].strip())
            if match:
                row["score"] = int(match[1]) / 100
                recovered += 1
    keyed = {(p["id"], p["side"], p["method"]): p for p in rows}
    human = json.loads(source_bytes(args.upstream, "human_labeled_data.json"))
    case_rows = human[::2]
    labels = np.array([r["overall_label"] for r in human]).reshape(280, 2)
    domains = np.array([r["dataset"] for r in case_rows])
    deltas, missing = {}, {}
    for method in ("answer_accuracy", "direct_rating"):
        scores = np.array([[keyed[r["instance_id"], side, method]["score"] for side in (0, 1)]
                           for r in case_rows], dtype=float)
        delta = scores[:, 1] - scores[:, 0]
        missing[method] = {"unscored_responses": int(np.isnan(scores).sum()),
                           "imputed_pairs": int(np.isnan(delta).sum())}
        delta[np.isnan(delta)] = np.nanmedian(delta)
        deltas[method] = delta
    scorer = native_correlation(args.upstream)
    write_json(args.out, {
        "status": "Post-hoc diagnostic after observing format failures; not preregistered or a replacement result",
        "leading_integer_pattern": r"\A(100|[1-9]?[0-9])(?=\s|$)",
        "recovered_direct_responses": recovered, "missing": missing,
        "overall": {m: correlations(x, labels, scorer) for m, x in deltas.items()},
        "paired_bootstrap": bootstrap(deltas, labels, domains),
        "remaining_limit": "QAG and direct rating prompts and parsing still differ; no new inference",
    })
    print(args.out)


if __name__ == "__main__":
    main()
