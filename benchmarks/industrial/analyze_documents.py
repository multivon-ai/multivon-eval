"""Descriptive source-paired analysis using SciPy's published interval/test APIs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

from multivon_eval import EvalReport


def analyze(paths: list[Path]) -> dict:
    runs, outcomes = [], []
    for path in paths:
        summary = json.loads((path / "summary.json").read_text())
        report = EvalReport.from_dict(json.loads((path / "report.json").read_text()))
        rows, results = defaultdict(list), {}
        failures = []
        for case in report.case_results:
            data = case.trials[0].data["case"]
            metadata = data["metadata"]
            family, modality = metadata["family"], metadata["modality"]
            check = next((r for r in case.results if r.evaluator == "ledger_outcome"), None)
            result = None if check is None else check.passed
            rows[(family, modality)].append((data["source_id"], result))
            results[case.case_id] = (family, modality, result)
            if result is not True:
                failures.append({"case_id": case.case_id, "source_id": data["source_id"],
                                 "observed": check.metadata if check else None,
                                 "model_error": case.model_error, "evaluator_error": case.evaluator_error})
        rates = []
        for (family, modality), values in sorted(rows.items()):
            if len({source for source, _ in values}) != len(values):
                raise ValueError("Correlated duplicates within a family/modality slice")
            measured = [result for _, result in values if result is not None]
            n, k = len(measured), sum(measured)
            ci = binomtest(k, n).proportion_ci(method="wilson") if n else None
            rates.append({"family": family, "modality": modality, "n": n, "passed": k,
                          "missing": len(values) - n, "wilson95": list(ci) if ci else None})
        runs.append({**summary, "rates": rates, "failures": failures})
        outcomes.append(results)
    paired = []
    if len(runs) == 2:
        if runs[0]["manifest_digest"] != runs[1]["manifest_digest"]:
            raise ValueError("Model comparison requires the identical prepared manifest")
        groups = defaultdict(list)
        for case_id in outcomes[0].keys() & outcomes[1].keys():
            family, modality, left = outcomes[0][case_id]
            right = outcomes[1][case_id][2]
            if left is not None and right is not None:
                groups[(family, modality)].append((left, right))
        for (family, modality), pairs in sorted(groups.items()):
            left_only = sum(a and not b for a, b in pairs)
            right_only = sum(b and not a for a, b in pairs)
            discordant = left_only + right_only
            p = binomtest(left_only, discordant, 0.5).pvalue if discordant else 1.0
            paired.append({"family": family, "modality": modality, "n": len(pairs),
                           "first_only": left_only, "second_only": right_only,
                           "exact_mcnemar_p_unadjusted": p,
                           "inference": "exploratory; no multiplicity adjustment or population ranking"})
    return {"runs": runs, "paired_by_source": paired,
            "uncertainty_scope": "CORD merchant independence unknown; synthetic intervals cover seed sampling within two fixed templates only"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.runs), indent=2))
