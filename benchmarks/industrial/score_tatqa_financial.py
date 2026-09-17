"""Score a completed TAT-QA financial study without invoking a model."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tatqa_financial import (
    GOLD_RELATIVE_PATH,
    GOLD_SHA256,
    TREATMENTS,
    UPSTREAM_REVISION,
    annotation_locations,
    location_scores,
    locations_are_valid,
    prediction_locations,
    sha256,
    write_json,
)

SEED = 17092026
BOOTSTRAP_RESAMPLES = 5000


def load_official_metric(upstream: Path):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected upstream {UPSTREAM_REVISION}, got {revision}")
    if sha256(upstream / GOLD_RELATIVE_PATH) != GOLD_SHA256:
        raise ValueError("Pinned TAT-QA gold hash mismatch")
    sys.path.insert(0, str(upstream))
    try:
        module = importlib.import_module("tatqa_metric")
    finally:
        sys.path.pop(0)
    return module.TaTQAEmAndF1


def load_run(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = json.loads((run / "protocol.json").read_text())
    execution = json.loads((run / "execution.json").read_text())
    if execution.get("jobs_completed") != 554:
        raise ValueError("Study is incomplete")
    rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    keyed = {(row["context_id"], row["treatment"]): row for row in rows}
    if len(rows) != 554 or len(keyed) != 554:
        raise ValueError("Missing or duplicate context/treatment rows")
    return protocol, rows


def official_scores(
    metric_class: Any,
    gold: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metric = metric_class()
    for source in gold:
        for question in source["questions"]:
            prediction = predictions.get(question["uid"])
            answer = prediction["answer"] if prediction else None
            scale = prediction["scale"] if prediction else ""
            metric(ground_truth=question, prediction=answer, pred_scale=scale)
    exact_match, f1, scale, _ = metric.get_overall_metric()
    raw = metric.get_raw()
    by_id = {
        row["uid"]: {
            "em": float(row["em"]),
            "f1": float(row["f1"]),
            "scale_match": bool(row["pred_scale"] == row["scale"] and row["pred"]),
            "answer_type": row["answer_type"],
            "answer_from": row["answer_from"],
        }
        for row in raw
    }
    return {
        "exact_match": exact_match,
        "f1": f1,
        "scale_accuracy": scale,
        "questions": len(raw),
    }, by_id


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_context_bootstrap(
    question_rows: list[dict[str, Any]],
    context_questions: dict[str, list[str]],
) -> dict[str, Any]:
    keyed = {(row["question_id"], row["treatment"]): row for row in question_rows}
    contexts = sorted(context_questions)
    rng = random.Random(SEED)
    distributions = {"em_difference": [], "f1_difference": []}
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = [rng.choice(contexts) for _ in contexts]
        question_ids = [question for context in sampled for question in context_questions[context]]
        for metric in ("em", "f1"):
            evidence = sum(keyed[q, "evidence_record"][metric] for q in question_ids) / len(question_ids)
            answer = sum(keyed[q, "answer_only"][metric] for q in question_ids) / len(question_ids)
            distributions[f"{metric}_difference"].append(evidence - answer)
    return {
        name: {
            "evidence_record_minus_answer_only": sum(values) / len(values),
            "ci95": [percentile(values, 0.025), percentile(values, 0.975)],
            "probability_greater_than_zero": sum(value > 0 for value in values) / len(values),
        }
        for name, values in distributions.items()
    } | {
        "resamples": BOOTSTRAP_RESAMPLES,
        "unit": "TAT-QA context; all questions within a sampled context retained",
        "seed": SEED,
    }


def exact_discordance(question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_question: dict[str, dict[str, int]] = defaultdict(dict)
    for row in question_rows:
        by_question[row["question_id"]][row["treatment"]] = int(row["em"] == 1)
    evidence_wins = sum(
        values["evidence_record"] == 1 and values["answer_only"] == 0
        for values in by_question.values()
    )
    answer_wins = sum(
        values["answer_only"] == 1 and values["evidence_record"] == 0
        for values in by_question.values()
    )
    discordant = evidence_wins + answer_wins
    if discordant:
        from scipy.stats import binomtest

        p_value = float(binomtest(min(evidence_wins, answer_wins), discordant, 0.5).pvalue)
    else:
        p_value = 1.0
    return {
        "evidence_record_only_correct": evidence_wins,
        "answer_only_only_correct": answer_wins,
        "discordant_questions": discordant,
        "two_sided_exact_mcnemar_p": p_value,
        "note": "Question-level test is secondary; context bootstrap is the primary uncertainty analysis",
    }


def usage_summary(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_treatment = {}
    pricing = protocol["pricing"]
    for treatment in TREATMENTS:
        selected = [row for row in rows if row["treatment"] == treatment]
        totals = Counter()
        for row in selected:
            for key, value in (row.get("usage") or {}).items():
                if isinstance(value, int):
                    totals[key] += value
        input_tokens = totals["input_tokens"]
        output_tokens = totals["output_tokens"]
        estimate = (
            input_tokens * pricing["input_per_million_usd"]
            + output_tokens * pricing["output_per_million_usd"]
        ) / 1_000_000
        by_treatment[treatment] = {
            "logical_requests": len(selected),
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "usage": dict(totals),
            "estimated_usd": estimate,
            "cost_is_invoice": False,
            "latency_seconds": {
                "median": percentile([row["seconds"] for row in selected], 0.5),
                "p95": percentile([row["seconds"] for row in selected], 0.95),
            },
            "responses_with_capture_gaps": sum(
                bool(row.get("coverage_gaps") or row.get("requests_without_response"))
                for row in selected
            ),
        }
    return {
        "by_treatment": by_treatment,
        "total_estimated_usd": sum(row["estimated_usd"] for row in by_treatment.values()),
        "pricing": pricing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol, run_rows = load_run(args.run)
    manifest = json.loads(args.manifest.read_text())
    if protocol["inputs_sha256"] != sha256(args.inputs) or manifest["inputs_sha256"] != sha256(args.inputs):
        raise ValueError("Scored input projection does not match the run")
    inputs = json.loads(args.inputs.read_text())
    gold_path = args.upstream / GOLD_RELATIVE_PATH
    gold = json.loads(gold_path.read_text())
    if [context["context_id"] for context in inputs] != [source["table"]["uid"] for source in gold]:
        raise ValueError("Input/gold context alignment mismatch")
    metric_class = load_official_metric(args.upstream)
    result_by_job = {(row["context_id"], row["treatment"]): row for row in run_rows}
    gold_questions = {
        question["uid"]: question for source in gold for question in source["questions"]
    }
    context_questions = {
        context["context_id"]: [question["id"] for question in context["questions"]]
        for context in inputs
    }
    predictions: dict[str, dict[str, dict[str, Any]]] = {treatment: {} for treatment in TREATMENTS}
    for context in inputs:
        for treatment in TREATMENTS:
            job = result_by_job[context["context_id"], treatment]
            if job["status"] == "scored":
                predictions[treatment].update({response["id"]: response for response in job["responses"]})
    official, per_question = {}, {}
    for treatment in TREATMENTS:
        official[treatment], per_question[treatment] = official_scores(
            metric_class, gold, predictions[treatment]
        )
    numeric_rows = []
    for context in inputs:
        for question in context["questions"]:
            question_id = question["id"]
            for treatment in TREATMENTS:
                job = result_by_job[context["context_id"], treatment]
                base = {
                    "context_id": context["context_id"],
                    "question_id": question_id,
                    "treatment": treatment,
                    "status": job["status"],
                    **per_question[treatment][question_id],
                }
                if treatment == "evidence_record" and question_id in predictions[treatment]:
                    response = predictions[treatment][question_id]
                    scores = location_scores(
                        prediction_locations(response), annotation_locations(gold_questions[question_id])
                    )
                    valid = locations_are_valid(response, context)
                    strict_accept = bool(
                        base["em"] == 1
                        and valid
                        and scores["annotation_location_recall"] == 1
                        and scores["predicted_locations"] > 0
                    )
                    base.update(**scores, locations_valid=valid, strict_workflow_accept=strict_accept)
                numeric_rows.append(base)
    evidence_rows = [row for row in numeric_rows if row["treatment"] == "evidence_record"]
    slices = {}
    for treatment in TREATMENTS:
        selected = [row for row in numeric_rows if row["treatment"] == treatment]
        slices[treatment] = {
            answer_type: {
                "questions": len(group),
                "exact_match": sum(row["em"] for row in group) / len(group),
                "f1": sum(row["f1"] for row in group) / len(group),
            }
            for answer_type in sorted({row["answer_type"] for row in selected})
            for group in [[row for row in selected if row["answer_type"] == answer_type]]
        }
    evidence = {
        "questions": len(evidence_rows),
        "valid_location_rate": sum(row.get("locations_valid", False) for row in evidence_rows) / len(evidence_rows),
        "mean_annotation_location_precision": sum(row.get("annotation_location_precision", 0) for row in evidence_rows) / len(evidence_rows),
        "mean_annotation_location_recall": sum(row.get("annotation_location_recall", 0) for row in evidence_rows) / len(evidence_rows),
        "mean_annotation_location_f1": sum(row.get("annotation_location_f1", 0) for row in evidence_rows) / len(evidence_rows),
        "strict_workflow_acceptance": sum(row.get("strict_workflow_accept", False) for row in evidence_rows) / len(evidence_rows),
        "limitation": "Location agreement is against one released annotation and does not establish semantic support or reject valid alternate evidence",
    }
    source_hashes = {
        "protocol.json": sha256(args.run / "protocol.json"),
        "predictions.jsonl": sha256(args.run / "predictions.jsonl"),
        "events.sqlite": sha256(args.run / "events.sqlite"),
        "execution.json": sha256(args.run / "execution.json"),
        "inputs.json": sha256(args.inputs),
        "manifest.json": sha256(args.manifest),
        "tatqa_eval.py": sha256(args.upstream / "tatqa_eval.py"),
        "tatqa_metric.py": sha256(args.upstream / "tatqa_metric.py"),
        "tatqa_utils.py": sha256(args.upstream / "tatqa_utils.py"),
    }
    results = {
        "study": protocol["study"],
        "population": {"contexts": 277, "questions": 1663, "treatments": 2},
        "official_tatqa": official,
        "paired_context_bootstrap": paired_context_bootstrap(numeric_rows, context_questions),
        "exact_match_discordance": exact_discordance(numeric_rows),
        "answer_type_slices": slices,
        "evidence_record": evidence,
        "provider": usage_summary(run_rows, protocol),
        "coverage": {
            treatment: {
                "contexts_scored": sum(
                    row["status"] == "scored" for row in run_rows if row["treatment"] == treatment
                ),
                "contexts_total": 277,
                "questions_with_predictions": len(predictions[treatment]),
                "questions_total": 1663,
            }
            for treatment in TREATMENTS
        },
        "protocol": protocol,
        "source_hashes": source_hashes,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "results.json", results)
    (args.out / "question-metrics.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in numeric_rows)
    )
    write_json(
        args.out / "artifact-hashes.json",
        {
            "source": source_hashes,
            "results.json": sha256(args.out / "results.json"),
            "question-metrics.jsonl": sha256(args.out / "question-metrics.jsonl"),
        },
    )
    print(json.dumps({"official_tatqa": official, "evidence_record": evidence}, indent=2))


if __name__ == "__main__":
    main()
