import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "industrial"))

from tatqa_financial import (
    annotation_locations,
    location_scores,
    locations_are_valid,
    output_schema,
    parse_response,
    prediction_locations,
    prompt_for,
)


@pytest.fixture
def context():
    return {
        "context_id": "context-1",
        "table": [["", "2025"], ["Revenue", "12"]],
        "paragraphs": [{"order": 1, "text": "Revenue increased."}],
        "questions": [
            {"id": "q1", "question": "What was revenue?"},
            {"id": "q2", "question": "What changed?"},
        ],
    }


def test_evidence_schema_and_prompt_contain_inputs_but_no_gold_fields(context):
    schema = output_schema(context, "evidence_record")
    item = schema["properties"]["responses"]["items"]
    assert schema["properties"]["responses"]["minItems"] == 1
    assert "maxItems" not in schema["properties"]["responses"]
    assert set(item["required"]) == {
        "id",
        "answer",
        "scale",
        "table_cells",
        "paragraphs",
        "calculation",
    }
    prompt = prompt_for(context, "evidence_record")
    assert "Revenue increased" in prompt
    assert "What was revenue?" in prompt
    assert '"answer"' not in prompt


def test_parse_response_orders_ids_and_rejects_missing_or_duplicate_ids(context):
    value = {
        "responses": [
            {"id": "q2", "answer": "increased", "scale": ""},
            {"id": "q1", "answer": "12", "scale": "million"},
        ]
    }
    parsed = parse_response(json.dumps(value), context, "answer_only")
    assert [row["id"] for row in parsed] == ["q1", "q2"]

    value["responses"][1]["id"] = "q2"
    with pytest.raises(ValueError, match="exactly once"):
        parse_response(json.dumps(value), context, "answer_only")


def test_locations_validate_bounds_and_score_annotation_agreement(context):
    response = {
        "table_cells": [{"row": 1, "column": 1}],
        "paragraphs": [1],
    }
    question = {
        "mappings": [
            {"table": [1, 1]},
            {"paragraph_1": [0, 7]},
        ]
    }
    predicted = prediction_locations(response)
    annotated = annotation_locations(question)
    assert predicted == {("table", 1, 1), ("paragraph", 1)}
    assert predicted == annotated
    assert locations_are_valid(response, context)
    assert location_scores(predicted, annotated)["annotation_location_f1"] == 1

    response["table_cells"] = [{"row": 9, "column": 1}]
    assert not locations_are_valid(response, context)


def test_annotation_location_scores_do_not_call_agreement_semantic_validity():
    scores = location_scores({("paragraph", 2)}, {("paragraph", 1)})
    assert scores["annotation_location_precision"] == 0
    assert scores["annotation_location_recall"] == 0
    assert scores["annotation_location_f1"] == 0
