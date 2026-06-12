"""Tests for span-grounded doc-QA + unanswerables (generate_from_text
upgrade). The LLM call (`generate._judge_call`) is mocked throughout —
same approach as test_generate_module.py.
"""
from __future__ import annotations

import hashlib
import json
from unittest.mock import patch

import pytest

from multivon_eval.generate import (
    _chunk_text_with_offsets,
    _locate_span,
    generate_from_file,
    generate_from_text,
)

TEXT = ("The Eiffel Tower is 330 metres tall. It was completed in 1889. "
        "Gustave Eiffel's company designed and built the structure.")

QA_RAW = json.dumps([
    {
        "question": "How tall is the Eiffel Tower?",
        "answer": "330 metres",
        "context_excerpt": "The Eiffel Tower is 330 metres tall.",
    },
    {
        "question": "Which company built the famous structure?",
        "answer": "Gustave Eiffel's company",
        "context_excerpt": "NOT A QUOTE FROM THE TEXT zzz",
    },
])

UNANSWERABLE_RAW = json.dumps([
    {"question": "How much did the tower cost to build?"},
    {"question": "Who was the mayor of Paris in 1889?"},
])


class TestSourceSpans:
    def test_span_offsets_anchor_into_source_text(self):
        with patch("multivon_eval.generate._judge_call", return_value=QA_RAW):
            cases, report = generate_from_text(TEXT, n=2, task="qa",
                                               return_report=True)
        assert report.kind == "doc_qa"
        located = cases[0].metadata["source_span"]
        quote = "The Eiffel Tower is 330 metres tall."
        assert located["start"] == TEXT.index(quote)
        assert located["end"] == located["start"] + len(quote)
        assert TEXT[located["start"]:located["end"]] == quote
        assert (located["sha256_chunk"]
                == hashlib.sha256(TEXT.encode("utf-8")).hexdigest())

    def test_unlocatable_quote_recorded_as_none_and_counted(self):
        with patch("multivon_eval.generate._judge_call", return_value=QA_RAW):
            cases, report = generate_from_text(TEXT, n=2, task="qa",
                                               return_report=True)
        assert cases[1].metadata["source_span"] is None  # honest, not invented
        assert report.spans_missing == 1
        assert "span" in report.summary_line()

    def test_record_spans_false_omits_metadata(self):
        with patch("multivon_eval.generate._judge_call", return_value=QA_RAW):
            cases = generate_from_text(TEXT, n=2, task="qa", record_spans=False)
        assert all("source_span" not in c.metadata for c in cases)

    def test_provenance_stamped(self):
        with patch("multivon_eval.generate._judge_call", return_value=QA_RAW):
            cases = generate_from_text(TEXT, n=2, task="qa")
        assert cases[0].metadata["_provenance"]["authored_by"] == "generator:doc_qa"
        assert cases[0].metadata["generation"] == {
            "kind": "doc_qa", "unanswerable": False,
        }

    def test_locate_span_uses_chunk_offset(self):
        span = _locate_span(50, "abc needle xyz", "needle")
        assert span["start"] == 54 and span["end"] == 60

    def test_chunk_offsets_match_chunks(self):
        text = "x" * 1500
        for off, chunk in _chunk_text_with_offsets(text, 300):
            assert text[off:off + len(chunk)] == chunk


class TestUnanswerables:
    def test_fraction_generates_refusal_bait(self):
        with patch("multivon_eval.generate._judge_call",
                   side_effect=[QA_RAW, UNANSWERABLE_RAW]):
            cases, report = generate_from_text(
                TEXT, n=4, task="qa", unanswerable_fraction=0.5,
                return_report=True,
            )
        unanswerable = [c for c in cases
                        if c.metadata["generation"]["unanswerable"]]
        assert len(unanswerable) == 2
        for c in unanswerable:
            assert c.expected_output is None
            assert c.metadata["expected_behavior"] == "refusal"
            assert "unanswerable" in c.metadata["note"]
            assert c.metadata["source_span"] is None  # nothing to ground
            assert c.context  # the doc chunk rides along as context
        assert report.accepted == 4

    def test_fraction_validation(self):
        with pytest.raises(ValueError, match="unanswerable_fraction"):
            generate_from_text(TEXT, n=2, unanswerable_fraction=1.5)

    def test_fraction_rejected_for_non_qa_tasks(self):
        with patch("multivon_eval.generate._judge_call",
                   return_value=json.dumps({"summary": "s"})):
            with pytest.raises(ValueError, match="task='qa' only"):
                generate_from_text(TEXT, n=2, task="summarization",
                                   unanswerable_fraction=0.5)


class TestBackwardCompatibleShape:
    def test_default_call_returns_plain_list(self):
        with patch("multivon_eval.generate._judge_call", return_value=QA_RAW):
            cases = generate_from_text(TEXT, n=2, task="qa")
        assert isinstance(cases, list)
        assert cases[0].expected_output == "330 metres"

    def test_return_report_accounts_for_gates(self):
        with patch("multivon_eval.generate._judge_call", return_value=QA_RAW):
            cases, report = generate_from_text(TEXT, n=2, task="qa",
                                               return_report=True)
        assert report.generated == report.accepted + report.dropped_malformed \
            + report.dropped_duplicate
        assert report.accepted == len(cases) == 2

    def test_generate_from_file_passes_params_through(self, tmp_path):
        src = tmp_path / "doc.txt"
        src.write_text(TEXT, encoding="utf-8")
        with patch("multivon_eval.generate._judge_call", return_value=QA_RAW):
            cases, report = generate_from_file(str(src), n=2,
                                               return_report=True)
        assert report.kind == "doc_qa"
        assert cases[0].metadata["source_span"] is not None

    def test_duplicate_questions_dropped_by_gate(self):
        dup_raw = json.dumps([
            {"question": "How tall is the Eiffel Tower?", "answer": "330 m",
             "context_excerpt": "The Eiffel Tower is 330 metres tall."},
            {"question": "How tall is the Eiffel Tower?", "answer": "330 m",
             "context_excerpt": "The Eiffel Tower is 330 metres tall."},
        ])
        with patch("multivon_eval.generate._judge_call", return_value=dup_raw):
            cases, report = generate_from_text(TEXT, n=2, task="qa",
                                               return_report=True)
        assert len(cases) == 1
        assert report.dropped_duplicate == 1
