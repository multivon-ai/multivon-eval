"""Tests for ``multivon_eval.discover`` — bootstrap pipeline.

These tests mock the LLM call (no Anthropic API hits) but exercise:
  - PII scan + redaction on real strings.
  - Trace summary stats math.
  - Heuristic recommendation paths per shape.
  - LLM response parsing + safety-net merge.
  - Threshold calibration math.
  - End-to-end artifact emission.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from multivon_eval import discover
from multivon_eval._pii_scan import scan, redact


# ─── PII scan + redact ───────────────────────────────────────────────────


def test_pii_scan_finds_email_and_ssn():
    text = "Email: alice@example.com, SSN 123-45-6789"
    detections = scan(text)
    labels = [d.label for d in detections]
    assert "email" in labels
    assert "us_ssn" in labels


def test_pii_redact_replaces_spans():
    text = "Contact alice@example.com about order"
    redacted, dets = redact(text)
    assert "alice@example.com" not in redacted
    assert "[REDACTED:email]" in redacted
    assert len(dets) == 1


def test_pii_scan_catches_aws_key():
    text = "config: AKIAIOSFODNN7EXAMPLE"
    labels = [d.label for d in scan(text)]
    assert "aws_access_key" in labels


def test_pii_scan_ignores_innocuous_text():
    assert scan("How do I reset my password?") == []


# ─── Trace summary + shape inference ─────────────────────────────────────


def test_summary_counts_fields_correctly():
    traces = [
        {"input": "q1", "context": "ctx1", "expected_output": "a1"},
        {"input": "q2", "context": "ctx2"},
        {"input": "q3", "expected_output": "a3"},
    ]
    summary, _ = discover.summarize_traces(traces)
    assert summary.count == 3
    assert summary.has_input_pct == 100
    assert summary.has_context_pct == pytest.approx(66.6666, rel=0.01)
    assert summary.has_expected_output_pct == pytest.approx(66.6666, rel=0.01)


def test_summary_redacts_email_by_default():
    traces = [{"input": "I'm alice@example.com, help?", "output": "Hi alice"}]
    summary, sanitized = discover.summarize_traces(traces)
    # Original email gone in sanitized version
    assert "alice@example.com" not in sanitized[0]["input"]
    assert "[REDACTED:email]" in sanitized[0]["input"]
    # Summary records the detection
    assert "email" in summary.pii_label_counts


def test_summary_strict_mode_raises_on_pii():
    traces = [{"input": "ssn: 123-45-6789"}]
    with pytest.raises(ValueError, match="strict-pii"):
        discover.summarize_traces(traces, pii_policy="strict")


def test_summary_allow_mode_keeps_pii_raw():
    traces = [{"input": "ssn: 123-45-6789"}]
    summary, sanitized = discover.summarize_traces(traces, pii_policy="allow")
    # Not redacted
    assert "123-45-6789" in sanitized[0]["input"]
    # But scan still ran and populated counts
    assert summary.pii_label_counts.get("us_ssn") == 1


# ─── Shape inference ─────────────────────────────────────────────────────


def _summary(**kw):
    """Build a TraceSummary with sensible defaults overridden by kw."""
    defaults = dict(
        count=10, has_input_pct=100, has_output_pct=100, has_context_pct=0,
        has_expected_output_pct=0, has_expected_tool_calls_pct=0,
        has_conversation_pct=0, has_images_pct=0, has_schema_pct=0,
        avg_input_chars=80, avg_output_chars=120,
    )
    defaults.update(kw)
    return discover.TraceSummary(**defaults)


def test_infer_rag_shape_from_context():
    assert discover.infer_product_shape(_summary(has_context_pct=80)) == "rag"


def test_infer_qa_shape_from_expected_output():
    assert discover.infer_product_shape(_summary(has_expected_output_pct=60)) == "qa"


def test_infer_agent_shape_from_tool_calls():
    assert discover.infer_product_shape(_summary(has_expected_tool_calls_pct=40)) == "agent"


def test_infer_conversation_shape():
    assert discover.infer_product_shape(_summary(has_conversation_pct=50)) == "conversation"


def test_infer_multimodal_shape():
    assert discover.infer_product_shape(_summary(has_images_pct=30)) == "multimodal"


def test_infer_bare_when_no_signal():
    s = _summary(has_input_pct=0, has_output_pct=0)
    assert discover.infer_product_shape(s) == "bare"


# ─── Heuristic recommendations ───────────────────────────────────────────


def test_heuristic_rag_includes_faithfulness_and_hallucination():
    recs = discover.heuristic_recommendations("rag", _summary(has_context_pct=80))
    names = [r.name for r in recs]
    assert "Faithfulness" in names
    assert "Hallucination" in names
    assert "NotEmpty" in names  # always-on guardrail


def test_heuristic_qa_includes_relevance():
    recs = discover.heuristic_recommendations("qa", _summary(has_expected_output_pct=60))
    names = [r.name for r in recs]
    assert "AnswerAccuracy" in names
    assert "Relevance" in names


def test_heuristic_adds_pii_guardrail_when_pii_detected():
    s = _summary()
    s.pii_label_counts["email"] = 5
    recs = discover.heuristic_recommendations("rag", s)
    names = [r.name for r in recs]
    assert "PIIEvaluator" in names


# ─── LLM proposal parsing ────────────────────────────────────────────────


def test_parse_proposal_extracts_valid_evaluators():
    text = json.dumps({
        "evaluators": [
            {"name": "Faithfulness", "tier": "primary", "threshold": 0.75,
             "rationale": "RAG shape"},
            {"name": "NotEmpty", "tier": "guardrail", "threshold": 1.0,
             "rationale": "sanity"},
        ],
        "discussion": "Two-evaluator mix for RAG.",
    })
    evals, disc = discover._parse_proposal(text)
    assert len(evals) == 2
    assert evals[0].name == "Faithfulness"
    assert evals[0].threshold == 0.75
    assert "Two-evaluator" in disc


def test_parse_proposal_drops_invented_evaluators():
    text = json.dumps({
        "evaluators": [
            {"name": "Faithfulness", "tier": "primary", "threshold": 0.7, "rationale": "ok"},
            {"name": "MadeUpEvaluator", "tier": "primary", "threshold": 0.5, "rationale": "bogus"},
        ],
        "discussion": "",
    })
    evals, _ = discover._parse_proposal(text)
    names = [e.name for e in evals]
    assert "Faithfulness" in names
    assert "MadeUpEvaluator" not in names


def test_parse_proposal_strips_markdown_fences():
    text = "```json\n" + json.dumps({"evaluators": [], "discussion": "wrapped"}) + "\n```"
    _, disc = discover._parse_proposal(text)
    assert disc == "wrapped"


def test_parse_proposal_handles_malformed_json():
    evals, disc = discover._parse_proposal("not json at all")
    assert evals == []
    assert disc.startswith("not json")


def test_parse_proposal_clamps_threshold_to_range():
    text = json.dumps({
        "evaluators": [
            {"name": "Faithfulness", "tier": "primary", "threshold": 99.0, "rationale": "x"},
            {"name": "NotEmpty", "tier": "guardrail", "threshold": -0.5, "rationale": "y"},
        ],
        "discussion": "",
    })
    evals, _ = discover._parse_proposal(text)
    assert evals[0].threshold == 1.0
    assert evals[1].threshold == 0.0


# ─── Merge: heuristic + LLM ──────────────────────────────────────────────


def test_merge_llm_wins_on_same_name():
    # Include a guardrail in heuristic so the safety net doesn't add one
    heuristic = [
        discover.RecommendedEvaluator("Faithfulness", "primary", 0.7, "heuristic rationale", "heuristic"),
        discover.RecommendedEvaluator("NotEmpty", "guardrail", 1.0, "sanity", "heuristic"),
    ]
    llm = [discover.RecommendedEvaluator(
        "Faithfulness", "primary", 0.85, "llm rationale", "llm",
    )]
    merged = discover.merge_recommendations(heuristic, llm)
    faithfulness = next(r for r in merged if r.name == "Faithfulness")
    # LLM's threshold + rationale win for the overlapping name
    assert faithfulness.threshold == 0.85
    assert "llm" in faithfulness.rationale
    # Source marked as 'merged' since both contributed
    assert faithfulness.source == "merged"


def test_merge_adds_heuristic_when_not_in_llm():
    heuristic = [
        discover.RecommendedEvaluator("Faithfulness", "primary", 0.7, "h1", "heuristic"),
        discover.RecommendedEvaluator("NotEmpty", "guardrail", 1.0, "h2", "heuristic"),
    ]
    llm = [discover.RecommendedEvaluator("Hallucination", "primary", 0.75, "l1", "llm")]
    merged = discover.merge_recommendations(heuristic, llm)
    names = {r.name for r in merged}
    assert names == {"Faithfulness", "NotEmpty", "Hallucination"}


def test_merge_caps_at_max_count():
    many = [
        discover.RecommendedEvaluator(f"NotEmpty", "guardrail", 1.0, str(i), "heuristic")
        for i in range(10)
    ]
    # Dedup by name happens first → cap kicks in only if we have unique names
    unique = [
        discover.RecommendedEvaluator("Faithfulness", "primary", 0.7, "f", "heuristic"),
        discover.RecommendedEvaluator("Hallucination", "primary", 0.7, "h", "heuristic"),
        discover.RecommendedEvaluator("Relevance", "secondary", 0.7, "r", "heuristic"),
        discover.RecommendedEvaluator("Coherence", "secondary", 0.7, "c", "heuristic"),
        discover.RecommendedEvaluator("NotEmpty", "guardrail", 1.0, "n", "heuristic"),
        discover.RecommendedEvaluator("BLEU", "secondary", 0.7, "b", "heuristic"),
        discover.RecommendedEvaluator("ROUGE", "secondary", 0.7, "r2", "heuristic"),
    ]
    merged = discover.merge_recommendations(unique, [], max_count=4)
    assert len(merged) == 4


def test_merge_always_includes_a_guardrail():
    # Even if no guardrail was proposed, safety net adds NotEmpty
    heuristic = [discover.RecommendedEvaluator("Faithfulness", "primary", 0.7, "f", "heuristic")]
    merged = discover.merge_recommendations(heuristic, [], max_count=6)
    assert any(r.tier == "guardrail" for r in merged)


# ─── Artifact emission ───────────────────────────────────────────────────


def test_render_eval_suite_py_imports_recommended_evaluators():
    evals = [
        discover.RecommendedEvaluator("Faithfulness", "primary", 0.7, "test", "llm"),
        discover.RecommendedEvaluator("NotEmpty", "guardrail", 1.0, "sanity", "heuristic"),
    ]
    py = discover._render_eval_suite_py("rag", evals)
    assert "from multivon_eval import" in py
    assert "Faithfulness" in py
    assert "NotEmpty" in py
    assert "rag" in py


def test_render_thresholds_yaml_is_valid():
    evals = [
        discover.RecommendedEvaluator("Faithfulness", "primary", 0.72, "test", "llm"),
    ]
    yaml_text = discover._render_thresholds_yaml(evals)
    assert "Faithfulness:" in yaml_text
    assert "threshold: 0.72" in yaml_text
    assert "tier: primary" in yaml_text


def test_render_report_md_includes_evaluator_table():
    evals = [
        discover.RecommendedEvaluator("Faithfulness", "primary", 0.72, "ground claims", "llm"),
    ]
    md = discover._render_report_md(
        "A support bot answering refund questions.",
        "rag",
        _summary(count=100, has_context_pct=80),
        evals,
        "RAG shape with strong context signal — Faithfulness is critical.",
    )
    assert "Discovery Report" in md
    assert "Faithfulness" in md
    assert "support bot" in md
    assert "rag" in md


# ─── End-to-end with mocked LLM ──────────────────────────────────────────


def _fake_judge_response(*_args, **_kw):
    """Simulate the LLM's JSON response for the proposal call."""
    return json.dumps({
        "evaluators": [
            {"name": "Faithfulness", "tier": "primary", "threshold": 0.75,
             "rationale": "RAG shape detected"},
            {"name": "Hallucination", "tier": "primary", "threshold": 0.80,
             "rationale": "context grounding required"},
            {"name": "Relevance", "tier": "secondary", "threshold": 0.70,
             "rationale": "topic-on-question check"},
            {"name": "NotEmpty", "tier": "guardrail", "threshold": 1.0,
             "rationale": "sanity"},
        ],
        "discussion": "RAG shape with strong context — Faithfulness is the main gate.",
    })


def test_end_to_end_with_mocked_llm(tmp_path):
    # Write inputs
    product_path = tmp_path / "product.md"
    product_path.write_text(
        "# Product\nA customer support bot answering refund questions.\n"
        "# Risks\nHallucination, off-topic responses.\n"
    )

    traces_path = tmp_path / "traces.jsonl"
    traces = [
        {"input": "What's the refund window?", "context": "Refunds within 30 days.",
         "output": "30 days from purchase."},
        {"input": "Can I return electronics?", "context": "Electronics: 14-day window.",
         "output": "Yes, within 14 days."},
        {"input": "Reset password?", "context": "See account page.",
         "output": "Visit your account page."},
    ]
    with traces_path.open("w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    out_dir = tmp_path / "out"

    # Patch the judge call to skip actual API hits + skip calibration (no LLM judges)
    with patch.object(discover, "_call_judge", side_effect=_fake_judge_response):
        result = discover.bootstrap(
            description_path=product_path,
            traces_path=traces_path,
            output_dir=out_dir,
            skip_seed_cases=True,
            skip_calibration=True,
        )

    # Shape inference picked RAG
    assert result.shape == "rag"
    # Evaluators include both the LLM picks + heuristic safety
    names = {r.name for r in result.evaluators}
    assert "Faithfulness" in names
    assert "NotEmpty" in names
    # Artifacts exist on disk
    assert result.artifacts["eval_suite"].exists()
    assert result.artifacts["report"].exists()
    assert result.artifacts["thresholds"].exists()
    # The emitted Python is parseable
    compile(result.artifacts["eval_suite"].read_text(), "eval_suite.py", "exec")
    # The report mentions the discussion text
    assert "RAG shape" in result.artifacts["report"].read_text()


def test_load_traces_skips_blank_and_comment_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(
        "\n"
        "# this is a comment\n"
        '{"input": "q1"}\n'
        "\n"
        '{"input": "q2"}\n'
    )
    rows = discover.load_traces(path)
    assert len(rows) == 2
    assert rows[0]["input"] == "q1"


def test_load_traces_skips_rows_missing_input(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(
        '{"input": "q1"}\n'
        '{"output": "no input here"}\n'
    )
    rows = discover.load_traces(path)
    assert len(rows) == 1
    assert rows[0]["input"] == "q1"


def test_load_traces_raises_on_malformed_interior_json(tmp_path):
    # A malformed INTERIOR line is data corruption — stays a loud
    # ValueError with file:line. (A malformed FINAL line is tolerated —
    # see test_load_traces_skips_malformed_final_line.)
    path = tmp_path / "t.jsonl"
    path.write_text('{"input": "q1"}\n{not json\n{"input": "q2"}\n')
    with pytest.raises(ValueError, match=r"t\.jsonl:2.*malformed"):
        discover.load_traces(path)


def test_load_traces_skips_malformed_final_line_with_warning(tmp_path, capsys):
    # Truncated streamed dump — the normal failure shape: tolerate the
    # tail, loudly, instead of failing the whole bootstrap.
    path = tmp_path / "t.jsonl"
    path.write_text('{"input": "q1"}\n{"input": "q2"}\n{"input": "tru')
    rows = discover.load_traces(path)
    assert [r["input"] for r in rows] == ["q1", "q2"]
    err = capsys.readouterr().err
    assert "final line is malformed JSON" in err
    assert "t.jsonl:3" in err


def test_load_traces_caps_at_10k_with_loud_warning(tmp_path, capsys):
    path = tmp_path / "t.jsonl"
    with path.open("w") as f:
        for i in range(10_050):
            f.write(json.dumps({"input": f"q{i}"}) + "\n")
    rows = discover.load_traces(path)
    assert len(rows) == 10_000
    err = capsys.readouterr().err
    assert "truncated to first 10,000 traces" in err


def test_cmd_bootstrap_malformed_traces_is_clean_exit_2(tmp_path, capsys):
    # CLI boundary: a corrupt interior traces line must exit 2 with
    # file:line, never a ValueError traceback. load_traces runs before any
    # LLM call, so no judge mocking is needed.
    from argparse import Namespace
    from multivon_eval.cli import cmd_bootstrap

    product = tmp_path / "product.md"
    product.write_text("# Product\nA bot.\n")
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"input": "q1"}\n{not json\n{"input": "q2"}\n')

    rcode = cmd_bootstrap(Namespace(
        product=str(product), traces=str(traces),
        output=str(tmp_path / "out"), judge_model="m",
        judge_provider="anthropic", judge_base_url=None,
        n_seed_cases=1, pii_policy="redact", skip_seed_cases=True,
        skip_calibration=True, validate=False, validate_n_shots=3,
        repo=str(tmp_path),
    ))
    assert rcode == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "traces.jsonl:2" in err


def test_emit_artifacts_is_atomic_no_partials_on_interrupt(tmp_path):
    # Ctrl-C mid-render must leave NO half-emitted eval_suite.py that looks
    # complete — and no .tmp- droppings.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    summary, _ = discover.summarize_traces([{"input": "q", "output": "a"}])
    evaluators = discover.heuristic_recommendations("qa", summary)
    kwargs = dict(
        out_dir=out_dir, description="d", shape="qa", summary=summary,
        evaluators=evaluators, discussion="", seed_cases=[],
    )
    with patch.object(discover, "_render_report_md",
                      side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            discover._emit_artifacts(**kwargs)
    # eval_suite.py was rendered before the interrupt, but must NOT have
    # been promoted into place; the tmp dir is cleaned up.
    assert list(out_dir.iterdir()) == []

    # Happy path: all four files land, no tmp dir remains.
    paths = discover._emit_artifacts(**kwargs)
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "DISCOVERY_REPORT.md", "eval_suite.py", "seed_cases.jsonl",
        "thresholds.yaml",
    ]
    for p in paths.values():
        assert p.exists()


# ─── Scaled + gated seed-case generation ─────────────────────────────────


from multivon_eval import EvalCase  # noqa: E402
from multivon_eval.case_gates import GenerationReport  # noqa: E402
from multivon_eval.judge import JudgeConfig  # noqa: E402

# Cost-estimation tests pin the judge explicitly: estimates are model-based
# (Haiku-only today) and other tests may leave a different global judge
# configured.
_HAIKU = JudgeConfig(provider="anthropic", model="claude-haiku-4-5-20251001")


def _distinct_case(i: int) -> EvalCase:
    """A generated-looking case whose token set is disjoint from every other
    index — never tripping the Jaccard dedupe gate by accident."""
    return EvalCase(
        input=f"q{i} alpha{i} beta{i} gamma{i} delta{i} epsilon{i} zeta{i} eta{i} theta{i}",
        expected_output=f"answer {i}",
        metadata={"stress_tests": ["NotEmpty"]},
    )


def _fake_batch_generator(batches: list[list[EvalCase]], calls: list[dict]):
    """Stand-in for auto.generate_adversarial_cases — records every call's
    prompt + n and serves pre-baked batches. Mirrors the existing
    patch.object mocking pattern (no network)."""
    def fake(seed_text, mode, n=5, *, judge=None):
        calls.append({"seed_text": seed_text, "mode": mode, "n": n})
        return batches[len(calls) - 1]
    return fake


def test_generate_seed_cases_batches_75_into_3_calls_with_digests():
    counter = iter(range(1000))
    batches = [
        [_distinct_case(next(counter)) for _ in range(30)],
        [_distinct_case(next(counter)) for _ in range(30)],
        [_distinct_case(next(counter)) for _ in range(15)],
    ]
    calls: list[dict] = []
    with patch("multivon_eval.auto.generate_adversarial_cases",
               side_effect=_fake_batch_generator(batches, calls)):
        cases, cost, report = discover.generate_seed_cases(
            "A refund-policy QA bot.", "qa", n=75, judge=_HAIKU,
        )

    # 3 LLM calls: 30 + 30 + 15
    assert [c["n"] for c in calls] == [30, 30, 15]
    # Batch 1 sees only the description — no digest.
    assert "ALREADY-ACCEPTED" not in calls[0]["seed_text"]
    # Batches 2+ carry the digest of already-accepted inputs.
    for later in calls[1:]:
        assert "ALREADY-ACCEPTED" in later["seed_text"]
        assert "do NOT duplicate" in later["seed_text"]
    # First 8 words of a batch-1 input appear in the batch-2 prompt.
    first_words = " ".join(batches[0][0].input.split()[:8])
    assert first_words in calls[1]["seed_text"]

    assert len(cases) == 75
    assert report.requested == 75
    assert report.generated == 75
    assert report.accepted == 75
    assert report.n_batches == 3
    # Per-case generation metadata records the batch index.
    assert cases[0].metadata["generation"]["batch"] == 1
    assert cases[-1].metadata["generation"]["batch"] == 3
    assert cases[0].metadata["generation"]["gates_passed"] == [
        "well_formed", "duplicate",
    ]
    assert cases[0].metadata["generation"]["hardness"] is None
    assert cost > 0


def test_generate_seed_cases_dedupes_across_batches():
    original = EvalCase(
        input="what is the standard refund window policy",
        expected_output="30 days",
        metadata={"stress_tests": ["NotEmpty"]},
    )
    exact_dup = EvalCase(
        input="what  is the standard\nrefund window policy",  # loose-identical
        expected_output="different answer",
        metadata={"stress_tests": ["NotEmpty"]},
    )
    near_dup = EvalCase(  # 6/7 token Jaccard = 0.857 ≥ 0.85
        input="what is the refund window policy",
        expected_output="30 days",
        metadata={"stress_tests": ["NotEmpty"]},
    )
    batches = [
        [original, _distinct_case(1)],
        [exact_dup, near_dup],
    ]
    calls: list[dict] = []
    with patch("multivon_eval.auto.generate_adversarial_cases",
               side_effect=_fake_batch_generator(batches, calls)):
        cases, _cost, report = discover.generate_seed_cases(
            "desc", "qa", n=4, batch_size=2,
        )

    assert len(calls) == 2
    assert [c.input for c in cases] == [original.input, _distinct_case(1).input]
    assert report.generated == 4
    assert report.accepted == 2
    assert report.dropped_duplicate == 2
    assert report.dropped_malformed == 0
    # Accounting invariant: every generated case lands in exactly one bucket.
    assert report.generated == (
        report.accepted + report.dropped_malformed
        + report.dropped_duplicate + report.dropped_hardness
    )


def test_generate_seed_cases_drops_malformed():
    batches = [[
        _distinct_case(1),
        EvalCase(input="", expected_output="orphan answer"),       # empty input
        EvalCase(input="no expected behavior here at all"),        # no answer key
    ]]
    calls: list[dict] = []
    with patch("multivon_eval.auto.generate_adversarial_cases",
               side_effect=_fake_batch_generator(batches, calls)):
        cases, _cost, report = discover.generate_seed_cases("desc", "qa", n=3)

    assert len(cases) == 1
    assert report.generated == 3
    assert report.dropped_malformed == 2
    assert report.accepted == 1
    assert report.generated == (
        report.accepted + report.dropped_malformed
        + report.dropped_duplicate + report.dropped_hardness
    )


def test_generate_seed_cases_hardness_gate_filters_and_stamps_scores():
    # NotEmpty is deterministic — no judge calls. Baseline answers the
    # "easy" case and goes silent on the "hard" ones.
    hard1 = EvalCase(input="hard question alpha beta",
                     expected_output="a", metadata={"stress_tests": ["NotEmpty"]})
    hard2 = EvalCase(input="hard question gamma delta tricky",
                     expected_output="a", metadata={"stress_tests": ["NotEmpty"]})
    easy = EvalCase(input="easy question zeta eta",
                    expected_output="a", metadata={"stress_tests": ["NotEmpty"]})

    def baseline(text: str) -> str:
        return "" if "hard" in text else "confident answer"

    calls: list[dict] = []
    with patch("multivon_eval.auto.generate_adversarial_cases",
               side_effect=_fake_batch_generator([[hard1, hard2, easy]], calls)):
        cases, _cost, report = discover.generate_seed_cases(
            "desc", "qa", n=3,
            validate_cases=True, baseline_model_fn=baseline, n_shots=3,
        )

    # Band default (0.5, 1.0): hard cases (failure_rate 1.0) kept,
    # easy case (0.0) dropped.
    assert cases == [hard1, hard2]
    assert report.dropped_hardness == 1
    assert report.accepted == 2
    assert not report.hardness_skipped
    assert report.generated == (
        report.accepted + report.dropped_malformed
        + report.dropped_duplicate + report.dropped_hardness
    )
    # Hardness scores land in metadata; the gate is recorded as passed.
    for c in cases:
        gen = c.metadata["generation"]
        assert gen["hardness"] == 1.0
        assert gen["gates_passed"] == ["well_formed", "duplicate", "hardness"]


def test_generate_seed_cases_hardness_skipped_paths_are_honest():
    batches = [[_distinct_case(1)]]
    # Default: no validate, no baseline.
    with patch("multivon_eval.auto.generate_adversarial_cases",
               side_effect=_fake_batch_generator(batches, [])):
        _cases, _cost, report = discover.generate_seed_cases("desc", "qa", n=1)
    assert report.hardness_skipped
    assert report.hardness_skip_reason == "no --validate-cases / baseline model"

    # validate_cases=True but no baseline model → still skipped, says why.
    with patch("multivon_eval.auto.generate_adversarial_cases",
               side_effect=_fake_batch_generator(batches, [])):
        _cases, _cost, report = discover.generate_seed_cases(
            "desc", "qa", n=1, validate_cases=True,
        )
    assert report.hardness_skipped
    assert "no baseline model" in report.hardness_skip_reason


def test_generate_seed_cases_budget_exceeded_fails_fast_pre_spend():
    calls: list[dict] = []
    with patch("multivon_eval.auto.generate_adversarial_cases",
               side_effect=_fake_batch_generator([], calls)):
        with pytest.raises(ValueError, match=r"budget.*--budget-usd"):
            # Haiku judge → 17 batches estimate ≈ $0.42 > $0.10.
            discover.generate_seed_cases(
                "desc", "qa", n=500, budget_usd=0.10, judge=_HAIKU,
            )
    # Fail-fast means ZERO LLM calls were made.
    assert calls == []


def test_generate_seed_cases_rejects_n_above_hard_cap():
    with pytest.raises(ValueError, match="hard cap"):
        discover.generate_seed_cases("desc", "qa", n=501)


def test_generate_seed_cases_emits_batch_progress_lines(capsys):
    batches = [
        [_distinct_case(1), _distinct_case(2)],
        [_distinct_case(3)],
    ]
    with patch("multivon_eval.auto.generate_adversarial_cases",
               side_effect=_fake_batch_generator(batches, [])):
        discover.generate_seed_cases("desc", "qa", n=3, batch_size=2)
    err = capsys.readouterr().err
    assert "[bootstrap] seed-case batch 1/2: 2 generated" in err
    assert "[bootstrap] seed-case batch 2/2: 1 generated" in err


def test_bootstrap_budget_check_runs_before_any_paid_call(tmp_path):
    product = tmp_path / "product.md"
    product.write_text("# Product\nA bot.\n")
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"input": "q", "output": "a"}) + "\n")

    with patch.object(discover, "_call_judge") as judge_mock:
        with pytest.raises(ValueError, match="budget"):
            discover.bootstrap(
                description_path=product, traces_path=traces,
                output_dir=tmp_path / "out",
                skip_calibration=True, judge=_HAIKU,
                n_seed_cases=500, budget_usd=0.05,
            )
    # The proposal LLM call never happened — pre-spend fail-fast.
    judge_mock.assert_not_called()


def test_bootstrap_report_renders_case_generation_section(tmp_path):
    product = tmp_path / "product.md"
    product.write_text("# Product\nA refund QA bot.\n")
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps(
        {"input": "refund?", "context": "Refunds in 30 days.", "output": "30 days"}
    ) + "\n")

    batches = [[
        _distinct_case(1),
        _distinct_case(2),
        EvalCase(input="", expected_output="x"),  # malformed
    ]]
    with patch.object(discover, "_call_judge", side_effect=_fake_judge_response), \
            patch("multivon_eval.auto.generate_adversarial_cases",
                  side_effect=_fake_batch_generator(batches, [])):
        result = discover.bootstrap(
            description_path=product, traces_path=traces,
            output_dir=tmp_path / "out",
            skip_calibration=True, n_seed_cases=3,
        )

    assert isinstance(result.generation_report, GenerationReport)
    assert result.generation_report.generated == 3
    assert result.generation_report.accepted == 2
    assert result.generation_report.dropped_malformed == 1

    report_md = result.artifacts["report"].read_text()
    assert "## Case generation" in report_md
    assert "generated 3, accepted 2 — dropped 0 duplicates, 1 malformed" in report_md
    # Hardness honesty: skipped is stated, never silent.
    assert "hardness gate skipped (no --validate-cases / baseline model)" in report_md

    # Per-case generation metadata flows into seed_cases.jsonl.
    rows = [json.loads(line) for line in
            result.artifacts["seed_cases"].read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        gen = row["metadata"]["generation"]
        assert gen["batch"] == 1
        assert gen["hardness"] is None
        assert "well_formed" in gen["gates_passed"]


def test_bootstrap_skip_seed_cases_report_says_skipped(tmp_path):
    product = tmp_path / "product.md"
    product.write_text("# Product\nA bot.\n")
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"input": "q", "output": "a"}) + "\n")

    with patch.object(discover, "_call_judge", side_effect=_fake_judge_response):
        result = discover.bootstrap(
            description_path=product, traces_path=traces,
            output_dir=tmp_path / "out",
            skip_seed_cases=True, skip_calibration=True,
        )

    assert result.generation_report is None
    report_md = result.artifacts["report"].read_text()
    assert "## Case generation" in report_md
    assert "skipped" in report_md
