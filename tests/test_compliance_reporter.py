"""Tests for the chained ComplianceReporter, control mappings, and coverage report."""
from __future__ import annotations

import json

from multivon_eval import ComplianceReporter, EvalSuite
from multivon_eval.compliance import (
    _CHAIN_VERSION,
    _EU_AI_ACT_BY_EVALUATOR,
    _EU_AI_ACT_CONTROLS,
    _GENESIS_HASH,
)
from multivon_eval.result import CaseResult, EvalReport, EvalResult


def _make_report(suite_name: str = "Test Suite", model_id: str = "test-model") -> EvalReport:
    case_results = [
        CaseResult(
            case_input="q1",
            actual_output="a1",
            results=[
                EvalResult(evaluator="faithfulness", score=0.9, passed=True),
                EvalResult(evaluator="hallucination", score=0.95, passed=True),
            ],
        ),
        CaseResult(
            case_input="q2",
            actual_output="a2",
            results=[EvalResult(evaluator="faithfulness", score=0.3, passed=False)],
        ),
    ]
    return EvalReport(suite_name=suite_name, case_results=case_results, model_id=model_id)


def _log_lines(tmp_path, suite_name: str = "Test Suite") -> list[dict]:
    log = tmp_path / f"{suite_name.replace(' ', '_')}.audit.ndjson"
    return [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]


class TestHashChain:
    def test_first_record_uses_genesis_prev_hash(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act")
        rep.record(_make_report())
        line = _log_lines(tmp_path)[0]
        assert line["prev_hash"] == _GENESIS_HASH
        assert line["chain_version"] == _CHAIN_VERSION
        assert line["record_hash"]

    def test_second_record_links_to_first(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act")
        rep.record(_make_report())
        rep.record(_make_report())
        lines = _log_lines(tmp_path)
        assert lines[1]["prev_hash"] == lines[0]["record_hash"]

    def test_chain_of_three_records(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act")
        for _ in range(3):
            rep.record(_make_report())
        lines = _log_lines(tmp_path)
        assert lines[1]["prev_hash"] == lines[0]["record_hash"]
        assert lines[2]["prev_hash"] == lines[1]["record_hash"]

    def test_verify_passes_for_intact_chain(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act")
        for _ in range(3):
            rep.record(_make_report())
        assert rep.verify("Test Suite") is True

    def test_verify_detects_in_place_edit(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act")
        for _ in range(3):
            rep.record(_make_report())
        log = tmp_path / "Test_Suite.audit.ndjson"
        lines = log.read_text().splitlines()
        middle = json.loads(lines[1])
        middle["summary"]["pass_rate"] = 0.0
        lines[1] = json.dumps(middle, separators=(",", ":"))
        log.write_text("\n".join(lines) + "\n")
        assert rep.verify("Test Suite") is False

    def test_verify_detects_mid_log_deletion(self, tmp_path):
        """The pre-fix gap: standalone hashes don't catch deleted middle records."""
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act")
        for _ in range(3):
            rep.record(_make_report())
        log = tmp_path / "Test_Suite.audit.ndjson"
        lines = log.read_text().splitlines()
        del lines[1]
        log.write_text("\n".join(lines) + "\n")
        assert rep.verify("Test Suite") is False


class TestArticleMapping:
    """Guards against the pre-fix bug where accuracy/privacy/bias were all attributed to Art. 9."""

    def test_faithfulness_maps_to_article_15_1(self):
        assert _EU_AI_ACT_BY_EVALUATOR["faithfulness"] == ["art_15_1"]
        assert _EU_AI_ACT_CONTROLS["art_15_1"].id == "Art. 15(1)"

    def test_hallucination_maps_to_article_15_1(self):
        assert _EU_AI_ACT_BY_EVALUATOR["hallucination"] == ["art_15_1"]

    def test_pii_detection_maps_to_article_10_5(self):
        assert _EU_AI_ACT_BY_EVALUATOR["pii_detection"] == ["art_10_5"]
        assert _EU_AI_ACT_CONTROLS["art_10_5"].id == "Art. 10(5)"

    def test_bias_maps_to_article_10_2(self):
        assert _EU_AI_ACT_BY_EVALUATOR["bias"] == ["art_10_2_fg"]

    def test_toxicity_maps_to_article_9_2_b(self):
        assert _EU_AI_ACT_BY_EVALUATOR["toxicity"] == ["art_9_2_b"]

    def test_robustness_evaluators_map_to_article_15_2(self):
        for ev in ["not_empty", "schema_compliance", "self_consistency", "latency", "max_latency"]:
            assert _EU_AI_ACT_BY_EVALUATOR[ev] == ["art_15_2"], ev

    def test_record_attaches_controls_to_evaluator_results(self, tmp_path):
        rep = ComplianceReporter(output_dir=str(tmp_path), framework="eu-ai-act")
        rep.record(_make_report())
        ev_results = _log_lines(tmp_path)[0]["evaluator_results"]
        faith = next(e for e in ev_results if e["evaluator"] == "faithfulness")
        assert faith["controls"] == [{"id": "Art. 15(1)", "description": "Accuracy"}]


class TestCoverageReport:
    def test_factory_covers_all_measurable_controls(self):
        suite = EvalSuite.eu_ai_act_high_risk()
        rep = ComplianceReporter(framework="eu-ai-act")
        cov = rep.coverage(suite)
        assert set(cov.covered.keys()) == set(_EU_AI_ACT_CONTROLS.keys())
        assert cov.missing == []

    def test_minimal_suite_surfaces_gaps(self):
        from multivon_eval.evaluators.deterministic import NotEmpty

        suite = EvalSuite("Minimal").add_evaluators(NotEmpty())
        rep = ComplianceReporter(framework="eu-ai-act")
        cov = rep.coverage(suite)
        assert "art_15_2" in cov.covered
        missing_ids = {c.id for c in cov.missing}
        assert {"Art. 15(1)", "Art. 10(5)", "Art. 10(2)(f-g)", "Art. 9(2)(b)"} <= missing_ids

    def test_process_controls_surfaced(self):
        suite = EvalSuite.eu_ai_act_high_risk()
        rep = ComplianceReporter(framework="eu-ai-act")
        cov = rep.coverage(suite)
        ids = {c.id for c in cov.process}
        assert {"Art. 11", "Art. 12", "Art. 13", "Art. 14", "Art. 15(4-5)"} <= ids

    def test_coverage_str_renders_articles_and_summary(self):
        suite = EvalSuite.eu_ai_act_high_risk()
        rep = ComplianceReporter(framework="eu-ai-act")
        rendered = str(rep.coverage(suite))
        assert "Art. 15(1)" in rendered
        assert "Art. 14" in rendered
        assert "Coverage:" in rendered

    def test_coverage_to_dict_is_json_serializable(self):
        suite = EvalSuite.eu_ai_act_high_risk()
        rep = ComplianceReporter(framework="eu-ai-act")
        d = rep.coverage(suite).to_dict()
        json.dumps(d)  # must not raise

    def test_nist_coverage(self):
        suite = EvalSuite.eu_ai_act_high_risk()
        rep = ComplianceReporter(framework="nist-ai-rmf")
        cov = rep.coverage(suite)
        # Same evaluator set covers performance, robustness, safety, privacy, fairness
        assert "measure_2_3" in cov.covered   # performance
        assert "measure_2_6" in cov.covered   # safety (toxicity)
        assert "measure_2_10" in cov.covered  # privacy (pii)
        assert "measure_2_11" in cov.covered  # fairness (bias)


class TestFactory:
    def test_wires_required_evaluators(self):
        from multivon_eval.evaluators.compliance import PIIEvaluator
        from multivon_eval.evaluators.deterministic import NotEmpty
        from multivon_eval.evaluators.llm_judge import (
            Bias,
            Faithfulness,
            Hallucination,
            Relevance,
            Toxicity,
        )

        suite = EvalSuite.eu_ai_act_high_risk()
        types = {type(e) for e in suite._evaluators}
        assert types == {
            NotEmpty,
            Faithfulness,
            Hallucination,
            Relevance,
            Toxicity,
            Bias,
            PIIEvaluator,
        }

    def test_factory_with_schema_adds_schema_evaluator(self):
        from pydantic import BaseModel

        from multivon_eval.evaluators.compliance import SchemaEvaluator

        class Out(BaseModel):
            answer: str

        suite = EvalSuite.eu_ai_act_high_risk(schema=Out)
        assert any(isinstance(e, SchemaEvaluator) for e in suite._evaluators)
