"""
ComplianceReporter — local-first, tamper-evident audit trail for AI evals.

Produces an append-only NDJSON log of every eval run, with each record
linked into a SHA-256 hash chain so the entire history is verifiable
end-to-end (deleting or editing a record mid-log is detectable).

Maps evaluator results to paragraph-accurate regulatory controls:

  • EU AI Act (high-risk obligations)
        Art. 9   Risk management system
        Art. 10  Data and data governance
        Art. 12  Record-keeping
        Art. 13  Transparency to deployers
        Art. 14  Human oversight
        Art. 15  Accuracy, robustness, cybersecurity

  • NIST AI RMF 1.0
        GOVERN / MAP / MEASURE / MANAGE subcategories

No cloud required. The log + verifier + coverage analysis all run locally.

Usage:
    from multivon_eval import ComplianceReporter, EvalSuite

    suite = EvalSuite.eu_ai_act_high_risk()
    suite.add_cases(cases)

    reporter = ComplianceReporter("./audit-logs", framework="eu-ai-act")
    report = suite.run(model_fn)
    reporter.record(report, tags={"system": "triage-bot", "version": "1.0"})

    # Pre-flight: which Articles does the suite actually exercise?
    print(reporter.coverage(suite))

    # Post-hoc: was the log tampered with?
    reporter.verify(report.suite_name)
"""
from __future__ import annotations
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .result import EvalReport

if TYPE_CHECKING:
    from .suite import EvalSuite


Framework = Literal["eu-ai-act", "nist-ai-rmf", "none"]

# Genesis prev_hash for the first record in any chain.
_GENESIS_HASH = "0" * 64
# Bump when the on-disk record format changes in a way that breaks hashing.
_CHAIN_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Control catalog
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Control:
    """
    A regulatory control that an evaluator may exercise.

    `id` is the canonical reference an auditor would cite, e.g. "Art. 15(1)"
    for the EU AI Act or "MEASURE 2.3" for NIST AI RMF.
    """
    id: str
    description: str
    framework: str = ""

    def __str__(self) -> str:
        return f"{self.id} — {self.description}"


# EU AI Act high-risk controls relevant to model evaluation.
# Paragraph references follow Regulation (EU) 2024/1689.
_EU_AI_ACT_CONTROLS: dict[str, Control] = {
    "art_9_2_b":    Control("Art. 9(2)(b)",  "Foreseeable misuse risk identification",     "eu-ai-act"),
    "art_10_2_fg": Control("Art. 10(2)(f-g)","Examination and mitigation of possible biases", "eu-ai-act"),
    "art_10_5":     Control("Art. 10(5)",    "Processing of personal data",                "eu-ai-act"),
    "art_15_1":     Control("Art. 15(1)",    "Accuracy",                                   "eu-ai-act"),
    "art_15_2":     Control("Art. 15(2)",    "Robustness",                                 "eu-ai-act"),
}

# Process controls — required for high-risk AI but not satisfiable by
# evaluator output alone. Surfaced separately in coverage reports so users
# know they still need organizational measures.
_EU_AI_ACT_PROCESS_CONTROLS: dict[str, Control] = {
    "art_11":       Control("Art. 11",       "Technical documentation",                    "eu-ai-act"),
    "art_12":       Control("Art. 12",       "Record-keeping (satisfied by this reporter)","eu-ai-act"),
    "art_13":       Control("Art. 13",       "Transparency and information to deployers",  "eu-ai-act"),
    "art_14":       Control("Art. 14",       "Human oversight",                            "eu-ai-act"),
    "art_15_45":    Control("Art. 15(4-5)",  "Cybersecurity and resilience",               "eu-ai-act"),
}

# Maps evaluator.name → Control ids that the evaluator exercises.
# Cross-referenced against the catalog above.
_EU_AI_ACT_BY_EVALUATOR: dict[str, list[str]] = {
    # Accuracy (Art. 15(1))
    "faithfulness":            ["art_15_1"],
    "hallucination":           ["art_15_1"],
    "relevance":               ["art_15_1"],
    "answer_accuracy":         ["art_15_1"],
    "context_precision":       ["art_15_1"],
    "context_recall":          ["art_15_1"],
    "summarization":           ["art_15_1"],
    "coherence":               ["art_15_1"],
    "bertscore":               ["art_15_1"],
    "bleu":                    ["art_15_1"],
    "rouge_l":                 ["art_15_1"],
    "step_faithfulness":       ["art_15_1"],
    "plan_quality":            ["art_15_1"],
    "task_completion":         ["art_15_1"],
    "tool_call_accuracy":      ["art_15_1"],
    "tool_argument_accuracy":  ["art_15_1"],
    "tool_call_necessity":     ["art_15_1"],
    "trajectory_efficiency":   ["art_15_1"],
    "conversation_relevance":  ["art_15_1"],
    "conversation_completeness": ["art_15_1"],
    "knowledge_retention":     ["art_15_1"],
    "g_eval":                  ["art_15_1"],
    "custom_rubric":           ["art_15_1"],
    # Robustness (Art. 15(2))
    "not_empty":               ["art_15_2"],
    "exact_match":             ["art_15_2"],
    "contains":                ["art_15_2"],
    "regex_match":             ["art_15_2"],
    "starts_with":             ["art_15_2"],
    "json_schema":             ["art_15_2"],
    "schema_compliance":       ["art_15_2"],
    "word_count":              ["art_15_2"],
    "latency":                 ["art_15_2"],
    "max_latency":             ["art_15_2"],
    "self_consistency":        ["art_15_2"],
    "turn_consistency":        ["art_15_2"],
    "agent_memory":            ["art_15_2"],
    # Bias & data governance (Art. 10)
    "bias":                    ["art_10_2_fg"],
    "pii_detection":           ["art_10_5"],
    # Foreseeable misuse (Art. 9)
    "toxicity":                ["art_9_2_b"],
}

# NIST AI RMF 1.0 subcategories.
_NIST_CONTROLS: dict[str, Control] = {
    "measure_2_3":  Control("MEASURE 2.3",   "AI system performance evaluation",           "nist-ai-rmf"),
    "measure_2_5":  Control("MEASURE 2.5",   "AI system robustness",                       "nist-ai-rmf"),
    "measure_2_6":  Control("MEASURE 2.6",   "AI system safety",                           "nist-ai-rmf"),
    "measure_2_10": Control("MEASURE 2.10",  "Privacy risk",                               "nist-ai-rmf"),
    "measure_2_11": Control("MEASURE 2.11",  "Fairness and harmful bias",                  "nist-ai-rmf"),
}

_NIST_PROCESS_CONTROLS: dict[str, Control] = {
    "govern_1_1":   Control("GOVERN 1.1",    "AI risk management policies",                "nist-ai-rmf"),
    "measure_2_7":  Control("MEASURE 2.7",   "Security and resilience",                    "nist-ai-rmf"),
    "measure_2_8":  Control("MEASURE 2.8",   "Transparency and accountability",            "nist-ai-rmf"),
    "measure_2_9":  Control("MEASURE 2.9",   "Explainability and interpretability",        "nist-ai-rmf"),
    "manage_4_1":   Control("MANAGE 4.1",    "Post-deployment monitoring",                 "nist-ai-rmf"),
}

_NIST_BY_EVALUATOR: dict[str, list[str]] = {
    # Performance
    "faithfulness":            ["measure_2_3"],
    "hallucination":           ["measure_2_3"],
    "relevance":               ["measure_2_3"],
    "answer_accuracy":         ["measure_2_3"],
    "context_precision":       ["measure_2_3"],
    "context_recall":          ["measure_2_3"],
    "summarization":           ["measure_2_3"],
    "coherence":               ["measure_2_3"],
    "bertscore":               ["measure_2_3"],
    "bleu":                    ["measure_2_3"],
    "rouge_l":                 ["measure_2_3"],
    "step_faithfulness":       ["measure_2_3"],
    "plan_quality":            ["measure_2_3"],
    "task_completion":         ["measure_2_3"],
    "tool_call_accuracy":      ["measure_2_3"],
    "tool_argument_accuracy":  ["measure_2_3"],
    "tool_call_necessity":     ["measure_2_3"],
    "trajectory_efficiency":   ["measure_2_3"],
    "conversation_relevance":  ["measure_2_3"],
    "conversation_completeness": ["measure_2_3"],
    "knowledge_retention":     ["measure_2_3"],
    "g_eval":                  ["measure_2_3"],
    "custom_rubric":           ["measure_2_3"],
    # Robustness
    "not_empty":               ["measure_2_5"],
    "exact_match":             ["measure_2_5"],
    "contains":                ["measure_2_5"],
    "regex_match":             ["measure_2_5"],
    "starts_with":             ["measure_2_5"],
    "json_schema":             ["measure_2_5"],
    "schema_compliance":       ["measure_2_5"],
    "word_count":              ["measure_2_5"],
    "latency":                 ["measure_2_5"],
    "max_latency":             ["measure_2_5"],
    "self_consistency":        ["measure_2_5"],
    "turn_consistency":        ["measure_2_5"],
    "agent_memory":            ["measure_2_5"],
    # Safety
    "toxicity":                ["measure_2_6"],
    # Privacy & fairness
    "pii_detection":           ["measure_2_10"],
    "bias":                    ["measure_2_11"],
}

_CATALOGS: dict[str, dict[str, dict[str, Control]]] = {
    "eu-ai-act": {"measurable": _EU_AI_ACT_CONTROLS, "process": _EU_AI_ACT_PROCESS_CONTROLS},
    "nist-ai-rmf": {"measurable": _NIST_CONTROLS, "process": _NIST_PROCESS_CONTROLS},
}
_BY_EVALUATOR: dict[str, dict[str, list[str]]] = {
    "eu-ai-act": _EU_AI_ACT_BY_EVALUATOR,
    "nist-ai-rmf": _NIST_BY_EVALUATOR,
}


def _controls_for(framework: str, evaluator_name: str) -> list[Control]:
    """Return the Controls that an evaluator exercises under `framework`."""
    if framework == "none":
        return []
    catalog = _CATALOGS.get(framework, {}).get("measurable", {})
    ids = _BY_EVALUATOR.get(framework, {}).get(evaluator_name, [])
    return [catalog[cid] for cid in ids if cid in catalog]


# ─────────────────────────────────────────────────────────────────────────────
# Audit record (chained)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    record_id: str
    suite_name: str
    model_id: str
    timestamp: str
    framework: str
    chain_version: int
    prev_hash: str
    summary: dict
    evaluator_results: list[dict]
    record_hash: str

    def to_ndjson(self) -> str:
        return json.dumps({
            "record_id": self.record_id,
            "suite_name": self.suite_name,
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "framework": self.framework,
            "chain_version": self.chain_version,
            "prev_hash": self.prev_hash,
            "summary": self.summary,
            "evaluator_results": self.evaluator_results,
            "record_hash": self.record_hash,
        }, separators=(",", ":"))


@dataclass
class CoverageReport:
    """Result of running ComplianceReporter.coverage() against a suite."""
    framework: str
    suite_name: str
    covered: dict[str, list[str]] = field(default_factory=dict)
    """control_id → evaluator names that exercise it."""
    missing: list[Control] = field(default_factory=list)
    """Measurable controls with no evaluator coverage."""
    process: list[Control] = field(default_factory=list)
    """Process controls — flagged so users know to address them organizationally."""
    unmapped_evaluators: list[str] = field(default_factory=list)
    """Evaluators in the suite with no entry in this framework's mapping."""

    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "suite_name": self.suite_name,
            "covered": self.covered,
            "missing": [{"id": c.id, "description": c.description} for c in self.missing],
            "process": [{"id": c.id, "description": c.description} for c in self.process],
            "unmapped_evaluators": self.unmapped_evaluators,
        }

    def __str__(self) -> str:
        lines: list[str] = []
        title = f"{self.framework} coverage for suite '{self.suite_name}'"
        lines.append(title)
        lines.append("─" * len(title))

        all_controls = _CATALOGS.get(self.framework, {}).get("measurable", {})
        for cid, control in all_controls.items():
            evs = self.covered.get(cid)
            if evs:
                lines.append(f"  [x] {control.id:<14} {control.description}")
                lines.append(f"      covered by: {', '.join(sorted(set(evs)))}")
            else:
                lines.append(f"  [ ] {control.id:<14} {control.description}  ← gap")

        if self.process:
            lines.append("")
            lines.append("  Process controls (not satisfiable by evaluators alone):")
            for c in self.process:
                lines.append(f"      {c.id:<14} {c.description}")

        total = len(all_controls)
        covered_count = sum(1 for cid in all_controls if cid in self.covered)
        lines.append("")
        lines.append(f"  Coverage: {covered_count}/{total} measurable controls exercised.")
        if self.missing:
            lines.append(f"  Gaps: {', '.join(c.id for c in self.missing)}")
        if self.unmapped_evaluators:
            lines.append(
                f"  Note: {len(self.unmapped_evaluators)} evaluator(s) have no mapping in this framework: "
                f"{', '.join(self.unmapped_evaluators)}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────────────────────────────────────

class ComplianceReporter:
    """
    Tamper-evident audit log for eval runs.

    Each `record()` appends one NDJSON line. The line is part of a SHA-256
    hash chain — `prev_hash` references the previous record's hash, and the
    final record_hash covers the whole payload. `verify()` walks the chain
    and reports any inconsistency.

    Args:
        output_dir: Directory to write audit logs (created if missing).
        framework:  "eu-ai-act" | "nist-ai-rmf" | "none"

    Files produced per suite:
        <output_dir>/<suite_name>.audit.ndjson   append-only chained log
        <output_dir>/<suite_name>.audit.sha256   running hash checkpoint (advisory)
    """

    def __init__(self, output_dir: str = "./audit-logs", framework: Framework = "eu-ai-act"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.framework = framework

    # ── recording ────────────────────────────────────────────────────────────

    def record(self, report: EvalReport, tags: dict[str, str] | None = None) -> str:
        """
        Append a chained audit record for this eval run.

        Returns the record_id (12-hex-char) for cross-reference.
        """
        record_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).isoformat()
        log_path = self._log_path(report.suite_name)
        hash_path = self._hash_path(report.suite_name)
        prev_hash = self._last_chain_hash(log_path)

        summary = {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "pass_rate": round(report.pass_rate, 4),
            "avg_score": round(report.avg_score, 4),
            "runs_per_case": report.runs_per_case,
            "flaky_count": report.flaky_count,
            "stability_score": round(report.stability_score, 4),
            "tags": tags or {},
        }

        evaluator_results: list[dict] = []
        for ev_name, score in report.scores_by_evaluator().items():
            entry: dict = {
                "evaluator": ev_name,
                "avg_score": round(score, 4),
                "pass_rate": round(report.passed_by_evaluator().get(ev_name, 0.0), 4),
            }
            controls = _controls_for(self.framework, ev_name)
            if controls:
                entry["controls"] = [{"id": c.id, "description": c.description} for c in controls]
            evaluator_results.append(entry)

        payload = {
            "record_id": record_id,
            "suite_name": report.suite_name,
            "model_id": report.model_id,
            "timestamp": timestamp,
            "framework": self.framework,
            "chain_version": _CHAIN_VERSION,
            "prev_hash": prev_hash,
            "summary": summary,
            "evaluator_results": evaluator_results,
        }
        record_hash = _hash_payload(payload)

        audit_record = AuditRecord(
            record_id=record_id,
            suite_name=report.suite_name,
            model_id=report.model_id,
            timestamp=timestamp,
            framework=self.framework,
            chain_version=_CHAIN_VERSION,
            prev_hash=prev_hash,
            summary=summary,
            evaluator_results=evaluator_results,
            record_hash=record_hash,
        )

        with open(log_path, "a") as f:
            f.write(audit_record.to_ndjson() + "\n")
        with open(hash_path, "a") as f:
            f.write(f"{record_hash}  {record_id}  {timestamp}\n")

        print(f"  [compliance] audit record → {record_id}  ({log_path.name})")
        if self.framework != "none":
            print(f"  [compliance] framework: {self.framework}")
        return record_id

    # ── verification ─────────────────────────────────────────────────────────

    def verify(self, suite_name: str) -> bool:
        """
        Walk the audit log for `suite_name` and verify the hash chain.

        Returns True iff every record is intact AND chain links are unbroken.
        Prints a per-record status line. Legacy unchained records (no
        chain_version field) are verified standalone, with a NOTE.
        """
        log_path = self._log_path(suite_name)
        if not log_path.exists():
            print(f"No audit log found: {log_path}")
            return False

        lines = log_path.read_text().strip().splitlines()
        prev_expected = _GENESIS_HASH
        all_ok = True

        for line in lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ERROR parsing record: {e}")
                all_ok = False
                continue

            record_id = data.get("record_id", "?")
            ts_short = data.get("timestamp", "")[:19]
            stored_hash = data.pop("record_hash", None)
            chain_version = data.get("chain_version")

            recomputed = _hash_payload(data)
            hash_ok = (stored_hash == recomputed)

            if chain_version is None:
                # Legacy unchained record — verify standalone hash only.
                status = "OK (legacy)" if hash_ok else "TAMPERED"
                if not hash_ok:
                    all_ok = False
                print(f"  {status}  {record_id}  {ts_short}")
                # Legacy records can't anchor a chain; reset expectation.
                prev_expected = recomputed if hash_ok else _GENESIS_HASH
                continue

            chain_ok = (data.get("prev_hash") == prev_expected)
            if hash_ok and chain_ok:
                status = "OK"
            elif not hash_ok:
                status = "TAMPERED"
                all_ok = False
            else:
                status = "CHAIN BROKEN"
                all_ok = False

            print(f"  {status}  {record_id}  {ts_short}")
            prev_expected = recomputed if hash_ok else _GENESIS_HASH

        print(f"\n  Verification: {'PASS — all records intact' if all_ok else 'FAIL — issues detected'}")
        return all_ok

    # ── coverage analysis ────────────────────────────────────────────────────

    def coverage(self, suite: "EvalSuite") -> CoverageReport:
        """
        Inspect a suite's evaluators and report which framework controls are
        exercised, which are gaps, and which require organizational measures
        outside of evaluation.
        """
        catalogs = _CATALOGS.get(self.framework, {"measurable": {}, "process": {}})
        measurable: dict[str, Control] = catalogs["measurable"]
        process: dict[str, Control] = catalogs["process"]
        mapping = _BY_EVALUATOR.get(self.framework, {})

        covered: dict[str, list[str]] = {}
        unmapped: list[str] = []
        for ev in suite._evaluators:
            ev_name = getattr(ev, "name", type(ev).__name__)
            control_ids = mapping.get(ev_name, [])
            if not control_ids and self.framework != "none":
                unmapped.append(ev_name)
            for cid in control_ids:
                covered.setdefault(cid, []).append(ev_name)

        missing = [ctrl for cid, ctrl in measurable.items() if cid not in covered]

        return CoverageReport(
            framework=self.framework,
            suite_name=suite.name,
            covered=covered,
            missing=missing,
            process=list(process.values()),
            unmapped_evaluators=sorted(set(unmapped)),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _log_path(self, suite_name: str) -> Path:
        return self.output_dir / f"{suite_name.replace(' ', '_')}.audit.ndjson"

    def _hash_path(self, suite_name: str) -> Path:
        return self.output_dir / f"{suite_name.replace(' ', '_')}.audit.sha256"

    @staticmethod
    def _last_chain_hash(log_path: Path) -> str:
        """Read the last record's record_hash; return genesis if log empty/missing."""
        if not log_path.exists():
            return _GENESIS_HASH
        last_line = ""
        with open(log_path, "rb") as f:
            for chunk in f:
                if chunk.strip():
                    last_line = chunk.decode("utf-8", errors="replace").strip()
        if not last_line:
            return _GENESIS_HASH
        try:
            return json.loads(last_line).get("record_hash", _GENESIS_HASH)
        except json.JSONDecodeError:
            return _GENESIS_HASH


def _hash_payload(payload: dict) -> str:
    """Canonical SHA-256 of the payload, excluding any embedded record_hash field."""
    sanitized = {k: v for k, v in payload.items() if k != "record_hash"}
    encoded = json.dumps(sanitized, separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(encoded).hexdigest()
