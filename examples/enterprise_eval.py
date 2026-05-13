"""
Enterprise eval pipeline — every feature added in v0.6 in one offline demo.

What this exercises (no API keys required):

  • Structured exceptions       — catch by class, not by string match
  • Secrets resolver            — env or your own backend, plugged in
  • Async + concurrent run      — suite.run_async overlaps cases + evaluators
  • Judge cache (sqlite)        — second identical call is a hit
  • Calibration provenance      — every threshold carries dataset/N/F1/date
  • Per-case audit records      — EU AI Act Art. 12 decision-level logging
  • External anchor             — tip hash shipped to GitHub Actions output
  • HTML compliance rollup      — auditor-attachable single-file report

Run::

    python examples/enterprise_eval.py
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from multivon_eval import (
    ComplianceHtmlReporter,
    ComplianceReporter,
    EvalCase,
    EvalSuite,
    JudgeConfig,
    StaticResolver,
    calibration_provenance,
    get_secret,
    github_actions_anchor,
    set_resolver,
)
from multivon_eval.evaluators.deterministic import NotEmpty, MaxLatency
from multivon_eval.exceptions import JudgeUnavailable, SecretsError
from multivon_eval.result import CaseResult, EvalReport, EvalResult


def section(title: str) -> None:
    print()
    print("─" * 64)
    print(title)
    print("─" * 64)


# ── 1. Pluggable secrets (no env contamination required) ────────────────────
section("1. Secrets resolver")

set_resolver(StaticResolver({"DEMO_API_KEY": "sk-fake-demo-1234"}))
print("  DEMO_API_KEY     :", get_secret("DEMO_API_KEY"))
print("  MISSING (default):", get_secret("MISSING", default="<unset>"))

try:
    get_secret("MISSING", required=True)
except SecretsError as exc:
    print("  Required-missing raises SecretsError →", type(exc).__name__)


# ── 2. Calibration provenance (auditable thresholds) ────────────────────────
section("2. Calibration provenance")

cfg = JudgeConfig(provider="anthropic", model="claude-haiku-4-5-20251001").resolve()
entry = calibration_provenance("hallucination", cfg)
print(f"  hallucination + {cfg.model}")
print(f"    threshold     : {entry.threshold}")
print(f"    dataset       : {entry.dataset}  ({entry.n} cases)")
print(f"    f1            : {entry.f1}")
print(f"    measured_at   : {entry.measured_at}")


# ── 3. Structured exceptions ────────────────────────────────────────────────
section("3. Structured exceptions")

bad = JudgeUnavailable("simulated 503", provider="openai", model="gpt-4o-mini")
print(f"  class={type(bad).__name__}  provider={bad.provider}  model={bad.model}")


# ── 4. Async + concurrent evaluation ────────────────────────────────────────
section("4. Async + concurrent evaluation")


async def fake_model(prompt: str) -> str:
    # Simulate a 50ms model call.
    await asyncio.sleep(0.05)
    return f"Refunds are processed within 14 days. (re: {prompt[:20]})"


async def run_async_demo() -> EvalReport:
    suite = EvalSuite("Async demo", model_id="fake-model-v1")
    suite.add_cases([
        EvalCase(input=f"Question {i}", tags=["demo"])
        for i in range(5)
    ])
    suite.add_evaluators(NotEmpty(), MaxLatency(max_ms=10_000))
    return await suite.run_async(fake_model, verbose=False, concurrency=5)


report = asyncio.run(run_async_demo())
print(f"  cases    : {report.total}")
print(f"  pass rate: {report.pass_rate:.1%}")
print(f"  avg score: {report.avg_score:.3f}")


# ── 5. Per-case audit records + external anchor + HTML rollup ───────────────
section("5. Compliance — per-case audit, anchor, HTML rollup")

# Synthesise a richer report so the HTML rollup has interesting content.
report = EvalReport(
    suite_name="Enterprise Demo Suite",
    model_id="fake-model-v1",
    case_results=[
        CaseResult(
            case_input="Summarize the refund policy.",
            actual_output="Refunds are processed within 14 days of receipt.",
            results=[
                EvalResult(evaluator="faithfulness",   score=0.92, passed=True, reason="all claims supported"),
                EvalResult(evaluator="hallucination",  score=0.96, passed=True),
                EvalResult(evaluator="relevance",      score=0.88, passed=True),
                EvalResult(evaluator="pii_detection",  score=1.0,  passed=True),
                EvalResult(evaluator="not_empty",      score=1.0,  passed=True),
            ],
            latency_ms=120.0,
        ),
        CaseResult(
            case_input="Tell me an unsupported fact.",
            actual_output="The refund window is 365 days.",
            results=[
                EvalResult(evaluator="faithfulness",   score=0.25, passed=False, reason="365 days not in context"),
                EvalResult(evaluator="hallucination",  score=0.30, passed=False),
                EvalResult(evaluator="relevance",      score=0.70, passed=True),
                EvalResult(evaluator="pii_detection",  score=1.0,  passed=True),
                EvalResult(evaluator="not_empty",      score=1.0,  passed=True),
            ],
            latency_ms=145.0,
        ),
    ],
)

audit_dir = Path("./audit-logs/enterprise_demo")
audit_dir.mkdir(parents=True, exist_ok=True)
for f in audit_dir.glob("*"):
    f.unlink()

anchored_tips: list[str] = []


def demo_anchor(tip_hash: str) -> None:
    """A real anchor would ship to Sigstore Rekor / S3 Object Lock / GHA output.
    For this demo we just collect the tip hashes in memory."""
    anchored_tips.append(tip_hash)
    github_actions_anchor(tip_hash)  # no-op without $GITHUB_OUTPUT, safe to call


reporter = ComplianceReporter(
    output_dir=str(audit_dir),
    framework="eu-ai-act",
    anchor_fn=demo_anchor,
    verbose=False,
)

# Per-case mode (Art. 12 decision-level)
record_ids = reporter.record(report, mode="case", tags={"system": "demo-bot", "version": "1.0"})
print(f"  per-case records : {len(record_ids)} written")
print(f"  anchored tip     : {anchored_tips[-1][:16]}…")

# Verify the chain
ok = reporter.verify(report.suite_name)
print(f"  verify           : {'PASS' if ok else 'FAIL'}")

# Render the HTML rollup
suite = EvalSuite.eu_ai_act_high_risk()
out_html = ComplianceHtmlReporter(reporter).write(
    audit_dir / "compliance.html",
    report,
    suite=suite,
)
print(f"  html rollup      : {out_html} ({out_html.stat().st_size} bytes)")


# ── 6. Judge cache demo (no API call required) ──────────────────────────────
section("6. Judge cache")

with tempfile.TemporaryDirectory() as td:
    from multivon_eval import JudgeCache, set_cache

    cache = JudgeCache(db_path=Path(td) / "judge.db")
    set_cache(cache)

    judge_cfg = JudgeConfig(provider="openai", model="gpt-4o-mini", cache=True).resolve()
    print(f"  initial size : {cache.size()}")
    cache.put("Is this faithful?", judge_cfg, "Yes.")
    print(f"  after put    : {cache.size()}")
    print(f"  hit          : {cache.get('Is this faithful?', judge_cfg)!r}")
    print(f"  miss         : {cache.get('different prompt', judge_cfg)}")
    print(f"  stats        : {cache.stats.as_dict()}")
    set_cache(None)


# ── 7. Tear-down ────────────────────────────────────────────────────────────
section("All set")
print(f"  audit log    : {audit_dir/(report.suite_name.replace(' ', '_'))}.audit.ndjson")
print(f"  hash chain   : {audit_dir/(report.suite_name.replace(' ', '_'))}.audit.sha256")
print(f"  html report  : {out_html}")
print()
print("  Open the HTML file in a browser to see the auditor-grade rollup.")
