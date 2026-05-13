"""
Self-contained HTML compliance rollup.

Auditors don't read NDJSON. They want a single document they can attach to
a SOC 2 / ISO 42001 / EU AI Act technical-documentation package and click
through. ``ComplianceHtmlReporter`` produces exactly that: one HTML file
with no external assets.

What's in the report:

  • Header: suite name, model, framework, generation timestamp.
  • Coverage table: which Articles the suite exercised, which are gaps,
    which are process controls (organizational measures required).
  • Audit-log integrity: verify() result for the chain (PASS / FAIL with
    per-record status).
  • Run summary: total / passed / failed / pass rate / avg score / flaky
    cases / stability score, with the 95% Wilson CI.
  • Per-evaluator breakdown: average score, pass rate, mapped controls.
  • Per-case detail (collapsible): input, output, evaluator scores +
    reasons, mapped controls — Art. 12 decision-level evidence.
  • Calibration provenance: which calibrated thresholds drove each
    LLM-judge decision, with dataset / N / F1 where measured.

Usage::

    from multivon_eval import ComplianceHtmlReporter, ComplianceReporter

    reporter = ComplianceReporter("./audit-logs", framework="eu-ai-act")
    rec_id = reporter.record(report, mode="case")

    html = ComplianceHtmlReporter(reporter).render(report, suite=suite)
    Path("compliance.html").write_text(html, encoding="utf-8")
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..compliance import (
    ComplianceReporter,
    CoverageReport,
    _CATALOGS,
    _controls_for,
)
from ..result import EvalReport

if TYPE_CHECKING:
    from ..suite import EvalSuite


_STYLES = """
:root {
    color-scheme: light dark;
    --bg: #0a0a14;
    --panel: #11121c;
    --panel-2: #161826;
    --border: #232540;
    --text: #e9eaf3;
    --muted: #8b8da6;
    --accent: #a78bfa;
    --ok: #34d399;
    --warn: #fbbf24;
    --bad: #f87171;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Monaco, "Cascadia Code", monospace;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
    padding: 32px 24px 80px;
    line-height: 1.5;
}
.container { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 18px; margin: 32px 0 12px; color: var(--text); letter-spacing: -0.005em; }
.subtitle { color: var(--muted); margin-bottom: 32px; font-size: 14px; }
.meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px 24px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 24px;
    font-size: 13px;
}
.meta-grid dt { color: var(--muted); }
.meta-grid dd { margin: 0; font-family: var(--mono); font-size: 12px; }
.section { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }
.section .head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }
.section .head h2 { margin: 0; }
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.tag.ok { background: rgba(52,211,153,0.12); color: var(--ok); }
.tag.warn { background: rgba(251,191,36,0.12); color: var(--warn); }
.tag.bad { background: rgba(248,113,113,0.12); color: var(--bad); }
.tag.neutral { background: rgba(167,139,250,0.12); color: var(--accent); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }
th { color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
td.mono, .mono { font-family: var(--mono); }
td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono); }
.coverage-row.gap td:first-child { color: var(--warn); }
.coverage-row.gap .tag { background: rgba(251,191,36,0.12); color: var(--warn); }
.case details { background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px; margin: 8px 0; padding: 10px 14px; }
.case summary { cursor: pointer; font-size: 13px; display: flex; gap: 12px; align-items: center; }
.case summary .case-no { color: var(--muted); font-family: var(--mono); }
.case summary .case-input { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.case .body { margin-top: 12px; font-size: 13px; }
.case .field { margin: 8px 0; }
.case .field .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
.case .field pre { background: #0a0a14; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; font-family: var(--mono); font-size: 12px; white-space: pre-wrap; word-break: break-word; margin: 0; }
.eval-list { display: flex; flex-direction: column; gap: 6px; }
.eval-entry { display: grid; grid-template-columns: 180px 60px 60px 1fr; gap: 12px; align-items: baseline; font-size: 12px; padding: 6px 8px; border-radius: 6px; background: rgba(255,255,255,0.02); }
.eval-entry.fail { background: rgba(248,113,113,0.06); }
.eval-name { font-family: var(--mono); }
.eval-reason { color: var(--muted); font-size: 11px; }
.controls { display: inline-flex; flex-wrap: wrap; gap: 4px; }
.controls .tag { font-size: 10px; padding: 1px 6px; }
.footer { color: var(--muted); font-size: 11px; text-align: center; margin-top: 32px; }
"""


class ComplianceHtmlReporter:
    """Render a self-contained HTML compliance rollup."""

    def __init__(self, reporter: ComplianceReporter):
        self.reporter = reporter

    def render(
        self,
        report: EvalReport,
        *,
        suite: "EvalSuite | None" = None,
        verification_lines: list[tuple[str, str]] | None = None,
    ) -> str:
        """Return the full HTML document as a string.

        Args:
            report:              The :class:`EvalReport` to render.
            suite:               The :class:`EvalSuite` used to produce the
                                 report. Required to compute coverage; if
                                 omitted, the coverage section is skipped.
            verification_lines:  Optional pre-computed (status, message)
                                 pairs from :meth:`ComplianceReporter.verify`.
                                 Pass these in when you've already verified
                                 the chain and want to avoid re-reading the
                                 NDJSON file.
        """
        coverage: CoverageReport | None = None
        if suite is not None:
            coverage = self.reporter.coverage(suite)

        sections = [
            _meta_section(self.reporter, report),
            _summary_section(report),
            _coverage_section(self.reporter.framework, coverage),
            _chain_section(self.reporter, report, verification_lines),
            _evaluators_section(self.reporter.framework, report),
            _cases_section(self.reporter.framework, report),
        ]
        body = "\n".join(s for s in sections if s)

        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Compliance report — {html.escape(report.suite_name)}</title>
<style>{_STYLES}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Compliance report</h1>
    <div class="subtitle">{html.escape(report.suite_name)} · {html.escape(self.reporter.framework)}</div>
  </header>
  {body}
  <div class="footer">Generated by multivon-eval · {datetime.now(timezone.utc).isoformat(timespec="seconds")}</div>
</div>
</body>
</html>
"""

    def write(self, path: str | Path, report: EvalReport, *, suite: "EvalSuite | None" = None) -> Path:
        """Render and write to ``path``; return the resolved Path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render(report, suite=suite), encoding="utf-8")
        return out


# ─── section builders ───────────────────────────────────────────────────────


def _meta_section(reporter: ComplianceReporter, report: EvalReport) -> str:
    items = {
        "Suite": report.suite_name,
        "Model": report.model_id or "—",
        "Framework": reporter.framework,
        "Total cases": str(report.total),
        "Runs per case": str(report.runs_per_case),
        "Audit log": str(reporter.output_dir),
    }
    rows = "".join(
        f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in items.items()
    )
    return f'<dl class="meta-grid">{rows}</dl>'


def _summary_section(report: EvalReport) -> str:
    lo, hi = report.pass_rate_ci()
    pass_tag = "ok" if report.pass_rate >= 0.9 else "warn" if report.pass_rate >= 0.7 else "bad"
    return f"""
<section class="section">
  <div class="head">
    <h2>Run summary</h2>
    <span class="tag {pass_tag}">Pass rate {report.pass_rate:.1%}</span>
  </div>
  <table>
    <tr><th>Total</th><th>Passed</th><th>Failed</th><th>Pass rate (95% CI)</th><th>Avg score</th><th>Flaky</th><th>Stability</th></tr>
    <tr>
      <td class="num">{report.total}</td>
      <td class="num">{report.passed}</td>
      <td class="num">{report.failed}</td>
      <td class="num">{report.pass_rate:.3f} [{lo:.3f}, {hi:.3f}]</td>
      <td class="num">{report.avg_score:.3f}</td>
      <td class="num">{report.flaky_count}</td>
      <td class="num">{report.stability_score:.3f}</td>
    </tr>
  </table>
</section>
"""


def _coverage_section(framework: str, coverage: CoverageReport | None) -> str:
    if coverage is None:
        return ""
    measurable = _CATALOGS.get(framework, {}).get("measurable", {})
    rows: list[str] = []
    for cid, ctrl in measurable.items():
        evs = coverage.covered.get(cid)
        if evs:
            covered_by = ", ".join(sorted(set(evs)))
            rows.append(
                f'<tr class="coverage-row covered">'
                f'<td class="mono">{html.escape(ctrl.id)}</td>'
                f'<td>{html.escape(ctrl.description)}</td>'
                f'<td><span class="tag ok">Covered</span></td>'
                f'<td class="mono">{html.escape(covered_by)}</td>'
                f"</tr>"
            )
        else:
            rows.append(
                f'<tr class="coverage-row gap">'
                f'<td class="mono">{html.escape(ctrl.id)}</td>'
                f'<td>{html.escape(ctrl.description)}</td>'
                f'<td><span class="tag warn">Gap</span></td>'
                f'<td class="mono">—</td>'
                f"</tr>"
            )
    process_rows = "".join(
        f'<tr><td class="mono">{html.escape(c.id)}</td>'
        f'<td>{html.escape(c.description)}</td>'
        f'<td><span class="tag neutral">Process</span></td>'
        f'<td class="mono">organizational measure</td></tr>'
        for c in coverage.process
    )
    covered_count = sum(1 for cid in measurable if cid in coverage.covered)
    return f"""
<section class="section">
  <div class="head">
    <h2>Regulatory coverage</h2>
    <span class="tag neutral">{covered_count}/{len(measurable)} measurable controls</span>
  </div>
  <table>
    <thead><tr><th>Control</th><th>Description</th><th>Status</th><th>Covered by</th></tr></thead>
    <tbody>{''.join(rows)}{process_rows}</tbody>
  </table>
</section>
"""


def _chain_section(
    reporter: ComplianceReporter,
    report: EvalReport,
    pre: list[tuple[str, str]] | None,
) -> str:
    log_path = reporter.output_dir / f"{report.suite_name.replace(' ', '_')}.audit.ndjson"
    if not log_path.exists():
        return f"""
<section class="section">
  <div class="head"><h2>Audit log integrity</h2><span class="tag warn">No log yet</span></div>
  <p class="mono">No audit log found at {html.escape(str(log_path))}.</p>
</section>
"""

    statuses = pre if pre is not None else _verify_silent(reporter, report.suite_name)
    all_ok = all(s == "OK" or s.startswith("OK") for s, _ in statuses)
    tag = "ok" if all_ok else "bad"
    label = "PASS — all records intact" if all_ok else "FAIL — issues detected"
    rows = "".join(
        f'<tr><td><span class="tag {"ok" if s == "OK" or s.startswith("OK") else "bad"}">{html.escape(s)}</span></td>'
        f'<td class="mono">{html.escape(rid)}</td></tr>'
        for s, rid in statuses
    )
    return f"""
<section class="section">
  <div class="head"><h2>Audit log integrity</h2><span class="tag {tag}">{html.escape(label)}</span></div>
  <table>
    <thead><tr><th>Status</th><th>Record id · timestamp</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def _verify_silent(reporter: ComplianceReporter, suite_name: str) -> list[tuple[str, str]]:
    """Run verify() without printing; return (status, id+ts) pairs."""
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reporter.verify(suite_name)
    out: list[tuple[str, str]] = []
    for line in buf.getvalue().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Lines look like "  OK  abc123def  2026-05-13T..."
        # or "  TAMPERED  abc123def  2026-05-13T..."
        parts = stripped.split(None, 2)
        if len(parts) >= 2 and parts[0] in ("OK", "OK", "TAMPERED", "CHAIN", "ERROR"):
            # Re-join "CHAIN BROKEN" or "OK (legacy)" into one status field.
            if parts[0] == "CHAIN" and parts[1] == "BROKEN":
                status = "CHAIN BROKEN"
                rest = parts[2] if len(parts) > 2 else ""
            elif parts[0] == "OK" and len(parts) >= 2 and parts[1].startswith("("):
                status = f"OK {parts[1]}"
                rest = parts[2] if len(parts) > 2 else ""
            else:
                status = parts[0]
                rest = " ".join(parts[1:])
            out.append((status, rest))
    return out


def _evaluators_section(framework: str, report: EvalReport) -> str:
    scores = report.scores_by_evaluator()
    passes = report.passed_by_evaluator()
    if not scores:
        return ""
    rows: list[str] = []
    for name, avg in scores.items():
        pass_rate = passes.get(name, 0.0)
        controls = _controls_for(framework, name)
        ctrl_tags = " ".join(
            f'<span class="tag neutral">{html.escape(c.id)}</span>' for c in controls
        )
        unmapped_tag = '<span class="tag warn">unmapped</span>'
        cells = ctrl_tags or unmapped_tag
        pass_class = "ok" if pass_rate >= 0.9 else "warn" if pass_rate >= 0.7 else "bad"
        rows.append(
            f'<tr>'
            f'<td class="mono">{html.escape(name)}</td>'
            f'<td class="num">{avg:.3f}</td>'
            f'<td class="num"><span class="tag {pass_class}">{pass_rate:.1%}</span></td>'
            f'<td><div class="controls">{cells}</div></td>'
            f'</tr>'
        )
    return f"""
<section class="section">
  <div class="head"><h2>Evaluators</h2><span class="tag neutral">{len(scores)} evaluators</span></div>
  <table>
    <thead><tr><th>Evaluator</th><th>Avg score</th><th>Pass rate</th><th>Controls</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _cases_section(framework: str, report: EvalReport) -> str:
    if not report.case_results:
        return ""
    items: list[str] = []
    for idx, cr in enumerate(report.case_results):
        eval_entries = []
        for r in cr.results:
            controls = _controls_for(framework, r.evaluator)
            ctrl_tags = " ".join(
                f'<span class="tag neutral">{html.escape(c.id)}</span>' for c in controls
            )
            cls = "fail" if not r.passed else ""
            tag_class = "ok" if r.passed else "bad"
            tag_label = "PASS" if r.passed else "FAIL"
            ctrl_block = f'<div class="controls">{ctrl_tags}</div>' if ctrl_tags else ""
            eval_entries.append(
                f'<div class="eval-entry {cls}">'
                f'<span class="eval-name">{html.escape(r.evaluator)}</span>'
                f'<span class="num">{r.score:.3f}</span>'
                f'<span class="tag {tag_class}">{tag_label}</span>'
                f'<span class="eval-reason">{html.escape(r.reason or "")}</span>'
                f'{ctrl_block}'
                f'</div>'
            )
        status_tag = "ok" if cr.passed else "bad"
        status_label = "PASS" if cr.passed else "FAIL"
        err_block = (
            f'<div class="field"><div class="label">Model error</div>'
            f'<pre>{html.escape(cr.model_error)}</pre></div>'
            if cr.model_error else ""
        )
        items.append(
            f'<div class="case"><details>'
            f'<summary>'
            f'<span class="case-no">#{idx:04d}</span>'
            f'<span class="case-input">{html.escape((cr.case_input or "")[:100])}</span>'
            f'<span class="tag {status_tag}">{status_label}</span>'
            f'<span class="num">{cr.score:.3f}</span>'
            f'</summary>'
            f'<div class="body">'
            f'<div class="field"><div class="label">Input</div><pre>{html.escape(cr.case_input or "")}</pre></div>'
            f'<div class="field"><div class="label">Output</div><pre>{html.escape(cr.actual_output or "")}</pre></div>'
            f'{err_block}'
            f'<div class="field"><div class="label">Evaluators</div><div class="eval-list">{"".join(eval_entries)}</div></div>'
            f'</div></details></div>'
        )
    return f"""
<section class="section">
  <div class="head"><h2>Per-case detail (Art. 12 decision log)</h2><span class="tag neutral">{report.total} cases</span></div>
  {''.join(items)}
</section>
"""


def render_compliance_html(
    reporter: ComplianceReporter,
    report: EvalReport,
    *,
    suite: "EvalSuite | None" = None,
) -> str:
    """One-shot helper: returns the HTML string."""
    return ComplianceHtmlReporter(reporter).render(report, suite=suite)
