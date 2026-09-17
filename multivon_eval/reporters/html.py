"""
Self-contained HTML report generator for EvalReport.

Produces a single .html file with no external dependencies —
dark theme, per-evaluator breakdown, per-case expandable table,
multi-run flakiness indicators.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..result import CaseResult, EvalReport

__all__ = ["to_html"]

from .evidence_html import case_filters, coverage_warning, trial_details
from .html_assets import _CSS, _JS


def _score_class(score: float) -> str:
    if score >= 0.7:
        return "s-green"
    if score >= 0.5:
        return "s-yellow"
    return "s-red"


def _status_pill(cr: CaseResult) -> str:
    """Render a status badge for one case.

    Surfaces the 0.7.0 EvalStatus enum so a reader sees at a glance
    whether a case PASSED, failed on QUALITY (a real model regression
    to investigate), errored on infrastructure (judge outage, model
    crash — retry-class, not a quality issue), or was deliberately
    skipped.

    Precedence: errors/skipped first; flaky only modifies pass/fail
    (a flaky outcome on top of a judge outage is misleading — the
    underlying signal is the outage).

    Each pill carries a tooltip AND an ``aria-label`` so the
    explanation reaches keyboard/touch/screen-reader users who
    can't hover the native ``title``.
    """
    from ..result import EvalStatus

    def _pill(cls: str, label: str, explanation: str | None = None) -> str:
        if not explanation:
            return f'<span class="pill {cls}">{label}</span>'
        safe = _h(explanation)
        return (
            f'<span class="pill {cls}" title="{safe}" aria-label="{label}: {safe}">'
            f'{label}</span>'
        )

    status = cr.status

    # Infra failures and skips dominate any per-run flakiness signal —
    # if the judge was unreachable, "flaky" is not the right framing.
    if status == EvalStatus.SKIPPED:
        return _pill("skipped", "SKIPPED", "Case was deliberately skipped")
    if status in (EvalStatus.MODEL_ERROR, EvalStatus.JUDGE_ERROR,
                  EvalStatus.EVALUATOR_ERROR, EvalStatus.TIMEOUT):
        label_map = {
            EvalStatus.MODEL_ERROR: ("MODEL ERR",
                                     "Your model_fn raised — not a quality issue"),
            EvalStatus.JUDGE_ERROR: ("JUDGE ERR",
                                     "Judge call failed (transient/auth) — not a quality issue"),
            EvalStatus.EVALUATOR_ERROR: ("EVAL ERR",
                                         "An evaluator itself crashed — likely a bug to file"),
            EvalStatus.TIMEOUT: ("TIMEOUT", "Case timed out"),
        }
        label, tooltip = label_map[status]
        return _pill("error", label, tooltip)

    # Quality outcome. Flakiness overrides only here — it's only
    # meaningful when the case actually completed evaluation across
    # multiple runs.
    if cr.is_flaky:
        return _pill("flaky", "FLAKY", "Case passed inconsistently across runs")
    if status == EvalStatus.PASSED:
        return _pill("pass", "PASS")
    return _pill("fail", "FAIL", "Quality threshold not met")


def _h(text: str) -> str:
    return html.escape(str(text))


def _truncate(text: str, n: int = 120) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "…"


def to_html(report: EvalReport) -> str:
    multi_run = report.runs_per_case > 1
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    meta_parts = []
    if report.model_id:
        meta_parts.append(f"Model: {_h(report.model_id)}")
    if multi_run:
        meta_parts.append(f"{report.runs_per_case} runs per case")
    meta_parts.append(f"Generated {ts}")
    meta_html = " &nbsp;·&nbsp; ".join(meta_parts)

    # ── Summary cards ─────────────────────────────────────────
    pr_pct = f"{report.pass_rate:.0%}"
    avg = f"{report.avg_score:.2f}"

    def card(val: str, lbl: str, cls: str = "") -> str:
        return (
            f'<div class="card {cls}">'
            f'<span class="val">{val}</span>'
            f'<span class="lbl">{lbl}</span>'
            f'</div>'
        )

    pr_cls = "c-pass" if report.pass_rate >= 0.8 else "c-warn" if report.pass_rate >= 0.5 else "c-fail"
    avg_cls = _score_class(report.avg_score)

    cards = [
        card(str(report.total), "Total"),
        card(str(report.passed), "Passed", "c-pass"),
        card(str(report.failed), "Failed", "c-fail"),
        card(pr_pct, "Pass Rate", pr_cls),
        card(avg, "Avg Score", f"c-accent score {avg_cls}"),
    ]
    # 0.7.0 — surface infrastructure errors as a separate count whenever
    # any are present, so readers can't conflate them with quality
    # failures. Tooltip explains how the count maps to status kinds.
    if report.errors:
        kinds = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in report.errors_by_kind.items())
        tooltip = f"Infrastructure failures (not quality): {kinds}"
        # title for hover, aria-label for keyboard / touch / screen readers.
        cards.append(
            f'<div class="card c-warn" title="{_h(tooltip)}" '
            f'aria-label="{report.errors} errors. {_h(tooltip)}">'
            f'<span class="val">{report.errors}</span>'
            f'<span class="lbl">Errors</span></div>'
        )
    if report.skipped:
        cards.append(card(str(report.skipped), "Skipped"))
    if multi_run:
        stab_cls = "c-pass" if report.stability_score >= 0.9 else "c-warn" if report.stability_score >= 0.7 else "c-fail"
        cards.append(card(f"{report.stability_score:.0%}", "Stability", stab_cls))
        cards.append(card(str(report.flaky_count), "Flaky", "c-warn" if report.flaky_count > 0 else ""))
    summary_html = '<div class="summary">' + "".join(cards) + '</div>'

    # ── Pass-rate bar (+ Wilson CI, matching console/JSON output) ─
    bar_pct = int(report.pass_rate * 100)
    ci_html = ""
    if report.evaluated > 0:
        ci_lo, ci_hi = report.pass_rate_ci()
        ci_html = (
            f'<div style="color:var(--muted);font-size:12px;margin-top:4px">'
            f'95% CI (Wilson): [{ci_lo:.1%}, {ci_hi:.1%}] '
            f'over {report.evaluated} evaluated case(s)</div>'
        )
    bar_html = (
        f'<div class="bar-wrap"><div class="bar-fill" style="width:{bar_pct}%"></div></div>'
        f'{ci_html}'
    )

    # ── Flaky callout ─────────────────────────────────────────
    flaky_html = ""
    if multi_run and report.flaky_count > 0:
        flaky_cases = [cr for cr in report.case_results if cr.is_flaky]
        items = "".join(
            f'<li>{_h(_truncate(cr.case_input, 80))} &nbsp;<span style="color:var(--yellow)">({cr.pass_count}/{cr.runs} runs passed)</span></li>'
            for cr in flaky_cases
        )
        flaky_html = (
            f'<div class="flaky-callout">'
            f'<strong>⚠ {report.flaky_count} flaky case(s)</strong> — passed inconsistently across {report.runs_per_case} runs'
            f'<ul class="flaky-list">{items}</ul>'
            f'</div>'
        )

    # ── Per-evaluator table ───────────────────────────────────
    ev_scores = report.scores_by_evaluator()
    ev_pass = report.passed_by_evaluator()
    ev_rows = ""
    for name, score in ev_scores.items():
        pass_rate = ev_pass.get(name, 0.0)
        sc = _score_class(score)
        ev_rows += (
            f'<tr>'
            f'<td>{_h(name)}</td>'
            f'<td class="r"><span class="score {sc}">{score:.2f}</span></td>'
            f'<td class="r">{pass_rate:.0%}</td>'
            f'</tr>'
        )
    ev_section = ""
    if ev_rows:
        ev_section = (
            f'<section>'
            f'<h2>By Evaluator</h2>'
            f'<table>'
            f'<thead><tr><th>Evaluator</th><th class="r">Avg Score</th><th class="r">Pass Rate</th></tr></thead>'
            f'<tbody>{ev_rows}</tbody>'
            f'</table>'
            f'</section>'
        )

    # ── Per-tag breakdown ─────────────────────────────────────
    tag_scores = report.scores_by_tag()
    tag_pass = report.passed_by_tag()
    tag_count = report.count_by_tag()
    tag_section = ""
    if tag_scores:
        tag_rows = ""
        for tag, score in sorted(tag_scores.items()):
            pass_rate = tag_pass.get(tag, 0.0)
            n = tag_count.get(tag, 0)
            sc = _score_class(score)
            tag_rows += (
                f'<tr>'
                f'<td><span style="font-size:12px;color:var(--accent-lt);background:rgba(124,58,237,.12);padding:2px 8px;border-radius:4px">{_h(tag)}</span></td>'
                f'<td class="r" style="color:var(--muted)">{n}</td>'
                f'<td class="r"><span class="score {sc}">{score:.2f}</span></td>'
                f'<td class="r">{pass_rate:.0%}</td>'
                f'</tr>'
            )
        tag_section = (
            f'<section>'
            f'<h2>By Tag</h2>'
            f'<table>'
            f'<thead><tr><th>Tag</th><th class="r">Cases</th><th class="r">Avg Score</th><th class="r">Pass Rate</th></tr></thead>'
            f'<tbody>{tag_rows}</tbody>'
            f'</table>'
            f'</section>'
        )

    # ── Per-case table ────────────────────────────────────────
    case_rows = ""
    for i, cr in enumerate(report.case_results):
        sc = _score_class(cr.score)
        std_html = (
            f'<span class="score-std">±{cr.score_std:.2f}</span>'
            if multi_run and cr.score_std > 0
            else ""
        )
        score_cell = f'<span class="score {sc}">{cr.score:.2f}</span>{std_html}'
        if cr.status.value not in {'passed', 'failed_quality'}:
            score_cell = '<span class="meta">unmeasured</span>'

        extra_cells = ""
        if multi_run:
            pr_c = "s-green" if cr.run_pass_rate >= 0.8 else "s-yellow" if cr.run_pass_rate >= 0.4 else "s-red"
            stab_pill = (
                '<span class="pill flaky">flaky</span>'
                if cr.is_flaky
                else '<span style="color:var(--green);font-size:12px">stable</span>'
            )
            extra_cells = (
                f'<td data-label="Run pass rate" class="r"><span class="score {pr_c}">{cr.run_pass_rate:.0%}</span></td>'
                f'<td data-label="Variability">{stab_pill}</td>'
            )

        tags_html = ""
        if cr.tags:
            tags_html = " ".join(
                f'<span style="font-size:11px;color:var(--accent-lt);background:rgba(124,58,237,.12);padding:1px 6px;border-radius:4px">{_h(t)}</span>'
                for t in cr.tags
            )

        # Main row
        case_rows += (
            f'<tr class="case-row" data-index="{i}" data-tags="{_h(json.dumps(cr.tags))}" data-status="{cr.status.value}" data-search="{_h(cr.case_input + " " + (cr.case_id or ""))}">'
            f'<td><button type="button" id="toggle-{i}" aria-expanded="false" aria-controls="d-{i}" onclick="toggle({i})">Case {i + 1} details</button></td>'
            f'<td data-label="Input" style="max-width:200px;word-break:break-word">{_h(_truncate(cr.case_input, 100))}</td>'
            f'<td data-label="Output" style="max-width:200px;word-break:break-word">{_h(_truncate(cr.actual_output, 100))}</td>'
            f'<td data-label="Score" class="r">{score_cell}</td>'
            f'{extra_cells}'
            f'<td data-label="Status">{_status_pill(cr)}</td>'
            f'<td data-label="Latency" class="r" style="color:var(--muted)">{"unknown" if cr.trials and cr.trials[-1].data["latency_ms"] is None else f"{cr.latency_ms:.0f}ms"}</td>'
            f'</tr>'
        )

        # Detail row (hidden by default)
        eval_rows = ""
        for r in cr.results:
            r_sc = _score_class(r.score)
            r_pass = ('<span class="pill">SKIPPED</span>' if r.metadata.get("skipped")
                      else '<span class="pill pass">✓</span>' if r.passed
                      else '<span class="pill fail">✗</span>')
            reason_cell = f'<span class="reason-text">{_h(r.reason)}</span>' if r.reason else '<span style="color:var(--muted)">—</span>'
            measured_score = '—' if r.metadata.get('skipped') or r.metadata.get('error_kind') else f'{r.score:.2f}'
            eval_rows += (
                f'<tr>'
                f'<td>{_h(r.evaluator)}</td>'
                f'<td class="r"><span class="score {r_sc}">{measured_score}</span></td>'
                f'<td class="r">{r_pass}</td>'
                f'<td>{reason_cell}</td>'
                f'</tr>'
            )

        colspan = 6 + (2 if multi_run else 0)
        detail_content = (
            f'<div class="detail-inner">'
            f'<div class="detail-block"><h3>Input</h3><div class="detail-text">{_h(cr.case_input)}</div></div>'
            f'<div class="detail-block"><h3>Output</h3><div class="detail-text">{_h(cr.actual_output)}</div></div>'
            f'</div>'
        )
        if eval_rows:
            detail_content += (
                f'<div style="padding:0 16px 16px">'
                f'<table class="eval-detail-table">'
                f'<thead><tr><th>Evaluator</th><th class="r">Score</th><th class="r">Pass</th><th>Reason</th></tr></thead>'
                f'<tbody>{eval_rows}</tbody>'
                f'</table>'
                f'</div>'
            )
        if tags_html:
            detail_content += f'<div style="padding:0 16px 14px">{tags_html}</div>'
        detail_content += trial_details(cr, prefix=f'case-{i}')

        case_rows += (
            f'<tr class="detail-row" id="d-{i}" hidden>'
            f'<td colspan="{colspan}">{detail_content}</td>'
            f'</tr>'
        )

    col_span_extra = '<th class="r">Pass Rate</th><th>Stability</th>' if multi_run else ""
    cases_section = (
        f'<section>'
        f'<h2>Cases</h2>{case_filters(report)}'
        f'{flaky_html}'
        f'<table class="case-table">'
        f'<thead><tr>'
        f'<th>#</th>'
        f'<th>Input</th>'
        f'<th>Output</th>'
        f'<th class="r">Score</th>'
        f'{col_span_extra}'
        f'<th>Status</th>'
        f'<th class="r">Latency</th>'
        f'</tr></thead>'
        f'<tbody>{case_rows}</tbody>'
        f'</table>'
        f'</section>'
    )

    title = _h(report.suite_name)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} — multivon-eval</title>
  <style>{_CSS}</style>
</head>
<body>
  <main>
  <header>
    <h1>{title}</h1>
    <p class="meta">{meta_html}</p>
  </header>
  {coverage_warning(report)}
  {summary_html}
  {bar_html}
  <br>
  {ev_section}
  {tag_section}
  {cases_section}
  </main>
  <footer class="footer">
    Generated by <a href="https://multivon.ai" target="_blank" rel="noopener">multivon-eval</a>
  </footer>
  <script>{_JS}</script>
</body>
</html>"""
