"""
Self-contained HTML report generator for EvalReport.

Produces a single .html file with no external dependencies —
dark theme, per-evaluator breakdown, per-case expandable table,
multi-run flakiness indicators.
"""
from __future__ import annotations
import html
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..result import EvalReport, CaseResult

__all__ = ["to_html"]

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0d0d12;
  --bg-card:   #14141c;
  --bg-table:  #111118;
  --bg-detail: #0b0b10;
  --border:    rgba(255,255,255,0.06);
  --text:      #e2e8f0;
  --muted:     rgba(255,255,255,0.35);
  --accent:    #7c3aed;
  --accent-lt: #a78bfa;
  --green:     #22c55e;
  --yellow:    #f59e0b;
  --orange:    #fb923c;   /* infra errors — distinct from quality failures */
  --red:       #ef4444;
  --slate:     #94a3b8;   /* skipped cases — neutral, not a failure */
  --radius:    10px;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  font-size: 14px;
  line-height: 1.5;
  padding: 32px 24px 80px;
}

a { color: var(--accent-lt); }

header {
  margin-bottom: 32px;
}
header h1 {
  font-size: 22px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 4px;
}
.meta {
  color: var(--muted);
  font-size: 12px;
}

/* ── Summary cards ────────────────────────────────────────── */
.summary {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 32px;
}
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 20px;
  min-width: 110px;
  text-align: center;
}
.card .val {
  font-size: 24px;
  font-weight: 700;
  color: #fff;
  display: block;
}
.card .lbl {
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .05em;
  margin-top: 2px;
  display: block;
}
.card.c-pass .val { color: var(--green); }
.card.c-fail .val { color: var(--red); }
.card.c-warn .val { color: var(--yellow); }
.card.c-accent .val { color: var(--accent-lt); }

/* ── Sections ─────────────────────────────────────────────── */
section { margin-bottom: 40px; }
section h2 {
  font-size: 13px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .07em;
  margin-bottom: 12px;
}

/* ── Tables ───────────────────────────────────────────────── */
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--bg-table);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
th {
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .06em;
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
}
th.r, td.r { text-align: right; }
td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
  color: var(--text);
}
tr:last-child td { border-bottom: none; }
tr.case-row { cursor: pointer; }
tr.case-row:hover td { background: rgba(255,255,255,0.025); }

/* ── Score badges ─────────────────────────────────────────── */
.score {
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.score.s-green { color: var(--green); }
.score.s-yellow { color: var(--yellow); }
.score.s-red { color: var(--red); }
.score-std { color: var(--muted); font-size: 12px; margin-left: 3px; }

/* ── Status pills ─────────────────────────────────────────── */
.pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 9999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .04em;
}
.pill.pass    { background: rgba(34,197,94,.15);  color: var(--green); }
.pill.fail    { background: rgba(239,68,68,.15);  color: var(--red); }
.pill.flaky   { background: rgba(245,158,11,.15); color: var(--yellow); }
/* 0.7.0 — infra errors are NOT quality failures; distinct color so the
   reader doesn't confuse a transient outage with a model regression. */
.pill.error   { background: rgba(251,146,60,.18); color: var(--orange); }
.pill.skipped { background: rgba(148,163,184,.18); color: var(--slate); }
.pill[title]  { cursor: help; border-bottom: 1px dotted currentColor; }

/* ── Detail rows ─────────────────────────────────────────── */
tr.detail-row td {
  padding: 0;
  background: var(--bg-detail);
}
.detail-inner {
  padding: 12px 16px 16px;
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
}
.detail-block { flex: 1; min-width: 280px; }
.detail-block h4 {
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .06em;
  margin-bottom: 8px;
}
.detail-text {
  font-size: 13px;
  color: rgba(255,255,255,0.75);
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  white-space: pre-wrap;
  word-break: break-word;
}
.eval-detail-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.eval-detail-table th {
  font-size: 11px;
  padding: 6px 10px;
}
.eval-detail-table td {
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
}
.eval-detail-table tr:last-child td { border-bottom: none; }
.reason-text {
  color: var(--muted);
  font-size: 12px;
  max-width: 420px;
}

/* ── Flaky callout ────────────────────────────────────────── */
.flaky-callout {
  background: rgba(245,158,11,.07);
  border: 1px solid rgba(245,158,11,.2);
  border-radius: var(--radius);
  padding: 14px 18px;
  margin-bottom: 20px;
  font-size: 13px;
}
.flaky-callout strong { color: var(--yellow); }
.flaky-list { margin-top: 8px; list-style: none; }
.flaky-list li { color: var(--muted); margin-top: 4px; }
.flaky-list li::before { content: "• "; color: var(--yellow); }

/* ── Progress bar ─────────────────────────────────────────── */
.bar-wrap {
  height: 4px;
  background: rgba(255,255,255,0.07);
  border-radius: 9999px;
  overflow: hidden;
  margin-top: 20px;
  max-width: 520px;
}
.bar-fill {
  height: 100%;
  border-radius: 9999px;
  background: var(--accent);
  transition: width .3s;
}

/* ── Footer ───────────────────────────────────────────────── */
.footer {
  margin-top: 48px;
  text-align: center;
  color: var(--muted);
  font-size: 11px;
}
.footer a { color: var(--muted); }
"""

_JS = """
function toggle(id) {
  var row = document.getElementById('d-' + id);
  if (!row) return;
  var hidden = row.getAttribute('hidden') !== null;
  if (hidden) {
    row.removeAttribute('hidden');
  } else {
    row.setAttribute('hidden', '');
  }
}
"""


def _score_class(score: float) -> str:
    if score >= 0.7:
        return "s-green"
    if score >= 0.5:
        return "s-yellow"
    return "s-red"


def _status_pill(cr: "CaseResult") -> str:
    """Render a status badge for one case.

    Surfaces the 0.7.0 EvalStatus enum so a reader sees at a glance
    whether a case PASSED, failed on QUALITY (a real model regression
    to investigate), errored on infrastructure (judge outage, model
    crash — retry-class, not a quality issue), or was deliberately
    skipped.

    Precedence: errors/skipped first; flaky only modifies pass/fail
    (a flaky outcome on top of a judge outage is misleading — the
    underlying signal is the outage). Codex round-2 caught the
    earlier ordering hiding infra errors behind a FLAKY badge.

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


def to_html(report: "EvalReport") -> str:
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
    case_header_extra = ""
    if multi_run:
        case_header_extra = '<th class="r">Pass Rate</th><th>Stability</th>'

    case_rows = ""
    for i, cr in enumerate(report.case_results):
        sc = _score_class(cr.score)
        std_html = (
            f'<span class="score-std">±{cr.score_std:.2f}</span>'
            if multi_run and cr.score_std > 0
            else ""
        )
        score_cell = f'<span class="score {sc}">{cr.score:.2f}</span>{std_html}'

        extra_cells = ""
        if multi_run:
            pr_c = "s-green" if cr.run_pass_rate >= 0.8 else "s-yellow" if cr.run_pass_rate >= 0.4 else "s-red"
            stab_pill = (
                '<span class="pill flaky">flaky</span>'
                if cr.is_flaky
                else '<span style="color:var(--green);font-size:12px">stable</span>'
            )
            extra_cells = (
                f'<td class="r"><span class="score {pr_c}">{cr.run_pass_rate:.0%}</span></td>'
                f'<td>{stab_pill}</td>'
            )

        tags_html = ""
        if cr.tags:
            tags_html = " ".join(
                f'<span style="font-size:11px;color:var(--accent-lt);background:rgba(124,58,237,.12);padding:1px 6px;border-radius:4px">{_h(t)}</span>'
                for t in cr.tags
            )

        # Main row
        case_rows += (
            f'<tr class="case-row" onclick="toggle({i})">'
            f'<td style="color:var(--muted)">{i + 1}</td>'
            f'<td style="max-width:200px;word-break:break-word">{_h(_truncate(cr.case_input, 100))}</td>'
            f'<td style="max-width:200px;word-break:break-word">{_h(_truncate(cr.actual_output, 100))}</td>'
            f'<td class="r">{score_cell}</td>'
            f'{extra_cells}'
            f'<td>{_status_pill(cr)}</td>'
            f'<td class="r" style="color:var(--muted)">{cr.latency_ms:.0f}ms</td>'
            f'</tr>'
        )

        # Detail row (hidden by default)
        eval_rows = ""
        for r in cr.results:
            r_sc = _score_class(r.score)
            r_pass = '<span class="pill pass">✓</span>' if r.passed else '<span class="pill fail">✗</span>'
            reason_cell = f'<span class="reason-text">{_h(r.reason[:300])}</span>' if r.reason else '<span style="color:var(--muted)">—</span>'
            eval_rows += (
                f'<tr>'
                f'<td>{_h(r.evaluator)}</td>'
                f'<td class="r"><span class="score {r_sc}">{r.score:.2f}</span></td>'
                f'<td class="r">{r_pass}</td>'
                f'<td>{reason_cell}</td>'
                f'</tr>'
            )

        colspan = 6 + (2 if multi_run else 0)
        detail_content = (
            f'<div class="detail-inner">'
            f'<div class="detail-block"><h4>Input</h4><div class="detail-text">{_h(cr.case_input)}</div></div>'
            f'<div class="detail-block"><h4>Output</h4><div class="detail-text">{_h(cr.actual_output)}</div></div>'
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

        case_rows += (
            f'<tr class="detail-row" id="d-{i}" hidden>'
            f'<td colspan="{colspan}">{detail_content}</td>'
            f'</tr>'
        )

    col_span_extra = '<th class="r">Pass Rate</th><th>Stability</th>' if multi_run else ""
    cases_section = (
        f'<section>'
        f'<h2>Cases <span style="color:var(--muted);font-size:11px;font-weight:400">— click a row to expand</span></h2>'
        f'{flaky_html}'
        f'<table>'
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
  <header>
    <h1>{title}</h1>
    <p class="meta">{meta_html}</p>
  </header>
  {summary_html}
  {bar_html}
  <br>
  {ev_section}
  {tag_section}
  {cases_section}
  <div class="footer">
    Generated by <a href="https://multivon.ai" target="_blank">multivon-eval</a>
  </div>
  <script>{_JS}</script>
</body>
</html>"""
