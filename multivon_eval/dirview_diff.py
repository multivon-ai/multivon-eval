"""Matched report comparison rendering for the local report browser."""
from __future__ import annotations

import html as _html

from .dirview import _page


def _cases_by_identity(report) -> dict:
    from collections import defaultdict
    groups = defaultdict(list)
    for row in report.case_results:
        key = ('id', row.case_id) if row.case_id else ('input', row.case_input)
        groups[key].append(row)
    return {key: rows[0] for key, rows in groups.items() if len(rows) == 1}


def _signed_pp(x: float) -> str:
    return ("+" if x >= 0 else "") + f"{x * 100:.1f}pp"


def _signed(x: float) -> str:
    return ("+" if x >= 0 else "") + f"{x:.3f}"


def _reason_block(who: str, reasons: list[str]) -> str:
    text = " ".join(_html.escape(r) for r in reasons) if reasons else "(no judge reason recorded)"
    return f'<div class="reason"><span class="who">{_html.escape(who)}</span><br>{text}</div>'


def render_diff(report_a, report_b, *, name_a: str = "", name_b: str = "") -> str:
    """Render report_a.compare(report_b) → ReportDiff as HTML. Pure."""
    diff = report_a.compare(report_b)
    name_a = name_a or diff.baseline_name or "A"
    name_b = name_b or diff.proposal_name or "B"

    cases_a = _cases_by_identity(report_a)
    cases_b = _cases_by_identity(report_b)

    pr_d = diff.pass_rate_delta
    sc_d = diff.avg_score_delta
    pr_cls = "delta up" if pr_d >= 0 else "delta down"
    sc_cls = "delta up" if sc_d >= 0 else "delta down"
    if diff.mcnemar_p is None:
        sig = '<span class="nsig">McNemar: n/a</span>'
    elif diff.mcnemar_p < 0.05:
        sig = f'<span class="sig">McNemar p={diff.mcnemar_p:.4f} (significant)</span>'
    else:
        sig = f'<span class="nsig">McNemar p={diff.mcnemar_p:.4f} (not significant)</span>'

    strip = (
        '<div class="strip">'
        f'<div><div class="m">comparing</div><div class="v num">'
        f'{_html.escape(name_a)} <span class="arrow">→</span> {_html.escape(name_b)}</div></div>'
        f'<div><div class="m">pass rate Δ</div><div class="v num {pr_cls}">{_signed_pp(pr_d)}</div></div>'
        f'<div><div class="m">avg score Δ</div><div class="v num {sc_cls}">{_signed(sc_d)}</div></div>'
        f'<div><div class="m">significance</div><div class="v" style="font-size:13px">{sig}</div></div>'
        "</div>"
    )

    # Bucket paired cases. STILL FAILING = paired, unchanged-direction,
    # and both sides not passing.
    from .result import EvalStatus
    regressed = diff.regressions
    fixed = diff.improvements
    still_failing = [
        c for c in diff.unchanged
        if c.baseline_status != EvalStatus.PASSED
        and c.proposal_status != EvalStatus.PASSED
        and c.baseline_status != EvalStatus.SKIPPED
        and c.proposal_status != EvalStatus.SKIPPED
    ]
    truly_unchanged = [c for c in diff.unchanged if c not in still_failing]

    counter = 0
    def case_row(c, *, expand_reasons: bool) -> str:
        from .reporters.evidence_html import trial_details
        nonlocal counter
        counter += 1
        key = ('id', c.case_id) if c.case_id else ('input', c.case_input)
        a, b = cases_a.get(key), cases_b.get(key)
        title = _html.escape(c.case_input[:120])
        statuses = _html.escape(c.baseline_status.value + ' → ' + c.proposal_status.value)
        def side(row, name, prefix):
            if row is None:
                return '<p>Case identity is ambiguous; no reasons were assigned.</p>'
            return (f'<section><h2>{_html.escape(name)}</h2><pre>{_html.escape(row.actual_output)}</pre>'
                    + _reason_block(name, [r.reason for r in row.results if r.reason])
                    + trial_details(row, prefix=prefix) + '</section>')
        body = (f'<p class="evidence-id">Case: {_html.escape(c.case_id or "legacy input match")}</p>'
                f'<div class="compare-columns">{side(a, name_a, f"a-{counter}")}{side(b, name_b, f"b-{counter}")}</div>')
        return f'<details class="row"><summary>{title} · {statuses}</summary><div class="body">{body}</div></details>'

    def section(title: str, cases, *, cls: str, open_: bool, expand: bool) -> str:
        rows = "".join(case_row(c, expand_reasons=expand) for c in cases) \
            or '<p class="dim">none</p>'
        attr = " open" if open_ else ""
        return (
            f'<details class="sect {cls}"{attr}>'
            f'<summary>{_html.escape(title)} ({len(cases)})</summary>{rows}</details>'
        )

    sections = (
        section("Regressed", regressed, cls="reg", open_=True, expand=True)
        + section("Fixed", fixed, cls="", open_=False, expand=True)
        + section("Still failing", still_failing, cls="", open_=False, expand=True)
        + section("Unchanged", truly_unchanged, cls="", open_=False, expand=False)
    )

    crumb = '<a href="/">← all reports</a>'
    issues = ''.join(f'<li>{_html.escape(issue)}</li>' for issue in diff.identity_issues)
    warning = ('<aside class="evidence-warning"><strong>Comparison evidence needs attention</strong><ul>'
               + issues + '</ul></aside>') if issues else ''
    unmatched = ''.join(f'<li>{_html.escape(row.case_id or "legacy")} · {_html.escape(row.case_input)}</li>'
                        for row in diff.added + diff.removed)
    unmatched_html = f'<h2>Unmatched cases ({len(diff.added)} added, {len(diff.removed)} removed)</h2><ul>{unmatched}</ul>' if unmatched else ''
    body = f"<h1>Compare reports</h1>{warning}{strip}{sections}{unmatched_html}"
    return _page("diff", body, crumb=crumb)

