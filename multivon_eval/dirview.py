"""Directory mode for ``multivon-eval view``.

When ``view`` is pointed at a directory instead of a single report JSON,
the local server stops serving one rendered file and instead routes by
URL path across every eval report in the directory:

    /                      INDEX  — sortable table of all valid reports
    /r/<idx>               OPEN   — one report's existing to_html(), verbatim
    /diff?a=<i>&b=<j>      DIFF   — report_a.compare(report_b), rendered

Everything is server-rendered, lazy (parse + render per request), and
strictly READ-ONLY: nothing is written to the user's tree and no parsed
report is held in memory across requests. The handler keeps only a list
of file PATHS discovered at launch; each request re-reads and re-parses
the files it needs.

The HTTP harness (reusable server, SIGTERM→KeyboardInterrupt, suppressed
access logs, delayed browser open, port-bind error handling) mirrors the
single-file ``cmd_view`` path exactly — see :func:`serve_directory`.
"""
from __future__ import annotations

import html as _html
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Hoisted out of f-string expressions: nesting a quoted string inside an
# f-string's {} braces requires PEP 701 (Python 3.12+); on 3.10/3.11 it is
# a SyntaxError. Module constants keep the f-strings expression-only.
_DIM_DASH = "<span class='dim'>—</span>"

# ── Validator ──────────────────────────────────────────────────────────────
# from_dict never raises on foreign JSON (the repo root holds 60+
# SECURITY_*.json that parse into empty reports — some even carry a
# ``summary`` key), so we validate POSITIVELY against to_json()'s shape:
# a non-empty ``cases`` list of case-shaped dicts AND a ``summary`` dict
# carrying ``pass_rate``. The case-level array is the load-bearing signal.

def is_eval_report(data: object) -> bool:
    """Return True iff ``data`` is a serialized EvalReport (to_json shape).

    Positive structural check — NOT "from_dict didn't raise". Rejects
    SECURITY_*.json-shaped dicts, bare ``{}``, and non-dicts.
    """
    if not isinstance(data, dict):
        return False
    summary = data.get("summary")
    if not isinstance(summary, dict) or "pass_rate" not in summary:
        return False
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        return False
    # First case must look like a case row: an input + a pass-state signal.
    first = cases[0]
    if not isinstance(first, dict) or "input" not in first:
        return False
    return any(k in first for k in ("status", "passed", "evaluators"))


# ── Report discovery (lazy: metadata only, never EvalReport objects) ────────

@dataclass
class ReportEntry:
    """Lightweight, cheap-to-build summary of one valid report file.

    Carries only what the INDEX table needs — parsed from the JSON dict
    directly, never via EvalReport, so rendering the index never forces a
    full report reconstruction.
    """
    idx: int
    path: Path
    base_dir: Path
    suite: str
    model: str
    n_cases: int
    pass_rate: float
    ci_low: float
    ci_high: float
    errors: int
    evaluated: int
    flaky: int
    total_cost: Optional[float]
    mtime: float

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def parent_prefix(self) -> str:
        """Parent dir relative to the scanned root, '' when at the root."""
        rel = self.path.parent.relative_to(self.base_dir)
        s = str(rel)
        return "" if s == "." else s

    @property
    def error_rate(self) -> float:
        """Share of ATTEMPTED (non-skipped) cases that errored."""
        denom = self.errors + self.evaluated
        return self.errors / denom if denom else 0.0


def _entry_from_dict(idx: int, path: Path, base_dir: Path, data: dict) -> ReportEntry:
    summary = data.get("summary") or {}
    ci = summary.get("pass_rate_ci_95") or [0.0, 0.0]
    try:
        ci_low, ci_high = float(ci[0]), float(ci[1])
    except (IndexError, TypeError, ValueError):
        ci_low, ci_high = 0.0, 0.0
    costs = summary.get("costs") or {}
    total_cost = costs.get("total_cost_usd") if isinstance(costs, dict) else None
    return ReportEntry(
        idx=idx,
        path=path,
        base_dir=base_dir,
        suite=str(data.get("suite") or ""),
        model=str(data.get("model") or ""),
        n_cases=len(data.get("cases") or []),
        pass_rate=float(summary.get("pass_rate") or 0.0),
        ci_low=ci_low,
        ci_high=ci_high,
        errors=int(summary.get("errors") or 0),
        evaluated=int(summary.get("evaluated") or 0),
        flaky=int(summary.get("flaky_count") or 0),
        total_cost=total_cost,
        mtime=path.stat().st_mtime,
    )


def discover(base_dir: Path, recursive: bool) -> tuple[list[ReportEntry], list[Path]]:
    """Scan ``base_dir`` for JSON files; split into valid reports + skipped.

    Lazy: reads + parses each JSON exactly once to classify and to pull
    the small metadata the index needs. Does NOT build EvalReport objects.
    Index is assigned in stable (sorted-path) order so /r/<idx> URLs are
    deterministic across requests.
    """
    pattern = "**/*.json" if recursive else "*.json"
    paths = sorted(p for p in base_dir.glob(pattern) if p.is_file())

    valid: list[ReportEntry] = []
    skipped: list[Path] = []
    idx = 0
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped.append(p)
            continue
        if is_eval_report(data):
            valid.append(_entry_from_dict(idx, p, base_dir, data))
            idx += 1
        else:
            skipped.append(p)
    return valid, skipped


def load_report(path: Path):
    """Reconstruct a full EvalReport from a file (lazy, per request)."""
    from .result import EvalReport
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalReport.from_dict(data)


# ── Shared styling (self-contained — does NOT import html.py CSS) ───────────

# Compact on purpose — calm, dense, one accent. Self-contained; does not
# touch html.py's private CSS constants.
_STYLE = (
    ":root{--bg:#0b0b10;--panel:#15151d;--line:#26263340;--fg:#e6e6ee;--muted:#8a8a9a;--accent:#5b8def;--bad:#e05a5a;--good:#4ec98f}"
    "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}"
    "a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.wrap{max-width:1100px;margin:0 auto;padding:24px 20px 60px}"
    "h1{font-size:18px;font-weight:600;margin:0 0 2px}.sub{color:var(--muted);font-size:12px;margin:0 0 20px}"
    ".num{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}table{width:100%;border-collapse:collapse;font-size:13px}"
    "th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:middle}th{color:var(--muted);font-weight:500;font-size:12px;white-space:nowrap}"
    "th a{color:var(--muted)}th a.active{color:var(--fg)}td.r,th.r{text-align:right}.dim{color:var(--muted)}"
    ".cibar{display:inline-block;width:90px;height:6px;border-radius:3px;background:#26263380;position:relative;vertical-align:middle;margin-left:8px}"
    ".cibar>i{position:absolute;top:0;height:6px;border-radius:3px;background:var(--accent);opacity:.45}.cibar>b{position:absolute;top:-2px;width:2px;height:10px;background:var(--accent)}"
    ".badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;font-variant-numeric:tabular-nums}.badge.err{background:#e05a5a22;color:var(--bad)}"
    ".badge.errflag{background:var(--bad);color:#fff}.badge.flaky{background:#e0a85a22;color:#e0a85a}.footnote{color:var(--muted);font-size:12px;margin-top:18px}.footnote a{font-size:12px}"
    ".skiplist{color:var(--muted);font-size:12px;margin:6px 0 0 0;padding-left:18px}select{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:4px;font-size:12px;padding:2px 4px}"
    ".crumb{font-size:12px;color:var(--muted);padding:10px 20px;border-bottom:1px solid var(--line);background:var(--panel)}"
    ".strip{display:flex;gap:28px;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:0 0 20px}"
    ".strip .m{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.strip .v{font-size:18px;font-variant-numeric:tabular-nums}"
    ".delta.up{color:var(--good)}.delta.down{color:var(--bad)}.sig{color:var(--good)}.nsig{color:var(--muted)}.sect{margin:18px 0}"
    ".sect>summary{cursor:pointer;font-weight:600;font-size:14px;padding:6px 0}.sect.reg>summary{color:var(--bad)}"
    ".row{border:1px solid var(--line);border-radius:6px;margin:8px 0;background:var(--panel)}.row>summary{cursor:pointer;padding:8px 12px;font-size:13px}"
    ".row .body{padding:0 14px 12px;border-top:1px solid var(--line)}.row .body .ci{color:var(--muted);font-size:12px;margin:8px 0 4px}"
    ".reason{font-size:13px;margin:6px 0 10px}.reason .who{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}.arrow{color:var(--muted)}"
)


def _page(title: str, body: str, *, crumb: str = "") -> str:
    crumb_html = f'<div class="crumb">{crumb}</div>' if crumb else ""
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{_html.escape(title)} — multivon-eval</title>"
        f"<style>{_STYLE}</style></head><body>{crumb_html}"
        f"<div class=\"wrap\">{body}</div></body></html>"
    )


def _rel_time(mtime: float, now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    d = max(0.0, now - mtime)
    if d < 60:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)}m ago"
    if d < 86400:
        return f"{int(d // 3600)}h ago"
    if d < 86400 * 30:
        return f"{int(d // 86400)}d ago"
    return f"{int(d // (86400 * 30))}mo ago"


# ── INDEX ───────────────────────────────────────────────────────────────────

_SORT_KEYS = {
    "run": lambda e: (e.parent_prefix.lower(), e.stem.lower()),
    "suite": lambda e: e.suite.lower(),
    "model": lambda e: e.model.lower(),
    "when": lambda e: e.mtime,
    "n": lambda e: e.n_cases,
    "pass_rate": lambda e: e.pass_rate,
    "cost": lambda e: (e.total_cost is None, e.total_cost or 0.0),
}


def _ci_bar(e: ReportEntry) -> str:
    lo = max(0.0, min(1.0, e.ci_low)) * 90
    hi = max(0.0, min(1.0, e.ci_high)) * 90
    mid = max(0.0, min(1.0, e.pass_rate)) * 90
    width = max(2.0, hi - lo)
    return (
        f'<span class="cibar" title="95% CI [{e.ci_low:.0%}, {e.ci_high:.0%}]">'
        f'<i style="left:{lo:.1f}px;width:{width:.1f}px"></i>'
        f'<b style="left:{mid:.1f}px"></b></span>'
    )


def _sort_header(label: str, key: str, sort: str, direction: str) -> str:
    active = key == sort
    next_dir = "desc" if (active and direction == "asc") else "asc"
    arrow = ("▲" if direction == "asc" else "▼") if active else ""
    cls = ' class="active"' if active else ""
    return (
        f'<a href="/?sort={key}&dir={next_dir}"{cls}>'
        f'{_html.escape(label)} {arrow}</a>'
    )


def render_index(
    reports: list[ReportEntry], skipped: list[Path], *,
    sort: str = "when", direction: str = "desc",
    base_dir: Optional[Path] = None, now: Optional[float] = None,
) -> str:
    """Render the sortable INDEX table. Pure — no server needed."""
    if sort not in _SORT_KEYS:
        sort = "when"
    if direction not in ("asc", "desc"):
        direction = "desc"
    ordered = sorted(reports, key=_SORT_KEYS[sort], reverse=(direction == "desc"))

    options = "".join(
        f'<option value="{e.idx}">{_html.escape(e.stem)}</option>'
        for e in reports
    )

    rows: list[str] = []
    for e in ordered:
        prefix = (
            f'<span class="dim">{_html.escape(e.parent_prefix)}/</span>'
            if e.parent_prefix else ""
        )
        run_cell = f'{prefix}<a href="/r/{e.idx}">{_html.escape(e.stem)}</a>'
        pr = f'{e.pass_rate:.0%}'
        flagged = e.error_rate >= 0.10
        badges: list[str] = []
        if e.errors:
            cls = "badge errflag" if flagged else "badge err"
            title = (
                f"error rate {e.error_rate:.0%} ≥ 10% — results may be unreliable"
                if flagged else f"{e.errors} error case(s)"
            )
            badges.append(f'<span class="{cls}" title="{title}">err {e.errors}</span>')
        if e.flaky:
            badges.append(f'<span class="badge flaky">flaky {e.flaky}</span>')
        badge_html = " ".join(badges)
        cost = "—" if e.total_cost is None else f"${e.total_cost:.4f}"
        # Per-row diff control: pick a baseline from the dropdown, jump to DIFF.
        diff_ctl = (
            f'<select onchange="if(this.value!=\'\')'
            f'location.href=\'/diff?a=\'+this.value+\'&b={e.idx}\'">'
            f'<option value="">diff vs…</option>{options}</select>'
        )
        suite_cell = _html.escape(e.suite) or _DIM_DASH
        model_cell = _html.escape(e.model) or _DIM_DASH
        rows.append(
            "<tr>"
            f'<td>{run_cell}</td>'
            f'<td>{suite_cell}</td>'
            f'<td>{model_cell}</td>'
            f'<td class="dim">{_rel_time(e.mtime, now)}</td>'
            f'<td class="r num">{e.n_cases}</td>'
            f'<td class="num">{pr}{_ci_bar(e)}</td>'
            f'<td>{badge_html}</td>'
            f'<td class="r num">{cost}</td>'
            f'<td>{diff_ctl}</td>'
            "</tr>"
        )

    if reports:
        head = (
            "<tr>"
            f'<th>{_sort_header("run", "run", sort, direction)}</th>'
            f'<th>{_sort_header("suite", "suite", sort, direction)}</th>'
            f'<th>{_sort_header("model / target", "model", sort, direction)}</th>'
            f'<th>{_sort_header("when", "when", sort, direction)}</th>'
            f'<th class="r">{_sort_header("n", "n", sort, direction)}</th>'
            f'<th>{_sort_header("pass_rate", "pass_rate", sort, direction)}</th>'
            "<th>flags</th>"
            f'<th class="r">{_sort_header("cost", "cost", sort, direction)}</th>'
            "<th></th></tr>"
        )
        table = f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"
    else:
        table = '<p class="dim">No eval reports found in this directory.</p>'

    foot = ""
    if skipped:
        items = "".join(
            f"<li>{_html.escape(p.name)}</li>" for p in skipped
        )
        foot = (
            '<div class="footnote">'
            f'{len(skipped)} file(s) skipped (not eval reports) '
            '<a href="#" onclick="document.getElementById(\'sk\').style.display='
            '(document.getElementById(\'sk\').style.display==\'block\'?\'none\':\'block\');'
            'return false">[expand]</a>'
            f'<ul class="skiplist" id="sk" style="display:none">{items}</ul>'
            "</div>"
        )

    src = str(base_dir) if base_dir else ""
    body = (
        "<h1>multivon-eval reports</h1>"
        f'<p class="sub num">{len(reports)} report(s) · {_html.escape(src)}</p>'
        f"{table}{foot}"
    )
    return _page("reports", body)


# ── OPEN ────────────────────────────────────────────────────────────────────

def render_open(report, entry: ReportEntry) -> str:
    """Serve the report's existing to_html() VERBATIM, plus a breadcrumb.

    The only modification to the report HTML is a single breadcrumb bar
    injected right after <body>. No section prepend, no renderer fork.
    """
    doc = report.to_html()
    label = _html.escape(entry.stem)
    # Self-contained inline style — the report doc carries its own CSS, so
    # the breadcrumb can't rely on dirview's _STYLE being present.
    crumb = (
        '<div style="font:12px/1.5 -apple-system,sans-serif;background:#15151d;'
        'color:#8a8a9a;padding:10px 20px;border-bottom:1px solid #26263340">'
        '<a href="/" style="color:#5b8def;text-decoration:none">← all reports</a>'
        f' &nbsp;/&nbsp; {label}</div>'
    )
    return doc.replace("<body>", "<body>" + crumb, 1) if "<body>" in doc else crumb + doc


# ── DIFF ─────────────────────────────────────────────────────────────────────

def _reasons_by_input(report) -> dict[str, list[str]]:
    """Map case_input → list of evaluator reasons for that case.

    Judge reasons live on CaseResult.results[].reason — NOT on CaseDiff —
    so DIFF re-derives them from the full report by case_input.
    """
    out: dict[str, list[str]] = {}
    for cr in report.case_results:
        reasons = [r.reason for r in cr.results if r.reason]
        out.setdefault(cr.case_input, reasons)
    return out


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

    reasons_a = _reasons_by_input(report_a)
    reasons_b = _reasons_by_input(report_b)

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

    def case_row(c, *, expand_reasons: bool) -> str:
        title = _html.escape(c.case_input[:120])
        statuses = (
            f'<span class="num">{c.baseline_status.value} '
            f'<span class="arrow">→</span> {c.proposal_status.value}</span>'
        )
        if not expand_reasons:
            return (
                f'<div class="row"><summary style="list-style:none">'
                f'{title} &nbsp; {statuses}</summary></div>'
            )
        ra = reasons_a.get(c.case_input, [])
        rb = reasons_b.get(c.case_input, [])
        body = (
            '<div class="body">'
            f'<div class="ci">{_html.escape(c.case_input)}</div>'
            f'{_reason_block(name_a, ra)}{_reason_block(name_b, rb)}'
            "</div>"
        )
        return (
            f'<details class="row"><summary>{title} &nbsp; {statuses}'
            f' <span class="num">({c.baseline_score:.2f}→{c.proposal_score:.2f})</span>'
            f'</summary>{body}</details>'
        )

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
    body = f"<h1>diff</h1>{strip}{sections}"
    return _page("diff", body, crumb=crumb)


# The HTTP harness wiring these renderers to routes lives in
# dirview_server (plumbing kept out of this rendering module).
def serve_directory(*args, **kwargs) -> int:
    """Start the directory-mode view server (see dirview_server)."""
    from .dirview_server import serve_directory as _serve
    return _serve(*args, **kwargs)

__all__ = [
    "is_eval_report", "discover", "load_report", "ReportEntry",
    "render_index", "render_open", "render_diff", "serve_directory",
]
