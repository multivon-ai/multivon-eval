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

import hashlib
import html as _html
import json
import time
from dataclasses import dataclass
from pathlib import Path

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
    total_cost: float | None
    mtime: float
    pass_hat_k: float | None = None  # absent in pre-pass^k reports
    saturated: bool = False  # absent in pre-saturation-monitor reports
    content_digest: str = ""

    @property
    def key(self) -> str:
        return hashlib.sha256(str(self.path.relative_to(self.base_dir)).encode()).hexdigest()

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
    phk = summary.get("pass_hat_k")
    pass_hat_k: float | None
    try:
        pass_hat_k = float(phk["value"]) if isinstance(phk, dict) and phk.get("value") is not None else None
    except (TypeError, ValueError):
        pass_hat_k = None
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
        pass_hat_k=pass_hat_k,
        saturated=bool(summary.get("saturated") or False),
    )


def discover(base_dir: Path, recursive: bool) -> tuple[list[ReportEntry], list[Path]]:
    """Scan ``base_dir`` for JSON files; split into valid reports + skipped.

    Lazy: reads + parses each JSON exactly once to classify and to pull
    the small metadata the index needs. Does NOT build EvalReport objects.
    Index is assigned in stable (sorted-path) order so /r/<idx> URLs are
    deterministic across requests.
    """
    base_dir = base_dir.resolve()
    pattern = "**/*.json" if recursive else "*.json"
    paths = sorted(p for p in base_dir.glob(pattern) if p.is_file())

    valid: list[ReportEntry] = []
    skipped: list[Path] = []
    idx = 0
    for p in paths:
        try:
            if p.is_symlink() or not p.resolve().is_relative_to(base_dir):
                skipped.append(p)
                continue
            payload = p.read_bytes()
            data = json.loads(payload)
            if is_eval_report(data):
                entry = _entry_from_dict(idx, p, base_dir, data)
                entry.content_digest = hashlib.sha256(payload).hexdigest()
                valid.append(entry)
                idx += 1
            else:
                skipped.append(p)
        except (OSError, ValueError, TypeError, OverflowError):
            skipped.append(p)
            continue
    return valid, skipped


def load_report(path: Path, *, expected_digest: str | None = None, base_dir: Path | None = None):
    """Reconstruct a full EvalReport from a file (lazy, per request)."""
    from .result import EvalReport
    resolved = path.resolve()
    if base_dir is not None and not resolved.is_relative_to(base_dir.resolve()):
        raise ValueError('Report path is outside the selected directory')
    payload = resolved.read_bytes()
    if expected_digest is not None and hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError('Report content changed; reload the index')
    data = json.loads(payload)
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
_STYLE += (
    ":root{--muted:#b3bdd0;--line:#373744;--bad:#fca5a5;--accent:#a6beff}"
    "a:focus-visible,button:focus-visible,select:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid #a78bfa;outline-offset:3px}"
    ".table-scroll{overflow:auto;max-width:100%}.wrap{overflow-wrap:anywhere}.compare-columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}"
    ".badge.errflag{background:#7f1d1d;color:#fecaca}"
    "pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{overflow-wrap:anywhere}select{min-height:36px;max-width:180px}"
    "@media(max-width:650px){.wrap{padding:16px 12px}.compare-columns{grid-template-columns:1fr}.strip{gap:14px}}"
)


def _page(title: str, body: str, *, crumb: str = "") -> str:
    from .reporters.html_assets import _EVIDENCE_CSS, _EVIDENCE_JS
    crumb_html = f'<nav class="crumb" aria-label="Reports">{crumb}</nav>' if crumb else ""
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{_html.escape(title)} — multivon-eval</title>"
        f"<style>{_STYLE}{_EVIDENCE_CSS}</style></head><body>{crumb_html}"
        f"<main class=\"wrap\">{body}</main><script>{_EVIDENCE_JS}</script></body></html>"
    )


def _rel_time(mtime: float, now: float | None = None) -> str:
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
    base_dir: Path | None = None, now: float | None = None,
) -> str:
    """Render the sortable INDEX table. Pure — no server needed."""
    if sort not in _SORT_KEYS:
        sort = "when"
    if direction not in ("asc", "desc"):
        direction = "desc"
    ordered = sorted(reports, key=_SORT_KEYS[sort], reverse=(direction == "desc"))

    options = "".join(
        f'<option value="{e.idx}&amp;ka={e.key}&amp;da={e.content_digest}">{_html.escape(e.stem)}</option>'
        for e in reports
    )

    rows: list[str] = []
    for e in ordered:
        prefix = (
            f'<span class="dim">{_html.escape(e.parent_prefix)}/</span>'
            if e.parent_prefix else ""
        )
        run_cell = f'{prefix}<a href="/r/{e.idx}?key={e.key}&amp;digest={e.content_digest}">{_html.escape(e.stem)}</a>'
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
        if e.saturated:
            badges.append(
                '<span class="badge flaky" title="all evaluated cases passed — '
                'this suite can no longer detect improvement">saturated</span>'
            )
        badge_html = " ".join(badges)
        phk_cell = _DIM_DASH if e.pass_hat_k is None else f"{e.pass_hat_k:.0%}"
        cost = "—" if e.total_cost is None else f"${e.total_cost:.4f}"
        # Per-row diff control: pick a baseline from the dropdown, jump to DIFF.
        diff_ctl = (
            f'<select aria-label="Compare {_html.escape(e.stem, quote=True)} with baseline" onchange="if(this.value!=\'\')'
            f'location.href=\'/diff?a=\'+this.value+\'&b={e.idx}&kb={e.key}&db={e.content_digest}\'">'
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
            f'<td class="r num">{phk_cell}</td>'
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
            '<th class="r">pass^k</th>'
            "<th>flags</th>"
            f'<th class="r">{_sort_header("cost", "cost", sort, direction)}</th>'
            "<th>Compare</th></tr>"
        )
        table = f'<div class="table-scroll" role="region" aria-label="Saved reports" tabindex="0"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'
    else:
        table = '<p class="dim">No eval reports found in this directory.</p>'

    foot = ""
    if skipped:
        items = "".join(
            f"<li>{_html.escape(p.name)}</li>" for p in skipped
        )
        foot = (
            '<details class="footnote">'
            f'<summary>{len(skipped)} file(s) skipped (not eval reports) — show files</summary>'
            f'<ul class="skiplist">{items}</ul></details>'
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
        '<nav aria-label="Reports" style="font:12px/1.5 -apple-system,sans-serif;background:#15151d;'
        'color:#b3bdd0;padding:10px 20px;border-bottom:1px solid #414150">'
        '<a href="/" style="color:#a6beff;text-decoration:none">← all reports</a>'
        f' &nbsp;/&nbsp; {label}</nav>'
    )
    return doc.replace("<body>", "<body>" + crumb, 1) if "<body>" in doc else crumb + doc


# ── DIFF ─────────────────────────────────────────────────────────────────────

def render_diff(*args, **kwargs) -> str:
    from .dirview_diff import render_diff as render
    return render(*args, **kwargs)


# The HTTP harness wiring these renderers to routes lives in
# dirview_server (plumbing kept out of this rendering module).
def serve_directory(*args, **kwargs) -> int:
    """Start the directory-mode view server (see dirview_server)."""
    from .dirview_server import serve_directory as _serve
    return _serve(*args, **kwargs)

__all__ = [
    "ReportEntry",
    "discover",
    "is_eval_report",
    "load_report",
    "render_diff",
    "render_index",
    "render_open",
    "serve_directory",
]
