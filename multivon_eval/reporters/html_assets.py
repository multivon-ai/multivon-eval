"""Offline report styling and keyboard/filter controls."""

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
  var button = document.getElementById('toggle-' + id);
  if (button) button.setAttribute('aria-expanded', String(hidden));
}
"""

_EVIDENCE_CSS = """
[hidden]{display:none!important}
.evidence-warning{border:1px solid #fca5a5;border-left:5px solid #fca5a5;padding:16px;margin:20px 0;color:#e2e8f0;background:#291d25}
.evidence-warning ul,.trial-body ul{padding-left:20px}.evidence-warning p{margin:6px 0}
.evidence-id{font:12px/1.6 ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere;color:#b3bdd0;margin:8px 0}
.evidence-note{font-size:13px;color:#b3bdd0;margin:8px 0 16px}
.trial-evidence{padding:16px;margin:0;min-width:0}.trial-evidence h3{font-size:15px;margin:0 0 8px}
.trial-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:12px;align-items:start}
.trial-card{border:1px solid #414150;border-radius:6px;min-width:0;background:#14141c}
.trial-card>summary{padding:12px;cursor:pointer;color:#e2e8f0;font-weight:600}
.trial-body{padding:0 12px 14px}.trial-body h4{font-size:13px;margin:12px 0 4px;color:#b3bdd0}
.trial-body p{white-space:pre-wrap;overflow-wrap:anywhere}.trial-body pre{font:12px/1.6 ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#0d0d12;color:#e2e8f0;padding:10px;border-radius:4px;max-height:440px;overflow:auto;margin:8px 0}
.trial-body details{margin:12px 0}.trial-body summary{cursor:pointer}.trial-grades li{margin:12px 0}
button,select,input{font:inherit;background:#14141c;color:#e2e8f0;border:1px solid #666578;border-radius:4px;padding:8px;max-width:100%}
button{cursor:pointer;min-height:40px}button:hover{border-color:#a78bfa}button:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible,a:focus-visible,[tabindex]:focus-visible{outline:3px solid #a78bfa;outline-offset:3px}
.case-filters{display:flex;flex-wrap:wrap;gap:14px;align-items:end;margin:16px 0}.case-filters .filter-field{display:flex;flex-direction:column;gap:6px;color:#b3bdd0}.case-filters p{color:#b3bdd0;font-size:13px;padding:8px 0}
.table-scroll{max-width:100%;overflow:auto}.detail-block{min-width:0}.reason-text{white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:650px){body{padding:20px 12px 40px}.case-table>thead{display:none}.case-table,.case-table>tbody,.case-table>tbody>tr,.case-table>tbody>tr>td{display:block;width:100%;max-width:none!important;text-align:left}.case-table>tbody>tr.case-row{padding:10px;border-bottom:1px solid #666578}.case-table>tbody>tr.case-row>td{border:0;padding:4px 0}.case-table>tbody>tr.case-row>td[data-label]::before{content:attr(data-label);display:block;color:#b3bdd0;font-size:11px}.detail-inner{display:block}.detail-block{margin-bottom:12px}.trial-grid{grid-template-columns:1fr}.case-filters .filter-field,.case-filters input,.case-filters select{width:100%}.card{flex:1;min-width:90px}.trial-evidence{padding:12px}}
"""

_EVIDENCE_JS = """
function downloadEvidence(id, filename) {
  var node = document.getElementById(id);
  if (!node) return;
  var url = URL.createObjectURL(new Blob([node.textContent], {type:'application/json'}));
  var link = document.createElement('a'); link.href=url; link.download=filename;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(function(){URL.revokeObjectURL(url);},1000);
}
function filterCases() {
  var tag=document.getElementById('filter-tag').value;
  var status=document.getElementById('filter-status').value;
  var query=document.getElementById('filter-text').value.toLocaleLowerCase();
  var count=0;
  document.querySelectorAll('tr.case-row[data-index]').forEach(function(row){
    var visible=(!tag || JSON.parse(row.dataset.tags).includes(tag)) && (!status || row.dataset.status===status) && (!query || row.dataset.search.toLocaleLowerCase().includes(query));
    row.hidden=!visible; if(visible) count++;
    var detail=document.getElementById('d-'+row.dataset.index);
    if(detail && !visible){detail.hidden=true;document.getElementById('toggle-'+row.dataset.index).setAttribute('aria-expanded','false');}
  });
  document.getElementById('filter-count').textContent=count+' cases shown';
}
"""

_CSS = _CSS.replace('rgba(255,255,255,0.35)', '#b3bdd0').replace('#ef4444', '#fca5a5').replace('.detail-block h4', '.detail-block h3') + _EVIDENCE_CSS
_JS += _EVIDENCE_JS
