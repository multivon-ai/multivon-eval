"""Task-specific retained evidence in existing HTML reports; native logs stay upstream."""
from __future__ import annotations

import html
import json

from ..regressions import regression_candidate
from ..trials import trial_integrity_issues


def _h(value):
    return html.escape(str(value), quote=True)


def _json(value):
    return _h(json.dumps(value, ensure_ascii=False, indent=2))


def coverage_warning(report) -> str:
    issues = list(report.evidence_issues)
    for row in report.case_results:
        issues.extend(trial_integrity_issues(row))
        if row.evidence_error:
            issues.append(row.evidence_error)
    if not issues:
        return ''
    items = ''.join(f'<li>{_h(issue)}</li>' for issue in sorted(set(issues)))
    return '<aside class="evidence-warning" aria-label="Evidence coverage"><strong>Evidence needs attention</strong><p>Passing checks alone do not establish release acceptance.</p><ul>' + items + '</ul></aside>'


def case_filters(report) -> str:
    tags = sorted({tag for row in report.case_results for tag in row.tags})
    statuses = sorted({row.status.value for row in report.case_results})
    def options(values):
        return ''.join(f'<option value="{_h(v)}">{_h(v)}</option>' for v in values)
    return ('<div class="case-filters" aria-label="Filter cases">'
            '<div class="filter-field"><label for="filter-tag">Tag</label><select id="filter-tag" onchange="filterCases()"><option value="">All tags</option>' + options(tags) + '</select></div>'
            '<div class="filter-field"><label for="filter-status">Status</label><select id="filter-status" onchange="filterCases()"><option value="">All statuses</option>' + options(statuses) + '</select></div>'
            '<div class="filter-field"><label for="filter-text">Search input or ID</label><input id="filter-text" type="search" oninput="filterCases()"></div>'
            f'<p id="filter-count" role="status">{len(report.case_results)} cases shown</p></div>'
            '<p class="evidence-note">Filters change the case list. Summary metrics above cover the full report.</p>')


def trial_details(row, *, prefix: str) -> str:
    if not row.trials:
        return '<p class="evidence-note">No saved trial evidence in this report. Legacy aggregates cannot reconstruct individual executions.</p>'
    intact = not row.evidence_error and not trial_integrity_issues(row)
    cards = []
    for index, trial in enumerate(row.trials):
        data = trial.data
        anchor = f'{prefix}-trial-{index}'
        grades = []
        for result in data['evaluators']:
            metadata = result.get('metadata', {})
            unmeasured = metadata.get('skipped') or metadata.get('error_kind')
            label = 'unmeasured' if unmeasured else 'pass' if result['passed'] else 'fail'
            score = '—' if unmeasured else f"{result['score']:.3f}"
            grades.append(f'<li><strong>{_h(result["name"])}</strong> · {_h(label)} · {_h(score)}<p>{_h(result["reason"])}</p></li>')
        upstream = data.get('upstream', {})
        reference = {key: value for key, value in upstream.items() if key != 'evidence'}
        native = upstream.get('evidence', {})
        if isinstance(native, dict) and native.get('digest'):
            reference['native_evidence_digest'] = native['digest']
        reference['trial_digest'] = trial.digest
        errors = {key: data.get(key) for key in ('model_error', 'judge_error', 'evaluator_error') if data.get(key)}
        gaps = ''.join(f'<li>{_h(gap)}</li>' for gap in data.get('evidence_gaps', []))
        candidate = ''
        if intact:
            candidate = (f'<pre id="{anchor}-candidate" hidden>{_json(regression_candidate(trial))}</pre>'
                         f'<button type="button" onclick="downloadEvidence(\'{anchor}-candidate\',\'regression-candidate.json\')">Download candidate for review</button>')
        cards.append(f'<details class="trial-card"><summary>Run {data["run_index"]} · attempt {data["attempt"]} · {_h(data["status"])}</summary>'
            f'<div class="trial-body"><p class="evidence-id">{_h(trial.digest)}</p>'
            f'<h4>Observed output</h4><pre>{_h(data["output"])}</pre>'
            f'<ul class="trial-grades">{"".join(grades)}</ul>'
            f'<h4>Evidence references</h4><pre>{_json(reference)}</pre>'
            + (f'<h4>Execution errors</h4><pre>{_json(errors)}</pre>' if errors else '')
            + (f'<details><summary>Recorded limitations</summary><ul>{gaps}</ul></details>' if gaps else '')
            + f'<details><summary>Full saved trial JSON</summary><pre id="{anchor}-json">{_json(data)}</pre>'
              f'<button type="button" onclick="downloadEvidence(\'{anchor}-json\',\'trial.json\')">Download trial JSON</button></details>'
            + candidate + '</div></details>')
    return (f'<section class="trial-evidence"><h3>Saved trials ({len(cards)})</h3>'
            f'<p class="evidence-id">Case: {_h(row.case_id)}<br>Content: {_h(row.case_digest)}</p>'
            '<p class="evidence-note">Open trials together to compare outputs and verdicts. Candidate downloads need review; they do not infer a correct answer.</p>'
            f'<div class="trial-grid">{"".join(cards)}</div></section>')
