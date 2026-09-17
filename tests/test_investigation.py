"""Evidence identity, reviewed case promotion and the actual local HTTP boundary."""
import html
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

from multivon_eval import EvalCase, EvalSuite, ExactMatch
from multivon_eval.dirview import discover, load_report, render_diff
from multivon_eval.integrations.label_studio import export_review_tasks, import_review_annotations
from multivon_eval.regressions import promote_reviewed_trial, regression_candidate


def report(outputs=('wrong A', 'wrong B')):
    cases = [EvalCase('Same prompt', 'hold', case_id=key, source_id='source-' + key,
                      context='Missing approval for ' + key, tags=[key]) for key in ('a', 'b')]
    return EvalSuite('fixture').add_evaluator(ExactMatch()).run_on_cases(list(zip(cases, outputs)), verbose=False)


def reviews_for(saved, choices=('Reject', 'Reject')):
    tasks = export_review_tasks(saved, 'exact_match', rubric='Synthetic fixture: approval is required')
    annotations = [{'id': i, 'completed_by': i, 'result': [
        {'from_name': 'verdict', 'to_name': 'output', 'type': 'choices', 'value': {'choices': [choice]}},
        {'from_name': 'review_reason', 'to_name': 'output', 'type': 'textarea',
         'value': {'text': ['Synthetic review rationale']}},
    ]} for i, choice in enumerate(choices, 1)]
    reviews = import_review_annotations([{**tasks[0], 'annotations': annotations}], tasks, reviewer_kind='synthetic')
    return tasks, reviews


def promote(saved, tasks, reviews, **kwargs):
    return promote_reviewed_trial(saved, saved.case_results[0].trials[0].digest, tasks, reviews,
        evaluator='exact_match', case_id='new-a', expected_output='hold',
        rationale='Fixture source specifies hold without approval', reviewer_kind='synthetic', **kwargs)


def test_candidate_never_promotes_failed_output_and_review_preserves_source():
    saved = report()
    saved.evidence_issues = ['Original capture was incomplete']
    trial = saved.case_results[0].trials[0]
    candidate = regression_candidate(trial)
    assert candidate['requires_review'] and candidate['source_case']['expected_output'] == 'hold'
    tasks, reviews = reviews_for(saved)
    promoted = promote(saved, tasks, reviews)
    assert promoted.manifest['splits'] == {'development': ['new-a']}
    case = promoted.cases[0]
    assert case.source_id == 'source-a' and case.expected_output == 'hold'
    assert case.agent_trace is None and case.reference_output is None
    assert promoted.manifest['provenance']['source_trial_digest'] == trial.digest
    assert promoted.manifest['provenance']['source_report_evidence_issues'] == saved.evidence_issues
    assert promoted.manifest['provenance']['source_trial_evidence_gaps'] == trial.data['evidence_gaps']
    assert saved.case_results[0].actual_output == 'wrong A'


@pytest.mark.parametrize('choices', [('Reject',), ('Reject', 'Accept'), ('Reject', 'Unknown')])
def test_unresolved_reviews_cannot_promote(choices):
    saved = report()
    tasks, reviews = reviews_for(saved, choices)
    with pytest.raises(ValueError, match='Resolve'):
        promote(saved, tasks, reviews)


def test_promotion_rejects_source_reassignment_and_tampered_task():
    saved = report()
    tasks, reviews = reviews_for(saved)
    with pytest.raises(ValueError, match='source group'):
        promote(saved, tasks, reviews, source_id='pretend-independent-source')
    tasks[0]['data']['output'] = 'forged'
    with pytest.raises(ValueError, match='original report'):
        promote(saved, tasks, reviews)


def test_same_prompt_comparison_reasons_follow_case_id():
    before, after = report(('hold', 'hold')), report()
    page = render_diff(before, after)
    a = page.index('Case: a')
    b = page.index('Case: b')
    assert 'wrong A' in page[a:b] and 'wrong B' not in page[a:b]
    assert 'wrong B' in page[b:] and 'wrong A' not in page[b:]
    after.evidence_issues.append('Missing target capture')
    assert 'Missing target capture' in render_diff(before, after)


def test_html_keeps_full_escaped_evidence_and_keyboard_controls():
    saved = report(('<script>window.injected=true</script>', 'wrong B'))
    saved.evidence_issues = ['<img src=x onerror=bad()>']
    page = saved.to_html()
    assert '<script>window.injected' not in page and '<img src=x' not in page
    assert '&lt;script&gt;window.injected' in page
    assert saved.case_results[0].trials[0].digest in page
    assert 'aria-controls="d-0"' in page and 'aria-expanded="false"' in page
    assert 'Full saved trial JSON' in page and 'Download candidate for review' in page
    assert 'Evidence needs attention' in page


def test_detached_case_headers_cannot_offer_a_promotion_candidate():
    saved = report()
    saved.case_results[0].actual_output = 'not the retained output'
    page = saved.to_html()
    assert 'Case output does not match' in page
    assert 'id="case-0-trial-0-candidate"' not in page
    tasks, reviews = reviews_for(report())
    with pytest.raises(ValueError, match='intact'):
        promote(saved, tasks, reviews)


def test_comparison_displays_added_and_removed_cases():
    before, after = report(), report()
    after.case_results.pop()
    page = render_diff(before, after)
    assert 'Unmatched cases (0 added, 1 removed)' in page


def test_discovery_excludes_symlinks_and_binds_content(tmp_path):
    root = tmp_path / 'reports'; root.mkdir()
    outside = tmp_path / 'outside.json'; report().save_json(str(outside))
    (root / 'linked.json').symlink_to(outside)
    report().save_json(str(root / 'inside.json'))
    valid, skipped = discover(root, False)
    assert [e.path.name for e in valid] == ['inside.json']
    assert len(skipped) == 1
    entry = valid[0]
    report(('changed', 'changed')).save_json(str(entry.path))
    with pytest.raises(ValueError, match='changed'):
        load_report(entry.path, expected_digest=entry.content_digest, base_dir=root)
    with pytest.raises(ValueError, match='outside'):
        load_report(outside, base_dir=root)


@pytest.fixture(params=['directory', 'single'])
def viewer(tmp_path, request):
    report().save_json(str(tmp_path / 'report.json'))
    target = str(tmp_path if request.param == 'directory' else tmp_path / 'report.json')
    proc = subprocess.Popen([sys.executable, '-u', '-m', 'multivon_eval', 'view', target,
                             '--port', '0', '--no-browser'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        url = None
        for _ in range(8):
            line = proc.stdout.readline()
            if 'http://127.0.0.1:' in line:
                url = line.split('→')[-1].strip()
                break
        assert url, 'Viewer did not start'
        yield url, tmp_path, request.param
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=5)


def test_real_viewer_blocks_cross_origin_and_rebinding(viewer):
    url, _, _ = viewer
    with urllib.request.urlopen(url, timeout=3) as response:
        assert response.status == 200
        assert response.headers['Cache-Control'] == 'no-store'
        assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
    for headers in ({'Host': 'evil.example'}, {'Origin': 'https://evil.example'}, {'Sec-Fetch-Site': 'cross-site'}):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=3)
        assert exc.value.code == 403


def test_directory_links_survive_insertions_and_reject_changed_content(viewer):
    url, root, mode = viewer
    if mode != 'directory':
        pytest.skip('Directory-only identity binding')
    entry = discover(root, False)[0][0]
    bound = url + f'r/{entry.idx}?key={entry.key}&digest={entry.content_digest}'
    report(('new before', 'new before')).save_json(str(root / 'aaa.json'))
    with urllib.request.urlopen(bound, timeout=3) as response:
        assert 'wrong A' in html.unescape(response.read().decode())
    report(('overwritten', 'overwritten')).save_json(str(root / 'report.json'))
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(bound, timeout=3)
    assert exc.value.code == 409
