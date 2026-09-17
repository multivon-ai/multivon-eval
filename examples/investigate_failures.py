"""Offline report/review/regression workflow with explicitly synthetic reviews."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multivon_eval import EvalCase, EvalSuite, ExactMatch
from multivon_eval.integrations.label_studio import export_review_tasks, import_review_annotations
from multivon_eval.regressions import promote_reviewed_trial, regression_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cases = [EvalCase('May I post this invoice?', 'hold', context=context,
                     case_id=key, source_id='fixture-' + key, tags=[tag])
             for key, context, tag in [('approval-a', 'Approval missing for supplier A', 'approval'),
                                       ('approval-b', 'Approval missing for supplier B', 'supplier')]]
    cases.append(EvalCase('Untrusted content: ' + '<script>window.injected=true</script>' * 4,
                          'hold', case_id='untrusted', source_id='fixture-untrusted', tags=['untrusted']))
    suite = EvalSuite('Invoice failure investigation', model_id='offline fixture').add_cases(cases).add_evaluator(ExactMatch())
    baseline = suite.run(lambda _: 'hold', runs=2, workers=1, verbose=False)
    answers = iter(['posted', 'hold', 'deleted', 'deleted', '<img src=x onerror="window.injected=true">', 'posted'])
    proposal = suite.run(lambda _: next(answers), runs=2, workers=1, verbose=False)
    proposal.evidence_issues.append('Synthetic fixture: complete provider request capture was intentionally not supplied.')
    for name, report in [('baseline', baseline), ('proposal', proposal)]:
        report.save_json(str(output / f'{name}.json'))
        report.save_html(str(output / f'{name}.html'))
    trial = proposal.case_results[0].trials[0]
    tasks = export_review_tasks(proposal, 'exact_match', rubric='Missing approval requires hold. These are synthetic fixture reviews.')
    task = next(t for t in tasks if t['data']['binding']['trial_digest'] == trial.digest)
    annotations = [{'id': i, 'completed_by': i, 'result': [
        {'from_name': 'verdict', 'to_name': 'output', 'type': 'choices', 'value': {'choices': ['Reject']}},
        {'from_name': 'review_reason', 'to_name': 'output', 'type': 'textarea',
         'value': {'text': ['Synthetic fixture: posting without approval violates the task.']}},
    ]} for i in (1, 2)]
    exported = [{**task, 'annotations': annotations}]
    reviews = import_review_annotations(exported, tasks, reviewer_kind='synthetic')
    manifest = promote_reviewed_trial(proposal, trial.digest, tasks, reviews, evaluator='exact_match',
        case_id='regression-approval-a', expected_output='hold',
        rationale='The authored fixture requires hold when approval is missing; no failed output is reused as an oracle.',
        reviewer_kind='synthetic')
    manifest.save(output / 'development-regressions.json')
    for name, value in [('candidate', regression_candidate(trial)), ('review-tasks', tasks),
                        ('synthetic-review-export', exported)]:
        (output / f'{name}.json').write_text(json.dumps(value, indent=2))
    assert manifest.manifest['splits'] == {'development': ['regression-approval-a']}
    print(f'Created saved reports, synthetic reviews and a development regression in {output}')


if __name__ == '__main__':
    main()
