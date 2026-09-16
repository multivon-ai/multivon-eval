"""Offline synthetic demonstration of source-disjoint review calibration.

Install the development checkout with `pip install -e '.[review]'`.
This tests workflow behavior; its invented scores do not measure judge quality.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from multivon_eval import CaseManifest, EvalCase, EvalSuite
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.integrations.label_studio import export_review_tasks, import_review_annotations
from multivon_eval.review_calibration import (
    collect_reviewed_scores,
    evaluate_review_threshold,
    fit_review_threshold,
)


class FixtureScore(Evaluator):
    name = 'fixture_score'

    def evaluate(self, case, output):
        return self._result(case.metadata['score'], 'Invented score for an offline demonstration')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for split in ['development', 'held_out']:
        for source in range(3):
            for good in [False, True]:
                score = 0.8 if good else 0.2
                if split == 'held_out' and source == 0 and not good:
                    score = 0.9
                cases.append(EvalCase(f'{split}-{source}-{good}', 'fixture',
                    case_id=f'{split}-{source}-{good}', source_id=f'{split}-{source}',
                    tags=['good' if good else 'bad'], metadata={'score': score}))
    manifest = CaseManifest('synthetic threshold workflow', cases,
        splits={split: [c.case_id for c in cases if c.case_id.startswith(split)]
                for split in ['development', 'held_out']})
    manifest.save(output / 'manifest.json')
    batches = {}
    for split in ['development', 'held_out']:
        report = EvalSuite('synthetic grader').add_cases(manifest.split(split)).add_evaluator(FixtureScore()).run(
            lambda _: 'fixture', runs=2, verbose=False)
        tasks = export_review_tasks(report, 'fixture_score', rubric='Synthetic good/bad labels for plumbing validation only.')
        exported = copy.deepcopy(tasks)
        for index, task in enumerate(exported):
            label = 'Accept' if task['data']['binding']['case_id'].endswith('True') else 'Reject'
            task['annotations'] = [{'id': index, 'completed_by': 'synthetic-oracle',
                'result': [{'from_name': 'verdict', 'to_name': 'output', 'type': 'choices',
                            'value': {'choices': [label]}},
                           {'from_name': 'review_reason', 'to_name': 'output', 'type': 'textarea',
                            'value': {'text': ['Code-authored fixture; no human review']}}]}]
        reviews = import_review_annotations(exported, tasks, reviewer_kind='synthetic')
        batch = collect_reviewed_scores(report, tasks, reviews, manifest, split=split,
                                       reviewer_kind='synthetic', min_reviewers=1)
        batches[split] = batch
        artifacts = {'report': json.loads(report.to_json()), 'tasks': tasks, 'export': exported,
                     'reviewed-scores': batch.to_dict()}
        for name, data in artifacts.items():
            (output / f'{split}-{name}.json').write_text(json.dumps(data, indent=2) + '\n')
    fitted = fit_review_threshold(batches['development'], false_accept_cost=5, false_reject_cost=1)
    result = evaluate_review_threshold(fitted, batches['held_out'])
    (output / 'fit.json').write_text(json.dumps(fitted.to_dict(), indent=2) + '\n')
    (output / 'held-out-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'synthetic_fixture_only': True, 'threshold': fitted.to_dict()['threshold'],
                      'held_out': result['metrics']}, indent=2))


if __name__ == '__main__':
    main()
