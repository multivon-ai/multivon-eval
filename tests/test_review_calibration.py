"""Saved-evidence calibration keeps fitting, holdout and source uncertainty separate."""
import copy
import json

import pytest

pytest.importorskip('sklearn')
pytest.importorskip('scipy')

from multivon_eval import CaseManifest, EvalCase, EvalSuite
from multivon_eval.case_manifest import digest
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.integrations.label_studio import export_review_tasks, import_review_annotations
from multivon_eval.review_calibration import (
    ReviewedScores,
    ThresholdFit,
    collect_reviewed_scores,
    evaluate_review_threshold,
    fit_review_threshold,
)


class FixtureScore(Evaluator):
    name = 'fixture_score'

    def evaluate(self, case, output):
        if case.metadata.get('skip'):
            return self._skipped('synthetic missing measurement')
        return self._result(case.metadata['score'], 'Synthetic score only')


def fixture(*, runs=1, unknown=False, skip=False, missing_source=False):
    cases = []
    for split in ['development', 'held_out']:
        for source in range(2):
            for good in [False, True]:
                score = 0.8 if good else 0.2
                if split == 'held_out' and source == 0 and not good:
                    score = 0.9
                cases.append(EvalCase(f'{split}-{source}-{good}', 'fixture',
                    case_id=f'{split}-{source}-{good}', source_id=None if missing_source else f'{split}-{source}',
                    tags=['good' if good else 'bad'],
                    metadata={'score': score, 'good': good, 'skip': skip and source == 0 and not good}))
    manifest = CaseManifest('synthetic review protocol', cases,
        splits={split: [c.case_id for c in cases if c.case_id.startswith(split)] for split in ['development','held_out']})
    batches = []
    for split in ['development', 'held_out']:
        report = EvalSuite('synthetic saved scores').add_cases(manifest.split(split)).add_evaluator(FixtureScore()).run(
            lambda _: 'fixture', runs=runs, verbose=False)
        tasks = export_review_tasks(report, 'fixture_score', rubric='Synthetic labels test the plumbing, not judge quality.')
        exported = copy.deepcopy(tasks)
        for index, task in enumerate(exported):
            label = 'Accept' if task['data']['binding']['case_id'].endswith('True') else 'Reject'
            if unknown and index == 0:
                label = 'Unknown'
            task['annotations'] = [{'id': index * 2 + reviewer, 'completed_by': reviewer,
                'result': [{'from_name': 'verdict', 'to_name': 'output', 'type': 'choices', 'value': {'choices': [label]}},
                           {'from_name': 'review_reason', 'to_name': 'output', 'type': 'textarea',
                            'value': {'text': ['Synthetic fixture annotation']}}]} for reviewer in [1, 2]]
        reviews = import_review_annotations(exported, tasks, reviewer_kind='synthetic')
        batches.append((report, tasks, reviews, manifest, split))
    return batches


def collect(args):
    report, tasks, reviews, manifest, split = args
    return collect_reviewed_scores(report, tasks, reviews, manifest, split=split, reviewer_kind='synthetic')


def fit(batch):
    return fit_review_threshold(batch, false_accept_cost=5, false_reject_cost=1)


def test_saved_review_roundtrip_fit_and_held_out_errors_need_no_new_target_calls():
    development, held = map(collect, fixture())
    fitted = fit(development)
    restored = ThresholdFit.from_dict(json.loads(json.dumps(fitted.to_dict())))
    result = evaluate_review_threshold(restored, ReviewedScores.from_dict(held.to_dict()))
    assert restored.to_dict()['threshold'] == 0.8
    assert result['status'] == 'complete'
    metrics = result['metrics']
    assert metrics['trials'] == 4 and metrics['sources'] == 2
    assert metrics['false_accepts'] == 1 and metrics['false_rejects'] == 0
    assert metrics['source_any_false_accept']['events'] == 1
    assert metrics['source_any_false_accept']['sources'] == 2
    assert metrics['source_any_false_accept']['interval'][0] < 0.5 < metrics['source_any_false_accept']['interval'][1]
    assert metrics['source_any_false_reject']['interval'][1] > 0
    assert result['slices']['bad']['false_accepts'] == 1
    assert result['slices']['good']['source_any_false_accept']['interval'] is None


def test_repeats_do_not_inflate_source_confidence():
    d1, h1 = map(collect, fixture(runs=1))
    d4, h4 = map(collect, fixture(runs=4))
    m1 = evaluate_review_threshold(fit(d1), h1)['metrics']
    m4 = evaluate_review_threshold(fit(d4), h4)['metrics']
    assert m4['false_accepts'] == 4 and m4['trials'] == 16
    assert m1['source_any_false_accept'] == m4['source_any_false_accept']


def test_unknown_reviews_cannot_fit_or_hide_in_held_out_coverage():
    clean = fixture()
    unknown = fixture(unknown=True)
    with pytest.raises(ValueError, match='coverage'):
        fit(collect(unknown[0]))
    result = evaluate_review_threshold(fit(collect(clean[0])), collect(unknown[1]))
    assert result['status'] == 'incomplete'
    assert result['metrics']['measured_trials'] == 3
    assert result['metrics']['complete_sources'] == 1


def test_skipped_measurements_cannot_become_scores_for_fitting():
    batch = collect(fixture(skip=True)[0])
    assert any(r['score'] is None for r in batch.to_dict()['rows'])
    with pytest.raises(ValueError, match='missing measurements'):
        fit(batch)


def test_missing_source_identity_is_not_inferred():
    with pytest.raises(ValueError, match='source_id'):
        collect(fixture(missing_source=True)[0])


def test_fitting_holdout_and_evaluating_development_are_rejected():
    development, held = map(collect, fixture())
    with pytest.raises(ValueError, match='development split'):
        fit(held)
    with pytest.raises(ValueError, match='held_out split'):
        evaluate_review_threshold(fit(development), development)


def test_changed_task_or_review_rubric_cannot_reuse_scores():
    report, tasks, reviews, manifest, split = fixture()[0]
    tasks[0]['data']['output'] = 'different'
    with pytest.raises(ValueError):
        collect_reviewed_scores(report, tasks, reviews, manifest, split=split, reviewer_kind='synthetic')


def reseal(data):
    data = copy.deepcopy(data)
    data.pop('digest')
    return {**data, 'digest': digest(data)}


def test_leakage_and_grader_contract_changes_block_evaluation():
    development, held = map(collect, fixture())
    fitted = fit(development)
    changed = held.to_dict()
    changed['contract']['rubric'] = 'different acceptance criterion'
    with pytest.raises(ValueError, match='contract changed'):
        evaluate_review_threshold(fitted, ReviewedScores.from_dict(reseal(changed)))
    changed = held.to_dict()
    changed['rows'][0]['source_id'] = development.to_dict()['rows'][0]['source_id']
    with pytest.raises(ValueError, match='leakage'):
        evaluate_review_threshold(fitted, ReviewedScores.from_dict(reseal(changed)))


def test_mutated_frozen_fit_is_detected():
    fitted = fit(collect(fixture()[0])).to_dict()
    fitted['threshold'] = 0.1
    with pytest.raises(ValueError, match='digest'):
        ThresholdFit.from_dict(fitted)


def test_reject_all_is_explicit_instead_of_nonfinite_json_threshold():
    batch = collect(fixture()[0]).to_dict()
    for row in batch['rows']:
        row['score'] = 1.0
    fitted = fit(ReviewedScores.from_dict(reseal(batch))).to_dict()
    assert fitted['rule'] == 'reject_all' and fitted['threshold'] is None
    json.dumps(fitted, allow_nan=False)


@pytest.mark.parametrize('cost', [0, -1, True, float('nan'), float('inf')])
def test_invalid_error_costs_are_rejected(cost):
    with pytest.raises(ValueError):
        fit_review_threshold(collect(fixture()[0]), false_accept_cost=cost, false_reject_cost=1)
