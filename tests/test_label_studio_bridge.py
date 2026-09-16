"""Native annotation exchange never invents consensus or silently moves labels."""
import copy
import dataclasses
import json

import pytest

from multivon_eval import EvalCase, EvalSuite, ExactMatch
from multivon_eval.integrations.label_studio import (
    LABEL_CONFIG,
    export_review_tasks,
    import_review_annotations,
    reconcile_reviews,
)


def tasks(runs=1, target=None):
    report = EvalSuite('review').add_case(EvalCase('q', 'yes', source_id='document')).add_evaluator(ExactMatch()).run(
        target or (lambda _: 'yes'), runs=runs, verbose=False)
    return report, export_review_tasks(report, 'exact_match', rubric='Compare the answer with the reference.')


def annotation(identifier=1, reviewer=10, label='Accept', cancelled=False):
    return {'id': identifier, 'completed_by': reviewer, 'was_cancelled': cancelled,
            'created_at': '2026-09-17T00:00:00Z', 'result': [
                {'from_name': 'verdict', 'to_name': 'output', 'type': 'choices', 'value': {'choices': [label]}},
                {'from_name': 'review_reason', 'to_name': 'output', 'type': 'textarea',
                 'value': {'text': ['Synthetic fixture review; no independent human label.']}}]}


def exported(original, annotations):
    result = copy.deepcopy(original)
    result[0]['id'] = 999  # Label Studio owns server task IDs.
    result[0]['annotations'] = annotations
    return result


def test_export_retains_trial_bindings_without_revealing_grader_verdicts():
    report, original = tasks(runs=2)
    assert len(original) == 2
    assert len({t['data']['review_key'] for t in original}) == 2
    assert {t['data']['binding']['trial_digest'] for t in original} == {t.digest for t in report.case_results[0].trials}
    assert all('score' not in t['data'] and 'passed' not in t['data'] for t in original)
    assert json.loads(json.dumps(original)) == original
    import xml.etree.ElementTree as ET
    root = ET.fromstring(LABEL_CONFIG)
    assert root.find('Choices').attrib['toName'] == 'output'


def test_real_report_annotation_round_trip_and_consensus_preserves_original():
    report, original = tasks()
    snapshot = report.to_json()
    labels = import_review_annotations(exported(original, [annotation(), annotation(2, 20)]), original,
                                      reviewer_kind='synthetic')
    result = reconcile_reviews(original, labels, reviewer_kind='synthetic')
    assert result['consensus_tasks'] == 1
    assert result['decisions'][0]['label'] is True
    assert result['decisions'][0]['measured_reviewers'] == 2
    assert report.to_json() == snapshot
    assert json.loads(json.dumps(labels[0].to_dict()))['reviewer_kind'] == 'synthetic'
    assert reconcile_reviews(original, labels)['consensus_tasks'] == 0  # Never relabel machine/fixture work as human.


@pytest.mark.parametrize('labels,status', [(['Accept', 'Reject'], 'disagreement'),
    (['Accept', 'Unknown'], 'incomplete'), (['Unknown', 'Unknown'], 'incomplete'),
    (['Reject', 'Reject'], 'consensus')])
def test_disagreement_and_unknown_are_explicit(labels, status):
    _, original = tasks()
    records = import_review_annotations(exported(original, [annotation(i, i, label) for i, label in enumerate(labels)]),
                                        original, reviewer_kind='synthetic')
    decision = reconcile_reviews(original, records, reviewer_kind='synthetic')['decisions'][0]
    assert decision['status'] == status
    assert decision['label'] is (False if status == 'consensus' else None)


def test_predictions_drafts_and_omitted_tasks_cannot_supply_completed_reviews():
    _, original = tasks(runs=2)
    payload = exported(original, [annotation()])[:1]
    payload[0]['predictions'] = [annotation(2, 20)]
    payload[0]['drafts'] = [annotation(3, 30)]
    records = import_review_annotations(payload, original, reviewer_kind='synthetic')
    result = reconcile_reviews(original, records, reviewer_kind='synthetic')
    assert len(records) == 1 and result['consensus_tasks'] == 0
    assert result['tasks'] == 2 and len(result['decisions']) == 2


def test_cancelled_annotations_remain_unknown():
    _, original = tasks()
    records = import_review_annotations(exported(original, [annotation(cancelled=True)]), original, reviewer_kind='synthetic')
    assert records[0].cancelled and records[0].label is None
    assert reconcile_reviews(original, records, min_reviewers=1, reviewer_kind='synthetic')['consensus_tasks'] == 0


@pytest.mark.parametrize('field', ['input', 'output', 'rubric', 'reference', 'trace'])
def test_edited_task_data_cannot_reuse_a_label(field):
    _, original = tasks()
    payload = exported(original, [annotation()])
    payload[0]['data'][field] = 'different evidence'
    with pytest.raises(ValueError, match='differs'):
        import_review_annotations(payload, original, reviewer_kind='synthetic')


def test_modified_original_contract_is_also_rejected():
    _, original = tasks()
    original[0]['data']['input'] = 'changed'
    with pytest.raises(ValueError, match='digest'):
        import_review_annotations(original, original, reviewer_kind='synthetic')


def test_duplicate_reviewer_requires_adjudication_not_extra_votes():
    _, original = tasks()
    records = import_review_annotations(exported(original, [annotation(), annotation(2)]), original, reviewer_kind='synthetic')
    with pytest.raises(ValueError, match='adjudication'):
        reconcile_reviews(original, records, reviewer_kind='synthetic')


@pytest.mark.parametrize('mutation', ['id', 'reviewer', 'verdict', 'reason', 'type', 'destination', 'multi_choice'])
def test_malformed_annotations_do_not_become_ground_truth(mutation):
    _, original = tasks()
    row = annotation()
    if mutation == 'id': row.pop('id')
    elif mutation == 'reviewer': row.pop('completed_by')
    elif mutation == 'verdict': row['result'] = row['result'][1:]
    elif mutation == 'reason': row['result'] = row['result'][:1]
    elif mutation == 'type': row['result'][0]['type'] = 'textarea'
    elif mutation == 'destination': row['result'][0]['to_name'] = 'different'
    else: row['result'][0]['value']['choices'] = ['Accept', 'Reject']
    with pytest.raises(ValueError):
        import_review_annotations(exported(original, [row]), original, reviewer_kind='synthetic')


def test_model_execution_error_cannot_be_reviewed_as_accepted_quality():
    def fail(_): raise RuntimeError('offline execution failure')
    _, original = tasks(target=fail)
    with pytest.raises(ValueError, match='execution error'):
        import_review_annotations(exported(original, [annotation()]), original, reviewer_kind='synthetic')


def test_missing_trial_evidence_cannot_be_exported():
    report, _ = tasks()
    report.case_results[0].trials = ()
    with pytest.raises(ValueError, match='intact saved trials'):
        export_review_tasks(report, 'exact_match', rubric='Compare exactly')


def test_direct_annotation_construction_cannot_smuggle_numeric_truth():
    _, original = tasks()
    record = import_review_annotations(exported(original, [annotation()]), original, reviewer_kind='synthetic')[0]
    with pytest.raises(ValueError, match='boolean'):
        dataclasses.replace(record, label=1)
    with pytest.raises(ValueError, match='binding'):
        reconcile_reviews(original, [dataclasses.replace(record, source_id='other')], reviewer_kind='synthetic')


def test_upstream_sdk_validates_config_and_generates_importable_regions():
    sdk = pytest.importorskip('label_studio_sdk.label_interface')
    interface = sdk.LabelInterface(LABEL_CONFIG)
    interface.validate()
    _, original = tasks()
    assert interface.validate_task(original[0])
    record = annotation()
    record['result'] = [json.loads(interface.get_tag(name).label(value).to_json())
                        for name, value in [('verdict', 'Reject'),
                                            ('review_reason', 'SDK-generated synthetic fixture')]]
    assert interface.validate_annotation(record)
    imported = import_review_annotations(exported(original, [record]), original, reviewer_kind='synthetic')
    assert imported[0].label is False


@pytest.mark.parametrize('empty', [None, {}])
def test_native_cancelled_annotation_can_have_no_results(empty):
    _, original = tasks()
    record = annotation(cancelled=True)
    record['result'] = empty
    imported = import_review_annotations(exported(original, [record]), original, reviewer_kind='synthetic')
    assert imported[0].label is None and imported[0].cancelled


@pytest.mark.parametrize('bad', [None, 2, [], {'value': None}, {'value': {'choices': [[]]}}])
def test_malformed_regions_fail_closed(bad):
    _, original = tasks()
    record = annotation()
    record['result'][0] = bad
    with pytest.raises(ValueError):
        import_review_annotations(exported(original, [record]), original, reviewer_kind='synthetic')
