"""Promote inspected evidence into explicitly reviewed development regression cases."""
from __future__ import annotations

from dataclasses import asdict

from .case_manifest import CaseManifest, canonical_json, case_from_dict
from .integrations.label_studio import export_review_tasks, reconcile_reviews
from .result import EvalReport
from .trials import TrialRecord, trial_integrity_issues


def regression_candidate(trial: TrialRecord) -> dict:
    """Export a review candidate, never infer an oracle from the observed answer."""
    data = trial.data
    return {'schema': 'multivon.regression-candidate/v1', 'requires_review': True,
            'trial_digest': trial.digest, 'case_id': data['case_id'],
            'case_digest': data['case_digest'], 'source_case': data['case'],
            'intended_split': 'development',
            'instructions': 'Review the original evidence and author the expected result before promotion. '
                            'Preserve source grouping. An inspected holdout is no longer untouched validation data.'}


def promote_reviewed_trial(report: EvalReport, trial_digest: str, original_tasks: list[dict], reviews: list, *,
                           evaluator: str, case_id: str, expected_output: str, rationale: str,
                           reviewer_kind: str, min_reviewers: int = 2,
                           source_id: str | None = None) -> CaseManifest:
    """Create a development-only manifest after bound review consensus.

    The caller explicitly supplies the new oracle and its rationale. Review
    consensus concerns the saved execution, not independent proof that the new
    oracle is correct. Label Studio owns review collection and identity.
    The observed trace and callable reference are never used as future execution.
    """
    if not isinstance(expected_output, str):
        raise TypeError('Supply an explicit expected output; the failed output is not an oracle')
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError('Explain the new expectation and its source')
    if not isinstance(case_id, str) or not case_id.strip() or any(r.case_id == case_id for r in report.case_results):
        raise ValueError('Supply a new nonempty regression case ID')
    if any(not row.trials or row.evidence_error or trial_integrity_issues(row) for row in report.case_results):
        raise ValueError('Promotion requires intact saved trial evidence')
    matches = [(row, trial) for row in report.case_results for trial in row.trials if trial.digest == trial_digest]
    if len(matches) != 1:
        raise ValueError('Select exactly one retained trial by its digest')
    row, trial = matches[0]
    selected = [task for task in original_tasks if task.get('data', {}).get('binding', {}).get('trial_digest') == trial_digest
                and task['data']['binding'].get('evaluator') == evaluator]
    if len(selected) != 1:
        raise ValueError('Exactly one original review task must bind this trial and evaluator')
    task = selected[0]
    generated = export_review_tasks(EvalReport(report.suite_name, [row]), evaluator, rubric=task['data']['rubric'])
    if not any(canonical_json(task) == canonical_json(item) for item in generated):
        raise ValueError('Review task does not match the original report evidence')
    consensus = reconcile_reviews(original_tasks, reviews, min_reviewers=min_reviewers, reviewer_kind=reviewer_kind)
    decision = next(d for d in consensus['decisions'] if d['review_key'] == task['data']['review_key'])
    if decision['status'] != 'consensus':
        raise ValueError('Resolve missing, unknown or disagreeing reviews before promotion')
    case = case_from_dict(trial.data['case'])
    if case.source_id is not None and source_id not in {None, case.source_id}:
        raise ValueError('Preserve the original source group when promoting a case')
    source_id = case.source_id or source_id
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError('Supply an explicit source group; do not invent independent sources from trial IDs')
    parent_id = trial.data['case_id']
    case.case_id, case.source_id = case_id, source_id
    case.expected_output, case.reference_output, case.agent_trace = expected_output, None, None
    case.revision = 'regression-v1'
    case.metadata = {**case.metadata, 'multivon_regression': {
        'parent_case_id': parent_id, 'parent_trial_digest': trial_digest,
        'intended_split': 'development', 'expectation_rationale': rationale}}
    annotations = [asdict(r) for r in reviews if r.review_key == task['data']['review_key']]
    return CaseManifest('reviewed development regressions', [case], splits={'development': [case_id]},
        provenance={'source_trial_digest': trial_digest, 'source_case_digest': trial.data['case_digest'],
                    'source_trial_status': trial.data['status'],
                    'source_report_evidence_issues': list(report.evidence_issues),
                    'source_trial_evidence_gaps': trial.data.get('evidence_gaps', []),
                    'review_task': task, 'reviews': annotations, 'review_consensus': decision,
                    'reviewer_kind': reviewer_kind, 'min_reviewers': min_reviewers,
                    'expectation_rationale': rationale,
                    'limits': 'Review identity and the new oracle are caller assertions. '
                              'Inspected/promoted sources cannot supply an untouched held-out evaluation.'})
