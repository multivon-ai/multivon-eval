"""Saved-score threshold selection with scikit-learn and source-level SciPy CIs.

These are empirical development fits, not calibrated probabilities or automatic
release approvals. Reviewer identities and source independence remain assertions.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .case_manifest import CaseManifest, canonical_json, digest
from .integrations.label_studio import (
    ReviewAnnotation,
    ReviewerKind,
    export_review_tasks,
    reconcile_reviews,
)
from .result import EvalReport


def _seal(body: dict) -> str:
    return canonical_json({**body, 'digest': digest(body)})


def _open(payload: str, schema: str) -> dict:
    data = json.loads(payload)
    claimed = data.pop('digest', None)
    if data.get('schema') != schema or digest(data) != claimed:
        raise ValueError('Artifact schema or content digest does not match')
    return {**data, 'digest': claimed}


def _number(value, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{name} must be a finite {"positive " if positive else ""}number')
    return float(value)


@dataclass(frozen=True)
class ReviewedScores:
    """Detached saved-score and review evidence. Serialize with ``to_dict``."""
    _json: str

    def __post_init__(self):
        data = self.to_dict()
        if data['split'] not in {'development', 'held_out'}:
            raise ValueError('Use explicit development or held_out split names')
        seen = set()
        if not data['rows']:
            raise ValueError('No reviewed trials')
        for row in data['rows']:
            for key in ('trial_digest', 'case_id', 'source_id', 'review_key'):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f'Each trial needs an explicit {key}')
            if row['trial_digest'] in seen:
                raise ValueError('Duplicate trial')
            seen.add(row['trial_digest'])
            if row['label'] is not None and type(row['label']) is not bool:
                raise ValueError('Labels must be boolean or unknown')
            if row['score'] is not None and not 0 <= _number(row['score'], 'score') <= 1:
                raise ValueError('Scores must be between zero and one')

    def to_dict(self) -> dict:
        return _open(self._json, 'multivon.reviewed-scores/v1')

    @classmethod
    def from_dict(cls, data: dict) -> ReviewedScores:
        return cls(canonical_json(data))


def collect_reviewed_scores(report: EvalReport, tasks: list[dict],
                            reviews: list[ReviewAnnotation], manifest: CaseManifest, *,
                            split: str, reviewer_kind: ReviewerKind = 'human',
                            min_reviewers: int = 2) -> ReviewedScores:
    """Join consensus labels to the exact original scores without calling a model.

    The manifest must freeze both source-disjoint development/held_out splits.
    All selected cases/trials remain in the artifact, including missing labels,
    skipped graders and execution errors. Fitting requires complete coverage.
    """
    if split not in {'development', 'held_out'}:
        raise ValueError('Use development or held_out')
    definition = manifest.manifest
    if set(definition['splits']) != {'development', 'held_out'} or any(
            not ids for ids in definition['splits'].values()):
        raise ValueError('Freeze nonempty development and held_out splits in the manifest')
    if any(not row['case'].get('source_id') for row in definition['cases']):
        raise ValueError('Every manifest case needs an explicit source_id')
    consensus = reconcile_reviews(tasks, reviews, min_reviewers=min_reviewers, reviewer_kind=reviewer_kind)
    evaluator = tasks[0]['data']['binding']['evaluator']
    rubric = tasks[0]['data']['rubric']
    regenerated = export_review_tasks(report, evaluator, rubric=rubric)
    by_key = lambda values: {t['data']['review_key']: t['data'] for t in values}
    if canonical_json(by_key(regenerated)) != canonical_json(by_key(tasks)):
        raise ValueError('Task set does not match the complete report and frozen rubric')
    cases = {r['id']: r for r in definition['cases']}
    ids = [r.case_id for r in report.case_results]
    if len(ids) != len(set(ids)) or set(ids) != set(definition['splits'][split]):
        raise ValueError('Report must contain each selected split case exactly once')
    if any(r.case_digest != cases[r.case_id]['digest'] for r in report.case_results):
        raise ValueError('Report case definitions differ from the frozen manifest')
    if report.suite_lock is None:
        raise ValueError('Recorded grader configuration is required')
    fingerprints = [asdict(e) for e in report.suite_lock.evaluators if e.name == evaluator]
    if len(fingerprints) != 1:
        raise ValueError('Exactly one recorded grader configuration is required')
    decisions = {d['review_key']: d for d in consensus['decisions']}
    trials = {t.digest: t.data for r in report.case_results for t in r.trials}
    rows = []
    for task in tasks:
        data = task['data']
        trial = trials[data['binding']['trial_digest']]
        grade = next(g for g in trial['evaluators'] if g['name'] == evaluator)
        measurement_error = (trial['status'] not in {'passed', 'failed_quality'} or
                             bool(grade['metadata'].get('skipped')))
        score = None if measurement_error else grade['score']
        review = decisions[data['review_key']]
        rows.append({'trial_digest': trial['digest'], 'review_key': data['review_key'],
                     'case_id': trial['case_id'], 'source_id': trial['case']['source_id'],
                     'tags': trial['case']['tags'], 'score': score, 'label': review['label'],
                     'review_status': review['status'], 'measurement_error': measurement_error})
    contract = {'manifest_digest': manifest.digest, 'evaluator': evaluator,
                'grader': fingerprints[0], 'library_version': report.suite_lock.library_version,
                'calibration_version': report.suite_lock.calibration_version, 'rubric': rubric,
                'reviewer_kind': reviewer_kind, 'min_reviewers': min_reviewers}
    return ReviewedScores(_seal({'schema': 'multivon.reviewed-scores/v1', 'split': split,
        'contract': contract, 'rows': sorted(rows, key=lambda r: r['trial_digest']),
        'reviews': sorted([r.to_dict() for r in reviews], key=lambda r: (r['review_key'], r['annotation_id'])),
        'report_evidence_issues': list(report.evidence_issues)}))


@dataclass(frozen=True)
class ThresholdFit:
    """A frozen development decision; a digest is not a trusted signature."""
    _json: str

    def __post_init__(self):
        data = self.to_dict()
        if data['rule'] not in {'score_gte', 'reject_all'}:
            raise ValueError('Unsupported threshold rule')
        if data['rule'] == 'score_gte':
            if not 0 <= _number(data['threshold'], 'threshold') <= 1:
                raise ValueError('Threshold must be between zero and one')
        elif data['threshold'] is not None:
            raise ValueError('reject_all has no numeric threshold')

    def to_dict(self) -> dict:
        return _open(self._json, 'multivon.threshold-fit/v1')

    @classmethod
    def from_dict(cls, data: dict) -> ThresholdFit:
        return cls(canonical_json(data))


def fit_review_threshold(development: ReviewedScores, *, false_accept_cost: float,
                         false_reject_cost: float) -> ThresholdFit:
    """Minimize empirical source-balanced error cost on development data only.

    scikit-learn supplies the complete ROC threshold candidates. Every source
    has equal total weight regardless of variant/epoch count. Costs must be
    chosen before inspecting held-out outcomes. Higher scores mean acceptance.
    Ties choose the highest threshold, including the reject-all rule.
    """
    try:
        import numpy as np
        import sklearn
        from sklearn.metrics import roc_curve
    except ImportError as exc:
        raise ImportError('Install multivon-eval[review] for threshold analysis') from exc
    fa_cost = _number(false_accept_cost, 'false_accept_cost', positive=True)
    fr_cost = _number(false_reject_cost, 'false_reject_cost', positive=True)
    data = development.to_dict()
    rows = data['rows']
    if data['split'] != 'development':
        raise ValueError('Threshold fitting accepts only the development split')
    if data['report_evidence_issues'] or any(r['score'] is None or r['label'] is None for r in rows):
        raise ValueError('Resolve all missing measurements and review coverage before fitting')
    if {r['label'] for r in rows} != {False, True}:
        raise ValueError('Development labels must contain both accepted and rejected outputs')
    counts = Counter(r['source_id'] for r in rows)
    weights = np.array([1 / counts[r['source_id']] for r in rows])
    labels = np.array([r['label'] for r in rows])
    scores = np.array([r['score'] for r in rows])
    fpr, tpr, thresholds = roc_curve(labels, scores, sample_weight=weights, drop_intermediate=False)
    risks = (fa_cost * fpr * weights[~labels].sum() +
             fr_cost * (1 - tpr) * weights[labels].sum()) / weights.sum()
    index = int(np.argmin(risks))
    threshold = float(thresholds[index])
    return ThresholdFit(_seal({'schema': 'multivon.threshold-fit/v1',
        'rule': 'score_gte' if math.isfinite(threshold) else 'reject_all',
        'threshold': threshold if math.isfinite(threshold) else None,
        'contract': data['contract'], 'development_digest': data['digest'],
        'development_sources': sorted(counts), 'development_cases': sorted({r['case_id'] for r in rows}),
        'development_trials': sorted(r['trial_digest'] for r in rows),
        'development_n': len(rows), 'development_sources_n': len(counts),
        'false_accept_cost': fa_cost, 'false_reject_cost': fr_cost,
        'development_risk': float(risks[index]), 'sklearn_version': sklearn.__version__,
        'scope': 'Empirical development fit; no held-out guarantee or probability calibration'}))


def _interval(events: int, n: int, confidence: float) -> dict:
    from scipy.stats import binomtest
    ci = binomtest(events, n).proportion_ci(confidence_level=confidence, method='exact') if n else None
    return {'events': events, 'sources': n, 'rate': events / n if n else None,
            'interval': [float(ci.low), float(ci.high)] if ci else None,
            'confidence': confidence, 'method': 'Clopper-Pearson; assumes independent sampled sources'}


def _metrics(rows: list[dict], fit: dict, confidence: float) -> dict:
    from sklearn.metrics import confusion_matrix
    measured = [r for r in rows if r['score'] is not None and r['label'] is not None]
    predicted = lambda r: fit['rule'] == 'score_gte' and r['score'] >= fit['threshold']
    tn, fp, fn, tp = confusion_matrix([r['label'] for r in measured],
        [predicted(r) for r in measured], labels=[False, True]).ravel().tolist() if measured else (0, 0, 0, 0)
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_id']].append(row)
    complete = [rs for rs in groups.values() if all(r['score'] is not None and r['label'] is not None for r in rs)]
    with_bad = [rs for rs in complete if any(not r['label'] for r in rs)]
    with_good = [rs for rs in complete if any(r['label'] for r in rs)]
    fa = sum(any(not r['label'] and predicted(r) for r in rs) for rs in with_bad)
    fr = sum(any(r['label'] and not predicted(r) for r in rs) for rs in with_good)
    return {'trials': len(rows), 'measured_trials': len(measured), 'sources': len(groups),
            'complete_sources': len(complete), 'false_accepts': fp, 'false_rejects': fn,
            'true_accepts': tp, 'true_rejects': tn,
            'false_accept_rate': fp / (fp + tn) if fp + tn else None,
            'false_reject_rate': fn / (fn + tp) if fn + tp else None,
            'source_any_false_accept': _interval(fa, len(with_bad), confidence),
            'source_any_false_reject': _interval(fr, len(with_good), confidence)}


def evaluate_review_threshold(fit: ThresholdFit, held_out: ReviewedScores, *,
                              confidence: float = 0.95) -> dict:
    """Apply a frozen fit once to held-out saved scores; no fitting or model calls.

    Source-event CIs measure at least one error per completely reviewed source,
    conditional on that source containing a reference-negative/positive output.
    They are not per-trial CIs. Missing sources/labels remain explicit. Repeated
    adaptive use of this holdout invalidates a fresh-test interpretation.
    """
    try:
        import scipy
        import sklearn
    except ImportError as exc:
        raise ImportError('Install multivon-eval[review] for threshold analysis') from exc
    confidence = _number(confidence, 'confidence')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be strictly between zero and one')
    frozen, data = fit.to_dict(), held_out.to_dict()
    if data['split'] != 'held_out':
        raise ValueError('Evaluation requires the held_out split')
    if canonical_json(frozen['contract']) != canonical_json(data['contract']):
        raise ValueError('Manifest, rubric, grader or reviewer contract changed after fitting')
    rows = data['rows']
    for field, saved in [('source_id', 'development_sources'), ('case_id', 'development_cases'),
                         ('trial_digest', 'development_trials')]:
        if {r[field] for r in rows} & set(frozen[saved]):
            raise ValueError('Development/held-out evidence leakage')
    metrics = _metrics(rows, frozen, confidence)
    tags = sorted({tag for r in rows for tag in r['tags']})
    complete = metrics['measured_trials'] == metrics['trials'] and not data['report_evidence_issues']
    both_labels = {r['label'] for r in rows if r['label'] is not None} == {False, True}
    trial_decisions = []
    for row in rows:
        prediction = (None if row['score'] is None else frozen['rule'] == 'score_gte'
                      and row['score'] >= frozen['threshold'])
        outcome = ('unknown' if prediction is None or row['label'] is None else
                   'false_accept' if prediction and not row['label'] else
                   'false_reject' if not prediction and row['label'] else 'agreement')
        trial_decisions.append({**row, 'predicted_accept': prediction, 'outcome': outcome})
    body = {'schema': 'multivon.threshold-evaluation/v1', 'fit_digest': frozen['digest'],
            'held_out_digest': data['digest'], 'status': 'complete' if complete and both_labels else 'incomplete',
            'metrics': metrics, 'slices': {tag: _metrics([r for r in rows if tag in r['tags']], frozen, confidence)
                                         for tag in tags},
            'report_evidence_issues': data['report_evidence_issues'],
            'review_status_counts': dict(Counter(r['review_status'] for r in rows)),
            'trial_decisions': trial_decisions,
            'scipy_version': scipy.__version__, 'sklearn_version': sklearn.__version__,
            'scope': 'Judge agreement with supplied labels; no automatic application release decision. '
                     'Source independence and reviewer identity are not authenticated. '
                     'Slice intervals are marginal, without multiplicity correction.'}
    return json.loads(_seal(body))
