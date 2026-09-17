"""Physical-unit forecast diagnostics and explicit tolerance checks."""
from __future__ import annotations

import math
from statistics import NormalDist

from .case_manifest import canonical_json, case_from_dict
from .dynamics import ForecastEvidence
from .result import CaseResult, EvalResult
from .trials import attach_trial, capture_case


def forecast_metrics(forecast: ForecastEvidence, *, interval_level: float = 0.9) -> dict:
    """Per-horizon errors and marginal Gaussian diagnostics, without unit pooling.

    Interval coverage is one observation per coordinate/horizon, not a calibrated
    confidence claim. Missing Gaussian parameters remain unavailable. Endpoints
    after the observed reference prefix are censored and never imputed.
    """
    if isinstance(interval_level, bool) or not isinstance(interval_level, (int, float)) or not 0 < interval_level < 1:
        raise ValueError('interval_level must lie strictly between zero and one')
    data = forecast.data
    rows = []
    z = -NormalDist().inv_cdf((1 - interval_level) / 2)
    for horizon in data['horizons']:
        row = {'horizon': horizon, 'status': data['status'], 'coordinates': []}
        rows.append(row)
        if data['status'] != 'measured':
            row['error'] = data['error']
            continue
        if horizon > len(data['reference']):
            row['status'] = 'censored'
            continue
        prediction, reference = data['prediction'], data['reference'][horizon - 1]
        for i, (name, unit) in enumerate(data['coordinates']):
            predicted, observed = prediction['states'][horizon - 1][i], reference[i]
            error = predicted - observed
            item = {'name': name, 'unit': unit, 'predicted': predicted, 'observed': observed,
                    'error': error, 'absolute_error': abs(error), 'interval': None}
            if prediction.get('standard_deviation') is not None:
                std = prediction['standard_deviation'][horizon - 1][i]
                standardized = error / std
                item['interval'] = {'level': interval_level, 'standard_deviation': std,
                    'lower': predicted - z * std, 'upper': predicted + z * std,
                    'covered': abs(error) <= z * std,
                    'negative_log_density': 0.5 * math.log(2 * math.pi) + math.log(std) + 0.5 * standardized * standardized}
            row['coordinates'].append(item)
        # Overflow is missing arithmetic evidence, not a portable infinite score.
        try:
            canonical_json(row)
        except ValueError:
            row.update(status='metric_error', coordinates=[], error='Non-finite diagnostic arithmetic')
    return {'forecast_digest': forecast.digest, 'interval_level': interval_level, 'rows': rows}


def forecast_case_result(forecast: ForecastEvidence, *, tolerances: dict[str, float]) -> CaseResult:
    """Apply caller-defined absolute-error tolerances; missing horizons are skipped.

    Use an acceptance policy requiring these checks so censored/missing evidence
    cannot pass by omission. Tolerances are physical-unit task decisions; this
    adapter supplies no universal world-model quality threshold.
    """
    data = forecast.data
    names = {name for name, _ in data['coordinates']}
    if set(tolerances) != names or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
        for x in tolerances.values()
    ):
        raise ValueError('Supply a finite nonnegative tolerance for every coordinate')
    case = case_from_dict(data['episode']['case'])
    case.input = canonical_json(data['request'])
    case.expected_output = canonical_json(data['reference'])
    case.reference_output = None
    case.metadata = {**case.metadata, 'dynamics_reference': {
        'episode_digest': data['episode']['digest'], 'contract': data['contract'],
        'coordinates': data['coordinates'], 'horizons': data['horizons']}}
    diagnostics = forecast_metrics(forecast)
    results = []
    for row in diagnostics['rows']:
        measured = {c['name']: c for c in row['coordinates']}
        for name, unit in data['coordinates']:
            check = f'dynamics/{name}/h{row["horizon"]}'
            metadata = {'forecast_digest': forecast.digest, 'model_id': data['model_id'],
                        'tolerance': tolerances[name], 'unit': unit, 'status': row['status']}
            if name not in measured:
                results.append(EvalResult(check, 0.0, False, row['status'], {**metadata, 'skipped': True}))
            else:
                item = measured[name]
                passed = item['absolute_error'] <= tolerances[name]
                results.append(EvalResult(check, float(passed), passed,
                    f'Absolute error {item["absolute_error"]:.6g} {unit}', {**metadata, 'diagnostics': item}))
    result = CaseResult(case.input, canonical_json(data['prediction']), results, tags=case.tags)
    if data['status'] == 'model_error':
        result.model_error = data['error']['message']
    elif data['status'] == 'simulator_error':
        result.evaluator_error = 'Simulator reference error: ' + data['error']['message']
    return attach_trial(result, capture_case(case), origin='dynamics_forecast', latency_known=False)
