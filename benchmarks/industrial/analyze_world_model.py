"""Recompute physical-unit diagnostics and planning outcomes from saved evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from multivon_eval.case_manifest import digest
from multivon_eval.dynamics import ForecastEvidence
from multivon_eval.dynamics_metrics import forecast_metrics
from multivon_eval.episode import EpisodeEvidence


def rmse(values):
    return math.hypot(*values) / math.sqrt(len(values)) if values else None


def analyze(root: Path) -> dict:
    def read(name):
        return json.loads((root / name).read_text())
    config = read('configuration.json')
    parameters = read('models.json')
    training = read('training.json')
    assert parameters['training_digest'] == digest(training)
    seeds, horizons = config['testing_seeds'], config['horizons']
    models = ['action_conditioned', 'action_blind', 'persistence']
    result = {'split': config['split'], 'forecast_sources': len(seeds),
              'training_transitions': len(read('training.json')['actions']), 'models': {},
              'limits': 'Same-environment state-space diagnostics; no video, hidden-memory or real-robot claim'}
    for model in models:
        forecasts = {(seed, variant): ForecastEvidence.from_dict(read(f'forecasts/{model}-{seed}-{variant}.json'))
                     for seed in seeds for variant in ['base', 'offset', 'action0', 'action1']}
        expected_id = f'{model}/v1/{digest(parameters["models"][model])}' if model != 'persistence' else 'persistence/v1'
        for (seed, variant), forecast in forecasts.items():
            data = forecast.data
            assert data['model_id'] == expected_id and data['contract'] == config['contract']
            assert data['episode']['seed'] == seed
            assert data['episode']['case']['source_id'] == f'cartpole:{seed}'
            assert data['action_scope'] == 'caller_planned'
            planned = [int(variant[-1])] if variant.startswith('action') else read('openloop_actions.json')[str(seed)]
            if data['request'] is not None:
                assert data['request']['actions'] == planned
        for seed in seeds:
            base, offset, action0, action1 = [forecasts[(seed, variant)].data
                                            for variant in ['base', 'offset', 'action0', 'action1']]
            if all(item['request'] is not None for item in [base, offset, action0, action1]):
                initial = base['request']['initial_state']
                assert action0['request']['initial_state'] == action1['request']['initial_state'] == initial
                assert offset['request']['initial_state'][1:] == initial[1:]
                assert math.isclose(offset['request']['initial_state'][0] - initial[0], 0.5, abs_tol=1e-6)
        metrics = {key: forecast_metrics(value) for key, value in forecasts.items()}
        coordinates = forecasts[(seeds[0], 'base')].data['coordinates']
        common = [seed for seed in seeds if all(row['status'] == 'measured' for row in metrics[(seed, 'base')]['rows'])]
        model_result = {'forecast_statuses': dict(Counter(f.data['status'] for f in forecasts.values())),
                        'by_horizon': {}, 'common_longest_horizon_sources': len(common),
                        'action_effect': {}, 'state_offset': {}, 'planning': {}}
        result['models'][model] = model_result
        for horizon in horizons:
            rows = {seed: next(r for r in metrics[(seed, 'base')]['rows'] if r['horizon'] == horizon) for seed in seeds}
            summary = {'statuses': dict(Counter(row['status'] for row in rows.values())), 'coordinates': {}}
            model_result['by_horizon'][str(horizon)] = summary
            for index, (name, unit) in enumerate(coordinates):
                points = [r['coordinates'][index] for r in rows.values() if r['status'] == 'measured']
                intervals = [p['interval'] for p in points if p['interval'] is not None]
                paired = [rows[seed]['coordinates'][index]['error'] for seed in common]
                summary['coordinates'][name] = {'unit': unit, 'n': len(points),
                    'rmse': rmse([p['error'] for p in points]), 'common_cohort_n': len(paired),
                    'common_cohort_rmse': rmse(paired), 'uncertainty_n': len(intervals),
                    'coverage_90': sum(i['covered'] for i in intervals) / len(intervals) if intervals else None,
                    'mean_width': sum(i['upper'] - i['lower'] for i in intervals) / len(intervals) if intervals else None,
                    'mean_negative_log_density': sum(i['negative_log_density'] for i in intervals) / len(intervals) if intervals else None}
        for probe, variants, probe_horizons in [('action_effect', ['action0', 'action1'], [1]),
                                                ('state_offset', ['base', 'offset'], horizons)]:
            for horizon in probe_horizons:
                pairs = []
                for seed in seeds:
                    a, b = [next(r for r in metrics[(seed, v)]['rows'] if r['horizon'] == horizon) for v in variants]
                    if a['status'] == b['status'] == 'measured':
                        pairs.append((seed, a['coordinates'], b['coordinates']))
                model_result[probe][str(horizon)] = {}
                for index, (name, unit) in enumerate(coordinates):
                    errors, observed, predicted = [], [], []
                    for _, a, b in pairs:
                        reference_effect = b[index]['observed'] - a[index]['observed']
                        model_effect = b[index]['predicted'] - a[index]['predicted']
                        errors.append(model_effect - reference_effect)
                        observed.append(reference_effect)
                        predicted.append(model_effect)
                    model_result[probe][str(horizon)][name] = {'unit': unit, 'n': len(pairs),
                        'excluded_sources': len(seeds) - len(pairs), 'effect_rmse': rmse(errors),
                        'mean_reference_effect': sum(observed) / len(pairs) if pairs else None,
                        'mean_predicted_effect': sum(predicted) / len(pairs) if pairs else None}
        planning_rows = []
        for seed in config['planning_seeds']:
            data = EpisodeEvidence.from_dict(read(f'episodes/plan-{seed}-{model}.json')).data
            plans = read(f'planning/{seed}-{model}.json')
            assert plans['episode_digest'] == data['digest']
            assert len(plans['plans']) == len(data['steps'])
            for step, selected in zip(data['steps'], plans['plans']):
                assert selected['evaluations'] == len(selected['candidate_costs']) == 64
                assert step['action'] == [selected['plan'][0]]
                assert selected['cost'] == min(c['cost'] for c in selected['candidate_costs'])
            complete = not data['errors'] and data['cleanup']['completed']
            steps = len(data['steps'])
            planning_rows.append({'seed': seed, 'steps': steps,
                'return': sum(row['transition']['reward'] for row in data['steps'] if row['transition']),
                'terminated': data['termination']['terminated'], 'truncated': data['termination']['truncated'],
                'reached_cap': complete and steps == config['planning_max_steps'] and not data['termination']['terminated'],
                'capture_complete': complete, 'errors': data['errors']})
        complete_rows = [row for row in planning_rows if row['capture_complete']]
        model_result['planning'] = {'requested_sources': len(planning_rows), 'measured_sources': len(complete_rows),
            'mean_steps': sum(r['steps'] for r in complete_rows) / len(complete_rows) if complete_rows else None,
            'reached_cap': sum(r['reached_cap'] for r in complete_rows), 'rows': planning_rows}
    result['paired_planning'] = {}
    for comparator in ['action_blind', 'persistence']:
        left = result['models']['action_conditioned']['planning']['rows']
        right = result['models'][comparator]['planning']['rows']
        differences = [{'seed': a['seed'], 'steps_difference': a['steps'] - b['steps']}
                       for a, b in zip(left, right) if a['capture_complete'] and b['capture_complete']]
        result['paired_planning'][comparator] = {'n': len(differences), 'differences': differences,
            'mean_step_difference': sum(d['steps_difference'] for d in differences) / len(differences) if differences else None}
    result['error_controls'] = {name: ForecastEvidence.from_dict(read(f'{name}.json')).data['status']
                               for name in ['model_error', 'simulator_error']}
    return result


def replay(root):
    root = root.resolve()
    checksums = json.loads((root / 'checksums.json').read_text())
    for name, expected in checksums.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing checksum file or path outside bundle')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    observed = analyze(root)
    expected = json.loads((root / 'summary.json').read_text())
    if observed != expected:
        raise ValueError('Recomputed diagnostics differ from saved summary')
    return {'verified_files': len(checksums), 'forecast_sources': observed['forecast_sources'],
            'error_controls': observed['error_controls'], 'target_calls': 0, 'simulator_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    print(json.dumps(replay(args.root) if args.replay else analyze(args.root), indent=2))
