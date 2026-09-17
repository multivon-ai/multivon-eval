"""Experimental vector-state forecasts bound to native environment evidence.

Models and environments stay upstream. This profile records observed prefixes,
not invented post-terminal states, and never supplies future truth to a model.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from time import perf_counter

from .case_manifest import canonical_json, digest
from .episode import EpisodeEvidence


def _contract(model_id, contract, coordinates, horizons):
    if any(not isinstance(v, str) or not v.strip() for v in (model_id, contract)):
        raise ValueError('Supply versioned model_id and task contract')
    if (not isinstance(coordinates, list) or not coordinates or any(
        not isinstance(pair, list) or len(pair) != 2 or any(
            not isinstance(v, str) or not v.strip() for v in pair) for pair in coordinates
    ) or len({pair[0] for pair in coordinates}) != len(coordinates)):
        raise ValueError('Supply ordered coordinate names and explicit units')
    if (not isinstance(horizons, list) or not horizons
            or any(type(h) is not int or h < 1 for h in horizons)
            or len(set(horizons)) != len(horizons)):
        raise ValueError('Horizons must be distinct positive integers')


def _vector(value, width):
    if (not isinstance(value, list) or len(value) != width or any(
        isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in value
    )):
        raise ValueError('State vectors must have the declared width and finite numeric coordinates')


def _prediction(value, steps, width):
    if (not isinstance(value, dict) or set(value) - {'states', 'standard_deviation', 'metadata'}
            or not isinstance(value.get('states'), list) or len(value['states']) != steps):
        raise ValueError('Prediction must contain one state per supplied action')
    for state in value['states']:
        _vector(state, width)
    deviation = value.get('standard_deviation')
    if deviation is not None:
        if not isinstance(deviation, list) or len(deviation) != steps:
            raise ValueError('Uncertainty must align with every predicted state')
        for state in deviation:
            _vector(state, width)
            if any(x <= 0 for x in state):
                raise ValueError('Gaussian marginal standard deviations must be positive')
    if not isinstance(value.get('metadata', {}), dict):
        raise TypeError('Prediction metadata must be an object')
    canonical_json(value)


def _reference(episode, width):
    data = episode.data
    if data['errors'] or not data['cleanup']['completed'] or data['initial'] is None:
        raise ValueError('Simulator capture has errors or incomplete setup/cleanup')
    initial = data['initial']['observation']
    if not isinstance(initial, list) or len(initial) != 1:
        raise ValueError('Expected one vector observation in the native space batch')
    _vector(initial[0], width)
    states, actions = [], []
    for row in data['steps']:
        if row['error'] is not None or row['transition'] is None or row['action'] is None:
            raise ValueError('Simulator transition is missing')
        observed = row['transition']['observation']
        action = row['action']
        if not isinstance(observed, list) or len(observed) != 1 or not isinstance(action, list) or len(action) != 1:
            raise ValueError('Expected native single-observation/action batches')
        _vector(observed[0], width)
        states.append(observed[0])
        actions.append(action[0])
    if not states:
        raise ValueError('No actual simulator transitions were captured')
    return {'initial_state': initial[0], 'actions': actions}, states


@dataclass(frozen=True)
class ForecastEvidence:
    """Detached forecast/reference record. A digest is not an authenticity proof."""
    _json: str

    def __post_init__(self):
        data = self.data
        claimed = data.pop('digest', None)
        if data.get('schema') != 'multivon.dynamics-forecast/v1' or digest(data) != claimed:
            raise ValueError('Invalid forecast schema or digest')
        _contract(data['model_id'], data['contract'], data['coordinates'], data['horizons'])
        episode = EpisodeEvidence.from_dict(data['episode'])
        if data['status'] not in {'measured', 'model_error', 'simulator_error'}:
            raise ValueError('Unknown forecast status')
        if data['status'] != 'measured' and not isinstance(data['error'], dict):
            raise ValueError('Unmeasured forecast must retain its error')
        if data['status'] != 'simulator_error':
            request, reference = _reference(episode, len(data['coordinates']))
            supplied = data['request']
            if (not isinstance(supplied, dict) or set(supplied) != {'initial_state', 'actions'}
                    or not isinstance(supplied['actions'], list)
                    or supplied['initial_state'] != request['initial_state']
                    or supplied['actions'][:len(request['actions'])] != request['actions']
                    or data['reference'] != reference):
                raise ValueError('Forecast reference does not match its bound episode')
            if data['action_scope'] not in {'observed_prefix', 'caller_planned'}:
                raise ValueError('Unknown action scope')
            if data['action_scope'] == 'observed_prefix' and supplied != request:
                raise ValueError('Observed-prefix actions do not match the episode')
        if data['status'] == 'measured':
            _prediction(data['prediction'], len(data['request']['actions']), len(data['coordinates']))
            if data['error'] is not None:
                raise ValueError('Measured forecast cannot contain an execution error')

    @property
    def data(self) -> dict:
        return json.loads(self._json)

    @property
    def digest(self) -> str:
        return self.data['digest']

    @classmethod
    def from_dict(cls, data):
        return cls(canonical_json(data))


def capture_forecast(episode: EpisodeEvidence, predict: Callable[[dict], dict], *,
                     model_id: str, contract: str, coordinates: dict[str, str],
                     horizons: list[int], planned_actions: list | None = None) -> ForecastEvidence:
    """Call a vector-state model once with only initial state and supplied actions.

    ``coordinates`` maps ordered names to physical units. ``predict`` returns
    states, optional positive marginal Gaussian standard deviations and metadata.
    A finite observed prefix is valid even without natural episode termination;
    horizons beyond it are censored. Simulator and model failures stay distinct.
    Callback exceptions are retained; BaseException/cancellation propagates.
    This synchronous capture helper cannot interrupt a blocked model callback.
    Supply actions fixed before reference execution to avoid exposing the actual
    termination time through request length. Without ``planned_actions``, this
    explicitly evaluates an observed prefix whose length is already known.
    """
    if not isinstance(coordinates, dict):
        raise TypeError('Supply ordered coordinate names and explicit units')
    _contract(model_id, contract, [list(pair) for pair in coordinates.items()], horizons)
    if planned_actions is not None and (not isinstance(planned_actions, list) or not planned_actions):
        raise ValueError('planned_actions must be a nonempty portable list')
    if planned_actions is not None:
        planned_actions = json.loads(canonical_json(planned_actions))
    body = {'schema': 'multivon.dynamics-forecast/v1', 'model_id': model_id,
            'contract': contract, 'coordinates': list(coordinates.items()),
            'horizons': sorted(horizons), 'episode': episode.data,
            'action_scope': 'observed_prefix' if planned_actions is None else 'caller_planned',
            'request': None, 'reference': None, 'prediction': None, 'error': None,
            'status': 'simulator_error', 'duration_ms': None}
    try:
        body['request'], body['reference'] = _reference(episode, len(coordinates))
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        body['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        return ForecastEvidence.from_dict({**body, 'digest': digest(body)})
    if planned_actions is not None:
        actual_actions = body['request']['actions']
        if planned_actions[:len(actual_actions)] != actual_actions:
            raise ValueError('Planned actions must match the actually executed prefix')
        body['request']['actions'] = planned_actions
    started = perf_counter()
    try:
        request = json.loads(canonical_json(body['request']))
        raw = predict(request)
        if request != body['request']:
            raise ValueError('Model mutated its request')
        try:
            body['prediction'] = json.loads(canonical_json(raw))
        except (TypeError, ValueError):
            body['prediction_repr'] = repr(raw)
            raise ValueError('Prediction is not portable JSON') from None
        _prediction(body['prediction'], len(body['request']['actions']), len(coordinates))
        body['status'] = 'measured'
    except Exception as exc:  # noqa: BLE001 — retain arbitrary model extension failures
        body['status'] = 'model_error'
        body['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    body['duration_ms'] = (perf_counter() - started) * 1000
    return ForecastEvidence.from_dict({**body, 'digest': digest(body)})
