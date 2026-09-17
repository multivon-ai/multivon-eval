import math

import pytest

gym = pytest.importorskip('gymnasium')

from multivon_eval import AcceptancePolicy, CheckRequirement, EvalCase, EvalReport
from multivon_eval.case_manifest import digest
from multivon_eval.dynamics import ForecastEvidence, capture_forecast
from multivon_eval.dynamics_metrics import forecast_case_result, forecast_metrics
from multivon_eval.integrations.gymnasium import capture_episode

COORDINATES = {'x': 'm', 'velocity': 'm/s', 'angle': 'rad', 'angular_velocity': 'rad/s'}


def episode(*, broken=False):
    holder = {}
    class Broken(gym.Wrapper):
        def step(self, action):
            raise RuntimeError('simulator offline')
    def factory():
        env = gym.make('CartPole-v1')
        holder['env'] = env
        return Broken(env) if broken else env
    def interact(env, initial, info):
        for action in [0, 1, 0]:
            env.step(action)
    return capture_episode(factory, interact, case=EvalCase('predict', case_id='episode', source_id='seed-1'),
                           environment_id='cartpole/v1', observer_id='native-state/v1', seed=1, max_steps=3,
                           observe=lambda: {'state': holder['env'].unwrapped.state.tolist()
                                            if hasattr(holder['env'].unwrapped.state, 'tolist')
                                            else list(holder['env'].unwrapped.state)})


def forecast(predict=None, reference=None, **kwargs):
    return capture_forecast(reference or episode(), predict or (lambda request: {
        'states': [request['initial_state'] for _ in request['actions']]}),
        model_id='test/v1', contract='vector/v1', coordinates=COORDINATES,
        horizons=kwargs.pop('horizons', [1, 3, 4]), **kwargs)


def test_native_prefix_is_measured_but_future_is_censored():
    record = forecast()
    assert record.data['status'] == 'measured'
    assert record.data['episode']['issues']  # no natural termination is not invented task success
    rows = forecast_metrics(record)['rows']
    assert [r['status'] for r in rows] == ['measured', 'measured', 'censored']
    assert rows[0]['coordinates'][0]['interval'] is None
    restored = ForecastEvidence.from_dict(record.data)
    assert restored.digest == record.digest
    record.data['request']['initial_state'][0] = 999
    assert record.data['request']['initial_state'][0] != 999


def test_model_sees_only_initial_observation_and_actions():
    def predict(request):
        assert set(request) == {'initial_state', 'actions'}
        assert request['actions'] == [0, 1, 0]
        return {'states': [request['initial_state']] * 3}
    assert forecast(predict).data['status'] == 'measured'


def test_gaussian_diagnostics_against_known_values():
    reference = episode()
    truth = [row['transition']['observation'][0] for row in reference.data['steps']]
    record = forecast(lambda _: {'states': truth, 'standard_deviation': [[1] * 4] * 3}, reference)
    point = forecast_metrics(record)['rows'][0]['coordinates'][0]
    assert point['absolute_error'] == 0
    assert point['interval']['covered'] is True
    assert point['interval']['negative_log_density'] == pytest.approx(math.log(2 * math.pi) / 2)
    result = forecast_case_result(record, tolerances=dict.fromkeys(COORDINATES, 0))
    assert sum(r.passed for r in result.results) == 8
    assert sum(r.metadata.get('skipped', False) for r in result.results) == 4
    assert len(result.trials) == 1


@pytest.mark.parametrize('output', [
    {'states': []}, {'states': [[0] * 3] * 3}, {'states': [[True] * 4] * 3},
    {'states': [[float('nan')] * 4] * 3},
    {'states': [[0] * 4] * 3, 'standard_deviation': [[0] * 4] * 3},
    {'states': [[0] * 4] * 3, 'standard_deviation': [[1] * 4]},
    {'states': [[0] * 4] * 3, 'unknown': 'unsupported'},
])
def test_invalid_predictions_are_model_errors_not_scores(output):
    record = forecast(lambda _: output)
    assert record.data['status'] == 'model_error'
    assert all(r['status'] == 'model_error' for r in forecast_metrics(record)['rows'])
    result = forecast_case_result(record, tolerances=dict.fromkeys(COORDINATES, 1))
    assert result.status == 'model_error'


def test_model_error_and_simulator_error_are_distinct():
    def broken(_):
        raise RuntimeError('model offline')
    record = forecast(broken)
    assert record.data['status'] == 'model_error'
    assert record.data['error']['message'] == 'model offline'
    reference = episode(broken=True)
    record = forecast(lambda _: pytest.fail('model must not run'), reference)
    assert record.data['status'] == 'simulator_error'
    assert record.data['episode']['errors'][0]['phase'] == 'step'
    result = forecast_case_result(record, tolerances=dict.fromkeys(COORDINATES, 1))
    assert result.status == 'evaluator_error'


def test_request_mutation_and_rehashed_reference_changes_rejected():
    def mutate(request):
        request['initial_state'][0] = 100
        return {'states': [[0] * 4] * 3}
    assert forecast(mutate).data['status'] == 'model_error'
    record = forecast().data
    record['reference'][0][0] = 100
    record.pop('digest')
    with pytest.raises(ValueError, match='bound episode'):
        ForecastEvidence.from_dict({**record, 'digest': digest(record)})


@pytest.mark.parametrize('horizons', [[0], [True], [1, 1], [1.5], []])
def test_invalid_horizons_rejected(horizons):
    with pytest.raises(ValueError, match='Horizons'):
        forecast(horizons=horizons)


def test_overflow_becomes_metric_error():
    record = forecast(lambda _: {'states': [[1e308] * 4] * 3,
                                 'standard_deviation': [[1e-308] * 4] * 3})
    assert forecast_metrics(record)['rows'][0]['status'] == 'metric_error'


def test_planned_request_does_not_expose_reference_termination_length():
    def predict(request):
        assert len(request['actions']) == 10
        return {'states': [request['initial_state']] * 10}
    record = forecast(predict, planned_actions=[0, 1, 0] + [1] * 7)
    assert record.data['action_scope'] == 'caller_planned'
    assert len(record.data['reference']) == 3
    assert forecast_metrics(record)['rows'][-1]['status'] == 'censored'
    with pytest.raises(ValueError, match='executed prefix'):
        forecast(planned_actions=[1, 1, 0])


def test_required_censored_horizon_cannot_pass_acceptance():
    record = forecast()
    row = forecast_case_result(record, tolerances=dict.fromkeys(COORDINATES, 100))
    report = EvalReport('forecast', [row])
    observed = AcceptancePolicy((CheckRequirement('dynamics/x/h1'),))
    missing = AcceptancePolicy((CheckRequirement('dynamics/x/h4'),))
    assert observed.evaluate(report).decision == 'accept'
    assert missing.evaluate(report).decision == 'indeterminate'
