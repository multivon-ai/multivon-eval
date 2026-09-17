"""Run the frozen Gymnasium/scikit-learn/SciPy state-space experiment locally."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
from cartpole_models import DeltaModel, Persistence, plan
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

from multivon_eval import EvalCase, EvalReport
from multivon_eval.case_manifest import canonical_json, digest
from multivon_eval.dynamics import capture_forecast
from multivon_eval.dynamics_metrics import forecast_case_result
from multivon_eval.integrations.gymnasium import capture_episode

COORDINATES = {'cart_position': 'm', 'cart_velocity': 'm/s',
               'pole_angle': 'rad', 'pole_angular_velocity': 'rad/s'}
HORIZONS = [1, 4, 8, 16, 32]
CONTRACT = 'cartpole-observed-vector-dynamics/v1'
TOLERANCES = dict(zip(COORDINATES, [0.05, 0.1, 0.02, 0.1]))


class OffsetReset(gym.Wrapper):
    def __init__(self, env, offset):
        super().__init__(env)
        self.offset = offset

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        state = np.array(self.unwrapped.state, copy=True)
        state[0] += self.offset
        self.unwrapped.state = tuple(state)
        return state.astype(observation.dtype), info


def episode(seed, *, variant, split, max_steps, actions=None, offset=0, model=None, broken=False):
    holder, plans = {}, []
    class Broken(gym.Wrapper):
        def step(self, action):
            raise RuntimeError('Deliberate simulator-error control')
    def factory():
        env = OffsetReset(gym.make('CartPole-v1'), offset)
        holder['env'] = env
        return Broken(env) if broken else env
    if actions is None:
        actions = np.random.default_rng(seed + 100000).integers(0, 2, size=max_steps).tolist()
    def interact(env, observation, info):
        for index in range(max_steps):
            if model is None:
                action = actions[index]
            else:
                action, evidence = plan(model, observation)
                plans.append(evidence)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        return 'Bounded interaction ended; inspect actual state and termination evidence.'
    captured = capture_episode(factory, interact,
        case=EvalCase('Predict/control CartPole under the frozen protocol',
                      case_id=f'cartpole:{seed}:{variant}', source_id=f'cartpole:{seed}',
                      tags=[split, variant], metadata={'initial_x_offset': offset}),
        environment_id=f'gymnasium-{gym.__version__}/CartPole-v1/offset-reset-v1',
        observer_id='native-cartpole-state/v1', seed=seed, max_steps=max_steps,
        observe=lambda: {'state': np.asarray(holder['env'].unwrapped.state).tolist()})
    return captured, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Development seeds only; 20 training episodes')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    (root / 'episodes').mkdir()
    (root / 'forecasts').mkdir()
    (root / 'planning').mkdir()
    source = root / 'source'
    shutil.copytree(repo / 'multivon_eval', source / 'multivon_eval',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['world_model_experiment.py', 'cartpole_models.py', 'analyze_world_model.py', 'WORLD_MODEL_PROTOCOL.md']:
        shutil.copy2(here / name, source / name)
    shutil.copy2(repo / 'LICENSE', source / 'LICENSE')
    training_seeds = list(range(1000, 1020 if args.smoke else 1200))
    testing_seeds = list(range(2000, 2005)) if args.smoke else list(range(3000, 3040))
    planning_seeds = list(range(2000, 2005)) if args.smoke else list(range(4000, 4010))
    split = 'development' if args.smoke else 'heldout'
    def save(name, data):
        (root / name).write_text(canonical_json(data) + '\n')
    save('configuration.json', {'contract': CONTRACT, 'coordinates': COORDINATES,
        'horizons': HORIZONS, 'training_seeds': training_seeds, 'testing_seeds': testing_seeds,
        'planning_seeds': planning_seeds, 'split': split, 'planning_horizon': 6,
        'planning_max_steps': 200, 'illustrative_tolerances': TOLERANCES,
        'tolerance_limits': 'Illustrative physical-unit checks, not independently calibrated acceptance limits'})
    save('runtime.json', {'base_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo)),
        'packages': {p: version(p) for p in ['gymnasium', 'scikit-learn', 'scipy', 'numpy', 'multivon-eval']},
        'upstream_cartpole_source_sha256': hashlib.sha256(inspect.getsource(CartPoleEnv).encode()).hexdigest(),
        'sources_captured_before_collection': True, 'provider_requests': 0})
    states, actions, next_states, origins = [], [], [], []
    for seed in training_seeds:
        record, _ = episode(seed, variant='training', split='training', max_steps=200)
        save(f'episodes/train-{seed}.json', record.data)
        data = record.data
        if data['errors']:
            raise RuntimeError('Training reference failed; stop before fitting')
        previous = data['initial']['observation'][0]
        for row in data['steps']:
            observed = row['transition']['observation'][0]
            states.append(previous)
            actions.append(row['action'][0])
            next_states.append(observed)
            origins.append({'episode_digest': record.digest, 'step': row['index'], 'source_id': data['case']['source_id']})
            previous = observed
    training = {'states': states, 'actions': actions, 'next_states': next_states, 'origins': origins}
    save('training.json', training)
    models = {name: DeltaModel.fit(states, actions, next_states, action_conditioned=conditioned)
              for name, conditioned in [('action_conditioned', True), ('action_blind', False)]}
    parameters = {name: model.parameters() for name, model in models.items()}
    save('models.json', {'training_digest': digest(training), 'models': parameters})
    models['persistence'] = Persistence()
    model_ids = {name: f'{name}/v1/{digest(parameters[name])}' if name in parameters else 'persistence/v1'
                 for name in models}
    print(f'Training frozen: {len(training_seeds)} episodes, {len(states)} transitions', flush=True)
    result_rows = {name: [] for name in models}
    first_reference = None
    openloop_actions = {str(seed): np.random.default_rng(seed + 100000).integers(0, 2, size=32).tolist()
                        for seed in testing_seeds}
    save('openloop_actions.json', openloop_actions)
    for seed in testing_seeds:
        planned_actions = openloop_actions[str(seed)]
        base, _ = episode(seed, variant='base', split=split, max_steps=32, actions=planned_actions)
        first_reference = first_reference or base
        variants = {'base': base,
            'offset': episode(seed, variant='offset', split=split, max_steps=32,
                              actions=planned_actions, offset=0.5)[0],
            'action0': episode(seed, variant='action0', split=split, max_steps=1, actions=[0])[0],
            'action1': episode(seed, variant='action1', split=split, max_steps=1, actions=[1])[0]}
        for variant, reference in variants.items():
            save(f'episodes/eval-{seed}-{variant}.json', reference.data)
            for name, model in models.items():
                forecast = capture_forecast(reference, model, model_id=model_ids[name], contract=CONTRACT,
                    coordinates=COORDINATES, horizons=[1] if variant.startswith('action') else HORIZONS,
                    planned_actions=[int(variant[-1])] if variant.startswith('action') else planned_actions)
                save(f'forecasts/{name}-{seed}-{variant}.json', forecast.data)
                if variant == 'base':
                    result_rows[name].append(forecast_case_result(forecast, tolerances=TOLERANCES))
    for name, rows in result_rows.items():
        (root / f'report-{name}.json').write_text(EvalReport(f'dynamics-{name}', rows).to_json())
    print(f'Forecasts saved: {len(testing_seeds)} sources x four variants x three models', flush=True)
    for seed in planning_seeds:
        for name, model in models.items():
            record, plans = episode(seed, variant=f'planning-{name}', split=split, max_steps=200, model=model)
            save(f'episodes/plan-{seed}-{name}.json', record.data)
            save(f'planning/{seed}-{name}.json', {'episode_digest': record.digest, 'model_id': model_ids[name], 'plans': plans})
        print(f'Planning seed {seed} complete', flush=True)
    def broken_model(request):
        raise RuntimeError('Deliberate model-error control')
    for name, reference, model in [('model_error', first_reference, broken_model),
        ('simulator_error', episode(9000, variant='error-control', split='control', max_steps=1, broken=True)[0],
         lambda _: (_ for _ in ()).throw(AssertionError('Model must not run after simulator error')))]:
        record = capture_forecast(reference, model, model_id=name + '/v1', contract=CONTRACT,
                                  coordinates=COORDINATES, horizons=[1])
        save(f'{name}.json', record.data)
    from analyze_world_model import analyze
    save('summary.json', analyze(root))
    save('checksums.json', {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(root.rglob('*')) if p.is_file()})
    print(f'Completed {root}', flush=True)


if __name__ == '__main__':
    main()
