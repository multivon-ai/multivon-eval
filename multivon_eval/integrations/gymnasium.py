"""Observe a bounded Gymnasium episode; keep environment semantics upstream."""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime, timezone
from numbers import Real
from time import perf_counter

try:
    import gymnasium as gym
except ImportError as exc:
    raise ImportError('Install multivon-eval[gymnasium] for environment capture') from exc

from ..case import EvalCase
from ..case_manifest import canonical_json, case_to_dict, digest
from ..episode import EpisodeEvidence


def _copy(value):
    return json.loads(canonical_json(value))


class _Recorder(gym.Wrapper):
    def __init__(self, env, data, observe):
        super().__init__(env)
        self.data, self.observe = data, observe
        self.exceptions = []
        self.started = self.finished = self.closed = False

    def _error(self, phase, exc):
        self.exceptions.append(exc)
        error = {'phase': phase, 'type': type(exc).__name__, 'message': str(exc)}
        self.data['errors'].append(error)
        self.data['issues'].append(f'{phase}: {type(exc).__name__}: {exc}')
        return error

    def _state(self, phase):
        try:
            state = self.observe()
            if not isinstance(state, dict):
                raise TypeError('State observer must return a portable JSON object')
            return _copy(state)
        except Exception as exc:  # noqa: BLE001 - retain observer failures as missing evidence
            self._error('observe_' + phase, exc)
            return None

    def _observation(self, observation):
        if not self.observation_space.contains(observation):
            raise ValueError('Observation is outside the declared Gymnasium space')
        return _copy(self.observation_space.to_jsonable([observation]))

    def reset(self, *, seed=None, options=None):
        if self.started or self.closed:
            exc = RuntimeError('One reset per captured episode; construct a fresh environment for repeats')
            self._error('lifecycle', exc)
            raise exc
        self.started = True
        try:
            observation, info = self.env.reset(seed=seed, options=options)
            if not isinstance(info, dict):
                raise TypeError('Gymnasium reset info must be a dict')
            self.data['initial'] = {'observation': self._observation(observation), 'info': _copy(info)}
        except Exception as exc:
            self.finished = True
            self._error('reset', exc)
            raise
        finally:
            self.data['initial_state'] = self._state('reset')
        return observation, info

    def step(self, action):
        if not self.started or self.finished or self.closed:
            exc = RuntimeError('Cannot step before reset or after episode completion/failure/close')
            self._error('lifecycle', exc)
            raise exc
        if len(self.data['steps']) >= self.data['max_steps']:
            self.finished = True
            exc = RuntimeError('Captured episode reached max_steps; no further action was executed')
            self._error('step_limit', exc)
            raise exc
        row = {'index': len(self.data['steps']) + 1, 'action': None,
               'transition': None, 'state': None, 'error': None}
        self.data['steps'].append(row)
        try:
            if not self.action_space.contains(action):
                raise ValueError('Action is outside the declared Gymnasium space')
            row['action'] = _copy(self.action_space.to_jsonable([action]))
            observation, reward, terminated, truncated, info = self.env.step(action)
            if type(terminated) is not bool or type(truncated) is not bool:
                raise TypeError('Gymnasium termination flags must be bool')
            if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(float(reward)):
                raise ValueError('Gymnasium reward must be finite numeric data')
            if not isinstance(info, dict):
                raise TypeError('Gymnasium step info must be a dict')
            row['transition'] = {'observation': self._observation(observation),
                'reward': float(reward), 'terminated': terminated, 'truncated': truncated,
                'info': _copy(info)}
            self.finished = terminated or truncated
        except Exception as exc:
            self.finished = True
            row['error'] = self._error('step', exc)
            raise
        finally:
            # An exception can follow a committed external write. Observe it
            # without assuming rollback or replaying the action.
            row['state'] = self._state('step')
        return observation, reward, terminated, truncated, info

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.data['final_state'] = self._state('final')
        finally:
            self.data['cleanup']['attempted'] = True
            try:
                self.env.close()
                self.data['cleanup']['completed'] = True
            except Exception as exc:  # noqa: BLE001 - preserve cleanup failure with prior errors
                self.data['cleanup']['error'] = self._error('close', exc)


def capture_episode(factory: Callable, interact: Callable, *, case: EvalCase,
                    environment_id: str, observer_id: str, observe: Callable[[], dict],
                    seed: int, max_steps: int, options: dict | None = None) -> EpisodeEvidence:
    """Capture one fresh Gymnasium environment, then always attempt its cleanup.

    ``interact(env, initial_observation, initial_info)`` owns the agent loop and
    returns text or None. The independent no-argument observer reads real state
    through a caller-owned read-only connection/API; it must not echo actions.
    Observations/actions use the native spaces' batched ``to_jsonable`` format.
    Exceptions are retained, never retried. BaseException (e.g. cancellation)
    propagates after cleanup. Use Inspect for durable logs and process isolation.
    This synchronous helper cannot interrupt a blocked callback or env.step.
    """
    if any(not isinstance(v, str) or not v.strip() for v in (environment_id, observer_id)):
        raise ValueError('Supply versioned environment_id and observer_id')
    if type(seed) is not int or seed < 0 or type(max_steps) is not int or max_steps < 1:
        raise ValueError('seed must be a nonnegative integer and max_steps a positive integer')
    if not all(callable(v) for v in (factory, interact, observe)):
        raise TypeError('factory, interact and observe must be callable')
    if options is not None and not isinstance(options, dict):
        raise TypeError('reset options must be a portable object or None')
    case_data = case_to_dict(case, include_reference=False)
    case_id, case_digest = case.identity()
    data = {'schema': 'multivon.episode/v1', 'case': case_data,
            'case_id': case_id, 'case_digest': case_digest,
            'environment_id': environment_id, 'observer_id': observer_id,
            'gymnasium_version': gym.__version__, 'seed': seed, 'max_steps': max_steps,
            'options': _copy(options), 'recorded_at': datetime.now(timezone.utc).isoformat(),
            'spaces': None, 'initial': None, 'initial_state': None, 'steps': [],
            'final_state': None, 'output': None, 'errors': [], 'issues': [],
            'cleanup': {'attempted': False, 'completed': False, 'error': None}}
    recorder, env = None, None
    started = perf_counter()
    try:
        env = factory()
        recorder = _Recorder(env, data, observe)
        data['spaces'] = {'action': repr(env.action_space), 'observation': repr(env.observation_space)}
        observation, info = recorder.reset(seed=seed, options=_copy(options))
        output = interact(recorder, observation, info)
        if output is not None and not isinstance(output, str):
            raise TypeError('Interaction output must be text or None')
        data['output'] = output
    except Exception as exc:  # noqa: BLE001 - environment/agent extensions may raise arbitrary exceptions
        if recorder is None:
            data['errors'].append({'phase': 'setup', 'type': type(exc).__name__, 'message': str(exc)})
            data['issues'].append(f'setup: {type(exc).__name__}: {exc}')
        elif not any(recorded is exc for recorded in recorder.exceptions):
            recorder._error('interaction', exc)
    finally:
        if recorder is not None:
            recorder.close()
        elif env is not None:
            data['cleanup']['attempted'] = True
            try:
                env.close()
                data['cleanup']['completed'] = True
            except Exception as exc:  # noqa: BLE001 - preserve setup and cleanup failures together
                error = {'phase': 'close', 'type': type(exc).__name__, 'message': str(exc)}
                data['cleanup']['error'] = error
                data['errors'].append(error)
                data['issues'].append(f'close: {type(exc).__name__}: {exc}')
    data['duration_ms'] = (perf_counter() - started) * 1000
    last = data['steps'][-1]['transition'] if data['steps'] else None
    data['termination'] = {'terminated': bool(last and last['terminated']),
                           'truncated': bool(last and last['truncated'])}
    if not data['termination']['terminated']:
        data['issues'].append('Episode did not reach natural termination; task completion is unverified')
    if data['termination']['truncated']:
        data['issues'].append('Episode was truncated; terminal task coverage is incomplete')
    if not data['cleanup']['completed']:
        data['issues'].append('Environment cleanup was not confirmed')
    data['issues'] = sorted(set(data['issues']))
    return EpisodeEvidence.from_dict({**data, 'digest': digest(data)})
