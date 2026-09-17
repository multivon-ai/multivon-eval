"""Target and runner snapshots; declarations do not discover hidden state."""
from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .case_manifest import canonical_json, digest
from .dependencies import _portable, declare_dependencies, evaluator_dependencies

SCHEMA = 'multivon.execution/v1'
_RUN = ContextVar('multivon_execution_run', default=None)
_TRIAL = ContextVar('multivon_execution_trial', default=None)


class DeclaredTarget:
    """Transparent sync/async and case-aware wrapper with a caller contract."""
    def __init__(self, target, *, version, configuration, dependencies, files):
        if not callable(target):
            raise TypeError('Target must be callable')
        self.target = target
        self.configuration = json.loads(canonical_json(configuration))
        declare_dependencies(self, version=version, dependencies=dependencies, files=files)

    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def _call_with_case(self, case):
        method = getattr(self.target, '_call_with_case', None)
        return method(case) if callable(method) else self.target(case.input)

    async def _acall_with_case(self, case):
        method = getattr(self.target, '_acall_with_case', None)
        result = method(case) if callable(method) else self.target(case.input)
        return await result


def declare_target(target, *, version: str, configuration: dict | None = None,
                   dependencies: dict[str, str], files: dict[str, str | Path] | None = None):
    """Declare target code/config/data revisions; named files are rehashed.

    The wrapper does not mutate the supplied callable. Configuration must be
    portable JSON. A declaration covers hidden callback/client state by caller
    assertion, not automatic discovery or proof of deterministic behavior.
    """
    if configuration is not None and not isinstance(configuration, dict):
        raise TypeError('Target configuration must be a mapping')
    return DeclaredTarget(target, version=version, configuration=configuration or {},
                          dependencies=dependencies, files=files)


def _clean(value):
    from .provider_http import _url
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clean(v) for v in value)
    if isinstance(value, str) and value.startswith(('http://', 'https://')):
        redacted, fields = _url(value)
        if fields:
            return {'redacted_url': redacted, 'original_digest': digest(value), 'redacted_fields': fields}
    return value


def _record(value):
    types = []
    return {'value': _portable(_clean(value), types, []), 'types': types}


def _target_snapshot(target):
    from .adapters import AnthropicAdapter, LiteLLMAdapter, OpenAIAdapter
    declared = target if isinstance(target, DeclaredTarget) else None
    actual = declared.target if declared else target
    issues = []
    record = {'schema': SCHEMA, 'callable': None, 'configuration': None, 'declaration': None,
              'client': None, 'issues': issues,
              'limits': 'Recorded settings and caller declarations; aliases and hidden runtime state are not immutable weights'}
    if actual is None:
        record['origin'] = 'saved_outputs'
        return {**record, 'digest': digest(record)}
    record['origin'] = 'executed'
    if inspect.isfunction(actual) or inspect.ismethod(actual):
        record['callable'] = (actual.__module__ or type(actual).__module__) + '.' + actual.__qualname__
    else:
        record['callable'] = type(actual).__module__ + '.' + type(actual).__qualname__
    if declared:
        # Reuse the file-hash/declaration contract; do not expose the callback or
        # local file paths through generic object introspection.
        holder = type('TargetContract', (), {})()
        holder._dependency_contract = declared._dependency_contract
        contract = evaluator_dependencies(holder)['dependencies']
        record['declaration'] = {**contract['declaration'], 'configuration': _record(declared.configuration)}
        issues.extend(contract['issues'])
    if type(actual) in (AnthropicAdapter, OpenAIAdapter, LiteLLMAdapter):
        record['configuration'] = {'model': actual.model, 'system_prompt': actual._system_prompt,
                                  'temperature': actual._temperature, 'max_tokens': actual._max_tokens}
        try:
            record['configuration']['extra'] = _record(actual._extra)
        except (TypeError, ValueError):
            issues.append('Opaque adapter request options require a caller declaration')
        provider = 'anthropic' if type(actual) is AnthropicAdapter else 'openai' if type(actual) is OpenAIAdapter else 'litellm'
        keys = {'anthropic': ['ANTHROPIC_BASE_URL'],
                'openai': ['OPENAI_BASE_URL', 'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID'], 'litellm': []}[provider]
        record['environment_routing'] = _record({key: os.environ[key] for key in keys if key in os.environ})
        client = getattr(actual, '_client', None)
        if client is None:
            record['client'] = {'mode': 'sdk_default', 'provider': provider,
                                'limits': 'SDK/transport defaults are not resolved without constructing a client; observe native requests'}
        else:
            record['client'] = {'mode': 'supplied', 'class': type(client).__module__ + '.' + type(client).__qualname__}
            settings = {}
            native_client = type(client).__module__ in ('openai', 'anthropic', 'openai._client', 'anthropic._client')
            for name in ('base_url', 'max_retries', 'timeout', 'organization', 'project') if native_client else ():
                try:
                    value = getattr(client, name)
                    if name == 'base_url':
                        value = str(value)
                    elif name == 'timeout' and hasattr(value, 'as_dict'):
                        value = value.as_dict()
                    settings[name] = _record(value)
                except (AttributeError, TypeError, ValueError):
                    continue
            record['client']['observed_settings'] = settings
            if not declared:
                issues.append('Supplied client transport/auth/hooks require a caller declaration')
        if provider == 'litellm' and not declared:
            issues.append('LiteLLM routing/global settings require a caller declaration')
    elif not declared:
        issues.append('Custom target requires a caller configuration/dependency declaration')
    if declared:
        issues[:] = [issue for issue in issues if 'require a caller declaration' not in issue]
    return {**record, 'digest': digest(record)}


def target_snapshot(target):
    try:
        return _target_snapshot(target)
    except Exception as exc:
        record = {'schema': SCHEMA, 'origin': 'executed', 'callable': None, 'configuration': None,
                  'declaration': None, 'client': None,
                  'issues': [f'Target snapshot unavailable ({type(exc).__name__})'],
                  'limits': 'The target still executes; its configuration could not be captured'}
        return {**record, 'digest': digest(record)}


@contextmanager
def execution_scope(function, args, kwargs):
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    from .execution_controls import validate_run_options
    validate_run_options(values)
    target = values.get('model_fn')
    policy = {key: values[key] for key in ('runs', 'workers', 'concurrency', 'evaluator_concurrency',
              'early_stop', 'fail_threshold', 'max_error_rate') if key in values}
    retry = values.get('judge_retry')
    policy['judge_retry'] = asdict(retry) if is_dataclass(retry) else None
    if function.__name__ == 'run':
        workers = values['workers']
        if workers is None:
            workers = 1 if values.get('tracer') is not None else max(1, min(8, len(values['self']._cases)))
        policy['effective_workers'] = workers
        policy['effective_early_stop'] = bool(values['early_stop'] and workers == 1)
    tracer = values.get('tracer')
    policy['tracer'] = None if tracer is None else type(tracer).__module__ + '.' + type(tracer).__qualname__
    state = {'target': target, 'before': target_snapshot(target), 'policy': _record(policy), 'method': function.__name__}
    token = _RUN.set(state)
    try:
        yield
    finally:
        _RUN.reset(token)


def execution_snapshot():
    state = _RUN.get()
    if state is None:
        return None
    after = target_snapshot(state['target'])
    record = {'schema': SCHEMA, 'method': state['method'], 'policy': state['policy'],
              'target_before': state['before'], 'target_after': after,
              'changed_during_execution': state['before']['digest'] != after['digest']}
    return {**record, 'digest': digest(record)}


@contextmanager
def trial_execution_scope():
    state = _RUN.get()
    token = _TRIAL.set(target_snapshot(state['target']) if state else None)
    try:
        yield
    finally:
        _TRIAL.reset(token)


def trial_execution_snapshot():
    state, before = _RUN.get(), _TRIAL.get()
    if state is not None and state['target'] is None and before is None:
        before = state['before']
    if state is None or before is None:
        return None
    after = target_snapshot(state['target'])
    record = {'schema': SCHEMA, 'target_before': before, 'target_after': after,
              'changed_during_execution': before['digest'] != after['digest']}
    return {**record, 'digest': digest(record)}


def validate_execution(record):
    """Check nested snapshot contents and their claimed drift, not authenticity."""
    from jsonschema import Draft202012Validator
    if record is None:
        return
    validator = Draft202012Validator({
        'type': 'object', 'required': ['schema', 'target_before', 'target_after', 'changed_during_execution', 'digest'],
        'properties': {'schema': {'const': SCHEMA}, 'changed_during_execution': {'type': 'boolean'},
                       'digest': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}},
    })
    target_validator = Draft202012Validator({
        'type': 'object', 'required': ['schema', 'origin', 'callable', 'configuration', 'declaration', 'client', 'issues', 'limits', 'digest'],
        'properties': {'schema': {'const': SCHEMA}, 'origin': {'enum': ['executed', 'saved_outputs']},
                       'issues': {'type': 'array', 'items': {'type': 'string'}},
                       'digest': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}},
    })
    if not validator.is_valid(record):
        raise ValueError('Invalid execution snapshot structure')
    for value, schema in ((record, validator), (record['target_before'], target_validator),
                          (record['target_after'], target_validator)):
        if not schema.is_valid(value):
            raise ValueError('Invalid target snapshot structure')
        data = dict(value)
        if data.pop('digest') != digest(data):
            raise ValueError('Execution snapshot digest mismatch')
    changed = record['target_before']['digest'] != record['target_after']['digest']
    if record['changed_during_execution'] != changed:
        raise ValueError('Execution snapshot drift flag does not match its contents')


def compare_execution(baseline, proposal):
    """Target interventions are expected; within-run drift is a separate issue."""
    notes, issues = [], []
    for label, report in [('baseline', baseline), ('proposal', proposal)]:
        evidence = report.execution
        if evidence is None:
            notes.append(f'{label}: target execution configuration was not recorded')
            continue
        try:
            validate_execution(evidence)
        except ValueError as exc:
            issues.append(f'{label}: {exc}')
            continue
        notes.extend(f'{label}: {issue}' for issue in evidence['target_before']['issues'])
        trial_drift = any(t.data.get('execution', {}).get('changed_during_execution', False)
                          for result in report.case_results for t in result.trials
                          if t.data.get('execution') is not None)
        if evidence['changed_during_execution'] or trial_drift:
            issues.append(f'{label}: target configuration or declared dependencies changed during execution')
    changes = {'target_paths': [], 'policy_paths': []}
    def paths(left, right, prefix=''):
        if isinstance(left, dict) and isinstance(right, dict):
            return [path for key in sorted(left.keys() | right.keys()) if key != 'digest'
                    for path in paths(left.get(key), right.get(key), f'{prefix}.{key}' if prefix else key)]
        return [prefix] if left != right else []
    if baseline.execution is not None and proposal.execution is not None and not issues:
        changes['target_paths'] = paths(baseline.execution['target_before'], proposal.execution['target_before'])
        changes['policy_paths'] = paths(baseline.execution.get('policy'), proposal.execution.get('policy'))
    return changes, notes, issues
