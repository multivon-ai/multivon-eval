"""Declared task compatibility around native Inspect execution and retry logs."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from copy import deepcopy
from types import SimpleNamespace

from ..case_manifest import digest
from ..dependencies import declare_dependencies, evaluator_dependencies, lock_issues
from ..dependency_schema import valid_dependency_record, SCHEMA as DEPENDENCY_SCHEMA
from ..execution_evidence import _record

TASK_KEY = 'multivon_task_contract_v1'
SAMPLE_KEY = 'multivon_task_contract_digest_v1'
GRADING_KEY = 'multivon_grading_contracts_v1'
SCHEMA = 'multivon.inspect-contract/v1'


def grader_snapshot(evaluator):
    from ..lockfile import fingerprint_evaluator
    try:
        return _record(asdict(fingerprint_evaluator(evaluator)))
    except Exception as exc:
        return {'unavailable': type(exc).__name__}


def _input_digest(value):
    value = deepcopy(value)
    if isinstance(value, list):
        for message in value:
            message.pop('id', None)
    return digest(value)


def _component(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [_component(item) for item in value]
    from inspect_ai.util import registry_info
    try:
        info = registry_info(value)
        return {'registry': info.model_dump(mode='json')}
    except ValueError:
        return {'callable': type(value).__module__ + '.' + type(value).__qualname__}


def _native_setting(value):
    from pydantic import BaseModel
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    if is_dataclass(value) and not isinstance(value, type):
        return _native_setting(asdict(value))
    if isinstance(value, dict):
        return {k: _native_setting(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native_setting(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    return {'caller_declared_type': type(value).__module__ + '.' + type(value).__qualname__}


def bind_inspect_task(task, *, version, dependencies, files=None, configuration=None,
                      previous_log=None):
    """Bind a caller-declared task contract; optionally reject an incompatible retry.

    Call inside the registered task factory on every reconstruction. The supplied
    Task and sample metadata are updated in place. Inspect still owns execution.
    Declarations cover hidden solver/client/environment state, not its discovery.
    Task construction/preparation may already have side effects before this call.
    """
    from inspect_ai import Task
    from ..suite import EvalSuite
    from .inspect import _case_from_metadata
    if not isinstance(task, Task) or task.sample_source is not None:
        raise ValueError('Task contracts require a static native Inspect Task dataset')
    if configuration is not None and not isinstance(configuration, dict):
        raise ValueError('configuration must be a portable mapping')
    holder = declare_dependencies(SimpleNamespace(), version=version, dependencies=dependencies, files=files)
    declaration = evaluator_dependencies(holder)['dependencies']
    graders = []
    for scorer in task.scorer or []:
        evaluator = getattr(scorer, '_multivon_evaluator', None)
        if evaluator is None:
            raise ValueError('Task contracts require as_inspect_scorer graders')
        graders.append(evaluator)
    names = [ev.name for ev in graders]
    if not names or len(set(names)) != len(names):
        raise ValueError('Task contracts require uniquely named Multivon graders')
    lock = EvalSuite('Inspect grading contract').add_evaluators(*graders).lock()
    issues = [*declaration['issues'], *lock_issues(lock)]
    if issues:
        raise ValueError('Unverifiable Inspect task contract: ' + '; '.join(issues))
    samples = []
    ids = set()
    for sample in task.dataset:
        if sample.id is None or str(sample.id) in ids:
            raise ValueError('Task contracts require unique explicit sample IDs')
        ids.add(str(sample.id))
        _case_from_metadata(sample.metadata or {}, sample.id)
        data = sample.model_dump(mode='json')
        data['metadata'] = {k: v for k, v in (data['metadata'] or {}).items()
                            if k not in (SAMPLE_KEY, GRADING_KEY)}
        if isinstance(data['input'], list):
            for message in data['input']:
                message.pop('id', None)  # Native message IDs are generated on construction.
        case = data['metadata'].get('multivon_case_v1', {})
        samples.append({'id': str(sample.id), 'sample_digest': digest(data),
                        'input_digest': _input_digest(data['input']), 'target_digest': digest(data['target']),
                        'case_digest': case.get('case_digest'),
                        'manifest_digest': case.get('manifest_digest')})
    settings = {name: getattr(task, name) for name in (
        'version', 'epochs', 'fail_on_error', 'continue_on_fail', 'score_on_error',
        'message_limit', 'token_limit', 'token_limit_type', 'turn_limit', 'time_limit',
        'working_limit', 'cost_limit')}
    settings['config'] = task.config.model_dump(mode='json')
    settings['dataset'] = {'name': task.dataset.name, 'shuffled': task.dataset.shuffled}
    settings['model'] = str(task.model) if task.model is not None else None
    for name in ('sandbox', 'checkpoint', 'model_roles', 'approval', 'early_stopping'):
        settings[name] = _native_setting(getattr(task, name))
    settings['metadata'] = {k: v for k, v in (task.metadata or {}).items() if k != TASK_KEY}
    components = {name: _component(getattr(task, name)) for name in
                  ('setup', 'solver', 'cleanup', 'on_checkpoint', 'on_resume')}
    record = {'schema': SCHEMA, 'declaration': declaration['declaration'],
              'configuration': _record(configuration or {}), 'task_settings': _record(settings),
              'components': _record(components), 'samples': samples, 'grader_lock': _record(lock.to_dict()),
              'graders': {ev.name: grader_snapshot(ev) for ev in graders},
              'limits': 'Caller declarations cover solver arguments, sandbox images, clients and external state; hashes are not signatures'}
    record['digest'] = digest(record)
    if previous_log is not None:
        previous = (previous_log.eval.metadata or {}).get(TASK_KEY)
        problems = contract_issues(previous)
        if problems or previous['digest'] != record['digest']:
            raise ValueError('Inspect retry contract is incompatible: ' + '; '.join(problems or ['task contract changed']))
    task.metadata = {**(task.metadata or {}), TASK_KEY: record}
    for sample in task.dataset:
        sample.metadata = {**(sample.metadata or {}), SAMPLE_KEY: record['digest']}
    return task


def contract_issues(record):
    if not isinstance(record, dict):
        return ['Missing declared Inspect task contract']
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator({
        'type': 'object', 'required': ['schema', 'declaration', 'configuration', 'task_settings',
                                     'components', 'samples', 'grader_lock', 'graders', 'digest'],
        'properties': {'schema': {'const': SCHEMA}, 'samples': {'type': 'array', 'items': {
            'type': 'object', 'required': ['id', 'case_digest', 'manifest_digest', 'sample_digest', 'input_digest', 'target_digest'],
            'properties': {'id': {'type': 'string'}, 'sample_digest': {'type': 'string'}}}},
            'graders': {'type': 'object'}, 'declaration': {'type': 'object'},
            'digest': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}},
    })
    if not validator.is_valid(record):
        return ['Invalid Inspect task contract structure']
    if not valid_dependency_record({'schema': DEPENDENCY_SCHEMA, 'declaration': record['declaration'],
                                     'opaque_fields': [], 'issues': [], 'limits': 'Caller assertion'}, 'grader'):
        return ['Invalid Inspect dependency declaration']
    if len({r['id'] for r in record['samples']}) != len(record['samples']):
        return ['Duplicate sample identity in Inspect task contract']
    data = dict(record)
    claimed = data.pop('digest')
    try:
        matches = digest(data) == claimed
    except (TypeError, ValueError):
        matches = False
    if not matches:
        return ['Inspect task contract digest mismatch']
    return []


def log_contract_issues(log):
    """Validate declared logs, and detect observable mixed legacy sample state."""
    issues = []
    record = (log.eval.metadata or {}).get(TASK_KEY)
    if record is not None:
        issues.extend(contract_issues(record))
    elif any(SAMPLE_KEY in (sample.metadata or {}) for sample in log.samples or []):
        issues.append('Missing declared Inspect task contract for bound samples')
    expected = {r['id']: r for r in record['samples']} if record is not None and not issues else None
    manifests, observed_graders = set(), {}
    for sample in log.samples or []:
        metadata = sample.metadata or {}
        envelope = metadata.get('multivon_case_v1', {})
        manifests.add(envelope.get('manifest_digest'))
        grading = metadata.get(GRADING_KEY, {})
        if not isinstance(grading, dict) or any(not isinstance(v, dict) for v in grading.values()):
            issues.append(f'Sample {sample.id}: invalid grader execution evidence')
            grading = {}
        if expected is not None:
            original = expected.get(str(sample.id))
            if (metadata.get(SAMPLE_KEY) != record['digest'] or original is None
                    or original['case_digest'] != envelope.get('case_digest')
                    or original['manifest_digest'] != envelope.get('manifest_digest')
                    or original['input_digest'] != _input_digest(sample.model_dump(mode='json')['input'])
                    or original['target_digest'] != digest(sample.target)):
                issues.append(f'Sample {sample.id}: task or dataset contract differs from this log')
        for name, snapshots in grading.items():
            before, after = snapshots.get('before'), snapshots.get('after')
            if not isinstance(before, dict) or before != after or 'unavailable' in before:
                issues.append(f'Sample {sample.id}: grader {name} changed during scoring or lacks a snapshot')
            observed_graders.setdefault(name, set()).add(digest(before))
            if expected is not None and record['graders'].get(name) != before:
                issues.append(f'Sample {sample.id}: grader {name} differs from the declared task')
        if expected is not None and sample.error is None and set(grading) != set(record['graders']):
            issues.append(f'Sample {sample.id}: missing declared grader execution evidence')
    if record is None and len(manifests) > 1:
        issues.append('Inspect log mixes different dataset manifests without a declared task contract')
    if any(len(values) > 1 for values in observed_graders.values()):
        issues.append('Inspect log mixes different grader configurations')
    return sorted(set(issues))


def retry_contract_issues(logs):
    issues = []
    declared = [(log.eval.metadata or {}).get(TASK_KEY) for log in logs]
    for record in declared:
        issues.extend(contract_issues(record))
    if not issues and len({record['digest'] for record in declared}) > 1:
        issues.append('Declared Inspect task contract changed across retry logs')
    def observed(log):
        spec = log.eval.model_dump(mode='json')
        return _record({'spec': {k: spec.get(k) for k in (
            'task', 'task_version', 'task_args', 'model', 'model_args', 'model_base_url', 'model_roles', 'sandbox')},
            'plan': log.plan.model_dump(mode='json'),
            'execution': {k: v for k, v in spec['config'].items()
                          if not k.startswith('log_') and k not in ('score_display', 'acp_server', 'ctl_server')}})
    if len({digest(observed(log)) for log in logs}) > 1:
        issues.append('Native Inspect task, plan or execution settings changed across retry logs')
    return sorted(set(issues))
