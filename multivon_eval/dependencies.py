"""Recorded dependency contracts, not automatic discovery of arbitrary program state."""
from __future__ import annotations

import hashlib
import json
import platform
import re
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from .case_manifest import canonical_json
from .dependency_schema import SCHEMA, valid_dependency_record

if TYPE_CHECKING:
    from .evaluators.base import Evaluator



def declare_dependencies(evaluator: Evaluator, *, version: str,
                         dependencies: dict[str, str], files: dict[str, str | Path] | None = None):
    """Declare externally managed code, model, schema and data revisions.

    ``version`` identifies the complete custom grading contract, including
    callback/closure semantics and hidden configuration. ``dependencies`` maps
    meaningful names to immutable revisions/digests; an empty map explicitly
    declares none. Named files are hashed again at each suite snapshot. Paths
    are local inputs, omitted from reports. Declarations are caller assertions,
    not proof of completeness, model immutability or deterministic execution.
    """
    if not isinstance(version, str) or not version.strip():
        raise ValueError('Dependency contract version must be a nonempty string')
    if not isinstance(dependencies, dict) or any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
        for k, v in dependencies.items()
    ):
        raise ValueError('dependencies must map nonempty names to revision strings')
    if files is not None and (not isinstance(files, dict) or any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, (str, Path))
        for k, v in files.items()
    )):
        raise ValueError('files must map nonempty names to local paths')
    evaluator._dependency_contract = {
        'version': version, 'dependencies': json.loads(canonical_json(dependencies)),
        'files': {k: str(Path(v).resolve()) for k, v in (files or {}).items()},
    }
    return evaluator


def _private_value(value):
    return {'sha256': hashlib.sha256(canonical_json(value).encode()).hexdigest()}


def _sensitive(key):
    return key.startswith('_') or any(word in key.lower() for word in (
        'secret', 'password', 'credential', 'api_key', 'authorization', 'token'))


def _portable(value, types, path):
    if isinstance(value, re.Pattern):
        types.append({'path': path, 'kind': 'regex'})
        return {'_kind': 'regex', 'pattern': value.pattern, 'flags': value.flags}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError('Non-string mapping key')
        result = {}
        for k, v in value.items():
            item = _portable(v, types, path + [k])
            result[k] = _private_value(item) if _sensitive(k) else item
        return result
    if isinstance(value, (tuple, list)):
        if isinstance(value, tuple):
            types.append({'path': path, 'kind': 'tuple'})
        return [_portable(v, types, path + [i]) for i, v in enumerate(value)]
    if value is None or type(value) in (str, int, bool, float):
        canonical_json(value)
        return value
    raise TypeError('Opaque value')


def evaluator_dependencies(evaluator):
    config, opaque, issues, types = {}, [], [], []
    for key, value in sorted(vars(evaluator).items()):
        if key in {'name', 'threshold', 'judge', '_judge', '_judge_cfg', '_dependency_contract'}:
            continue
        try:
            portable = _portable(value, types, [key])
            config[key] = _private_value(portable) if _sensitive(key) else portable
        except (TypeError, ValueError):
            opaque.append(key)
    contract = getattr(evaluator, '_dependency_contract', None)
    declared = None
    if contract is not None:
        declared = {'version': contract['version'], 'dependencies': dict(contract['dependencies']), 'files': {}}
        for name, path in sorted(contract['files'].items()):
            try:
                declared['files'][name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError as exc:
                issues.append(f'Declared file {name!r} is unavailable ({type(exc).__name__})')
    if declared is None:
        if not type(evaluator).__module__.startswith('multivon_eval.evaluators.'):
            issues.append('Custom grader requires a declared dependency contract')
        if opaque:
            issues.append('Opaque grader fields require a declared dependency contract: ' + ', '.join(opaque))
        if type(evaluator).__name__ == 'BERTScore' or getattr(evaluator, 'use_ner', False):
            issues.append('External model artifacts require a declared dependency contract')
    return {'config': config, 'config_types': types, 'dependencies': {
        'schema': SCHEMA, 'declaration': declared, 'opaque_fields': opaque, 'issues': issues,
        'limits': 'Caller declarations cover hidden code/data; remote model aliases are not immutable weights',
    }}


def engine_dependencies():
    """Use installed distribution metadata and local source bytes conservatively.

    All installed distributions are recorded, so unrelated environment changes
    also block compatibility. This is metadata inventory, not a lock resolver
    or proof that every third-party installed file matches its distribution.
    """
    root = Path(__file__).parent
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.rglob('*.py'))}
    packages = []
    for distribution in metadata.distributions():
        info = distribution.metadata
        if info['Name']:
            packages.append([info['Name'], info['Version']])
    packages.sort()
    return {'schema': SCHEMA, 'source_digest': hashlib.sha256(canonical_json(sources).encode()).hexdigest(),
            'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'system': platform.system(), 'release': platform.release(), 'machine': platform.machine(),
            'packages': packages}


def seal_lock(lock):
    data = lock.to_dict()
    data.pop('suite_hash')
    lock.suite_hash = hashlib.sha256(canonical_json(data).encode()).hexdigest()
    return lock


def lock_issues(lock):
    if lock is None:
        return ['Missing suite lock; grader dependencies cannot be verified']
    if not isinstance(lock.extra, dict):
        return ['Invalid suite dependency metadata']
    recorded = lock.extra.get('issues', [])
    issues = list(recorded) if isinstance(recorded, list) and all(isinstance(i, str) for i in recorded) else [
        'Invalid suite dependency issue list']
    engine = lock.extra.get('engine')
    if not valid_dependency_record(engine, 'engine'):
        issues.append('Missing or invalid engine dependency snapshot')
    try:
        data = lock.to_dict()
        data.pop('suite_hash')
        if hashlib.sha256(canonical_json(data).encode()).hexdigest() != lock.suite_hash:
            issues.append('Suite lock digest does not match its recorded contents')
    except (TypeError, ValueError):
        issues.append('Suite lock contents are not portable JSON')
    for row in lock.evaluators:
        evidence = row.extra.get('dependencies') if isinstance(row.extra, dict) else None
        if row.version != '3' or not valid_dependency_record(evidence, 'grader'):
            issues.append(f'{row.name}: missing or invalid grader dependency snapshot')
            continue
        issues.extend(f'{row.name}: {issue}' for issue in evidence['issues'])
        if evidence['declaration'] is None and (evidence['opaque_fields'] or
                not row.class_path.startswith('multivon_eval.evaluators.')):
            issues.append(f'{row.name}: undeclared custom or opaque grader dependencies')
    return issues


def finish_lock(suite, before):
    """Preserve the pre-run snapshot and mark observed execution-time drift."""
    from .suite import _safe_lock
    after = _safe_lock(suite)
    if before is None:
        return None
    issues = list(before.extra.get('issues', []))
    if after is None:
        issues.append('Post-run dependency snapshot unavailable')
    elif before.suite_hash != after.suite_hash:
        issues.append('Suite configuration or dependencies changed during execution')
        before.extra['post_run_lock'] = json.loads(after.to_json())
    if issues:
        before.extra['issues'] = issues
    return seal_lock(before)


def comparison_issues(baseline, proposal):
    from dataclasses import asdict
    issues = [f'{label}: {issue}' for label, lock in [('baseline', baseline), ('proposal', proposal)]
              for issue in lock_issues(lock)]
    if baseline is None or proposal is None:
        return issues
    try:
        if sorted(canonical_json(asdict(e)) for e in baseline.evaluators) != sorted(
            canonical_json(asdict(e)) for e in proposal.evaluators
        ):
            issues.append('Recorded evaluator configuration changed between runs')
    except (TypeError, ValueError):
        issues.append('Recorded evaluator configuration is not portable JSON')
    if baseline.library_version != proposal.library_version:
        issues.append('Evaluation engine version changed between runs')
    b_engine = baseline.extra.get('engine') if isinstance(baseline.extra, dict) else None
    p_engine = proposal.extra.get('engine') if isinstance(proposal.extra, dict) else None
    try:
        if canonical_json(b_engine) != canonical_json(p_engine):
            issues.append('Evaluation engine source or environment changed between runs')
    except (TypeError, ValueError):
        issues.append('Evaluation engine snapshot is not portable JSON')
    if baseline.calibration_version != proposal.calibration_version:
        issues.append('Calibration version changed between runs')
    return issues
