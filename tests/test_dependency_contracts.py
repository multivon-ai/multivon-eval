"""Comparisons require recorded grader contracts, not equal class names alone."""
import asyncio
import json
from pathlib import Path

import pytest

from multivon_eval import (
    Contains,
    EvalCase,
    EvalReport,
    EvalSuite,
    ExactMatch,
    JudgeConfig,
    LockMismatch,
    ToolCallNecessity,
    compare_reports,
    configure,
    declare_dependencies,
    regrade,
)
from multivon_eval.dependencies import engine_dependencies, lock_issues
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.evaluators.compliance import SchemaEvaluator
from multivon_eval.judge import get_global_judge
from multivon_eval.lockfile import fingerprint_evaluator


class CallbackGrader(Evaluator):
    name = 'callback'
    def __init__(self, callback):
        super().__init__()
        self._callback = callback
    def evaluate(self, case, output):
        return self._result(float(self._callback(output)), 'callback result')


def suite(evaluator=None):
    return EvalSuite('deps').add_case(EvalCase('x', 'yes')).add_evaluator(evaluator or ExactMatch())


def run(evaluator=None):
    return suite(evaluator).run(lambda _: 'yes', verbose=False)


def test_opaque_callback_cannot_claim_verified_comparison():
    report = run(CallbackGrader(lambda _: True))
    diff = compare_reports(report, report)
    assert diff.identity_issues and diff.mcnemar_p is None
    assert 'Opaque' in ' '.join(diff.identity_issues)
    with pytest.raises(LockMismatch, match='declared dependency'):
        suite(CallbackGrader(lambda _: True)).verify_lock(report.suite_lock)


def test_declared_custom_grader_can_compare_and_changed_revision_cannot():
    def grader(version):
        return declare_dependencies(CallbackGrader(lambda _: True), version=version,
                                    dependencies={'callback': 'git:0123456789'})
    a, b = run(grader('v1')), run(grader('v1'))
    assert not compare_reports(a, b).identity_issues
    changed = run(grader('v2'))
    assert compare_reports(a, changed).mcnemar_p is None
    assert 'configuration changed' in ' '.join(compare_reports(a, changed).identity_issues)


def test_named_file_rehashed_without_leaking_its_local_path(tmp_path):
    path = tmp_path / 'private-customer-schema.json'
    path.write_text('{"required":["amount"]}')
    grader = declare_dependencies(CallbackGrader(lambda _: True), version='v1', dependencies={},
                                 files={'schema': path})
    a = run(grader)
    assert str(path) not in a.to_json()
    path.write_text('{"required":["total"]}')
    b = run(grader)
    assert compare_reports(a, b).identity_issues
    path.unlink()
    unavailable = run(grader)
    assert 'unavailable' in ' '.join(lock_issues(unavailable.suite_lock))
    assert unavailable.case_results[0].passed


def test_private_json_schema_is_not_silently_omitted():
    a = fingerprint_evaluator(SchemaEvaluator({'type': 'object', 'required': ['a']}))
    b = fingerprint_evaluator(SchemaEvaluator({'type': 'object', 'required': ['b']}))
    assert a.extra['config']['_json_schema'] != b.extra['config']['_json_schema']


def test_global_and_environment_judge_changes_are_recorded(monkeypatch):
    original = get_global_judge()
    try:
        configure(JudgeConfig(provider='anthropic', model='judge-a', timeout=10))
        a = fingerprint_evaluator(ToolCallNecessity())
        configure(JudgeConfig(provider='anthropic', model='judge-b', timeout=20))
        b = fingerprint_evaluator(ToolCallNecessity())
        assert a.judge['model'] == 'judge-a' and b.judge['timeout'] == 20
        assert a.judge != b.judge
        configure(JudgeConfig())
        monkeypatch.setenv('JUDGE_MODEL', 'environment-judge')
        assert fingerprint_evaluator(ToolCallNecessity()).judge['model'] == 'environment-judge'
    finally:
        configure(original)


def test_judge_extra_is_bound_without_recording_secret_values():
    secret = 'test-private-configuration-value'
    config = JudgeConfig(provider='anthropic', model='x', extra={'secret': secret})
    before = fingerprint_evaluator(ToolCallNecessity(judge=config))
    config.extra['secret'] = 'changed'
    after = fingerprint_evaluator(ToolCallNecessity(judge=config))
    assert before.judge['extra_digest'] != after.judge['extra_digest']
    assert secret not in json.dumps(before.judge)


@pytest.mark.parametrize('mode', ['sync', 'parallel', 'async', 'saved', 'regrade'])
def test_configuration_changed_during_run_is_explicit(mode):
    grader = Contains(['yes'])
    s = suite(grader)
    def target(_):
        grader.substrings[:] = ['changed']
        return 'yes'
    async def atarget(prompt):
        return target(prompt)
    if mode in ('saved', 'regrade'):
        original = grader.evaluate
        def evaluate(case, output):
            target('')
            return original(case, output)
        # An instance override is itself opaque; declare it and still detect config drift.
        grader.evaluate = evaluate
        declare_dependencies(grader, version='mutation-fixture/v1', dependencies={})
        if mode == 'saved':
            report = s.run_on_cases([(EvalCase('x', 'yes'), 'yes')], verbose=False)
        else:
            report = regrade(run(), s)
    elif mode == 'async':
        report = asyncio.run(s.run_async(atarget, verbose=False))
    else:
        report = s.run(target, workers=2 if mode == 'parallel' else 1, verbose=False)
    assert report.suite_lock.evaluators[0].extra['config']['substrings'] == ['yes']
    assert 'changed during execution' in ' '.join(lock_issues(report.suite_lock))
    after = report.suite_lock.extra['post_run_lock']
    assert after['evaluators'][0]['extra']['config']['substrings'] == ['changed']
    restored = EvalReport.from_dict(json.loads(report.to_json()))
    assert restored.suite_lock.to_dict() == report.suite_lock.to_dict()
    assert 'digest does not match' not in ' '.join(lock_issues(restored.suite_lock))
    assert compare_reports(restored, restored).mcnemar_p is None


def test_engine_sources_rehashed_from_actual_bytes(tmp_path, monkeypatch):
    import multivon_eval.dependencies as module
    source = tmp_path / 'dependencies.py'
    source.write_text('VERSION = 1')
    monkeypatch.setattr(module, '__file__', str(source))
    a = engine_dependencies()
    source.write_text('VERSION = 2')
    b = engine_dependencies()
    assert a['source_digest'] != b['source_digest']
    assert a['packages'] == b['packages'] and a['packages']


def test_missing_or_modified_dependency_metadata_blocks_comparison():
    report = run()
    assert not compare_reports(report, report).identity_issues
    copy = EvalReport.from_dict(json.loads(report.to_json()))
    copy.suite_lock.extra['engine']['python'] = 'different'
    issues = compare_reports(report, copy).identity_issues
    assert 'digest does not match' in ' '.join(issues)
    assert 'environment changed' in ' '.join(issues)
    copy.suite_lock = None
    assert compare_reports(copy, copy).mcnemar_p is None
    assert 'Missing suite lock' in ' '.join(compare_reports(copy, copy).identity_issues)


def test_undeclared_pydantic_validator_remains_opaque():
    from pydantic import BaseModel, field_validator
    class Model(BaseModel):
        amount: int
        @field_validator('amount')
        @classmethod
        def positive(cls, value):
            if value < 0:
                raise ValueError('negative')
            return value
    fp = fingerprint_evaluator(SchemaEvaluator(Model))
    assert '_pydantic_model' in fp.extra['dependencies']['opaque_fields']
    assert fp.extra['dependencies']['issues']


@pytest.mark.parametrize('kwargs', [
    {'version': '', 'dependencies': {}}, {'version': 'v1', 'dependencies': {'x': ''}},
    {'version': 'v1', 'dependencies': {'x': 4}}, {'version': 'v1', 'dependencies': {}, 'files': {4: Path('x')}},
])
def test_invalid_declarations_rejected(kwargs):
    with pytest.raises(ValueError):
        declare_dependencies(ExactMatch(), **kwargs)


def test_declared_files_detach_from_mutable_caller_map(tmp_path):
    path = tmp_path / 'data'; path.write_text('data')
    files, dependencies = {'data': path}, {'model': 'sha256:fixture'}
    grader = declare_dependencies(CallbackGrader(lambda _: True), version='v1',
                                 dependencies=dependencies, files=files)
    files.clear(); dependencies.clear()
    declaration = fingerprint_evaluator(grader).extra['dependencies']['declaration']
    assert declaration['dependencies'] and declaration['files']


@pytest.mark.parametrize('where,value', [
    ('engine', None), ('engine', {'schema': 'multivon.dependencies/v1'}),
    ('grader', None), ('grader', {'schema': 'multivon.dependencies/v1'}),
    ('issues', 'silently ignore me'), ('extra', []),
])
def test_malformed_record_cannot_become_verified_even_if_rehashed(where, value):
    from multivon_eval.dependencies import seal_lock
    report = run()
    if where == 'grader':
        report.suite_lock.evaluators[0].extra['dependencies'] = value
    elif where == 'extra':
        report.suite_lock.extra = value
    else:
        report.suite_lock.extra[where] = value
    seal_lock(report.suite_lock)
    diff = compare_reports(report, report)
    assert diff.identity_issues and diff.mcnemar_p is None


def test_private_and_credential_configuration_bound_without_plaintext():
    grader = CallbackGrader(lambda _: True)
    grader._api_key = 'test-private-key-value'
    grader.headers = {'Authorization': 'Bearer test-private-key-value'}
    before = fingerprint_evaluator(grader)
    assert 'test-private-key-value' not in json.dumps(before.extra)
    grader._api_key = 'changed'
    assert fingerprint_evaluator(grader).extra['config'] != before.extra['config']


def test_tuple_and_regex_cannot_collide_with_plain_json_config():
    import re
    grader = CallbackGrader(lambda _: True)
    grader.options = {'nested': [(1, 2)]}
    before = fingerprint_evaluator(grader)
    grader.options = {'nested': [[1, 2]]}
    after = fingerprint_evaluator(grader)
    assert before.extra['config'] == after.extra['config']
    assert before.extra['config_types'] != after.extra['config_types']
    grader.options = re.compile('x')
    before = fingerprint_evaluator(grader)
    grader.options = {'_kind': 'regex', 'pattern': 'x', 'flags': 32}
    assert before.extra != fingerprint_evaluator(grader).extra


def test_saved_and_model_error_regrading_do_not_add_preparation_calls():
    class Prepared(Evaluator):
        def prepare(self):
            raise AssertionError('Preparation would make an unwanted provider call')
        def evaluate(self, case, output):
            return self._result(1)
    target = EvalSuite('prepared').add_evaluator(Prepared())
    result = target.run_on_cases([(EvalCase('x'), 'done')], verbose=False)
    assert result.case_results[0].passed
    def broken(_):
        raise RuntimeError('target failed before producing an output')
    saved = suite().run(broken, verbose=False)
    assert regrade(saved, target).case_results[0].status.value == 'model_error'
