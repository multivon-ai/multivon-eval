"""Declared native task contracts and actual retry preservation under drift."""
import json

import pytest

pytest.importorskip('inspect_ai')
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.solver import solver

from multivon_eval import AcceptancePolicy, CaseManifest, CheckRequirement, EvalCase, EvalReport, EvalSuite, ExactMatch, regrade
from multivon_eval.integrations.inspect import (
    as_inspect_scorer, bind_inspect_task, from_inspect_log, to_inspect_dataset,
)
from multivon_eval.integrations.inspect_contract import GRADING_KEY, SAMPLE_KEY, TASK_KEY
from benchmarks.industrial.retry_contract_experiment import experiment


@solver
def answer():
    async def solve(state, generate):
        state.output = ModelOutput.from_content('fixture', 'yes')
        return state
    return solve


def native_task(*, prompt='x', grader=None):
    return Task(dataset=to_inspect_dataset(CaseManifest('fixture', [EvalCase(prompt, 'yes', case_id='a')])),
                solver=answer(), scorer=as_inspect_scorer(grader or ExactMatch()),
                config=GenerateConfig(temperature=0.2))


def bound_task(**kwargs):
    return bind_inspect_task(native_task(), version='fixture/v1', dependencies={}, **kwargs)


@pytest.fixture
def log(tmp_path):
    return inspect_eval(bound_task(), model='mockllm/model', display='none', log_dir=str(tmp_path))[0]


def policy():
    return AcceptancePolicy((CheckRequirement('exact_match'),))


def test_native_retry_mixed_grades_and_preflight_calls(tmp_path):
    result = experiment(tmp_path / 'probe')
    assert result['blocked_target_calls'] == 0 and result['mixed_native_passed'] == 3
    assert result['mixed_import_decision'] == 'indeterminate' and result['fresh_passed'] == 2


def test_identical_reconstruction_ignores_generated_message_ids_and_roundtrips(log):
    equivalent = bound_task(previous_log=log)
    assert equivalent.metadata[TASK_KEY]['digest'] == log.eval.metadata[TASK_KEY]['digest']
    report = from_inspect_log(log)
    assert not report.evidence_issues and policy().evaluate(report).decision == 'accept'
    loaded = EvalReport.from_dict(json.loads(report.to_json()))
    assert loaded.case_results[0].trials == report.case_results[0].trials


@pytest.mark.parametrize('field', ['grader', 'dataset', 'input', 'target', 'generation', 'epochs', 'sandbox', 'version'])
def test_preflight_rejects_observed_task_changes(log, field):
    current = native_task(prompt='changed' if field == 'dataset' else 'x')
    if field == 'grader':
        current.scorer = [as_inspect_scorer(ExactMatch(threshold=0.9))]
    elif field == 'input':
        current.dataset[0].input = 'different actual prompt'
    elif field == 'target':
        current.dataset[0].target = 'different target'
    elif field == 'generation':
        current.config.temperature = 0.8
    elif field == 'epochs':
        current.epochs = 2
    elif field == 'sandbox':
        current.sandbox = ('docker', 'changed-compose.yaml')
    with pytest.raises(ValueError, match='incompatible'):
        bind_inspect_task(current, version='fixture/v2' if field == 'version' else 'fixture/v1',
                          dependencies={}, previous_log=log)


def test_named_files_and_credential_configuration_are_bound(tmp_path):
    path = tmp_path / 'policy.json'
    path.write_text('{"minimum": 1}')
    original = bound_task(files={'policy': path}, configuration={
        'api_key': 'fixture-secret', 'endpoint': 'https://example.invalid/?api_key=query-secret'})
    encoded = json.dumps(original.metadata)
    assert 'fixture-secret' not in encoded and 'query-secret' not in encoded
    assert str(path) not in encoded
    log = inspect_eval(original, model='mockllm/model', display='none', log_dir=str(tmp_path / 'logs'))[0]
    path.write_text('{"minimum": 2}')
    with pytest.raises(ValueError, match='incompatible'):
        bound_task(files={'policy': path}, configuration={
            'api_key': 'fixture-secret', 'endpoint': 'https://example.invalid/?api_key=query-secret'}, previous_log=log)


@pytest.mark.parametrize('mutation', ['root_missing', 'root_digest', 'root_structure', 'sample_binding', 'sample_input', 'sample_target', 'grading'])
def test_import_cannot_accept_missing_or_changed_binding(log, mutation):
    if mutation == 'root_missing':
        log.eval.metadata.pop(TASK_KEY)
    elif mutation == 'root_digest':
        log.eval.metadata[TASK_KEY]['digest'] = '0' * 64
    elif mutation == 'root_structure':
        log.eval.metadata[TASK_KEY] = 'unsupported contract format'
    elif mutation == 'sample_binding':
        log.samples[0].metadata[SAMPLE_KEY] = '0' * 64
    elif mutation == 'sample_input':
        log.samples[0].input = 'changed native input'
    elif mutation == 'sample_target':
        log.samples[0].target = 'changed native reference'
    else:
        log.samples[0].metadata[GRADING_KEY] = []
    report = from_inspect_log(log)
    assert report.evidence_issues
    assert policy().evaluate(report).decision == 'indeterminate'
    report.evidence_issues.clear()
    assert AcceptancePolicy((CheckRequirement('exact_match'),), max_error_rate=1).evaluate(report).decision == 'indeterminate'
    reviewed = regrade(report, EvalSuite('new grades').add_evaluator(ExactMatch()))
    assert policy().evaluate(reviewed).decision == 'indeterminate'


def test_legacy_retry_history_has_unknown_contract_not_invented_compatibility(log):
    log.eval.metadata.pop(TASK_KEY)
    for sample in log.samples:
        sample.metadata.pop(SAMPLE_KEY)
    report = from_inspect_log(log, previous_logs=[log.model_copy(deep=True)])
    assert any('Missing declared' in issue for issue in report.evidence_issues)
    assert policy().evaluate(report).decision == 'indeterminate'


def test_native_execution_override_is_detected_in_retry_chain(log):
    original = log.model_copy(deep=True)
    log.plan.config.temperature = 0.7
    report = from_inspect_log(log, previous_logs=[original])
    assert any('execution settings changed' in issue for issue in report.evidence_issues)
    assert policy().evaluate(report).decision == 'indeterminate'


def test_observed_grader_mutation_blocks_acceptance(tmp_path):
    class Changing(ExactMatch):
        def evaluate(self, case, output):
            result = super().evaluate(case, output)
            self.threshold = 0.1
            return result
    from multivon_eval import declare_dependencies
    grader = declare_dependencies(Changing(), version='changing-fixture/v1', dependencies={})
    task = bind_inspect_task(native_task(grader=grader), version='fixture/v1', dependencies={})
    log = inspect_eval(task, model='mockllm/model', display='none', log_dir=str(tmp_path))[0]
    report = from_inspect_log(log)
    assert any('changed during scoring' in issue for issue in report.evidence_issues)
    assert policy().evaluate(report).decision == 'indeterminate'
