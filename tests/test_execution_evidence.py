"""Declared target interventions, within-run drift and native prompt forwarding."""
import asyncio
import importlib
import json

import pytest

from multivon_eval import (AnthropicAdapter, OpenAIAdapter, EvalCase, EvalReport, EvalSuite,
                           ExactMatch, JudgeRetry, declare_target, provider_http_hooks, regrade)
from multivon_eval.adapters import ModelAdapter
from multivon_eval.case_manifest import digest
from multivon_eval.compare import compare_reports
from multivon_eval.execution_evidence import target_snapshot
from multivon_eval.result import EvalResult
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.exceptions import JudgeUnavailable


def suite():
    return EvalSuite('target evidence').add_cases([EvalCase('q1', 'Yes'), EvalCase('q2', 'Yes')]).add_evaluator(ExactMatch())


@pytest.mark.parametrize('provider', ['anthropic', 'openai'])
def test_prompt_copy_preserves_native_adapter_context_and_wire_evidence(provider):
    sdk = importlib.import_module(provider)
    base = importlib.import_module(provider + '._base_client')
    http = getattr(base, 'httpx2', None) or base.httpx
    if provider == 'anthropic':
        reply = {'id': 'fixture', 'type': 'message', 'role': 'assistant', 'model': 'fixture',
                 'content': [{'type': 'text', 'text': 'Yes'}], 'usage': {'input_tokens': 2, 'output_tokens': 1}}
        native, adapter_class = sdk.Anthropic, AnthropicAdapter
    else:
        reply = {'id': 'fixture', 'object': 'chat.completion', 'created': 1, 'model': 'fixture',
                 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Yes'}, 'finish_reason': 'stop'}],
                 'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}
        native, adapter_class = sdk.OpenAI, OpenAIAdapter
    with native(api_key='fixture-secret', http_client=sdk.DefaultHttpxClient(
            transport=http.MockTransport(lambda _: http.Response(200, json=reply)),
            event_hooks=provider_http_hooks())) as client:
        original = adapter_class('fixture', client=client, system_prompt='original', temperature=0.2, max_tokens=13)
        target = original.with_system_prompt('replacement')
        evaluation = EvalSuite('prompt').add_case(EvalCase('question', 'Yes', context='source text')).add_evaluator(ExactMatch())
        report = evaluation.run(target, verbose=False)
    assert type(target) is type(original) and target._client is original._client
    assert original._system_prompt == 'original'
    before = report.execution['target_before']
    assert before['configuration']['system_prompt'] == 'replacement'
    assert before['configuration']['temperature'] == 0.2 and before['configuration']['max_tokens'] == 13
    assert before['client']['observed_settings']['max_retries']['value'] == 2
    trial = report.case_results[0].trials[0].data
    request = next(e for e in trial['provider_evidence']['events'] if e['kind'] == 'http_request')
    payload = request['request']['body']['value']
    wire = payload['system'] if provider == 'anthropic' else payload['messages'][0]['content']
    assert 'replacement' in wire and 'source text' in wire and 'original' not in wire
    assert not trial['execution']['changed_during_execution']


def test_generic_adapter_cannot_silently_ignore_a_system_prompt():
    class Custom(ModelAdapter):
        def __call__(self, text):
            return text
    with pytest.raises(NotImplementedError):
        Custom().with_system_prompt('must matter')


def test_custom_client_properties_are_not_invoked_for_metadata():
    class Client:
        @property
        def base_url(self):
            raise AssertionError('Metadata must not invoke custom properties')
    evidence = target_snapshot(AnthropicAdapter(client=Client()))
    assert evidence['client']['observed_settings'] == {}
    assert evidence['issues']


def test_credentials_in_declared_options_and_urls_are_bound_without_plaintext():
    target = declare_target(lambda text: text, version='v1', dependencies={}, configuration={
        'api_key': 'private-key-fixture', 'nested': {'Authorization': 'Bearer private-auth-fixture'},
        'endpoint': 'https://private-user:private-password@example.test/path?api_key=private-query&X-Amz-Signature=private-signature#private-fragment',
    })
    serialized = json.dumps(target_snapshot(target))
    for secret in ('private-key-fixture', 'private-auth-fixture', 'private-user', 'private-password',
                   'private-query', 'private-signature', 'private-fragment'):
        assert secret not in serialized
    target.configuration['api_key'] = 'changed-key-fixture'
    assert target_snapshot(target)['digest'] != json.loads(serialized)['digest']


@pytest.mark.parametrize('mode', ['sync', 'parallel', 'async'])
def test_target_declarations_runner_policy_and_trials_survive(mode, tmp_path):
    class Transient(Evaluator):
        name = 'temporary'
        def evaluate(self, case, output):
            if output == 'first':
                raise JudgeUnavailable('temporary')
            return EvalResult(self.name, 1, True)
    calls = []
    def target(prompt):
        calls.append(prompt)
        return 'first' if len(calls) == 1 else 'Yes'
    async def async_target(prompt):
        return target(prompt)
    declared = declare_target(async_target if mode == 'async' else target, version='target/v1',
                              configuration={'prompt_revision': 'p1'}, dependencies={'model': 'fixture/1'})
    evaluation = EvalSuite('repeats').add_case(EvalCase('q', 'Yes')).add_evaluators(ExactMatch(), Transient())
    kwargs = {'runs': 2, 'judge_retry': JudgeRetry(max_attempts=2, base_backoff=0), 'verbose': False}
    if mode == 'async':
        report = asyncio.run(evaluation.run_async(declared, concurrency=2, **kwargs))
    else:
        path = tmp_path / 'report.json'
        report = evaluation.run(declared, workers=2 if mode == 'parallel' else 1, save_json=str(path), **kwargs)
        assert json.loads(path.read_text())['execution'] == report.execution
    restored = EvalReport.from_dict(json.loads(report.to_json()))
    assert restored.execution == report.execution
    assert len(calls) == 4 and len(report.case_results[0].trials) == 4
    assert report.execution['policy']['value']['runs'] == 2
    assert report.execution['policy']['value']['judge_retry']['max_attempts'] == 2
    for trial in report.case_results[0].trials:
        evidence = trial.data['execution']
        assert evidence['target_before']['declaration']['version'] == 'target/v1'
        assert not evidence['changed_during_execution']
    reviewed = regrade(restored, EvalSuite('regrade').add_evaluator(ExactMatch()))
    assert len(calls) == 4
    for result in reviewed.case_results:
        data = result.trials[0].data
        assert data['execution']['target_before']['origin'] == 'saved_outputs'
        assert data['inherited_execution'][0]['execution']['target_before']['declaration']['version'] == 'target/v1'


def test_target_intervention_is_described_without_invalidating_pairing():
    baseline = suite().run(declare_target(lambda _: 'Yes', version='baseline', dependencies={}), verbose=False)
    proposal = suite().run(declare_target(lambda _: 'No', version='candidate', dependencies={}), verbose=False)
    diff = compare_reports(baseline, proposal)
    assert not diff.identity_issues and diff.mcnemar_p is not None
    assert 'declaration.version' in diff.execution_changes['target_paths']
    assert diff.to_dict()['execution_changes'] == diff.execution_changes
    assert 'Execution changes' in diff.to_markdown()


def test_declared_file_drift_is_retained_and_blocks_controlled_comparison(tmp_path):
    path = tmp_path / 'prompt.txt'
    path.write_text('old')
    baseline = suite().run(declare_target(lambda _: 'Yes', version='v1', dependencies={}, files={'prompt': path}), verbose=False)
    def changing(_):
        path.write_text('new')
        return 'Yes'
    proposal = suite().run(declare_target(changing, version='v1', dependencies={}, files={'prompt': path}), workers=1, verbose=False)
    assert proposal.execution['changed_during_execution']
    assert proposal.case_results[0].trials[0].data['execution']['changed_during_execution']
    diff = compare_reports(baseline, proposal)
    assert diff.mcnemar_p is None
    assert any('target configuration' in issue for issue in diff.identity_issues)


def test_unknown_target_contract_is_not_an_invented_identity():
    report = suite().run(lambda _: 'Yes', verbose=False)
    assert report.execution['target_before']['issues']
    diff = compare_reports(report, report)
    assert diff.execution_notes and not diff.identity_issues
    assert diff.mcnemar_p is not None


def test_execution_tampering_and_forged_drift_are_rejected():
    data = json.loads(suite().run(lambda _: 'Yes', verbose=False).to_json())
    data['execution']['changed_during_execution'] = True
    with pytest.raises(ValueError, match='digest'):
        EvalReport.from_dict(data)
    record = data['execution']
    record.pop('digest')
    record['digest'] = digest(record)
    with pytest.raises(ValueError, match='drift flag'):
        EvalReport.from_dict(data)


def test_missing_declared_file_is_an_explicit_issue(tmp_path):
    target = declare_target(lambda _: 'Yes', version='v1', dependencies={}, files={'policy': tmp_path / 'missing'})
    report = suite().run(target, verbose=False)
    assert report.passed == 2
    assert any('unavailable' in issue for issue in report.execution['target_before']['issues'])


def test_snapshot_failure_does_not_invoke_or_prevent_target():
    calls = []
    def callback(prompt):
        calls.append(prompt)
        return 'Yes'
    target = declare_target(callback, version='v1', dependencies={})
    target.configuration['opaque'] = object()
    report = suite().run(target, workers=1, verbose=False)
    assert len(calls) == 2 and report.passed == 2
    assert report.execution['target_before']['issues'] == ['Target snapshot unavailable (TypeError)']
