"""Native transport and process-boundary tests; no external services or keys."""
import asyncio
import json
import select
import subprocess
import sys
from unittest.mock import patch

import pytest

from multivon_eval import (
    EvalCase, EvalSuite, ExactMatch, JudgeConfig, ProviderJournal,
    capture_provider_events, provider_http_hooks,
)
from multivon_eval.judge import make_judge_call, make_judge_call_async
from multivon_eval.provider_http import google_http_options


def test_sigkill_keeps_request_committed_before_transport_dispatch(tmp_path):
    journal_path = tmp_path / 'killed.sqlite'
    script = '''
import importlib, sys, threading
import anthropic
from multivon_eval import (AnthropicAdapter, EvalCase, EvalSuite, ExactMatch,
    ProviderJournal, capture_provider_events, provider_http_hooks)
module = importlib.import_module('anthropic._base_client')
http = getattr(module, 'httpx2', None) or module.httpx
def handler(request):
    print('dispatched', flush=True)
    threading.Event().wait()
with ProviderJournal(sys.argv[1]) as journal, capture_provider_events(journal=journal):
    with anthropic.Anthropic(api_key='fixture', http_client=anthropic.DefaultHttpxClient(
        transport=http.MockTransport(handler), event_hooks=provider_http_hooks())) as client:
        suite = EvalSuite('interrupted').add_case(EvalCase('wait', 'Yes')).add_evaluator(ExactMatch())
        suite.run(AnthropicAdapter('fixture', client=client, system_prompt='frozen prompt',
                                   temperature=0.4, max_tokens=10), verbose=False)
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(journal_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 30)[0], 'Child did not reach transport'
        line = process.stdout.readline().strip()
        assert line == 'dispatched', process.stderr.read() if process.poll() is not None else line
        process.kill()
        process.communicate(timeout=10)
        assert process.returncode < 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    with ProviderJournal(journal_path) as recovered:
        events = recovered.events()
    assert [e['kind'] for e in events] == ['capture_started', 'capture_started', 'capture_started',
                                        'operation_started', 'http_request']
    started = next(e for e in events if e['kind'] == 'capture_started' and e['capture_kind'] == 'trial')
    configuration = started['execution']['target_before']['configuration']
    assert configuration['system_prompt'] == 'frozen prompt'
    assert configuration['temperature'] == 0.4 and configuration['max_tokens'] == 10
    assert events[-1]['request']['body']['value']['messages'][0]['content'] == 'wait'
    assert not any(e['kind'] == 'http_response' for e in events)


def test_journal_rejects_changed_storage_identity(tmp_path):
    with ProviderJournal(tmp_path / 'changed.sqlite') as journal:
        with capture_provider_events(journal=journal):
            pass
        journal._connection.execute("UPDATE provider_events SET event_id='changed' WHERE sequence=1")
        journal._connection.commit()
        with pytest.raises(ValueError, match='digest mismatch'):
            journal.events()


@pytest.mark.parametrize('asynchronous', [False, True])
def test_google_native_wire_with_explicit_async_httpx(asynchronous, monkeypatch):
    pytest.importorskip('google.genai')
    import httpx
    monkeypatch.setenv('GOOGLE_API_KEY', 'fixture-secret-never-save')
    usage = {'promptTokenCount': 12, 'candidatesTokenCount': 3,
             'thoughtsTokenCount': 4, 'cachedContentTokenCount': 2, 'totalTokenCount': 19}
    def handler(request):
        return httpx.Response(200, json={'candidates': [{'content': {'role': 'model',
            'parts': [{'text': 'Yes'}]}, 'finishReason': 'STOP'}], 'usageMetadata': usage})
    config = JudgeConfig(provider='google', model='fixture', temperature=0.3, timeout=9).resolve()
    options = google_http_options(9)
    options['client_args']['transport'] = httpx.MockTransport(handler)
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                event_hooks=provider_http_hooks(asynchronous=True)) as client:
            options['httpx_async_client'] = client
            return await make_judge_call_async('Question?', config)
    with patch('multivon_eval.judge.google_http_options', return_value=options):
        with capture_provider_events() as capture:
            result = asyncio.run(execute()) if asynchronous else make_judge_call('Question?', config)
    evidence = capture.snapshot()
    assert result == 'Yes'
    requests = [e for e in evidence['events'] if e['kind'] == 'http_request']
    responses = [e for e in evidence['events'] if e['kind'] == 'http_response']
    assert len(requests) == len(responses) == 1
    assert requests[0]['request']['body']['value']['generationConfig']['temperature'] == 0.3
    assert responses[0]['response']['usage'] == usage
    assert 'fixture-secret-never-save' not in json.dumps(evidence)
    assert evidence['coverage_gaps'] == []


def test_google_default_async_discloses_unobserved_wire(monkeypatch):
    pytest.importorskip('google.genai')
    import httpx
    monkeypatch.setenv('GOOGLE_API_KEY', 'fixture')
    options = google_http_options(9)
    assert 'async_client_args' not in options
    options['async_client_args'] = {'transport': httpx.MockTransport(lambda _: httpx.Response(200,
        json={'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Yes'}]}}]}))}
    config = JudgeConfig(provider='google', model='fixture').resolve()
    with patch('multivon_eval.judge.google_http_options', return_value=options):
        with capture_provider_events() as capture:
            assert asyncio.run(make_judge_call_async('Question?', config)) == 'Yes'
    evidence = capture.snapshot()
    assert evidence['coverage_gaps'][0]['reason'] == 'No instrumented HTTP request observed'
    assert not any(e['kind'] == 'http_request' for e in evidence['events'])


def test_saved_report_and_returned_snapshot_match(tmp_path):
    path = tmp_path / 'report.json'
    with ProviderJournal(tmp_path / 'run.sqlite') as journal:
        with capture_provider_events(journal=journal):
            report = EvalSuite('snapshot').add_case(EvalCase('q', 'Yes')).add_evaluator(ExactMatch()).run(
                lambda _: 'Yes', verbose=False, save_json=str(path))
        events = journal.events()
    saved = json.loads(path.read_text())
    assert saved['provider_evidence'] == report.provider_evidence
    # The report snapshots the measurement before export/gates; the journal
    # records the later method closure, including failures during export/gates.
    assert report.provider_evidence['state'] == 'open'
    assert report.case_results[0].trials[0].data['provider_evidence']['state'] == 'closed'
    assert any(e['kind'] == 'capture_finished' and e['capture_kind'] == 'run' for e in events)
