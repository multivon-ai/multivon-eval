"""Use real SDK serialization/retry machinery with local transports, never API keys."""
import asyncio
import importlib
import json
from unittest.mock import patch

import pytest

from multivon_eval import (
    AnthropicAdapter, EvalCase, EvalReport, EvalSuite, ExactMatch, JudgeConfig,
    ProviderJournal, capture_provider_events, provider_http_hooks, regrade,
)
from multivon_eval.judge import make_judge_call, make_judge_call_async


def transport_module(sdk):
    module = importlib.import_module(sdk.__name__ + '._base_client')
    return getattr(module, 'httpx2', None) or module.httpx


def reply(provider, *, usage=True):
    if provider == 'anthropic':
        data = {'id': 'msg_fixture', 'type': 'message', 'role': 'assistant', 'model': 'fixture-model',
                'content': [{'type': 'text', 'text': 'Yes'}], 'stop_reason': 'end_turn', 'stop_sequence': None}
        if usage:
            data['usage'] = {'input_tokens': 11, 'output_tokens': 3, 'cache_read_input_tokens': 7,
                             'cache_creation_input_tokens': 5, 'server_tool_use': {'web_search_requests': 1}}
    else:
        data = {'id': 'chat_fixture', 'object': 'chat.completion', 'created': 1, 'model': 'fixture-model',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Yes'}, 'finish_reason': 'stop'}]}
        if usage:
            data['usage'] = {'prompt_tokens': 11, 'completion_tokens': 3, 'total_tokens': 14,
                             'prompt_tokens_details': {'cached_tokens': 7},
                             'completion_tokens_details': {'reasoning_tokens': 2}}
    return data


def native_factory(sdk, handler, *, asynchronous=False):
    http = transport_module(sdk)
    cls = sdk.DefaultAsyncHttpxClient if asynchronous else sdk.DefaultHttpxClient
    return cls(transport=http.MockTransport(handler), event_hooks=provider_http_hooks(asynchronous=asynchronous))


@pytest.mark.parametrize('provider', ['anthropic', 'openai'])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_native_sdk_attempts_full_usage_and_temperature(provider, asynchronous, monkeypatch, tmp_path):
    sdk = importlib.import_module(provider)
    http = transport_module(sdk)
    monkeypatch.setenv('ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY', 'fixture-secret-never-save')
    requests = []
    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return http.Response(429, json={'error': {'type': 'rate_limit_error', 'message': 'retry'}},
                                 headers={'retry-after-ms': '1'})
        return http.Response(200, json=reply(provider), headers={'request-id': 'req_fixture'})
    config = JudgeConfig(provider=provider, model='fixture-model', temperature=0.6, max_tokens=123, timeout=9).resolve()
    with ProviderJournal(tmp_path / 'events.sqlite') as journal, capture_provider_events(journal=journal) as capture:
        with patch('multivon_eval.judge.sdk_http_client',
                   side_effect=lambda sdk, asynchronous=False: native_factory(sdk, handler, asynchronous=asynchronous)):
            result = asyncio.run(make_judge_call_async('Question?', config)) if asynchronous else make_judge_call('Question?', config)
        evidence = capture.snapshot()
        persisted = journal.events()
    assert result == 'Yes' and len(requests) == 2
    assert persisted == evidence['events']
    assert 'fixture-secret-never-save' not in json.dumps(evidence)
    sent = [e for e in evidence['events'] if e['kind'] == 'http_request']
    received = [e for e in evidence['events'] if e['kind'] == 'http_response']
    assert len(sent) == len(received) == 2
    assert sent[0]['request']['body']['value']['temperature'] == 0.6
    assert sent[0]['request']['body']['value']['messages'][0]['content'] == 'Question?'
    assert sent[0]['request']['timeout']['read'] == 9
    assert received[0]['response']['usage'] is None
    assert received[1]['response']['usage'] == reply(provider)['usage']
    assert evidence['requests_without_complete_response'] == []


def test_missing_usage_stays_unknown(monkeypatch):
    import anthropic
    http = transport_module(anthropic)
    client = anthropic.Anthropic(api_key='fixture', http_client=native_factory(
        anthropic, lambda _: http.Response(200, json=reply('anthropic', usage=False))))
    with client, capture_provider_events() as capture:
        assert AnthropicAdapter('fixture-model', client=client)('question') == 'Yes'
    response = next(e for e in capture.snapshot()['events'] if e['kind'] == 'http_response')
    assert response['response']['usage'] is None


@pytest.mark.parametrize('workers', [1, 3])
def test_target_trial_binding_roundtrip_and_regrade(workers, tmp_path):
    import anthropic
    http = transport_module(anthropic)
    client = anthropic.Anthropic(api_key='fixture', http_client=native_factory(
        anthropic, lambda _: http.Response(200, json=reply('anthropic'))))
    suite = EvalSuite('requests').add_cases([EvalCase(f'question-{i}', 'Yes') for i in range(3)]).add_evaluator(ExactMatch())
    with client:
        report = suite.run(AnthropicAdapter('fixture-model', client=client), workers=workers, verbose=False)
    restored = EvalReport.from_dict(json.loads(report.to_json()))
    assert restored.provider_evidence == report.provider_evidence
    for index, result in enumerate(restored.case_results):
        evidence = result.trials[0].data['provider_evidence']
        requests = [e for e in evidence['events'] if e['kind'] == 'http_request']
        assert len(requests) == 1 and requests[0]['labels']['case_id'] == result.case_id
        assert requests[0]['request']['body']['value']['messages'][0]['content'] == f'question-{index}'
        assert requests[0]['role'] == 'target'
    reviewed = regrade(restored, EvalSuite('saved').add_evaluator(ExactMatch()))
    for result in reviewed.case_results:
        data = result.trials[0].data
        assert not any(e['kind'] == 'http_request' for e in data['provider_evidence']['events'])
        assert data['inherited_provider_evidence'][0]['evidence']['events']


def test_custom_client_discloses_missing_wire_evidence():
    from types import SimpleNamespace
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
        content=[SimpleNamespace(text='Yes')], usage=None)))
    report = EvalSuite('custom').add_case(EvalCase('question', 'Yes')).add_evaluator(ExactMatch()).run(
        AnthropicAdapter('fixture-model', client=client), verbose=False)
    evidence = report.case_results[0].trials[0].data['provider_evidence']
    assert [e['kind'] for e in evidence['events']] == ['capture_started', 'operation_started', 'operation_finished', 'capture_finished']
    assert 'uninstrumented' in evidence['coverage']


def test_streaming_body_is_not_consumed_by_capture():
    import anthropic
    http = transport_module(anthropic)
    reads = []
    class Stream(http.SyncByteStream):
        def __iter__(self):
            reads.append(True)
            yield b'data: example\n\n'
    client = native_factory(anthropic, lambda _: http.Response(200, stream=Stream(),
                             headers={'content-type': 'text/event-stream'}))
    with client, capture_provider_events() as capture:
        with client.stream('POST', 'https://example.test/messages', json={'stream': True}) as response:
            assert reads == []
            assert response.read() == b'data: example\n\n'
    recorded = [e for e in capture.snapshot()['events'] if e['kind'] == 'http_response'][0]
    assert recorded['response']['body'] is None and recorded['response']['capture_gap']


def test_async_cancellation_preserves_started_request(tmp_path):
    import anthropic
    reached = asyncio.Event()
    async def handler(request):
        reached.set()
        await asyncio.Event().wait()
    async def execute():
        client = anthropic.AsyncAnthropic(api_key='fixture', http_client=native_factory(
            anthropic, handler, asynchronous=True))
        with ProviderJournal(tmp_path / 'cancel.sqlite') as journal, capture_provider_events(journal=journal) as capture:
            async with client:
                task = asyncio.create_task(client.messages.create(model='fixture', max_tokens=10,
                    messages=[{'role': 'user', 'content': 'wait'}]))
                await asyncio.wait_for(reached.wait(), 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert capture.snapshot()['requests_without_complete_response']
            assert [e['kind'] for e in journal.events()] == ['capture_started', 'http_request']
    asyncio.run(execute())


@pytest.mark.parametrize('provider', ['anthropic', 'openai'])
def test_vision_requests_retain_media_and_native_usage(provider, monkeypatch):
    from multivon_eval.vision import call_vision
    sdk = importlib.import_module(provider)
    http = transport_module(sdk)
    monkeypatch.setenv('ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY', 'fixture')
    config = JudgeConfig(provider=provider, model='fixture', temperature=0.4, timeout=9).resolve()
    uri = 'data:image/png;base64,aW1hZ2UtZml4dHVyZQ=='
    with patch('multivon_eval.vision.sdk_http_client',
               side_effect=lambda sdk: native_factory(sdk, lambda _: http.Response(200, json=reply(provider)))):
        with capture_provider_events() as capture:
            assert call_vision('Read image', [uri], config, max_tokens=31) == 'Yes'
    evidence = capture.snapshot()
    request = next(e for e in evidence['events'] if e['kind'] == 'http_request')
    response = next(e for e in evidence['events'] if e['kind'] == 'http_response')
    body = request['request']['body']['value']
    assert 'aW1hZ2UtZml4dHVyZQ==' in json.dumps(body)
    assert body['temperature'] == 0.4 and body['max_tokens'] == 31
    assert request['role'] == 'judge' and request['request']['timeout']['read'] == 9
    assert response['response']['usage'] == reply(provider)['usage']
