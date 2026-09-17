"""Actual upstream cost calculator; no provider requests or remote price fetches."""
import copy
import importlib.util
import json
import socket
import tarfile
from pathlib import Path

import pytest

from multivon_eval import account_provider_events
from multivon_eval.integrations.litellm_pricing import LiteLLMPricer

pytestmark = pytest.mark.skipif(importlib.util.find_spec('litellm') is None, reason='Optional LiteLLM pricing extra')


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    import litellm
    monkeypatch.setattr(litellm, 'telemetry', False)
    def forbidden(*args, **kwargs):
        raise AssertionError('Pricing must not dispatch network requests')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    return litellm


def live_events():
    path = Path(__file__).parents[1] / 'benchmarks/industrial/results/provider-capture-2026-09-17/evidence.tar.gz'
    with tarfile.open(path) as archive:
        return json.load(archive.extractfile('events.json'))


def pair():
    events = live_events()
    request = next(e for e in events if e['kind'] == 'http_request')
    response = next(e for e in events if e['kind'] == 'http_response' and e['request_id'] == request['request_id'])
    return request, response


def test_price_all_four_actual_calls_without_new_requests(upstream):
    costs = account_provider_events(live_events(), price_estimator=LiteLLMPricer(upstream),
                                   coverage_declaration='Frozen four-call protocol, complete journal')
    assert costs.complete and costs.total_calls == 4 and costs.total_tokens == 68
    assert costs.total_cost_usd == pytest.approx(48 * 1e-6 + 20 * 5e-6)
    for row in costs.evidence['requests']:
        assert row['pricing']['version']
        assert row['pricing']['catalog_entry']['input_cost_per_token'] == 1e-6
        assert len(row['pricing']['catalog_digest']) == 64


def test_anthropic_cache_windows_use_upstream_native_conversion(upstream):
    request, response = pair()
    response['response']['body']['value']['usage'] = {
        'input_tokens': 100, 'output_tokens': 10, 'cache_creation_input_tokens': 100,
        'cache_read_input_tokens': 200,
        'cache_creation': {'ephemeral_5m_input_tokens': 60, 'ephemeral_1h_input_tokens': 40},
    }
    estimate = LiteLLMPricer(upstream)(request, response)
    assert estimate['cost_usd'] == pytest.approx(100e-6 + 60 * 1.25e-6 + 40 * 2e-6 + 200 * 0.1e-6 + 10 * 5e-6)


def test_openai_cache_and_reasoning_tokens_are_not_billed_twice(upstream):
    request, response = pair()
    request['provider'] = 'openai'
    request['request']['url'] = 'https://api.openai.com/v1/chat/completions'
    request['request']['body']['value']['model'] = 'gpt-4o-mini'
    response['response']['body']['value'] = {'model': 'gpt-4o-mini', 'choices': [], 'usage': {
        'prompt_tokens': 1000, 'completion_tokens': 50, 'total_tokens': 1050,
        'prompt_tokens_details': {'cached_tokens': 200},
        'completion_tokens_details': {'reasoning_tokens': 10}}}
    estimate = LiteLLMPricer(upstream)(request, response)
    assert estimate['cost_usd'] == pytest.approx(800 * 0.15e-6 + 200 * 0.075e-6 + 50 * 0.6e-6)


@pytest.mark.parametrize('change', ['endpoint', 'unknown_model', 'tier', 'server_tool', 'region', 'missing_usage', 'empty_usage', 'configured_server_tool'])
def test_unvalidated_pricing_dimensions_are_unknown(upstream, change):
    request, response = pair()
    usage = response['response']['body']['value']['usage']
    if change == 'endpoint':
        request['request']['url'] = 'https://proxy.example/v1/messages'
    elif change == 'unknown_model':
        request['request']['body']['value']['model'] = 'unknown-pricing-model'
    elif change == 'tier':
        usage['service_tier'] = 'priority'
    elif change == 'server_tool':
        usage['server_tool_use'] = {'web_search_requests': 1}
    elif change == 'region':
        usage['inference_geo'] = 'us'
    elif change == 'empty_usage':
        response['response']['body']['value']['usage'] = {}
    elif change == 'configured_server_tool':
        request['request']['body']['value']['tools'] = [{'type': 'web_search_20250305', 'name': 'web_search'}]
    else:
        response['response']['body']['value']['usage'] = None
    with pytest.raises(ValueError):
        LiteLLMPricer(upstream)(request, response)


def test_catalog_mutation_is_detected(upstream, monkeypatch):
    pricer = LiteLLMPricer(upstream)
    model = 'claude-haiku-4-5'
    altered = copy.deepcopy(upstream.model_cost[model])
    altered['input_cost_per_token'] = 0
    monkeypatch.setitem(upstream.model_cost, model, altered)
    with pytest.raises(ValueError, match='changed'):
        pricer(*pair())


def test_documented_native_accounting_workflow(upstream, monkeypatch, tmp_path):
    import anthropic._base_client as base
    import re
    from multivon_eval import provider_http_hooks
    http = getattr(base, 'httpx2', None) or base.httpx
    _, response = pair()
    body = response['response']['body']['value']
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fixture-key')
    monkeypatch.setattr('multivon_eval.adapters.sdk_http_client', lambda sdk: sdk.DefaultHttpxClient(
        transport=http.MockTransport(lambda _: http.Response(200, json=body)),
        event_hooks=provider_http_hooks()))
    source = (Path(__file__).parents[1] / 'docs/guides/provider-accounting.mdx').read_text()
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for code in re.findall(r'```python\n(.*?)```', source, re.S):
        exec(compile(code, 'provider-accounting.mdx', 'exec'), namespace)
    costs = namespace['report'].costs
    assert costs.complete and costs.total_calls == 1
    assert costs.total_cost_usd == pytest.approx(37e-6)
    assert json.loads((tmp_path / 'accounted-report.json').read_text())['summary']['costs']['complete']
