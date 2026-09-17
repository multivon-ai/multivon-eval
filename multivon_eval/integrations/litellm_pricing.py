"""Use upstream native-response cost calculation with pinned pricing evidence."""
from __future__ import annotations

import importlib.metadata
import json
import os
from urllib.parse import urlsplit

from ..case_manifest import canonical_json, digest
from ..provider_accounting import native_tokens


class LiteLLMPricer:
    """Price supported native responses without dispatching model requests.

    Set LITELLM_LOCAL_MODEL_COST_MAP=True before importing LiteLLM to avoid its
    default remote catalog fetch. Supply the imported module so applications
    own its initialization/settings. Its loaded catalog must stay unchanged.
    Estimates are list-price approximations, never provider invoices.
    """
    def __init__(self, litellm):
        if os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP', '').lower() != 'true':
            raise ValueError('Set LITELLM_LOCAL_MODEL_COST_MAP=True before importing LiteLLM')
        self._sdk = litellm
        self._catalog = json.loads(canonical_json(litellm.model_cost))
        self._digest = digest(self._catalog)
        self._version = importlib.metadata.version('litellm')

    def __call__(self, request_event, response_event):
        if digest(self._sdk.model_cost) != self._digest:
            raise ValueError('LiteLLM catalog changed after estimator creation')
        request = request_event['request']
        response = response_event['response']
        url = urlsplit(request['url'])
        provider = request_event['provider']
        allowed = {'anthropic': ('api.anthropic.com', '/v1/messages'),
                   'openai': ('api.openai.com', '/v1/chat/completions')}
        if provider not in allowed or (url.hostname, url.path) != allowed[provider] or url.scheme != 'https':
            raise ValueError('Endpoint has no validated native LiteLLM pricing bridge')
        if request['body']['encoding'] != 'json' or response['body']['encoding'] != 'json':
            raise ValueError('Native JSON bodies are required')
        payload, native = request['body']['value'], response['body']['value']
        # Use the effective request model for routing/billing, retain the native
        # resolved model too. Never price a custom endpoint from a familiar ID.
        model = payload['model']
        catalog_key = next((key for key in (f'{provider}/{model}', model) if key in self._catalog), None)
        if catalog_key is None:
            raise ValueError('Model missing from the frozen LiteLLM catalog')
        if not isinstance(native.get('usage'), dict):
            raise ValueError('Native usage required; never infer tokens from text')
        usage = native['usage']
        native_tokens(provider, usage)
        tier = native.get('service_tier', usage.get('service_tier'))
        if tier not in (None, 'standard', 'default'):
            raise ValueError('Only standard served tiers are validated by this pricing bridge')
        if tier is None and payload.get('service_tier') not in (None, 'standard', 'default'):
            raise ValueError('Requested tier has no confirmed served-tier pricing')
        if payload.get('inference_geo') not in (None, 'global') or payload.get('speed') not in (None, 'standard'):
            raise ValueError('Requested regional or fast-mode pricing is not validated')
        if usage.get('inference_geo') not in (None, 'global', 'not_available') or usage.get('speed') not in (None, 'standard'):
            raise ValueError('Regional or fast-mode pricing requires a validated estimator')
        if any(value for value in (usage.get('server_tool_use') or {}).values()):
            raise ValueError('Server-tool fees require a validated estimator')
        if any(tool.get('type') not in (None, 'custom', 'function') for tool in payload.get('tools', [])):
            raise ValueError('Configured server-tool fees require a validated estimator')
        if payload.get('modalities') not in (None, ['text']):
            raise ValueError('Nontext output pricing requires a validated estimator')
        amount = self._sdk.completion_cost(completion_response=json.loads(canonical_json(native)),
                                           model=model, custom_llm_provider=provider,
                                           service_tier=native.get('service_tier'))
        if digest(self._sdk.model_cost) != self._digest:
            raise ValueError('LiteLLM catalog changed during estimation')
        return {'cost_usd': amount, 'provenance': {
            'estimator': 'litellm.completion_cost', 'version': self._version,
            'catalog_digest': self._digest, 'catalog_key': catalog_key,
            'catalog_entry': self._catalog[catalog_key], 'request_model': model,
            'response_model': native.get('model'),
            'source': 'https://github.com/BerriAI/litellm',
            'limits': 'Upstream list-price estimate; not a bill; discounts, tax and external services excluded',
        }}
