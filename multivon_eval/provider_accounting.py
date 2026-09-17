"""Reconcile observed attempts; pricing stays with an upstream/caller estimator."""
from __future__ import annotations

import json
import math

from .case_manifest import canonical_json, digest
from .costs import Costs, ProviderUsage
from .provider_evidence import SCHEMA


def provider_events(report):
    """Current-run snapshots only; inherited regrade events are not new spend.

    Report-level snapshots precede run closure. Use the closed journal's events
    for a complete lifecycle audit. Identical event copies are deduplicated.
    """
    bundles = [report.provider_evidence]
    for result in report.case_results:
        bundles.extend(t.data.get('provider_evidence') for t in result.trials)
    return _events(e for b in bundles if b is not None for e in b['events'])


def _events(events):
    unique = {}
    for original in events:
        event = json.loads(canonical_json(original))
        claimed = event.pop('digest', None)
        if event.get('schema') != SCHEMA or claimed != digest(event):
            raise ValueError('Invalid provider event schema or digest')
        event['digest'] = claimed
        identity = event.get('event_id')
        if not isinstance(identity, str) or not identity:
            raise ValueError('Provider event has no identity')
        if identity in unique and unique[identity] != event:
            raise ValueError('Conflicting provider event identity')
        unique[identity] = event
    return list(unique.values())


def _count(usage, name, *, optional=False):
    value = usage.get(name, 0 if optional else None)
    if type(value) is not int or value < 0:
        raise ValueError(f'Missing or invalid native usage field {name}')
    return value


def native_tokens(provider, usage):
    """Normalize token counts, preserving the original usage separately.

    Reasoning tokens are already in OpenAI completion counts. Anthropic cache
    tokens are additional input; Google thoughts are additional output.
    """
    if not isinstance(usage, dict):
        raise ValueError('Native usage is missing')
    if provider == 'anthropic':
        input_tokens = (_count(usage, 'input_tokens') + _count(usage, 'cache_read_input_tokens', optional=True)
                        + _count(usage, 'cache_creation_input_tokens', optional=True))
        return input_tokens, _count(usage, 'output_tokens')
    if provider == 'openai':
        if 'prompt_tokens' in usage:
            incoming, outgoing = _count(usage, 'prompt_tokens'), _count(usage, 'completion_tokens')
        else:
            incoming, outgoing = _count(usage, 'input_tokens'), _count(usage, 'output_tokens')
        if 'total_tokens' in usage and _count(usage, 'total_tokens') != incoming + outgoing:
            raise ValueError('Native token total is inconsistent')
        return incoming, outgoing
    if provider == 'google':
        incoming = _count(usage, 'promptTokenCount')
        outgoing = _count(usage, 'candidatesTokenCount') + _count(usage, 'thoughtsTokenCount', optional=True)
        # Tool-use prompt tokens are separate usage and cannot be dropped.
        incoming += _count(usage, 'toolUsePromptTokenCount', optional=True)
        if 'totalTokenCount' in usage and _count(usage, 'totalTokenCount') != incoming + outgoing:
            raise ValueError('Native token total is inconsistent')
        return incoming, outgoing
    raise ValueError(f'No native token normalization for provider {provider!r}')


def account_provider_events(events, *, price_estimator=None, coverage_declaration=None):
    """Return Costs from retained events, never from retokenized output text.

    ``price_estimator(request_event, response_event)`` returns a dict containing
    finite nonnegative ``cost_usd`` and a nonempty ``provenance`` dict. Missing
    estimates leave dollars unknown without discarding observed token counts.
    ``coverage_declaration`` is the caller's explanation that this closed set
    includes all run provider usage, including targets, preparation and judges.
    It cannot override observed gaps and is not independent attestation.
    """
    events = _events(events)
    requests, responses, started, finished, captures, closed = {}, {}, {}, set(), set(), set()
    gaps = []
    for event in events:
        kind = event['kind']
        if kind == 'http_request':
            key = event['request_id']
            if key in requests:
                raise ValueError('Duplicate physical request identity')
            requests[key] = event
        elif kind == 'http_response':
            key = event['request_id']
            if key in responses:
                raise ValueError('Duplicate physical response identity')
            responses[key] = event
        elif kind == 'operation_started':
            started[event['operation_id']] = event
        elif kind == 'operation_finished':
            finished.add(event['operation_id'])
        elif kind == 'capture_started':
            captures.add(event['capture_id'])
        elif kind == 'capture_finished':
            closed.add(event['capture_id'])
    if not events:
        gaps.append('No provider lifecycle evidence supplied')
    gaps.extend(f'Capture {key} has no start' for key in sorted({e['capture_id'] for e in events} - captures))
    gaps.extend(f'Capture {key} has no closure' for key in sorted(captures - closed))
    gaps.extend(f'Response {key} has no retained request' for key in sorted(responses.keys() - requests.keys()))
    observed_operations = {r.get('operation_id') for r in requests.values()}
    gaps.extend(f'Operation {key} has no start' for key in sorted((observed_operations - {None}) - started.keys()))
    for key in started:
        if key not in observed_operations:
            gaps.append(f'Operation {key} has no observed HTTP request')
        if key not in finished:
            gaps.append(f'Operation {key} has no closure')
    entries = {}
    rows = []
    for key, request in requests.items():
        response = responses.get(key)
        provider = request.get('provider', 'unknown')
        body = (request.get('request', {}).get('body') or {}).get('value')
        model = body.get('model', request.get('model', '')) if isinstance(body, dict) else request.get('model', '')
        pair = provider, model
        entry = entries.setdefault(pair, ProviderUsage(provider, model))
        entry.calls += 1
        row = {'request_id': key, 'request_event': request['digest'], 'provider': provider,
               'model': model, 'role': request.get('role'), 'input_tokens': None, 'output_tokens': None,
               'cost_usd': None, 'pricing': None, 'issues': []}
        if response is None:
            row['issues'].append('No complete response; execution and billing unknown')
        else:
            native = response['response']
            row.update(response_event=response['digest'], status_code=native['status_code'], usage=native.get('usage'))
            if response.get('operation_id') != request.get('operation_id') or response['capture_id'] != request['capture_id']:
                row['issues'].append('Response binding differs from request')
            if native.get('capture_gap'):
                row['issues'].append(native['capture_gap'])
            raw_body = native.get('body') or {}
            raw_payload = raw_body.get('value')
            if raw_body.get('encoding') == 'json' and isinstance(raw_payload, dict):
                if native.get('usage') != raw_payload.get('usage', raw_payload.get('usageMetadata')):
                    row['issues'].append('Usage differs from retained response body')
            if not 200 <= native['status_code'] < 300:
                row['issues'].append('Unsuccessful HTTP attempt; billing is not established')
            try:
                incoming, outgoing = native_tokens(provider, native.get('usage'))
                row.update(input_tokens=incoming, output_tokens=outgoing)
                entry.input_tokens += incoming
                entry.output_tokens += outgoing
            except ValueError as exc:
                row['issues'].append(str(exc))
            if not row['issues'] and price_estimator is not None:
                try:
                    estimate = price_estimator(json.loads(canonical_json(request)), json.loads(canonical_json(response)))
                    amount = estimate['cost_usd']
                    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
                        raise ValueError('Estimator returned an invalid cost')
                    provenance = estimate['provenance']
                    if not isinstance(provenance, dict) or not provenance:
                        raise ValueError('Estimator returned no pricing provenance')
                    row.update(cost_usd=amount, pricing=json.loads(canonical_json(provenance)))
                except Exception as exc:
                    row['pricing_error'] = f'{type(exc).__name__}: {exc}'
        if request.get('request', {}).get('capture_gap'):
            row['issues'].append(request['request']['capture_gap'])
        gaps.extend(f'Request {key}: {issue}' for issue in row['issues'])
        if row['cost_usd'] is None:
            entry.cost_usd = None
        elif entry.cost_usd is not None:
            entry.cost_usd += row['cost_usd']
        rows.append(row)
    return Costs(by_model=sorted(entries.values(), key=lambda entry: (entry.provider, entry.model)),
                 scope='run_provider_usage', coverage_declaration=coverage_declaration,
                 evidence_gaps=gaps, evidence={'schema': 'multivon.provider-accounting/v1',
                    'event_digests': [e['digest'] for e in events], 'requests': rows,
                    'limits': 'Observed provider API usage; excludes infrastructure, tax, discounts and external services'})
