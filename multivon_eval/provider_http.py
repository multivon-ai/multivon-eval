"""Native SDK HTTPX hooks: serialized attempts, responses and native usage."""
from __future__ import annotations

import base64
import hashlib
import json
import uuid
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .provider_evidence import _ACTIVE, _OPERATION

_HEADERS = {'content-type', 'anthropic-version', 'anthropic-beta', 'openai-beta',
            'request-id', 'x-request-id', 'retry-after', 'retry-after-ms', 'x-should-retry'}
_CREDENTIALS = {'key', 'api_key', 'apikey', 'token', 'access_token', 'auth', 'authorization'}


def _body(raw):
    try:
        value = {'encoding': 'json', 'value': json.loads(raw)}
    except (ValueError, UnicodeError):
        value = {'encoding': 'base64', 'value': base64.b64encode(raw).decode('ascii')}
    return {**value, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def _url(url):
    parts = urlsplit(str(url))
    redacted = []
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in _CREDENTIALS:
            redacted.append('query.' + key)
            value = '[redacted]'
        query.append((key, value))
    if '@' in parts.netloc:
        redacted.append('url.userinfo')
    return (urlunsplit((parts.scheme, parts.netloc.rsplit('@', 1)[-1], parts.path,
                       urlencode(query), '')), redacted)


def request_hook(request):
    capture = _ACTIVE.get()
    if capture is None:
        return
    operation = dict(_OPERATION.get() or {'operation_id': None, 'provider': 'unknown', 'role': 'unknown', 'model': ''})
    request_id = uuid.uuid4().hex
    url, redacted = _url(request.url)
    try:
        body, gap = _body(request.content), None
    except Exception as exc:
        body, gap = None, f'Request body unavailable: {type(exc).__name__}'
    payload = body.get('value') if body and body['encoding'] == 'json' else None
    streaming = isinstance(payload, dict) and payload.get('stream') is True
    event = capture.emit('http_request', **operation, request_id=request_id,
                         request={'method': request.method, 'url': url,
                                  'headers': {k: v for k, v in request.headers.items() if k.lower() in _HEADERS},
                                  'body': body, 'timeout': request.extensions.get('timeout'),
                                  'redacted': redacted, 'capture_gap': gap})
    with capture._lock:
        capture.requests[id(request)] = {'request_id': request_id, 'operation_id': operation['operation_id'],
                                         'responded': False, 'streaming': streaming,
                                         'request': request, 'event': event}


def _response_context(response):
    capture = _ACTIVE.get()
    if capture is None:
        return None, None
    with capture._lock:
        request = capture.requests.get(id(response.request))
    return capture, request


def _record_response(response, capture, request, raw, gap):
    body = _body(raw) if raw is not None else None
    payload = body.get('value') if body and body['encoding'] == 'json' else None
    usage = None
    if isinstance(payload, dict):
        usage = payload.get('usage', payload.get('usageMetadata'))
    capture.emit('http_response', request_id=request['request_id'], operation_id=request['operation_id'],
                 response={'status_code': response.status_code,
                           'headers': {k: v for k, v in response.headers.items() if k.lower() in _HEADERS},
                           'body': body, 'usage': usage, 'capture_gap': gap})
    with capture._lock:
        request['responded'] = True


def _streaming(response, request):
    return ('text/event-stream' in response.headers.get('content-type', '')
            or (request['streaming'] and 200 <= response.status_code < 300))


def _response_headers(response, capture, request):
    capture.emit('http_response_headers', request_id=request['request_id'],
                 operation_id=request['operation_id'], status_code=response.status_code,
                 headers={k: v for k, v in response.headers.items() if k.lower() in _HEADERS})


def response_hook(response):
    capture, request = _response_context(response)
    if request is None:
        return
    _response_headers(response, capture, request)
    if _streaming(response, request):
        _record_response(response, capture, request, None, 'Streaming body/usage not captured')
        return
    try:
        raw, gap = response.read(), None
    except Exception as exc:
        _record_response(response, capture, request, None, f'Response body unavailable: {type(exc).__name__}')
        raise
    _record_response(response, capture, request, raw, gap)


async def async_request_hook(request):
    request_hook(request)


async def async_response_hook(response):
    capture, request = _response_context(response)
    if request is None:
        return
    _response_headers(response, capture, request)
    if _streaming(response, request):
        _record_response(response, capture, request, None, 'Streaming body/usage not captured')
        return
    try:
        raw, gap = await response.aread(), None
    except Exception as exc:
        _record_response(response, capture, request, None, f'Response body unavailable: {type(exc).__name__}')
        raise
    _record_response(response, capture, request, raw, gap)


def provider_http_hooks(*, asynchronous=False):
    """Public hooks for an explicitly configured custom native HTTPX client.

    SDK defaults (TLS, pooling, retries) stay with the SDK/client owner. Streaming
    response bodies are not consumed. HTTP request headers use a small allowlist.
    """
    return {'request': [async_request_hook if asynchronous else request_hook],
            'response': [async_response_hook if asynchronous else response_hook]}


def sdk_http_client(sdk, *, asynchronous=False):
    factory = sdk.DefaultAsyncHttpxClient if asynchronous else sdk.DefaultHttpxClient
    return factory(event_hooks=provider_http_hooks(asynchronous=asynchronous))


def google_http_options(timeout=None):
    # Google may select aiohttp for async requests. HTTPX-only kwargs there can
    # break that transport. Leave selection with the SDK; disclose absent wire
    # evidence instead of silently replacing its TLS/auth/transport defaults.
    options = {'client_args': {'event_hooks': provider_http_hooks()}}
    if timeout is not None:
        options['timeout'] = int(timeout * 1000)
    return options
