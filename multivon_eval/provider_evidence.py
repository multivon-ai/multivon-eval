"""Context-local native provider evidence; execution/recovery remains upstream."""
from __future__ import annotations

import functools
import inspect
import json
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from .case_manifest import canonical_json, digest

SCHEMA = 'multivon.provider-evidence/v1'
_ACTIVE = ContextVar('multivon_provider_capture', default=None)
_OPERATION = ContextVar('multivon_provider_operation', default=None)
_POSITION = ContextVar('multivon_provider_position', default={})


class ProviderCapture:
    def __init__(self, *, kind='manual', labels=None, journal=None):
        self.capture_id = uuid.uuid4().hex
        self.kind = kind
        self.labels = dict(labels or {})
        self.journal = journal
        self.events = []
        self.requests = {}
        self._lock = threading.RLock()

    def emit(self, kind, **data):
        event = {'schema': SCHEMA, 'event_id': uuid.uuid4().hex,
                 'capture_id': self.capture_id, 'capture_kind': self.kind,
                 'labels': self.labels, 'recorded_at': datetime.now(timezone.utc).isoformat(),
                 'kind': kind, **data}
        event['digest'] = digest(event)
        event = json.loads(canonical_json(event))
        with self._lock:
            if self.journal is not None:
                self.journal.append(event)
            self.events.append(event)
        return event

    def snapshot(self):
        with self._lock:
            observed = {r['operation_id'] for r in self.requests.values()}
            gaps = [{'operation_id': e['operation_id'], 'reason': 'No instrumented HTTP request observed'}
                    for e in self.events if e['kind'] == 'operation_started'
                    and e['operation_id'] not in observed]
            gaps.extend({'request_id': e['request_id'], 'reason': e['response']['capture_gap']}
                        for e in self.events if e['kind'] == 'http_response'
                        and e['response']['capture_gap'])
            return json.loads(canonical_json({
                'schema': SCHEMA, 'capture_id': self.capture_id, 'kind': self.kind,
                'events': self.events,
                'state': 'closed' if any(e['kind'] == 'capture_finished' for e in self.events) else 'open',
                'coverage_gaps': gaps,
                'requests_without_complete_response': [r['request_id'] for r in self.requests.values()
                                                       if not r['responded']],
                'coverage': 'Observed instrumented calls only; uninstrumented transports may exist',
            }))


@contextmanager
def capture_provider_events(*, kind='manual', labels=None, journal=None):
    """Capture native events, optionally durably in a ProviderJournal.

    Child trial captures inherit the journal, while retaining separate events.
    An empty capture cannot prove that an arbitrary callback made no API calls.
    """
    parent = _ACTIVE.get()
    if journal is None and parent is not None:
        journal = parent.journal
    capture = ProviderCapture(kind=kind, labels={**(parent.labels if parent else {}),
                                               **_POSITION.get(), **(labels or {})}, journal=journal)
    if kind == 'trial':
        capture.labels.setdefault('attempt', 1)
        capture.labels.setdefault('run_index', 1)
    if kind == 'run':
        capture.labels['run_id'] = capture.capture_id
    capture.emit('capture_started')
    token = _ACTIVE.set(capture)
    error = None
    try:
        yield capture
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        try:
            capture.emit('capture_finished', error=error)
        finally:
            _ACTIVE.reset(token)


def trial_provider_evidence():
    capture = _ACTIVE.get()
    return capture.snapshot() if capture is not None and capture.kind == 'trial' else None


def capture_trial(function):
    @functools.wraps(function)
    def wrapped(self, case, *args, **kwargs):
        try:
            case_id, case_digest = case.identity()
        except (TypeError, ValueError, AttributeError):
            case_id = case_digest = None
        with capture_provider_events(kind='trial', labels={'case_id': case_id, 'case_digest': case_digest}) as capture:
            result = function(self, case, *args, **kwargs)
        return finish_trial_capture(result, capture)
    return wrapped


def capture_run(function):
    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def async_wrapped(*args, **kwargs):
            with capture_provider_events(kind='run') as capture:
                result = await function(*args, **kwargs)
            if result.provider_evidence is None:
                result.provider_evidence = capture.snapshot()
            return result
        return async_wrapped
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with capture_provider_events(kind='run') as capture:
            result = function(*args, **kwargs)
        if result.provider_evidence is None:
            result.provider_evidence = capture.snapshot()
        return result
    return wrapped


@contextmanager
def provider_operation(provider, role, model):
    capture = _ACTIVE.get()
    if capture is None:
        yield
        return
    operation = {'operation_id': uuid.uuid4().hex, 'provider': provider, 'role': role, 'model': model}
    capture.emit('operation_started', **operation)
    token = _OPERATION.set(operation)
    error = None
    try:
        yield
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        # A missing response, including one before an SDK retry, is unknown billing.
        with capture._lock:
            pending = [r['request_id'] for r in capture.requests.values()
                       if r['operation_id'] == operation['operation_id'] and not r['responded']]
        try:
            capture.emit('operation_finished', **operation, error=error,
                         requests_without_complete_response=pending)
        finally:
            _OPERATION.reset(token)


def observe_provider(provider, role):
    """Label a built-in adapter/judge call without changing its retry behavior."""
    def decorate(function):
        def model(args, kwargs):
            bound = inspect.signature(function).bind(*args, **kwargs).arguments
            obj = bound.get('self') if role == 'target' else bound.get('config', bound.get('judge'))
            return getattr(obj, 'model', '')
        if inspect.iscoroutinefunction(function):
            @functools.wraps(function)
            async def awrapped(*args, **kwargs):
                with provider_operation(provider, role, model(args, kwargs)):
                    return await function(*args, **kwargs)
            return awrapped
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with provider_operation(provider, role, model(args, kwargs)):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def cache_observation(prompt, config, answer):
    capture = _ACTIVE.get()
    if capture is not None:
        capture.emit('cache_hit', provider=config.provider, model=config.model,
                     prompt=prompt, answer=answer, usage=None,
                     limits='Cached verdict; no new provider usage established by this event')


def run_provider_evidence():
    capture = _ACTIVE.get()
    return capture.snapshot() if capture is not None and capture.kind == 'run' else None


@contextmanager
def provider_position(**values):
    token = _POSITION.set({**_POSITION.get(), **values})
    try:
        yield
    finally:
        _POSITION.reset(token)


def finish_trial_capture(result, capture):
    from .trials import TrialRecord
    if result.trials:
        data = result.trials[0].data
        data.pop('digest')
        data['provider_evidence'] = capture.snapshot()
        result.trials = (TrialRecord.from_dict({**data, 'digest': digest(data)}),)
    return result
