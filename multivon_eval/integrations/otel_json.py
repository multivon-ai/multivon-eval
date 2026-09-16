"""OTLP JSON's hex-ID adaptation around the upstream protobuf JSON parser."""
from __future__ import annotations

import base64
import json


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate OTLP JSON field: {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'Invalid JSON numeric constant: {value}')


def _hex_field(record, name, size):
    if name not in record or record[name] == '':
        return
    value = record[name]
    if not isinstance(value, str) or len(value) != size * 2:
        raise ValueError(f'OTLP JSON {name} requires {size * 2} hexadecimal characters')
    raw = bytes.fromhex(value)
    if len(raw) != size:
        raise ValueError(f'Invalid OTLP JSON {name}')
    record[name] = base64.b64encode(raw).decode('ascii')


def _items(record, field):
    values = record.get(field, [])
    if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
        raise ValueError(f'OTLP JSON {field} must be an array of objects')
    return values


def parse_otlp_json(payload: bytes):
    """Parse standard OTLP JSON while callers retain the unchanged original bytes.

    OTLP differs from protobuf JSON for trace/span IDs and integer enum values.
    Unknown fields are ignored for interpretation per OTLP, not deleted from the
    caller's evidence. Generic protobuf JSON with base64 IDs is not OTLP JSON.
    """
    from google.protobuf.json_format import ParseDict
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    data = json.loads(payload, object_pairs_hook=_unique, parse_constant=_invalid_constant)
    if not isinstance(data, dict):
        raise ValueError('OTLP JSON request must be an object')  # noqa: TRY004 - malformed wire content
    for resource in _items(data, 'resourceSpans'):
        for scope in _items(resource, 'scopeSpans'):
            for span in _items(scope, 'spans'):
                _hex_field(span, 'traceId', 16)
                _hex_field(span, 'spanId', 8)
                _hex_field(span, 'parentSpanId', 8)
                for link in _items(span, 'links'):
                    _hex_field(link, 'traceId', 16)
                    _hex_field(link, 'spanId', 8)
                status = span.get('status', {})
                if not isinstance(status, dict):
                    raise ValueError('OTLP JSON status must be an object')  # noqa: TRY004 - malformed wire content
                for value in [span.get('kind', 0), status.get('code', 0)]:
                    if type(value) is not int:
                        raise ValueError('OTLP JSON enum fields must use integer values')
    return ParseDict(data, ExportTraceServiceRequest(), ignore_unknown_fields=True)
