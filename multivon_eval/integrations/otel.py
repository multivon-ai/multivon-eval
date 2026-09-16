"""Native OTLP evidence bridge; transport, storage and instrumentation stay upstream."""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, replace

from ..case import AgentStep, EvalCase, ToolCall
from ..case_manifest import canonical_json, digest
from ..result import CaseResult, EvalReport
from ..trials import TrialRecord, attach_trial, capture_case

GENAI_CONVENTIONS_REVISION = 'c88d504ab3d9879f8e50d3cc87e69775e11db234'
GENAI_PROFILE = 'otel-genai-development-2026-09-17'


def _id(value: str, size: int) -> str:
    if not isinstance(value, str) or len(value) != size * 2:
        raise ValueError('Trace/span IDs must be fixed-width nonzero hexadecimal strings')
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError('Invalid hexadecimal trace/span ID') from exc
    if len(raw) != size or not any(raw):
        raise ValueError('Invalid trace/span ID')
    return raw.hex()


def _value(value):
    kind = value.WhichOneof('value')
    if kind == 'array_value':
        return [_value(v) for v in value.array_value.values]
    if kind == 'kvlist_value':
        return _attributes(value.kvlist_value.values)
    return getattr(value, kind) if kind else None


def _attributes(values):
    attributes = {}
    for item in values:
        if item.key in attributes:
            raise ValueError(f'Duplicate OTLP attribute: {item.key}')
        attributes[item.key] = _value(item.value)
    return attributes


def _structured(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _message_text(value, role: str) -> str:
    messages = _structured(value)
    if not isinstance(messages, list) or not messages:
        raise ValueError('Native GenAI messages are absent or not an array')
    selected = [m for m in messages if isinstance(m, dict) and m.get('role') == role]
    if role == 'assistant' and len(selected) != 1:
        raise ValueError('Select one output message; multiple candidates cannot be silently combined')
    if not selected:
        raise ValueError(f'No native {role} message')
    parts = selected[-1].get('parts')
    if not isinstance(parts, list) or not parts or any(
            not isinstance(p, dict) or p.get('type') != 'text' or
            not isinstance(p.get('content'), str) for p in parts):
        raise ValueError('The text projection requires text-only message parts; native media remains upstream')
    return ''.join(p['content'] for p in parts)


@dataclass(frozen=True)
class OtelTrace:
    """An immutable selection over retained native OTLP ExportTraceServiceRequests.

    Payloads are uncompressed protobuf or standard OTLP JSON bodies, with resource/scope
    metadata and unknown fields. A batch may contain other traces; those bytes
    remain in the retained evidence. No collector or global tracer is installed.
    """
    payloads: tuple[bytes, ...]
    trace_id: str
    root_span_id: str
    profile: str
    tool_coverage_complete: bool = False
    encoding: str = "protobuf"

    def __post_init__(self):
        if type(self.payloads) is not tuple or not self.payloads or any(type(p) is not bytes or not p for p in self.payloads):
            raise ValueError('Supply a nonempty tuple of native OTLP request bodies')
        object.__setattr__(self, 'trace_id', _id(self.trace_id, 16))
        object.__setattr__(self, 'root_span_id', _id(self.root_span_id, 8))
        if self.profile != GENAI_PROFILE:
            raise ValueError('Unsupported GenAI convention profile; no automatic schema migration is performed')
        if type(self.tool_coverage_complete) is not bool:
            raise ValueError('Tool coverage must be an explicit boolean assertion')
        if self.encoding not in {'protobuf', 'json'}:
            raise ValueError('OTLP encoding must be protobuf or json')
        self._records()

    def _records(self) -> dict:
        try:
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceRequest,
            )
        except ImportError as exc:
            raise ImportError('Install multivon-eval[otel] for OTLP ingestion') from exc
        records = {}
        for payload in self.payloads:
            if self.encoding == 'json':
                from .otel_json import parse_otlp_json
                request = parse_otlp_json(payload)
            else:
                request = ExportTraceServiceRequest.FromString(payload)
            for resource in request.resource_spans:
                for scope in resource.scope_spans:
                    for span in scope.spans:
                        if span.trace_id.hex() != self.trace_id:
                            continue
                        span_id = _id(span.span_id.hex(), 8)
                        if span.parent_span_id:
                            _id(span.parent_span_id.hex(), 8)
                        signature = (span.SerializeToString(deterministic=True),
                                     resource.resource.SerializeToString(deterministic=True), resource.schema_url,
                                     scope.scope.SerializeToString(deterministic=True), scope.schema_url)
                        if span_id in records and records[span_id]['signature'] != signature:
                            raise ValueError('Conflicting duplicate span ID in OTLP payloads')
                        records[span_id] = {'span': span, 'resource': resource.resource,
                            'scope': scope.scope, 'schema_urls': [resource.schema_url, scope.schema_url],
                            'signature': signature}
        for origin in records:
            seen = set()
            current = origin
            while current in records:
                if current in seen:
                    raise ValueError('Cyclic parent relationships in the native trace')
                seen.add(current)
                current = records[current]['span'].parent_span_id.hex()
        if self.root_span_id not in records:
            raise ValueError('The selected root span is absent from the native payloads')
        return records

    def selected(self) -> tuple[list[dict], list[str]]:
        records = self._records()
        selected = {self.root_span_id}
        while True:
            children = {sid for sid, row in records.items() if row['span'].parent_span_id.hex() in selected}
            if children <= selected:
                break
            selected |= children
        issues = []
        if not self.tool_coverage_complete:
            issues.append('Complete tool instrumentation/capture was not asserted; sampling cannot prove absence')
        if any(row['span'].parent_span_id and row['span'].parent_span_id.hex() not in records
               for sid, row in records.items() if sid != self.root_span_id):
            issues.append('Disconnected spans prevent verifying the selected execution boundary')
        rows = [records[sid] for sid in selected]
        root = records[self.root_span_id]['span']
        for row in rows:
            span = row['span']
            if span.status.code not in {0, 1, 2}:
                issues.append('Unsupported native span status')
            if (span.span_id.hex() != self.root_span_id and
                    (span.start_time_unix_nano < root.start_time_unix_nano or
                     span.end_time_unix_nano > root.end_time_unix_nano)):
                issues.append('Descendant span timing falls outside the selected execution interval')
            if span.start_time_unix_nano == 0 or span.end_time_unix_nano < span.start_time_unix_nano:
                issues.append('Missing or invalid native span timestamps')
            if (span.dropped_attributes_count or span.dropped_events_count or span.dropped_links_count or
                    row['resource'].dropped_attributes_count or row['scope'].dropped_attributes_count or
                    any(e.dropped_attributes_count for e in span.events) or
                    any(link.dropped_attributes_count for link in span.links)):
                issues.append('Native OTLP reports dropped attributes, events or links')
        rows.sort(key=lambda r: (r['span'].start_time_unix_nano, r['span'].span_id))
        return rows, sorted(set(issues))

    def to_dict(self) -> dict:
        body = {'schema': 'multivon.otel-evidence/v1', 'trace_id': self.trace_id,
                'root_span_id': self.root_span_id, 'profile': self.profile,
                'conventions_revision': GENAI_CONVENTIONS_REVISION,
                'tool_coverage_complete': self.tool_coverage_complete, 'encoding': self.encoding,
                'payloads_base64': [base64.b64encode(p).decode('ascii') for p in self.payloads],
                'payloads_sha256': [hashlib.sha256(p).hexdigest() for p in self.payloads]}
        return {**body, 'digest': digest(body)}

    @classmethod
    def from_dict(cls, data: dict) -> OtelTrace:
        candidate = cls(tuple(base64.b64decode(p, validate=True) for p in data['payloads_base64']),
                        data['trace_id'], data['root_span_id'], data['profile'], data['tool_coverage_complete'], data['encoding'])
        if canonical_json(candidate.to_dict()) != canonical_json(data):
            raise ValueError('OTLP evidence digest, schema or content changed')
        return candidate


def _project_tools(rows: list[dict], complete: bool) -> tuple[list[AgentStep] | None, list[str]]:
    steps, issues, tools = [], [], []
    for row in rows:
        span = row['span']
        attrs = _attributes(span.attributes)
        if attrs.get('gen_ai.operation.name') != 'execute_tool' and attrs.get('mcp.method.name') != 'tools/call':
            continue
        tools.append(span)
        name = attrs.get('gen_ai.tool.name')
        arguments = _structured(attrs.get('gen_ai.tool.call.arguments'))
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            issues.append('Tool names/arguments are missing or unsupported; empty arguments cannot be invented')
            continue
        result = _structured(attrs.get('gen_ai.tool.call.result'))
        if 'gen_ai.tool.call.result' not in attrs:
            issues.append('Tool result was not captured; successful completion cannot be verified')
        if span.status.code == 2 or attrs.get('error.type'):
            issues.append('Tool execution error remains native; AgentStep cannot represent its error status faithfully')
            result = None
        canonical_json(arguments)
        canonical_json(result)
        steps.append(AgentStep(tool_calls=[ToolCall(name, arguments, result)]))
    for index, span in enumerate(tools):
        if any(other.start_time_unix_nano < span.end_time_unix_nano and
               span.start_time_unix_nano < other.end_time_unix_nano for other in tools[index + 1:]):
            issues.append('Overlapping/nested tool spans have no verified total execution order or deduplication')
            break
    return (steps if complete and not issues else None), sorted(set(issues))


def score_otel_trace(suite, trace: OtelTrace, case: EvalCase, *,
                     output_span_id: str | None = None) -> EvalReport:
    """Grade a captured text/tool execution without invoking its target again.

    Input must match the root's last native user message. Output is selected
    explicitly from the root (default) or a descendant. Missing/redacted outputs
    remain unmeasured. The original case identity stays separate from its observed
    trace. Native bytes, schema URLs and tool-projection gaps remain in the trial.
    """
    rows, issues = trace.selected()
    by_id = {r['span'].span_id.hex(): r for r in rows}
    root = by_id[trace.root_span_id]['span']
    output_id = _id(output_span_id, 8) if output_span_id is not None else trace.root_span_id
    if output_id not in by_id:
        raise ValueError('Output span is outside the selected execution boundary')
    if _message_text(_attributes(root.attributes).get('gen_ai.input.messages'), 'user') != case.input:
        raise ValueError('Native input does not match the evaluation case')
    output_span = by_id[output_id]['span']
    output_attrs = _attributes(output_span.attributes)
    try:
        output = _message_text(output_attrs.get('gen_ai.output.messages'), 'assistant')
    except ValueError as exc:
        output = None
        issues.append(str(exc))
    steps, tool_issues = _project_tools(rows, trace.tool_coverage_complete and not issues)
    issues.extend(tool_issues)
    projected = replace(case, agent_trace=steps)
    latency = ((root.end_time_unix_nano - root.start_time_unix_nano) / 1_000_000
               if root.start_time_unix_nano and root.end_time_unix_nano >= root.start_time_unix_nano else None)
    target_error = (root.status.code == 2 or output_span.status.code == 2 or
                    bool(_attributes(root.attributes).get('error.type')) or bool(output_attrs.get('error.type')))
    if target_error or output is None:
        cr = CaseResult(case.input, output or '', [], tags=case.tags, agent_trace=steps,
                        model_error='Selected native operation recorded ERROR' if target_error else None,
                        skipped=not target_error, latency_ms=latency or 0.0)
        report = EvalReport(suite_name=suite.name, case_results=[cr], model_id=suite.model_id)
    else:
        report = suite.run_on_cases([(projected, output)], latencies_ms=[latency], verbose=False)
        cr = report.case_results[0]
    attach_trial(cr, capture_case(case), origin='otel', evaluation_snapshot=capture_case(projected),
                 latency_known=latency is not None)
    if not cr.trials:
        raise ValueError('Evaluation case or projected trace is not portable evidence')
    trial = cr.trials[0].data
    trial.pop('digest')
    trial['upstream'] = {'format': 'otlp/' + trace.encoding, 'evidence': trace.to_dict(),
                         'output_span_id': output_id,
                         'schema_urls': sorted({url for r in rows for url in r['schema_urls'] if url})}
    trial['evidence_gaps'] = sorted(set(issues + [
        'Source instrumentation and completeness assertions are not independently authenticated',
        'AgentStep is a text/tool projection; native spans preserve timing, usage, media and errors',
        'Grader API requests are not automatically captured in this imported target trace']))
    cr.trials = (TrialRecord.from_dict({**trial, 'digest': digest(trial)}),)
    report.evidence_issues = sorted(set(report.evidence_issues + issues))
    return report
