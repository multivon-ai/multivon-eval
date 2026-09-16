"""Native OTel SDK/OTLP paths preserve scope, errors and incomplete evidence."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

pytest.importorskip('opentelemetry.sdk')
pytest.importorskip('opentelemetry.exporter.otlp.proto.http')
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from multivon_eval import (
    AcceptancePolicy,
    CheckRequirement,
    EvalCase,
    EvalSuite,
    ExactMatch,
    ToolCallAccuracy,
)
from multivon_eval.integrations.otel import GENAI_PROFILE, OtelTrace, score_otel_trace
from multivon_eval.integrations.otel_export import emit_evaluation_events
from multivon_eval.trials import trial_integrity_issues


def messages(role, content):
    return json.dumps([{'role': role, 'parts': [{'type': 'text', 'content': content}]}])


@pytest.fixture
def native_trace():
    """The official HTTP exporter produces real protobuf, not a mirrored encoder."""
    payloads = []
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            payloads.append(self.rfile.read(int(self.headers['Content-Length'])))
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-protobuf')
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(
        endpoint=f'http://127.0.0.1:{server.server_port}/v1/traces')))
    tracer = provider.get_tracer('fixture-app', '1.0', schema_url='https://example.invalid/otel/schema/fixture-v1')
    with tracer.start_as_current_span('invoke_agent fixture', attributes={
            'gen_ai.operation.name': 'invoke_agent', 'gen_ai.input.messages': messages('user', 'q'),
            'gen_ai.output.messages': messages('assistant', 'yes'), 'vendor.opaque': 'keep me'}) as root:
        root_id = root.get_span_context()
        with tracer.start_as_current_span('execute_tool lookup', attributes={
                'gen_ai.operation.name': 'execute_tool', 'gen_ai.tool.name': 'lookup',
                'gen_ai.tool.call.arguments': json.dumps({'id': 7}),
                'gen_ai.tool.call.result': json.dumps({'found': True}),
                'gen_ai.usage.input_tokens': 9}):
            pass
    assert provider.force_flush()
    provider.shutdown()
    server.shutdown(); server.server_close(); worker.join()
    return OtelTrace(tuple(payloads), f'{root_id.trace_id:032x}', f'{root_id.span_id:016x}',
                     GENAI_PROFILE, tool_coverage_complete=True)


def suite_case():
    case = EvalCase('q', 'yes', case_id='case-1', source_id='source-1', expected_tool_calls=['lookup'])
    suite = EvalSuite('native telemetry').add_case(case).add_evaluators(ExactMatch(), ToolCallAccuracy())
    return suite, case


def edited(trace, edit):
    payloads = []
    for payload in trace.payloads:
        request = ExportTraceServiceRequest.FromString(payload)
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                for span in list(scope.spans):
                    edit(span, resource, scope)
        payloads.append(request.SerializeToString())
    return OtelTrace(tuple(payloads), trace.trace_id, trace.root_span_id, GENAI_PROFILE, True)


def test_real_http_otlp_import_identity_and_native_roundtrip(native_trace):
    suite, case = suite_case()
    report = score_otel_trace(suite, native_trace, case)
    assert report.passed == 1 and not report.evidence_issues
    row = report.case_results[0]
    assert (row.case_id, row.case_digest) == case.identity()
    assert case.agent_trace is None
    assert row.agent_trace[0].tool_calls[0].arguments == {'id': 7}
    assert row.agent_trace[0].tool_calls[0].result == {'found': True}
    assert not trial_integrity_issues(row)
    native = row.trials[0].data['upstream']['evidence']
    restored = OtelTrace.from_dict(json.loads(json.dumps(native)))
    assert restored.payloads == native_trace.payloads
    assert b'vendor.opaque' in b''.join(restored.payloads)
    assert row.trials[0].data['upstream']['schema_urls'] == ['https://example.invalid/otel/schema/fixture-v1']
    assert AcceptancePolicy((CheckRequirement('exact_match'), CheckRequirement('tool_call_accuracy'))).evaluate(report).decision == 'accept'


def test_events_use_native_logger_parent_context_and_hide_content_by_default(native_trace):
    suite, case = suite_case()
    report = score_otel_trace(suite, native_trace, case)
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    count = emit_evaluation_events(report, provider.get_logger('multivon.fixture'))
    assert provider.force_flush()
    logs = [r.log_record for r in exporter.get_finished_logs()]
    assert count == 3 and len(logs) == 3
    grades = [r for r in logs if r.event_name == 'gen_ai.evaluation.result']
    assert len(grades) == 2
    for log in grades:
        assert log.trace_id == int(native_trace.trace_id, 16)
        assert log.span_id == int(native_trace.root_span_id, 16)
        assert log.attributes['gen_ai.evaluation.score.value'] == 1.0
        assert 'gen_ai.evaluation.explanation' not in log.attributes
        assert log.attributes['multivon.trial.digest'] == report.case_results[0].trials[0].digest
    provider.shutdown()


def test_unknown_capture_cannot_prove_no_forbidden_tools(native_trace):
    incomplete = OtelTrace(native_trace.payloads, native_trace.trace_id, native_trace.root_span_id, GENAI_PROFILE)
    suite, case = suite_case()
    case.expected_tool_calls = []
    report = score_otel_trace(suite, incomplete, case)
    assert report.case_results[0].agent_trace is None
    assert report.case_results[0].results[1].metadata['skipped']
    assert AcceptancePolicy((CheckRequirement('exact_match'),)).evaluate(report).decision == 'indeterminate'


@pytest.mark.parametrize('fault', ['drop', 'arguments', 'result', 'tool_error', 'overlap'])
def test_lossy_tool_projection_stays_unmeasured(native_trace, fault):
    def edit(span, resource, scope):
        if span.span_id.hex() == native_trace.root_span_id:
            if fault == 'drop':
                span.dropped_events_count = 1
            return
        if fault in {'arguments', 'result'}:
            key = 'gen_ai.tool.call.' + fault
            keep = [a for a in span.attributes if a.key != key]
            del span.attributes[:]
            span.attributes.extend(keep)
        elif fault == 'tool_error':
            span.status.code = 2
        elif fault == 'overlap':
            duplicate = scope.spans.add()
            duplicate.CopyFrom(span)
            duplicate.span_id = bytes.fromhex('1111111111111111')
    changed = edited(native_trace, edit)
    suite, case = suite_case()
    report = score_otel_trace(suite, changed, case)
    assert report.evidence_issues
    assert AcceptancePolicy((CheckRequirement('exact_match'),)).evaluate(report).decision == 'indeterminate'
    if fault != 'drop':
        assert report.case_results[0].agent_trace is None


def test_target_error_and_redacted_output_never_grade_empty_string_as_quality(native_trace):
    suite, case = suite_case()
    def fail(span, *_):
        if span.span_id.hex() == native_trace.root_span_id:
            span.status.code = 2
    report = score_otel_trace(suite, edited(native_trace, fail), case)
    assert report.errors == 1 and not report.case_results[0].results
    def redact(span, *_):
        for attr in span.attributes:
            if attr.key == 'gen_ai.output.messages':
                attr.value.string_value = '[]'
    missing = score_otel_trace(suite, edited(native_trace, redact), case)
    assert missing.skipped == 1 and missing.evidence_issues
    assert not missing.case_results[0].results


def test_transport_retries_deduplicate_exact_spans_but_conflicts_fail(native_trace):
    retry = OtelTrace(native_trace.payloads * 2, native_trace.trace_id, native_trace.root_span_id, GENAI_PROFILE, True)
    assert len(retry.selected()[0]) == 2
    def rename(span, *_):
        span.name = 'changed'
    altered = edited(native_trace, rename)
    with pytest.raises(ValueError, match='Conflicting'):
        OtelTrace(native_trace.payloads + altered.payloads, native_trace.trace_id, native_trace.root_span_id, GENAI_PROFILE)


def test_wrong_case_outside_output_profile_and_cycles_are_rejected(native_trace):
    suite, case = suite_case()
    with pytest.raises(ValueError, match='input'):
        score_otel_trace(suite, native_trace, EvalCase('different', 'yes'))
    with pytest.raises(ValueError, match='outside'):
        score_otel_trace(suite, native_trace, case, output_span_id='1111111111111111')
    with pytest.raises(ValueError, match='profile'):
        OtelTrace(native_trace.payloads, native_trace.trace_id, native_trace.root_span_id, 'future-profile')
    def cycle(span, *_):
        span.parent_span_id = span.span_id
    with pytest.raises(ValueError, match='Cyclic'):
        edited(native_trace, cycle)


def test_native_evidence_mutation_is_detected(native_trace):
    value = native_trace.to_dict()
    value['tool_coverage_complete'] = False
    with pytest.raises(ValueError, match='changed'):
        OtelTrace.from_dict(value)


def test_invalid_native_latency_cannot_pass_a_latency_requirement(native_trace):
    from multivon_eval import MaxLatency
    def corrupt(span, *_):
        if span.span_id.hex() == native_trace.root_span_id:
            span.start_time_unix_nano = 0
    suite, case = suite_case()
    suite.add_evaluator(MaxLatency(10000))
    report = score_otel_trace(suite, edited(native_trace, corrupt), case)
    row = report.case_results[0]
    assert row.trials[0].data['latency_ms'] is None
    assert row.results[-1].metadata['skipped']
    assert AcceptancePolicy((CheckRequirement('max_latency'),)).evaluate(report).decision == 'indeterminate'


def test_unknown_protobuf_fields_survive_without_being_reinterpreted(native_trace):
    # Protobuf field 1000, wire type 2, payload xyz. It is outside the current schema.
    bodies = tuple(body + b'\xc2\x3e\x03xyz' for body in native_trace.payloads)
    native = OtelTrace(bodies, native_trace.trace_id, native_trace.root_span_id, GENAI_PROFILE, True)
    assert OtelTrace.from_dict(native.to_dict()).payloads == bodies
    assert len(native.selected()[0]) == 2


def test_missing_parent_and_dropped_scope_are_visible(native_trace):
    def disconnect(span, resource, scope):
        if span.span_id.hex() != native_trace.root_span_id:
            span.parent_span_id = bytes.fromhex('1111111111111111')
        scope.scope.dropped_attributes_count = 1
    changed = edited(native_trace, disconnect)
    suite, case = suite_case()
    report = score_otel_trace(suite, changed, case)
    assert any('Disconnected' in issue for issue in report.evidence_issues)
    assert any('dropped' in issue for issue in report.evidence_issues)
    assert report.case_results[0].agent_trace is None


def test_observed_no_tools_requires_capture_assertion(native_trace):
    root_payload = []
    for body in native_trace.payloads:
        request = ExportTraceServiceRequest.FromString(body)
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                keep = [s for s in scope.spans if s.span_id.hex() == native_trace.root_span_id]
                del scope.spans[:]
                scope.spans.extend(keep)
        root_payload.append(request.SerializeToString())
    trace = OtelTrace(tuple(root_payload), native_trace.trace_id, native_trace.root_span_id, GENAI_PROFILE, True)
    suite, case = suite_case()
    case.expected_tool_calls = []
    report = score_otel_trace(suite, trace, case)
    assert report.case_results[0].agent_trace == []
    assert report.case_results[0].results[1].passed


def test_error_trial_and_skipped_grader_export_without_fabricated_score(native_trace):
    suite, case = suite_case()
    incomplete = OtelTrace(native_trace.payloads, native_trace.trace_id, native_trace.root_span_id, GENAI_PROFILE)
    report = score_otel_trace(suite, incomplete, case)
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    emit_evaluation_events(report, provider.get_logger('fixture'))
    logs = [r.log_record for r in exporter.get_finished_logs()]
    skipped = next(r for r in logs if r.attributes.get('gen_ai.evaluation.name') == 'tool_call_accuracy')
    assert 'gen_ai.evaluation.score.value' not in skipped.attributes
    assert skipped.attributes['gen_ai.evaluation.score.label'] == 'unmeasured'
    assert skipped.attributes['multivon.report.has_evidence_issues'] is True
    exporter.clear()
    def fail(span, *_):
        if span.span_id.hex() == native_trace.root_span_id:
            span.status.code = 2
    failed = score_otel_trace(suite, edited(native_trace, fail), case)
    assert emit_evaluation_events(failed, provider.get_logger('fixture')) == 1
    status = exporter.get_finished_logs()[0].log_record
    assert status.event_name == 'multivon.evaluation.trial'
    assert status.attributes['multivon.trial.status'] == 'model_error'
    assert 'gen_ai.evaluation.score.value' not in status.attributes
    provider.shutdown()


def collector_json():
    from pathlib import Path
    directory = Path(__file__).parent / 'fixtures' / 'otel'
    metadata = json.loads((directory / 'provenance.json').read_text())
    payloads = tuple((directory / 'collector-traces.jsonl').read_bytes().splitlines())
    return OtelTrace(payloads, metadata['trace_id'], metadata['root_span_id'], GENAI_PROFILE, True, encoding='json')


def test_actual_collector_json_fixture_import_and_unknown_field_preservation():
    trace = collector_json()
    case = EvalCase('Run the fixture check', 'pass', expected_tool_calls=['eval_tool_call_accuracy'])
    suite = EvalSuite('collector json').add_evaluators(ExactMatch(), ToolCallAccuracy())
    report = score_otel_trace(suite, trace, case)
    assert report.passed == 1 and not report.evidence_issues
    assert report.case_results[0].trials[0].data['upstream']['format'] == 'otlp/json'
    bodies = []
    for body in trace.payloads:
        parsed = json.loads(body)
        parsed['futureTransportField'] = {'keep': 'opaque data'}
        bodies.append(json.dumps(parsed).encode())
    extended = OtelTrace(tuple(bodies), trace.trace_id, trace.root_span_id, GENAI_PROFILE, True, encoding='json')
    assert OtelTrace.from_dict(extended.to_dict()).payloads == tuple(bodies)
    assert len(extended.selected()[0]) == 2


@pytest.mark.parametrize('bad', [b'{"resourceSpans":null}', b'{"resourceSpans":[null]}',
                               b'{"resourceSpans":[],"resourceSpans":[]}', b'{"x":NaN}'])
def test_malformed_json_fails_without_guessing(bad):
    trace = collector_json()
    with pytest.raises(ValueError):
        OtelTrace((bad,), trace.trace_id, trace.root_span_id, GENAI_PROFILE, encoding='json')


def test_protobuf_json_base64_ids_and_symbolic_enums_are_not_otlp_json():
    import base64
    trace = collector_json()
    data = json.loads(trace.payloads[0])
    span = data['resourceSpans'][0]['scopeSpans'][0]['spans'][0]
    span['traceId'] = base64.b64encode(bytes.fromhex(span['traceId'])).decode()
    with pytest.raises(ValueError, match='hexadecimal'):
        OtelTrace((json.dumps(data).encode(),), trace.trace_id, trace.root_span_id, GENAI_PROFILE, encoding='json')
    data = json.loads(trace.payloads[0])
    data['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['kind'] = 'SPAN_KIND_CLIENT'
    with pytest.raises(ValueError, match='integer'):
        OtelTrace((json.dumps(data).encode(),), trace.trace_id, trace.root_span_id, GENAI_PROFILE, encoding='json')
