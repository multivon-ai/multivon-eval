"""Local OTel SDK -> native OTLP -> evaluation -> OTel log-event round trip.

Development checkout: pip install -e '.[otel]'. No provider/model calls.
Optional --mcp-python uses a real multivon-mcp server over stdio; install the
MCP client SDK in this environment and multivon-mcp in that server environment.
The loopback HTTP receiver is a test fixture, not a production collector.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode

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


def message(role, content):
    return json.dumps([{'role': role, 'parts': [{'type': 'text', 'content': content}]}])


async def mcp_call(python, tool, arguments):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    params = StdioServerParameters(command=python, args=['-m', 'multivon_mcp.server'])
    async with (
        stdio_client(params) as (reader, writer),
        ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=20)) as session,
    ):
        await session.initialize()
        result = await session.call_tool(tool, arguments)
        if result.isError:
            raise RuntimeError('MCP fixture tool returned an execution error')
        return result.structuredContent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mcp-python', help='Python executable with published multivon-mcp installed')
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    received = {'/v1/traces': [], '/v1/logs': []}
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path not in received:
                self.send_error(404)
                return
            received[self.path].append(self.rfile.read(int(self.headers['Content-Length'])))
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-protobuf')
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f'http://127.0.0.1:{server.server_port}'
    trace_provider, log_provider = TracerProvider(), LoggerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(
        endpoint=endpoint + '/v1/traces', compression=Compression.NoCompression, timeout=5)))
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(OTLPLogExporter(
        endpoint=endpoint + '/v1/logs', compression=Compression.NoCompression, timeout=5)))
    try:
        tracer = trace_provider.get_tracer('multivon.otel.fixture', '1')
        tool = 'eval_tool_call_accuracy' if args.mcp_python else 'lookup'
        parameters = {'expected_tool_calls': [], 'agent_trace': []} if args.mcp_python else {'id': 7}
        tool_attributes = {'gen_ai.operation.name': 'execute_tool', 'gen_ai.tool.name': tool,
                           'gen_ai.tool.call.arguments': json.dumps(parameters)}
        if args.mcp_python:
            tool_attributes['mcp.method.name'] = 'tools/call'
        with tracer.start_as_current_span('invoke_agent fixture', attributes={
                'gen_ai.operation.name': 'invoke_agent',
                'gen_ai.input.messages': message('user', 'Run the fixture check')}) as root:
            parent = root.get_span_context()
            with tracer.start_as_current_span('execute_tool ' + tool,
                    kind=SpanKind.CLIENT if args.mcp_python else SpanKind.INTERNAL,
                    attributes=tool_attributes) as call:
                response = asyncio.run(mcp_call(args.mcp_python, tool, parameters)) if args.mcp_python else {'passed': True}
                call.set_attribute('gen_ai.tool.call.result', json.dumps(response))
                if response['passed'] is not True:
                    call.set_status(Status(StatusCode.ERROR, 'Fixture check failed'))
                    raise RuntimeError('Fixture did not pass')
            root.set_attribute('gen_ai.output.messages', message('assistant', 'pass'))
        if not trace_provider.force_flush():
            raise RuntimeError('OTel trace flush did not finish')
        trace = OtelTrace(tuple(received['/v1/traces']), f'{parent.trace_id:032x}', f'{parent.span_id:016x}',
                          GENAI_PROFILE, tool_coverage_complete=True)
        # This assertion is justified only for the bounded, fully instrumented
        # local fixture. Sampling flags alone would not justify it in production.
        case = EvalCase('Run the fixture check', 'pass', case_id='otel-fixture',
                        source_id='otel-fixture-source', expected_tool_calls=[tool])
        suite = EvalSuite('OTLP evidence fixture').add_case(case).add_evaluators(ExactMatch(), ToolCallAccuracy())
        report = score_otel_trace(suite, trace, case)
        policy = AcceptancePolicy((CheckRequirement('exact_match'), CheckRequirement('tool_call_accuracy')))
        assert policy.evaluate(report).decision == 'accept'
        report.save_json(str(output / 'report.json'))
        (output / 'policy.json').write_text(json.dumps(policy.to_dict(), indent=2))
        (output / 'native-evidence.json').write_text(json.dumps(trace.to_dict(), indent=2))
        submitted = emit_evaluation_events(report, log_provider.get_logger('multivon.evaluation'))
        if not log_provider.force_flush():
            raise RuntimeError('OTel event flush did not finish')
        records = [record for body in received['/v1/logs']
                   for resource in ExportLogsServiceRequest.FromString(body).resource_logs
                   for scope in resource.scope_logs for record in scope.log_records]
        assert len(records) == submitted == 3
        assert all(r.trace_id.hex() == trace.trace_id and r.span_id.hex() == trace.root_span_id for r in records)
        assert [r.event_name for r in records].count('gen_ai.evaluation.result') == 2
        for index, payload in enumerate(received['/v1/traces']):
            (output / f'trace-{index}.pb').write_bytes(payload)
        for index, payload in enumerate(received['/v1/logs']):
            (output / f'events-{index}.pb').write_bytes(payload)
        incomplete = OtelTrace(trace.payloads, trace.trace_id, trace.root_span_id, GENAI_PROFILE)
        partial_report = score_otel_trace(suite, incomplete, case)
        assert policy.evaluate(partial_report).decision == 'indeterminate'
        partial_report.save_json(str(output / 'incomplete-report.json'))
        decisions = {}
        if args.mcp_python:
            for filename, expected in [('report.json', 'accept'), ('incomplete-report.json', 'indeterminate')]:
                decision = asyncio.run(mcp_call(args.mcp_python, 'eval_acceptance_report', {
                    'report_json_path': str(output / filename), 'policy_json_path': str(output / 'policy.json')}))
                assert decision['decision'] == expected
                decisions[filename] = decision['decision']
        summary = {'fixture_only': True, 'provider_calls': 0, 'tool': tool,
                   'trace_payloads': len(trace.payloads), 'evaluation_events': submitted,
                   'complete_decision': 'accept', 'incomplete_decision': 'indeterminate',
                   'mcp_decisions': decisions}
        (output / 'validation.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        trace_provider.shutdown()
        log_provider.shutdown()
        server.shutdown(); server.server_close(); worker.join()


if __name__ == '__main__':
    main()
