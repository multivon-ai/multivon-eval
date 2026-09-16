# OpenTelemetry interoperability validation — 2026-09-17

This is a synthetic compatibility check, not an application-quality benchmark.
One instrumented task calls published multivon-mcp over stdio. No model API
calls or independent production instrumentation claims are involved.

## Reused components

- OpenTelemetry Python API, SDK, protobuf types and OTLP HTTP exporters 1.44.0.
- Official Collector Contrib 0.161.0, using its OTLP HTTP receiver and file exporter.
- Published multivon-mcp 0.4.0 with multivon-eval 0.18.0 in an isolated server
  environment. The calling environment uses the development evidence bridge.
- GenAI Development conventions pinned to
  `c88d504ab3d9879f8e50d3cc87e69775e11db234`.

Multivon implements evidence selection, conservative projection to existing
graders and policy decisions. Transport, telemetry types, instrumentation,
collection and event delivery remain upstream. No upstream source was copied.

## Actual SDK, MCP and Collector checks

Run from a development checkout with the `otel` extra and official `mcp` client
installed. Use a separate Python environment with `multivon-mcp==0.4.0`:

```bash
python examples/otel_evidence.py --output-dir otel-demo \
  --mcp-python /path/to/mcp-environment/bin/python
```

The example retains two native protobuf export requests: the root task and its
MCP client tool span. The actual `eval_tool_call_accuracy` call checks an observed
empty trace against an empty expectation. The bridge evaluates the outer task's
captured text and tool call, then exports three native OTLP log events: one trial
status and two standard evaluation results. All three preserve the target trace
and selected output span IDs. The example parses the exported protobuf to verify
this instead of merely checking an in-memory callback.

The complete fixture's policy decision is `accept`. Removing its explicit
capture-completeness assertion produces `indeterminate`. Two further real MCP
calls to `eval_acceptance_report` reproduce those decisions from saved JSON.
There are three MCP calls overall and zero provider inference calls.

We also sent the two trace protobuf bodies to the official Collector's loopback
OTLP receiver. Both requests returned HTTP 200. This configuration produced
native newline-delimited OTLP JSON:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
exporters:
  file:
    path: /artifacts/traces.jsonl
    format: json
    flush_interval: 100ms
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [file]
```

The pinned image is
`otel/opentelemetry-collector-contrib@sha256:fd328de2552466ad78385e1b1289c3f2402b1c45f265b252aab1955b42845ac1`.
Bind the receiver to loopback when reproducing this local fixture. Allow the
file exporter to flush before reading its output. The task-owned Collector was
stopped after validation.

The actual exported JSON is committed in
[`tests/fixtures/otel/collector-traces.jsonl`](../../tests/fixtures/otel/collector-traces.jsonl),
with version, ID and SHA-256 provenance alongside it. Its SHA-256 is
`9fd8d33700bc682bdac0ef827d21d7e1dbedfecfd76fffbeefa35af87d8ba8a1`.
Reimport preserves the original bytes and reproduces measured passing checks.
Additional tests preserve unknown protobuf and JSON fields without interpreting
them. The fixture does not invent a schema URL from the SDK version.

Validation passed 1,642 library tests on Python 3.12, with four skips and seven
warnings. All 23 focused telemetry tests passed on Python 3.10. All 65 MDX
pages compiled. These APIs remain a development preview, not PyPI 0.18.0.

## Failure behavior and limits

Regression checks cover transport retries, conflicting duplicate spans, changed
evidence, cycles, wrong inputs, output selection outside the task, missing
parents, dropped data, redacted output, missing arguments/results, native errors,
overlapping tool calls, invalid latency, absent capture assertions and malformed
OTLP JSON. Errors/skips never acquire an invented numeric evaluation score.

The bridge requires an explicit supported profile; schema URLs are retained
without automatic migration. GenAI conventions remain Development, and the
Collector file exporter is alpha upstream. This validates the versions and
bounded fixture above, not arbitrary vendors, backends, future schemas or
distributed instrumentation. The current projection covers text and sequential
tools; native media, usage and other fields are retained but not automatically
interpreted. Completeness and source authenticity are not independently proven.
Judge API accounting and backend delivery guarantees remain outside this bridge.

Sources: [GenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai/tree/c88d504ab3d9879f8e50d3cc87e69775e11db234),
[OTLP JSON](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding),
[Collector file exporter](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/v0.161.0/exporter/fileexporter).
