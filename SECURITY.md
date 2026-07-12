# Security Policy

## Reporting a vulnerability

Email <hello@multivon.ai>. Include a reproduction if you can. You'll get a
human reply; we'll coordinate a fix and credit you unless you'd rather stay
anonymous. Please don't open a public issue for anything exploitable before
we've had a chance to patch.

## Scope

multivon-eval is a **local-first library and CLI**. There is no hosted
service, no server component, and no telemetry — everything runs in your
process on your machine or your CI.

In scope:

- Code execution or file-write vulnerabilities triggered by untrusted eval
  inputs (datasets, traces, model outputs, downloaded suite files).
- Secrets leaking into artifacts the tool writes (reports, audit logs, cache
  files, prompt recordings).
- Tampering weaknesses in the hash-chained compliance audit trail.

Out of scope:

- Vulnerabilities in the model providers you point the tool at (OpenAI,
  Anthropic, etc.) or in optional third-party dependencies — report those
  upstream.
- Prompt injection against your own models. The evaluators can help you
  *measure* that; it isn't a vulnerability in this tool.
- Anything requiring a malicious local user who already has your shell.

## Secrets handling

API keys are read from environment variables or a local `.env` file
(see `.env.example`). The library never transmits keys anywhere except to
the provider you configured, and never writes them into reports or logs.
If you find a code path that does, that's a valid report — see above.
