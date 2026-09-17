# Provider accounting validation — 2026-09-17

Reconciliation of the **4 retained live responses** produced **48 input tokens,
20 output tokens and a $0.000148 list-price estimate**. The experiment blocked
socket connections before importing LiteLLM and observed **zero attempted
connections**. No provider calls were repeated.

This is an accounting check over a small transport smoke study, not a model
benchmark or invoice reconciliation. The original two strict quality checks
remain failed; repricing does not change their outputs or judgments.

## Frozen inputs and execution

- [Original native capture](PROVIDER_EVIDENCE_VALIDATION.md), produced from source
  `a11e344eccf9dc42886d90a94ace6c55efb57946`.
- Input archive SHA-256:
  `67adbd7274e91a6e41476acd141a174f29caced93308670771335b6e332eaefd`.
- [Reconciliation script](reconcile_provider_capture.py), executed with a clean
  checkout at `f98748921e1291b83775a240b5a43e2aca4272e1`.
- Actual upstream LiteLLM **1.101.0**, local catalog option enabled before import.
  Loaded catalog digest:
  `386574df9d610aff7819b47c00a1511c2954658d9a97db85ab2a98deb3fb8e25`.

The measured scope includes both parallel targets and the two standalone
sync/async judge calls in the original smoke protocol. The declaration relies on
that inspected execution configuration and closed journal. It does not claim
automatic discovery of all networking in arbitrary callbacks.

## Independent arithmetic and upstream behavior

The provider's published standard Haiku 4.5 rates are $1 per million base input
tokens and $5 per million output tokens. These four responses reported zero
cache reads/writes and standard service. Thus `(48 × 1 + 20 × 5) / 1,000,000 =
0.000148`, agreeing with upstream calculation within floating-point precision.
Source: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing),
checked 2026-09-17. This is current list pricing applied to retained evidence.

Separate native-usage fixtures check Anthropic's mixed five-minute/one-hour cache
writes and cache reads, and OpenAI's cached-input and reasoning-token treatment.
They use LiteLLM's actual response conversion and calculator, rather than a
replacement token estimator. Unsupported endpoints/tariffs, missing usage and
catalog mutation are negative controls and leave prices unknown.

## Verification and artifacts

The [accounting JSON](results/provider-accounting-2026-09-17/accounting.json)
retains native usage, normalized dimensions, per-attempt event references,
estimator version, catalog hash and selected tariff entry. Its
[summary](results/provider-accounting-2026-09-17/summary.json) records source and
reconciliation commits. [Environment metadata](results/provider-accounting-2026-09-17/environment.json)
was collected afterward from the unchanged environment; it is not a verified
build manifest. The [validation index](results/provider-accounting-2026-09-17/validation.json)
contains file hashes and round-trip checks. Historical capture artifacts were
not rewritten.

The reconciled ledger passed a 68-token and $0.001 post-run budget. Tests reject
missing/legacy/partial accounting, incomplete lifecycles, invalid usage/prices,
unknown dollar tariffs and nonfinite/negative limits. Submicrodollar estimates
are not rounded down to zero before gating.

Validation: **1,958 passed, 18 skipped** in the main Python 3.12 environment;
**111 focused checks passed** on Python 3.10; **44 checks passed** in the isolated
actual-LiteLLM environment, including all **13 optional pricing checks** skipped
in the main environment. The optional environment emitted one pytest warning
because pytest-asyncio was not installed; those checks do not use async tests.
All **71 MDX pages** parsed, and the new documentation example executed through
an actual SDK with a local transport.

## Remaining limits

Coverage declarations are caller assertions. Unknown responses and unsupported
transport paths cannot be converted into measured zero usage. This bridge
validates standard direct Anthropic Messages/OpenAI Chat Completions with text
output; Google pricing, server-tool fees, nonstandard tiers, regional pricing,
other endpoints and nontext output remain unknown. Upstream prices can change or
contain errors. Discounts, tax, infrastructure and external services are excluded.
Post-run budget gates do not enforce a prepaid execution cap. These limits remain
part of the [industrial program](../../plans/industrial-evaluation.md).
