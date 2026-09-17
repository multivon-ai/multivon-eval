# Regulated enterprise target and moat study

September 17, 2026. Owner direction: regulated enterprise teams are the closest
customer segment. Recommendations below are hypotheses, not customer discovery
results. No customer has been contacted and no regulated customer data obtained.

## Recommendation

Start with **verification of financial-document answers and their saved review
records**. The first application is an analyst assistant that reads a supplied
financial table and surrounding text, calculates an answer, cites evidence and
creates a draft case record for review. This fits the existing document, numeric
oracle, provenance, outcome and retry work. It does not automate a lending,
investment or compliance decision.

Use TAT-QA for the public task, its official scorer for benchmark comparability,
and a separately labeled workflow projection for saved-state checks. Use
RAGChecker's human meta-evaluation and, after permitted access, LLM-AggreFact to
measure whether Multivon's *verifier* makes correct judgments. An answer benchmark
measures the answering system; it cannot by itself establish evaluator accuracy.

Contract review with CUAD is the second candidate if accessible customers are
legal/procurement teams. Do not build several vertical products before testing
whether one recurring release-review workflow is useful.

## Public application and dataset candidates

| Candidate | What to reuse | Fit and limitations | Decision |
|---|---|---|---|
| [TAT-QA](https://github.com/NExTplusplus/TAT-QA) | Financial tables/text, answer/scale labels, arithmetic derivations and official EM/F1 scorer; dataset README specifies CC BY 4.0 | Good numeric evidence-review task; supplied contexts do not test full-report retrieval. Our file audit found identifier and coverage differences between original test and released gold | First public task; pin the exact labeled release and disclose the projection |
| [FinQA](https://github.com/czyssrs/FinQA) | Financial reasoning programs, supporting facts and execution/program accuracy | Useful second numerical baseline. Upstream documents a historical retrieval-label leak and its correction; use corrected code and keep gold support out of target inputs. Repository [license](https://github.com/czyssrs/FinQA/blob/main/LICENSE) is MIT | Add after the first task; reuse the native evaluator |
| [CUAD](https://www.atticusprojectai.org/cuad/) | 510 contracts, 41 clause types and over 13,000 expert-supervised labels; author page states CC BY 4.0 | Strong evidence-extraction target. Clause presence is not a legal risk decision or customer procurement policy | Preferred alternate vertical |
| [MAUD](https://www.atticusprojectai.org/maud/) | 152 merger agreements, 92 questions and expert-supervised labels; author page states CC BY 4.0 | Credible legal interpretation benchmark; narrower M&A workflow and harder domain validation | Later, with a relevant design partner |
| [DocILE](https://docile.rossum.ai/) | Native invoice/business-document loader, localization and line-item evaluation | Strong invoice fit. Official access is by request for research; access terms must be resolved before commercial distribution. Extraction labels do not authorize payment | Keep as a candidate; do not bypass its access process |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) | Native tasks, attacks, defenses and utility/security checks; [MIT license](https://github.com/ethz-spylab/agentdojo/blob/main/LICENSE) | Tests agents consuming untrusted tool content. A controlled environment, not real banking customer validation | Separate agent-security regression track |
| [WorkArena](https://github.com/ServiceNow/WorkArena) | BrowserGym/AgentLab and native ServiceNow tasks; code [Apache 2.0](https://github.com/ServiceNow/WorkArena/blob/main/LICENSE) | Closer to actual enterprise software. Instance access requires a gated request and terms; code licensing does not grant instance access | Later integration when the customer workflow warrants it |
| [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) | Public financial QA sample and references | Dataset card specifies CC BY-NC 4.0 | Do not make it the default commercial product dataset |

For multimodal follow-up, [TAT-DQA](https://nextplusplus.github.io/TAT-DQA/)
provides financial PDF pages, converted content and QA annotations. Its current
project page states CC BY 4.0, while the [original paper abstract](https://arxiv.org/abs/2207.11871)
describes non-commercial release. Verify the downloaded release's terms before
redistribution. Do not assume TAT-QA and TAT-DQA IDs or question populations align.
Attribution and original-source notices should travel with every reused corpus.

## Actual TAT-QA file audit

Downloaded four JSON files from official commit
`870accc41953dcde885aabeb963d94aabdc0fbc3`; no inference or relabeling occurred.
The reproducible [audit script](../benchmarks/industrial/audit_public_financial_data.py)
records hashes, counts, fields and cross-file comparisons in
[this artifact](../benchmarks/industrial/results/regulated-target-research-2026-09-17/tatqa-audit.json).

| File | Contexts | Questions |
|---|---:|---:|
| Train | 2,201 | 13,215 |
| Development | 278 | 1,668 |
| Original unlabeled test | 278 | 1,669 |
| Released test gold | 277 | 1,663 |

There are **zero shared question IDs** between the original test and test-gold
files. After excluding identifiers, exact table/ordered-paragraph/question
content matches for **1,626** questions, without duplicate matching keys. There
are 43 original-only and 37 gold-only exact contents. This is not proof that
those 80 questions are semantically different: formatting and revisions can
also change exact content. Do not join labels by row position or assume this
is merely six missing annotations.

Train/development/original-test context-ID intersections are empty. The inspected
objects have no explicit source-report/company field, so this cannot prove
report-level or company-level independence. Keep the official split for public
comparisons and describe confidence intervals at the observed context level.
A customer-transfer claim needs separately identified source reports and cases.

The labeled file's extra `facts`, `mappings`, `tree_derivation` and all answers,
derivations and support annotations must stay out of model inputs. These are
grading evidence. Public test exposure also prevents a claim of contamination-free
generalization. Pin the scoring code as well as the data.

## Concrete acceptance target

Proposed task contract, to freeze before inference:

1. Supply only table cells, surrounding paragraphs and the user question. The
   model returns an answer, scale/unit and references to the supplied evidence.
2. Grade answer/scale through the [official TAT-QA evaluator](https://github.com/NExTplusplus/TAT-QA/blob/master/tatqa_eval.py).
   Keep the official benchmark metric separate from our workflow decision.
3. Independently inspect a draft case record: exact input revision, answer and
   unit, valid evidence references, one permitted record per request and no
   changes to unrelated records. A citation locator alone does not prove support.
4. Exercise interrupted writes, idempotent retry, changed policies, missing
   observations and explicit refusal. Preserve failures and missing evidence.
5. Review false accepts, false rejects and abstentions on frozen held-out cases.
   Validate numerical labels/units independently on a development sample before
   scaling. Do not silently repair test labels after viewing model outcomes.

The workflow controls are **Multivon's synthetic projection**, not annotations
supplied by TAT-QA. Reuse the existing SQLite/Inspect fixtures for transaction
tests. Keep AgentDojo attack scores in their native task definitions; transporting
an attack to another workflow creates a new experiment, not an official score.

## What can become a moat

The hypothesis is that regulated teams will pay for **faster, independently
reviewable release decisions when an AI workflow changes**. A useful artifact
connects the task requirement to the exact inputs, observed outcome, evaluator
version, review decision and deployment candidate. Re-evaluation must expose
which evidence became invalid when a document, model, tool or policy changed.

The possible durable advantage is accumulated domain validation and repeatable
integration work: vetted outcome checks, reviewed failure cases, adjudicated
disagreements, customer-controlled deployment patterns and credible evidence
that reviewers can use. Customer data remains customer-owned. Reusable knowledge
requires permission; there is no assumed right to pool private cases for training.

Several important features are already available elsewhere:

| Existing capability | Primary evidence | Implication |
|---|---|---|
| Versioned datasets and experiments | [Langfuse dataset versioning](https://langfuse.com/docs/evaluation/experiments/datasets) | Dataset IDs, snapshots and experiment UI are not a moat |
| Self-hosted tracing infrastructure | [Langfuse self-hosting](https://langfuse.com/self-hosting) | Integrate with established tracing/storage rather than replacing it |
| Enterprise audit controls | [Promptfoo audit logging](https://www.promptfoo.dev/docs/enterprise/audit-logging/) | Administrative logs and RBAC are expected controls, not novel evaluation science |
| Native evaluation logs | [Inspect logs](https://inspect.aisi.org.uk/eval-logs.html) | Preserve upstream logs; demonstrate added decision correctness over a well-configured baseline |
| Artifact signing | [Sigstore](https://docs.sigstore.dev/about/overview/) and [in-toto](https://in-toto.io/) | Reuse signing/attestation tooling; a hash or signature does not prove ground truth |

A serious comparison should give native Inspect the same outcome grader,
task rules and dependencies. Comparing Multivon to a deliberately incomplete
default grader would overstate the contribution. Measure integration effort,
misleading release decisions caught, review time and supported failure coverage.

## Feature priorities for this segment

1. **Verifier accuracy:** grounded claims, numerical consistency, explicit
   uncertainty, human disagreement and false-accept analysis. Benchmark plan:
   [public SoTA targets](sota-benchmark-program.md).
2. **Controlled evidence lifecycle:** minimize captured data, distinguish raw
   private artifacts from redacted exports, define retention/deletion behavior,
   and preserve a clear missing-evidence state when content expires. Use existing
   object stores and identity systems. [Langfuse's retention semantics](https://langfuse.com/docs/administration/data-retention)
   illustrate why keeping a dataset item is not the same as retaining its trace.
3. **Review and change control:** role-separated approval records, policy version
   binding, change impact, issue-system export and customer-controlled signing.
   Do not publish sensitive artifact identities to a public transparency log by
   default. Deployment and identity boundaries need an explicit threat model.
4. **Reproducible private operation:** installable packages, pinned dependencies,
   offline regrading, no-required-cloud workflow and operational documentation.
   Existing Python packages alone are not proof of production isolation.

Multimodal financial documents strengthen this application. Agentic controls
matter when systems write records or call tools. World-model work remains an
experimental track until an accessible customer has a simulation/planning task;
the existing CartPole demonstration does not establish this market's need.

## Evidence needed before positioning

Proposed discovery protocol: work with three permissioned teams, each supplying
one recent model/prompt/tool change and its current review process. Observe one
baseline and one Multivon-assisted review. Track reviewer time, unresolved
questions, false accepts, data-sharing constraints and repeat use on the next
change. Three teams is a planning target, not a measured sample or proof of fit.
Continue only if teams use the evidence in an actual release decision and want
the workflow again; an attractive benchmark alone is insufficient.

Use risk frameworks as vocabulary for evidence organization, not certification.
[NIST describes AI RMF as voluntary](https://www.nist.gov/itl/ai-risk-management-framework).
For US banking discussions, the Federal Reserve's
[SR 26-2, issued April 17, 2026](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm),
supersedes SR 11-7; do not base current sales copy on an outdated reference.
Applicability and control sufficiency require the customer's domain owners.

R17 remains unproven: public financial text and simulated workflows are credible
engineering substrates, not access to a customer's production acceptance target.
