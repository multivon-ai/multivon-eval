# Controlled robustness validation — 2026-09-17

The existing mutation API copied labels for presumed invariant transformations,
while hardness filtering counted baseline exceptions as quality failures and
discarded evaluator exceptions. Both behaviors could manufacture apparent
adversarial signal. This development checkpoint corrects those paths and tests
an explicit oracle-validation boundary. It is not a model benchmark.

## Reused methodology and bounded contribution

[CheckList](https://aclanthology.org/2020.acl-main.442/) provides behavioral test
types, including invariance and directional expectations. We use that distinction
without claiming a new methodology. [TextAttack](https://github.com/QData/TextAttack)
already supplies NLP transformation, constraint and search machinery; this change
does not implement another search engine. [Hypothesis](https://hypothesis.readthedocs.io/en/latest/)
6.168.0 supplies generated property examples and shrinking.

Multivon's contribution here is an integration boundary: preserve the original
candidate and its validation evidence; distinguish invalid from unknown; bind the
derived oracle to the exact cases, contract and source group; use the existing
CaseManifest and trial infrastructure. Digests and a callback contract do not
prove correctness or independence. This mechanism is readily reproducible and
is not, alone, a defensible moat. Domain contracts and a permissioned corpus of
independently reviewed failures remain more plausible sources of differentiation.

## Task and protocol

The local fixture reads `TOTAL EUR <amount>` with comma decimals, optional
horizontal whitespace, and integer euros when no comma is present. Four authored
amounts are €1.25, €23.45, €18,900.25 and €12,345.67. This narrow grammar is
synthetic test data, not a production invoice parser or a competing dataset.

Each source produces:

- A trailing-whitespace invariant, validated by reparsing both inputs.
- A one-euro counterfactual with a freshly derived answer.
- A comma-removal candidate whose claimed invariance is rejected.
- An explicitly unreviewed candidate whose validity remains unknown.

The Decimal oracle parses the actual inputs; it does not copy the candidate's
label or inspect a model answer. A separate integer implementation provides the
correct parser control. The intentionally flawed parser removes decimal commas.
Both parsers receive the same frozen 12 cases: four bases and eight validated
variants. All four source groups are development fixtures; no held-out or
customer-domain generalization is claimed.

Before execution, the example copies the entire runtime package and its own
source into the artifact directory. The archive includes that snapshot, manifest,
validation records, complete trial reports, paired results and hashes. The
underlying Git revision is recorded as a dirty development base; the archived
source is the precise implementation used, not an assertion of a clean release.

## Observed results

| Check | Observed count |
| --- | --- |
| Candidate validity | 8 valid, 4 invalid, 4 unknown across 4 sources |
| Correct integer parser | 12/12 scored cases correct |
| Deliberately comma-blind parser | 0/12 scored cases correct |
| Whitespace output consistency | 4/4 pairs for each parser |
| Counterfactual output changes | 4/4 pairs for each parser |
| Hypothesis property examples | 200 executed; all assertions passed |

The incorrect parser satisfies both output-consistency expectations while being
wrong everywhere. This is why consistency or a changed answer alone is
insufficient. Evaluate each side against its own oracle and retain both outcomes.
The eight pairs remain only four independent authored sources. No confidence
interval or model-ranking inference is warranted from these deterministic controls.

Hypothesis checks 200 positive integer-cent values up to 999,999 cents. For each,
it checks the source oracle, integer parser, valid invariant, valid one-euro
counterfactual and invalid comma removal. These generated regression checks are
not additional independent evaluation sources.

Three hardness controls use two requested shots per case. A fully measured wrong
answer has failure rate 1.0 and is kept; a baseline that raises has unknown
hardness and is excluded; a missing evaluator remains a not-run record. Baseline
and grader exceptions are no longer interchangeable with quality failures.

There were no provider or judge requests. The final execution made 24 suite
parser calls and four hardness-baseline calls; the property checks additionally
invoked the integer parser 200 times. Two development preflights failed in the
example's orchestration: an unsupported suite constructor argument before model
execution, then an incorrect summary attribute after 12 local parser calls. Their
console records and available raw reports are retained under `rehearsals/` in
the public bundle. One completed rehearsal preceded the final CLI identity-export
correction; its summary matches the final execution. The completed rehearsal
made another 24 suite calls and four hardness calls. Each of the four script
invocations also completed its 200 property examples. Rehearsal sources are not
snapshotted separately; `source/` is the final execution's pre-run snapshot.

## Reproduce and replay

```bash
pip install -e . 'hypothesis==6.168.0'
python examples/controlled_robustness.py --output-dir /tmp/robustness-new
mkdir /tmp/robustness-replay
tar -xzf benchmarks/industrial/results/robustness-2026-09-17/evidence.tar.gz \
  -C /tmp/robustness-replay
python benchmarks/industrial/replay_robustness.py /tmp/robustness-replay
```

[Evidence archive](results/robustness-2026-09-17/evidence.tar.gz): 435,141 bytes,
130 checksummed files plus the checksum index. SHA-256:
`fce8c77d529311a18b078640869343ad1686ce94a2f4852d9e82d9f38f62c9bb`.
Offline replay from the extracted archive verified file hashes, validation
digests, manifest reconstruction, saved trial outputs and exact regrading with
zero target calls. It checks consistency of the evidence, not oracle truth.

Validation: full Python 3.12 suite **1,776 passed, 5 skipped, 7 warnings**;
focused Python 3.10 checks **160 passed**; **69 MDX pages** compiled; scoped lint
and whitespace checks passed. Mutation JSONL export also now preserves case IDs,
source groups and all portable case fields.

## Remaining limits

The public API's initial profile supports explicit string answers and invariant
or counterfactual relations. It does not implement general directional scores,
tool/state oracle comparison, multimodal transformation validation or human-review
collection. Existing Label Studio and media bridges supply adjacent capabilities;
task-specific validation must connect them. A validator can still be wrong,
circular or dishonest. Validation records make that claim inspectable, not true
by construction. This experiment does not establish customer usefulness, broad
linguistic coverage, model robustness, or state-of-the-art performance.
