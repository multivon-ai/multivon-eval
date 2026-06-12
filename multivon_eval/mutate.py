"""Deterministic case generation — mutators + template grids. $0, instant.

CheckList-style robustness suites without judge spend:

- :func:`mutate_cases` applies registry mutations (typo / whitespace /
  case noise, unicode confusables, punctuation strip, conservative
  negation flips) to existing cases. Every mutant records the
  transformation and the *expectation*: ``invariant`` (model output
  should not change materially) or ``flip`` (the label inverts — the
  source ``expected_output`` is dropped, never silently kept).
- :func:`cases_from_template` expands ``"Refund for {item} bought
  {when}"`` over axis values — full product (capped) or a greedy
  pairwise covering array.

Both are deterministic per ``seed`` (modulo the provenance stamp's
``case_uid`` / ``authored_at``, which are intentionally fresh per run —
see ``provenance.py``) and route accepted cases through the
``case_gates`` accounting contract: every generated case lands in
exactly one :class:`~multivon_eval.case_gates.GenerationReport` bucket.

Dedupe note: mutation batches dedupe on *exact* mutated-input identity
(vs the source and vs other accepted mutants) instead of
``gate_duplicate``'s loose-normalized Jaccard — mutants are near
duplicates of their source *by construction* (that is the whole point of
a robustness suite), so the Jaccard gate would reject nearly everything
it exists to create. Template grids use ``gate_duplicate`` as-is.
"""
from __future__ import annotations

import itertools
import random
import re
import string
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .case import EvalCase
from .case_gates import GenerationReport, gate_duplicate, gate_well_formed
from .provenance import git_info, read_provenance, stamp_metadata_inplace

#: A mutator is a pure function (text, rng) -> mutated text, or None when
#: the mutation does not apply to this text (no eligible site).
Mutator = Callable[[str, random.Random], "str | None"]

# Full product above this size is an error, not a silent truncation.
TEMPLATE_PRODUCT_CAP = 2000


# ─── Mutators ─────────────────────────────────────────────────────────────

# Words eligible for typo noise: pure-letter runs of ≥5 chars. Digits are
# never part of a match, so typos can never land inside numbers.
_TYPO_WORD_RE = re.compile(r"[A-Za-z]{5,}")
_CASE_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Latin → visually-identical Cyrillic confusables. Borrowed from
# ``auto.generate_unicode_obfuscation_cases``'s function-local
# _HOMOGLYPH_MAP (not importable from there). The digit "0" entry is
# deliberately dropped: mutators never touch numbers.
_CONFUSABLES = {
    "a": "а", "c": "с", "e": "е", "i": "і", "j": "ј", "o": "о", "p": "р",
    "s": "ѕ", "x": "х", "y": "у", "A": "А", "B": "В", "C": "С", "E": "Е",
    "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х",
}

# Longest alternatives FIRST — "is not" must win over "is" and "not".
_NEGATABLE_RE = re.compile(r"\b(?:is not|cannot|is|can|not)\b")
_NEGATION_MAP = {"is not": "is", "cannot": "can", "is": "is not", "can": "cannot"}


def typo_noise(text: str, rng: random.Random) -> str | None:
    """Swap/drop/duplicate 1-2 adjacent chars inside one word of ≥5 letters."""
    matches = list(_TYPO_WORD_RE.finditer(text))
    if not matches:
        return None
    m = matches[rng.randrange(len(matches))]
    w = list(m.group())
    for _ in range(rng.randint(1, 2)):
        i = rng.randrange(len(w) - 1)
        op = rng.choice(("swap", "drop", "dup"))
        if op == "swap":
            w[i], w[i + 1] = w[i + 1], w[i]
        elif op == "drop" and len(w) > 4:
            del w[i]
        else:
            w.insert(i, w[i])
    mutated = "".join(w)
    if mutated == m.group():  # e.g. swapping identical letters no-ops
        return None
    return text[: m.start()] + mutated + text[m.end():]


def whitespace_noise(text: str, rng: random.Random) -> str | None:
    """Double a space, add trailing spaces, or swap a space for a tab."""
    if not text.strip():
        return None
    space_idxs = [i for i, c in enumerate(text) if c == " "]
    ops = ["trailing"] + (["double", "tab"] if space_idxs else [])
    op = rng.choice(ops)
    if op == "trailing":
        return text + "  "
    i = space_idxs[rng.randrange(len(space_idxs))]
    sub = "  " if op == "double" else "\t"
    return text[:i] + sub + text[i + 1:]


def case_noise(text: str, rng: random.Random) -> str | None:
    """RANDOM-cAsE or ALL-CAPS a word or two."""
    matches = list(_CASE_WORD_RE.finditer(text))
    if not matches:
        return None
    picks = rng.sample(range(len(matches)), k=min(len(matches), rng.randint(1, 2)))
    out = text
    for mi in sorted(picks, reverse=True):  # right-to-left keeps offsets valid
        m = matches[mi]
        word = m.group()
        if rng.random() < 0.5:
            new = "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in word)
            if new == word:
                new = word.swapcase()
        else:
            new = word.upper() if word != word.upper() else word.lower()
        out = out[: m.start()] + new + out[m.end():]
    return out if out != text else None


def unicode_confusable(text: str, rng: random.Random) -> str | None:
    """Map 1-3 latin chars to visually-identical Cyrillic confusables."""
    idxs = [i for i, c in enumerate(text) if c in _CONFUSABLES]
    if not idxs:
        return None
    chars = list(text)
    for i in rng.sample(idxs, min(len(idxs), rng.randint(1, 3))):
        chars[i] = _CONFUSABLES[chars[i]]
    return "".join(chars)


def punctuation_strip(text: str, rng: random.Random) -> str | None:
    """Remove terminal punctuation or commas."""
    stripped = text.rstrip()
    ops = []
    if stripped and stripped[-1] in ".!?":
        ops.append("terminal")
    if "," in text:
        ops.append("commas")
    if not ops:
        return None
    op = rng.choice(ops)
    if op == "terminal":
        return stripped[:-1] + text[len(stripped):]
    return text.replace(",", "")


def negation_flip(text: str, rng: random.Random) -> str | None:
    """CONSERVATIVE rule-based negation: applies ONLY when exactly one
    negatable site exists in the text ("is"↔"is not", "can"↔"cannot",
    bare " not" removal); otherwise inapplicable. The mutant's label is
    inverted — ``mutate_cases`` drops the source expected_output."""
    matches = list(_NEGATABLE_RE.finditer(text))
    if len(matches) != 1:
        return None
    m = matches[0]
    token = m.group()
    if token == "not":
        start = m.start()
        if start > 0 and text[start - 1] == " ":
            start -= 1  # collapse the now-doubled space
        return text[:start] + text[m.end():]
    return text[: m.start()] + _NEGATION_MAP[token] + text[m.end():]


#: Registry of deterministic mutators, applied by :func:`mutate_cases`.
MUTATIONS: dict[str, Mutator] = {
    "typo_noise": typo_noise,
    "whitespace_noise": whitespace_noise,
    "case_noise": case_noise,
    "unicode_confusable": unicode_confusable,
    "punctuation_strip": punctuation_strip,
    "negation_flip": negation_flip,
}

#: Mutations whose expected label INVERTS (everything else is invariant).
FLIP_MUTATIONS = frozenset({"negation_flip"})

_FLIP_NOTE = (
    "label inverted by negation_flip — the source expected_output no "
    "longer applies and was dropped; relabel before scoring"
)


def mutate_cases(
    cases: Sequence[EvalCase],
    mutations: Sequence[str] | None = None,
    seed: int = 0,
    per_case: int = 1,
) -> tuple[list[EvalCase], GenerationReport]:
    """Apply deterministic mutations to ``cases``. No LLM calls, $0.

    Args:
        cases:     Source cases (their inputs get mutated).
        mutations: Mutation names from :data:`MUTATIONS` (default: all).
        seed:      Determinism seed — same inputs + seed ⇒ same mutants.
        per_case:  Mutant attempts per (case, mutation) pair.

    Returns:
        ``(mutants, GenerationReport)``. Invariant mutants carry the
        source ``expected_output``/``context`` unchanged; flip mutants
        carry ``expected_output=None`` plus a metadata note explaining
        that the label inverted. Inapplicable mutations are simply not
        generated (visible as ``requested - generated``).
    """
    if mutations is None:
        names = list(MUTATIONS)
    else:
        names = list(mutations)
        unknown = [m for m in names if m not in MUTATIONS]
        if unknown:
            raise ValueError(
                f"unknown mutation(s) {unknown}; known: {sorted(MUTATIONS)}"
            )
    if per_case < 1:
        raise ValueError("per_case must be >= 1")

    report = GenerationReport(
        requested=len(cases) * len(names) * per_case, kind="mutation",
    )
    git = git_info(".")
    accepted: list[EvalCase] = []
    seen_inputs: set[str] = set()

    for idx, case in enumerate(cases):
        status, prov = read_provenance(case.metadata)
        src_uid = prov.get("case_uid") if (status == "ok" and prov) else None
        for name in names:
            fn = MUTATIONS[name]
            for k in range(per_case):
                # Per-(case, mutation, k) rng: stable regardless of which
                # other mutations/cases are in the batch.
                rng = random.Random(f"{seed}:{idx}:{name}:{k}")
                mutated = fn(case.input or "", rng)
                if mutated is None or mutated == case.input:
                    continue  # inapplicable — accounted as requested - generated
                report.generated += 1
                flip = name in FLIP_MUTATIONS
                metadata: dict[str, Any] = {
                    "generation": {
                        "kind": "mutation",
                        "mutation": name,
                        "expectation": "flip" if flip else "invariant",
                        "seed": seed,
                        "source_case_uid": src_uid,
                    },
                }
                if flip:
                    metadata["generation"]["note"] = _FLIP_NOTE
                    # Doubles as the gate_well_formed expected-behavior text.
                    metadata["expected_behavior"] = _FLIP_NOTE
                stamp_metadata_inplace(
                    metadata, authored_by="generator:mutation", git=git, targets=[],
                )
                mutant = EvalCase(
                    input=mutated,
                    expected_output=None if flip else case.expected_output,
                    context=case.context,
                    metadata=metadata,
                    tags=list(case.tags),
                )
                if not gate_well_formed(mutant).passed:
                    report.dropped_malformed += 1
                    continue
                # Exact-identity dedupe vs the batch (see module docstring
                # for why gate_duplicate's Jaccard is wrong for mutants).
                if mutated in seen_inputs:
                    report.dropped_duplicate += 1
                    continue
                seen_inputs.add(mutated)
                accepted.append(mutant)

    report.accepted = len(accepted)
    return accepted, report


# ─── Template grids ───────────────────────────────────────────────────────


def cases_from_template(
    template: str,
    axes: Mapping[str, Sequence[Any]],
    sample: str = "all",
    n: int | None = None,
    seed: int = 0,
    expected_output: str | None = None,
    context: "str | list[str] | None" = None,
    expected_behavior: str | None = None,
) -> tuple[list[EvalCase], GenerationReport]:
    """Expand a ``{placeholder}`` template over axis values.

    ``sample="all"`` is the full product (error above
    :data:`TEMPLATE_PRODUCT_CAP`); ``sample="pairwise"`` is a greedy
    covering array — every pair of values across every axis pair appears
    in at least one row. Deterministic per ``seed``. ``n`` optionally
    subsamples the rows (seeded).

    ``expected_output`` may itself contain the same placeholders — it is
    formatted per row. Rows without ``expected_output`` or
    ``expected_behavior`` are still valid: judge evaluators (Relevance,
    Toxicity, add_check, ...) score outputs without a reference answer.
    Template rows are gated on non-empty substituted input + dedupe only;
    no label is ever invented for them.
    """
    if not isinstance(axes, Mapping) or not axes:
        raise ValueError("axes must be a non-empty mapping of axis -> values")
    keys = list(axes)
    for key in keys:
        vals = axes[key]
        if isinstance(vals, str) or not isinstance(vals, Sequence) or not vals:
            raise ValueError(f"axis {key!r} must be a non-empty list of values")
    fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    if missing := fields - set(keys):
        raise ValueError(f"template placeholders {sorted(missing)} have no axis values")
    if unused := set(keys) - fields:
        raise ValueError(f"axes {sorted(unused)} do not appear in the template")

    if sample == "all":
        total = 1
        for key in keys:
            total *= len(axes[key])
        if total > TEMPLATE_PRODUCT_CAP:
            raise ValueError(
                f"full product is {total} cases — above the "
                f"{TEMPLATE_PRODUCT_CAP} cap. Use sample='pairwise' or trim "
                f"the axes."
            )
        rows = [dict(zip(keys, combo))
                for combo in itertools.product(*(axes[k] for k in keys))]
    elif sample == "pairwise":
        rows = _pairwise_rows(keys, axes, seed)
    else:
        raise ValueError(f"sample must be 'all' or 'pairwise', got {sample!r}")

    if n is not None and 0 <= n < len(rows):
        rng = random.Random(f"{seed}:template-n")
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), n))]

    git = git_info(".")
    report = GenerationReport(requested=len(rows), kind="template")
    accepted: list[EvalCase] = []
    for row in rows:
        report.generated += 1
        expected = expected_output
        if isinstance(expected, str):
            try:
                expected = expected.format(**row)
            except (KeyError, IndexError, ValueError):
                pass  # literal braces that aren't axis placeholders — keep raw
        metadata: dict[str, Any] = {
            "generation": {
                "kind": "template",
                "template": template,
                "axes": dict(row),
                "sample": sample,
                "seed": seed,
            },
        }
        if expected_behavior:
            metadata["expected_behavior"] = expected_behavior
        stamp_metadata_inplace(
            metadata, authored_by="generator:template", git=git, targets=[],
        )
        case = EvalCase(
            input=template.format(**row), expected_output=expected,
            context=context, metadata=metadata,
        )
        # Templates gate on input only: a row without expected_output is
        # valid for judge evaluators, which need no reference answer.
        # gate_well_formed stays for mutants (they inherit expectations).
        if not (case.input or "").strip():
            report.dropped_malformed += 1
            continue
        if not gate_duplicate(case, accepted).passed:
            report.dropped_duplicate += 1
            continue
        accepted.append(case)

    report.accepted = len(accepted)
    return accepted, report


def _pair(i: int, a: int, j: int, b: int) -> tuple[int, int, int, int]:
    """Canonical (axis, value-idx) pair key with axis order normalized."""
    return (i, a, j, b) if i < j else (j, b, i, a)


def _pairwise_rows(
    keys: list[str], axes: Mapping[str, Sequence[Any]], seed: int,
) -> list[dict[str, Any]]:
    """Greedy pairwise covering array: loop until every (value, value)
    pair across every axis pair is covered by at least one row.
    Deterministic per seed (rng only breaks score ties)."""
    if len(keys) == 1:
        return [{keys[0]: v} for v in axes[keys[0]]]
    rng = random.Random(f"{seed}:pairwise")
    sizes = [len(axes[k]) for k in keys]
    uncovered: set[tuple[int, int, int, int]] = {
        (i, a, j, b)
        for i in range(len(keys)) for j in range(i + 1, len(keys))
        for a in range(sizes[i]) for b in range(sizes[j])
    }
    rows: list[dict[str, Any]] = []
    while uncovered:
        i, a, j, b = min(uncovered)  # deterministic anchor pair
        assign: dict[int, int] = {i: a, j: b}
        for axis in range(len(keys)):
            if axis in assign:
                continue
            best_vals: list[int] = []
            best_gain = -1
            for v in range(sizes[axis]):
                gain = sum(
                    1 for other, ov in assign.items()
                    if _pair(axis, v, other, ov) in uncovered
                )
                if gain > best_gain:
                    best_gain, best_vals = gain, [v]
                elif gain == best_gain:
                    best_vals.append(v)
            assign[axis] = rng.choice(best_vals)
        items = sorted(assign.items())
        for x in range(len(items)):
            for y in range(x + 1, len(items)):
                (i2, a2), (j2, b2) = items[x], items[y]
                uncovered.discard((i2, a2, j2, b2))
        rows.append({keys[ax]: axes[keys[ax]][v] for ax, v in items})
    return rows


__all__ = [
    "MUTATIONS",
    "FLIP_MUTATIONS",
    "TEMPLATE_PRODUCT_CAP",
    "mutate_cases",
    "cases_from_template",
    "typo_noise",
    "whitespace_noise",
    "case_noise",
    "unicode_confusable",
    "punctuation_strip",
    "negation_flip",
]
