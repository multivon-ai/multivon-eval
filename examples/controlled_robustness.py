"""Controlled, code-validated numeric transformations. No inference or judge calls.

Synthetic EUR-total task: comma is the decimal separator, integers are euros,
and horizontal/trailing whitespace is irrelevant. This deliberately narrow
grammar is not a production invoice parser or a substitute for a public dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path

from multivon_eval import CaseManifest, EvalCase, EvalSuite, ExactMatch
from multivon_eval.case_manifest import canonical_json
from multivon_eval.hardness import validate_adversarial_cases
from multivon_eval.mutate import mutate_cases
from multivon_eval.robustness import OracleVerdict, validate_variant

CONTRACT = 'synthetic-eur-comma-total/v1'
PATTERN = re.compile(r'TOTAL[ \t]+EUR[ \t]+([0-9]+(?:,[0-9]{2})?)[ \t]*')


def oracle(case):
    match = PATTERN.fullmatch(case.input)
    if match is None:
        raise ValueError('Input outside the declared numeric task grammar')
    return format(Decimal(match[1].replace(',', '.')), '.2f')


def validate_totals(base, candidate):
    return OracleVerdict(True, 'Parsed both inputs independently using the task grammar',
                         oracle(base), oracle(candidate), {'contract': CONTRACT,
                         'parser': PATTERN.pattern, 'arithmetic': 'Python decimal.Decimal'})


def source_case(cents, index):
    expected = f'{cents // 100}.{cents % 100:02}'
    return EvalCase(f'TOTAL EUR {expected.replace(".", ",")}', expected,
                    case_id=f'invoice-{index}:base', source_id=f'invoice-{index}',
                    tags=['synthetic', 'base'], metadata={'task_contract': CONTRACT})


def candidates(base):
    invariant = replace(base, input=base.input + '  ', expected_output=None,
                        case_id=base.case_id + ':whitespace', tags=['synthetic', 'whitespace'])
    # Transform authored cents, not a target answer. Validation reparses both inputs.
    whole, fraction = base.input.rsplit(' ', 1)[1].split(',')
    changed = replace(base, input=f'TOTAL EUR {int(whole) + 1},{fraction}', expected_output=None,
                      case_id=base.case_id + ':plus-one-euro', tags=['synthetic', 'counterfactual'])
    broken, _ = mutate_cases([base], mutations=['punctuation_strip'])
    return [('invariant', invariant), ('counterfactual', changed), ('invariant', broken[0])]


def comma_blind(text):
    """Deliberately flawed control: removes decimal commas as if grouping marks."""
    return f'{int(text.split()[-1].replace(",", ""))}.00'


def cents_parser(text):
    """Independent integer implementation, distinct from the Decimal oracle."""
    token = text.split()[-1]
    parts = token.split(',')
    cents = int(parts[0]) * 100 + (int(parts[1]) if len(parts) == 2 else 0)
    return f'{cents // 100}.{cents % 100:02}'


def property_checks():
    from hypothesis import given, settings
    from hypothesis import strategies as st

    count = 0
    @settings(max_examples=200, derandomize=True, database=None, deadline=None)
    @given(st.integers(min_value=1, max_value=999999))
    def check(cents):
        nonlocal count
        count += 1
        base = source_case(cents, 'property')
        assert oracle(base) == base.expected_output == cents_parser(base.input)
        records = [validate_variant(base, c, relation=r, contract=CONTRACT,
                                    validator=validate_totals) for r, c in candidates(base)]
        assert [r.status for r in records] == ['valid', 'valid', 'invalid']
        assert records[1].data['verdict']['variant_expected'] == f'{(cents + 100) // 100}.{cents % 100:02}'
    check()
    return {'library': 'hypothesis', 'version': version('hypothesis'),
            'examples_executed': count, 'status': 'passed', 'derandomize': True,
            'scope': 'Generated synthetic cents, two valid relations and one invalid relation each'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    # Freeze the full package and executable example before any experiment code runs.
    source = root / 'source'
    shutil.copytree(repo / 'multivon_eval', source / 'multivon_eval',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, source / 'controlled_robustness.py')
    shutil.copy2(repo / 'LICENSE', source / 'LICENSE')
    runtime = {'base_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
               'worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo)),
               'source_snapshot': 'source/', 'source_captured_before_experiment': True,
               'contract': CONTRACT, 'provider_requests': 0, 'local_model_calls': 28, 'judge_calls': 0,
               'packages': {p: version(p) for p in ['multivon-eval', 'hypothesis']}}
    def save(name, value):
        (root / name).write_text(canonical_json(value) + '\n')
    save('runtime.json', runtime)
    save('property-checks.json', property_checks())
    records, cases, pairs = [], {}, []
    for index, cents in enumerate([125, 2345, 1890025, 1234567]):
        base = source_case(cents, index)
        for relation, candidate in candidates(base):
            record = validate_variant(base, candidate, relation=relation, contract=CONTRACT,
                                      validator=validate_totals)
            records.append(record.data)
            if record.status == 'valid':
                validated = record.manifest('validated pair').cases
                cases.update({c.case_id: c for c in validated})
                pairs.append({'base': base.case_id, 'variant': candidate.case_id,
                              'source': base.source_id, 'relation': relation})
        unknown = validate_variant(base, replace(base, case_id=base.case_id + ':unreviewed'),
            relation='invariant', contract=CONTRACT,
            validator=lambda *_: OracleVerdict(None, 'Deliberate missing-review control'))
        records.append(unknown.data)
    save('validations.json', records)
    manifest = CaseManifest('controlled numeric development fixture', cases.values(),
        splits={'development': list(cases)}, provenance={'task_contract': CONTRACT,
        'source_kind': 'synthetic', 'paired_cases': pairs})
    manifest.save(root / 'manifest.json')
    summary = {'sources': 4, 'cases': len(cases), 'validation_counts': dict(Counter(r['status'] for r in records)),
               'models': {}, 'limitations': 'Local deterministic controls; no customer, model-ranking or general robustness claim.'}
    for name, model in [('comma_blind', comma_blind), ('cents_parser', cents_parser)]:
        suite = EvalSuite(name).add_evaluator(ExactMatch())
        suite.add_cases(manifest.cases)
        report = suite.run(model)
        (root / f'{name}.json').write_text(report.to_json())
        rows = {row.case_id: row for row in report.case_results}
        paired = []
        for pair in pairs:
            a, b = rows[pair['base']], rows[pair['variant']]
            paired.append({**pair, 'base_correct': a.passed, 'variant_correct': b.passed,
                           'same_output': a.actual_output == b.actual_output})
        summary['models'][name] = {'correct': sum(row.passed for row in rows.values()),
                                    'n': len(rows), 'pairs': paired}
    hard_cases = [EvalCase('wrong', 'expected', metadata={'stress_tests': ['ExactMatch']}),
                  EvalCase('crash', 'expected', metadata={'stress_tests': ['ExactMatch']}),
                  EvalCase('unconfigured', 'expected')]
    def baseline(text):
        if text == 'crash':
            raise RuntimeError('Deliberate unavailable-baseline control')
        return 'wrong answer'
    kept, reports = validate_adversarial_cases(hard_cases, baseline, n_shots=2)
    assert kept == [hard_cases[0]]
    assert [r.failure_rate for r in reports] == [1.0, None, None]
    save('hardness-controls.json', [asdict(r) for r in reports])
    save('summary.json', summary)
    save('checksums.json', {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(root.rglob('*')) if p.is_file()})
    print(canonical_json(summary))


if __name__ == '__main__':
    main()
