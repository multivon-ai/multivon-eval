"""Verify hashes and regrade the saved local robustness fixture without target calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from multivon_eval import CaseManifest, EvalReport, ExactMatch
from multivon_eval.case_manifest import canonical_json
from multivon_eval.robustness import VariantValidation


def replay(root: Path) -> dict:
    root = root.resolve()
    def read(name):
        return json.loads((root / name).read_text())
    checksums = read('checksums.json')
    for name, expected in checksums.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing file or checksum path outside bundle')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    summary = read('summary.json')
    validations = [VariantValidation(canonical_json(r)) for r in read('validations.json')]
    assert dict(Counter(v.status for v in validations)) == summary['validation_counts']
    manifest = CaseManifest.load(root / 'manifest.json')
    cases = {c.case_id: c for c in manifest.cases}
    reconstructed = {}
    for validation in validations:
        if validation.status == 'valid':
            for case in validation.manifest('replay pair').cases:
                reconstructed[case.case_id] = case.identity()
    assert reconstructed == {key: case.identity() for key, case in cases.items()}
    assert len({c.source_id for c in cases.values()}) == summary['sources']
    results = {}
    for name, expected in summary['models'].items():
        report = EvalReport.from_dict(read(f'{name}.json'))
        rows = {r.case_id: r for r in report.case_results}
        assert set(rows) == set(cases)
        measured = 0
        for key, row in rows.items():
            assert row.case_digest == cases[key].identity()[1]
            assert row.status in {'passed', 'failed_quality'}
            verdict = ExactMatch().evaluate(cases[key], row.actual_output)
            assert verdict.passed == row.passed
            assert len(row.trials) == 1 and row.trials[0].data['output'] == row.actual_output
            measured += verdict.passed
        assert measured == expected['correct'] and len(rows) == expected['n']
        for pair in expected['pairs']:
            a, b = rows[pair['base']], rows[pair['variant']]
            assert (a.passed, b.passed, a.actual_output == b.actual_output) == (
                pair['base_correct'], pair['variant_correct'], pair['same_output'])
        results[name] = {'correct': measured, 'n': len(rows)}
    hardness = read('hardness-controls.json')
    assert [r['failure_rate'] for r in hardness] == [1.0, None, None]
    assert [r['in_hardness_band'] for r in hardness] == [True, False, False]
    return {'verified_files': len(checksums), 'sources': summary['sources'],
            'validation_counts': summary['validation_counts'], 'results': results,
            'target_calls': 0, 'limits': 'Regrades saved outputs; does not independently prove oracle truth.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(replay(parser.parse_args().root), indent=2))
