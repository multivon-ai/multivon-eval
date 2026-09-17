"""Verify the committed media bundle and re-import native logs without target calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from multivon_eval import AcceptancePolicy
from multivon_eval.integrations.inspect import from_inspect_log
from multivon_eval.media import MediaArtifact


def replay(root: Path) -> dict:
    root = root.resolve()
    checksums = json.loads((root / 'checksums.json').read_text())
    for name, expected in checksums.items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError('Checksum path is missing or outside the bundle')
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    for item in json.loads((root / 'assets.json').read_text()).values():
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Asset path escapes bundle')
        MediaArtifact.from_dict(item['artifact']).verify(path.read_bytes())
    expected = json.loads((root / 'summary.json').read_text())
    policy = AcceptancePolicy.from_dict(json.loads((root / 'policy.json').read_text()))
    results = {}
    for path in sorted((root / 'logs').glob('*.eval')):
        log = read_eval_log(path, resolve_attachments=True)
        report = from_inspect_log(log)
        name = 'documents' if log.eval.model.startswith('anthropic/') else 'transport'
        observed = [{'case_id': r.case_id, 'output': r.actual_output, 'passed': r.passed} for r in report.case_results]
        if observed != expected[name]['outputs']:
            raise ValueError('Re-imported outputs differ from archived summary')
        results[name] = {'cases': report.total, 'passed': report.passed, 'errors': report.errors}
        if name == 'documents':
            results[name]['decision'] = policy.evaluate(report).decision
    if set(results) != {'documents', 'transport'}:
        raise ValueError('Missing expected native logs')
    return {'verified_files': len(checksums), 'results': results, 'target_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(replay(parser.parse_args().root), indent=2))
