"""Offline Inspect retry compatibility, native preservation and policy drift."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from inspect_ai import eval as inspect_eval, eval_retry
from inspect_ai.log import read_eval_log

import multivon_eval
from multivon_eval import AcceptancePolicy, CheckRequirement
from multivon_eval.integrations.inspect import from_inspect_log


def experiment(root: Path):
    root.mkdir(parents=True, exist_ok=False)
    initial_policy = {'a': 100, 'b': 200, 'c': 300}
    changed_policy = {**initial_policy, 'a': 999}
    policy_file = root / 'policy.json'
    policy_file.write_text(json.dumps(initial_policy))
    with sqlite3.connect(root / 'ledger.sqlite') as connection:
        connection.execute('CREATE TABLE entries (id TEXT PRIMARY KEY, amount INTEGER)')
    task_path = str(Path(__file__).with_name('inspect_contract_task.py')) + '@contract_probe'
    def calls():
        return [json.loads(row)['case'] for row in (root / 'calls.jsonl').read_text().splitlines()]
    options = {'display': 'none', 'max_samples': 1, 'log_buffer': 1, 'log_model_api': True}
    original = inspect_eval(task_path, task_args={'workdir': str(root)}, model='mockllm/model',
                            log_dir=str(root / 'original'), **options)[0]
    assert original.status == 'error' and calls() == ['a', 'b', 'c']
    assert [s.id for s in original.samples if s.error is None] == ['a', 'b']
    (root / 'resume_allowed').write_text('Explicit replay authorization for this local idempotent fixture')
    guard = root / 'expected_log.txt'
    guard.write_text(original.location)
    policy_file.write_text(json.dumps(changed_policy))
    blocked = None
    try:
        eval_retry(original.location, log_dir=str(root / 'blocked'), **options)
    except Exception as exc:
        blocked = {'type': type(exc).__name__, 'message': str(exc)}
    assert blocked and 'incompatible' in blocked['message'], blocked
    assert calls() == ['a', 'b', 'c'], 'Incompatible retry executed a target'
    policy_file.write_text(json.dumps(initial_policy))
    resumed = eval_retry(original.location, log_dir=str(root / 'compatible'), **options)[0]
    assert resumed.status == 'success' and calls() == ['a', 'b', 'c', 'c']
    original_ids = {s.id: s.uuid for s in original.samples if s.error is None}
    assert all(s.uuid == original_ids[s.id] for s in resumed.samples if s.id in original_ids)
    checks = (CheckRequirement('exact_match'), CheckRequirement('ledger_requirement', critical=True))
    policy = AcceptancePolicy(checks, min_cases=3)
    final = replace(policy, trial_scope='final_attempt')
    compatible = from_inspect_log(read_eval_log(resumed.location), previous_logs=[read_eval_log(original.location)])
    assert not compatible.evidence_issues, compatible.evidence_issues
    assert policy.evaluate(compatible).decision == 'indeterminate'
    assert final.evaluate(compatible).decision == 'accept'
    compatible.save_json(str(root / 'compatible-report.json'))
    # Negative control: deliberately omit preflight and let native retry preserve
    # old successful scores despite a changed business rule.
    guard.unlink()
    policy_file.write_text(json.dumps(changed_policy))
    mixed = eval_retry(original.location, log_dir=str(root / 'unguarded'), **options)[0]
    assert mixed.status == 'success' and calls() == ['a', 'b', 'c', 'c', 'c']
    assert all(score.value['passed'] for sample in mixed.samples for score in sample.scores.values())
    mixed_report = from_inspect_log(read_eval_log(mixed.location))
    assert mixed_report.evidence_issues
    assert final.evaluate(mixed_report).decision == 'indeterminate'
    mixed_report.save_json(str(root / 'mixed-report.json'))
    fresh = inspect_eval(task_path, task_args={'workdir': str(root)}, model='mockllm/model',
                         log_dir=str(root / 'fresh'), **options)[0]
    assert fresh.status == 'success'
    fresh_report = from_inspect_log(read_eval_log(fresh.location))
    assert not fresh_report.evidence_issues, fresh_report.evidence_issues
    assert policy.evaluate(fresh_report).decision == 'reject'
    fresh_report.save_json(str(root / 'fresh-report.json'))
    assert calls() == ['a', 'b', 'c', 'c', 'c', 'a', 'b', 'c']
    with sqlite3.connect(root / 'ledger.sqlite') as connection:
        rows = dict(connection.execute('SELECT id,amount FROM entries'))
    assert rows == initial_policy
    result = {'cases': 3, 'initial_policy': initial_policy, 'changed_policy': changed_policy,
              'blocked_retry': blocked, 'blocked_target_calls': 0, 'preserved_completed_ids': sorted(original_ids),
              'compatible_all_attempts': policy.evaluate(compatible).decision,
              'compatible_final_attempt': final.evaluate(compatible).decision,
              'mixed_native_passed': 3, 'mixed_import_decision': final.evaluate(mixed_report).decision,
              'mixed_issues': mixed_report.evidence_issues, 'fresh_changed_policy': policy.evaluate(fresh_report).decision,
              'fresh_passed': fresh_report.passed, 'persisted_rows': rows, 'all_target_invocations': calls(),
              'logs': {name: str(Path(log.location).relative_to(root)) for name, log in
                       [('original', original), ('compatible', resumed), ('mixed', mixed), ('fresh', fresh)]},
              'limits': 'Three synthetic local cases; declaration/file/metadata checks, not remote state or checkpoint authenticity'}
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repository, text=True).strip():
        raise RuntimeError('Freeze a clean source commit before the measured experiment')
    protocol = {'schema': 'multivon.inspect-retry-contract/v1',
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
                'python': sys.version, 'inspect': importlib.metadata.version('inspect-ai'),
                'core_module': multivon_eval.__version__, 'core_distribution': importlib.metadata.version('multivon-eval'),
                'scenarios': ['original interruption', 'changed guard rejection', 'compatible retry',
                              'unguarded mixed retry', 'fresh changed-policy run'], 'api_requests': 0}
    import socket
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError('No network is allowed in this offline experiment')
    socket.socket.connect = forbidden
    root = args.output.resolve()
    # Preserve the frozen protocol even if a later assertion fails.
    root.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = root.with_suffix('.protocol.json')
    with protocol_path.open('x') as output:
        json.dump(protocol, output, indent=2)
    result = experiment(root)
    assert not attempts
    result['attempted_network_connections'] = len(attempts)
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
