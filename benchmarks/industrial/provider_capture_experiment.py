"""Four intended live calls; transport-evidence smoke study, not an accuracy test.

Run with ANTHROPIC_API_KEY already in the environment. Never reads key files.
Uses the native SDK's default retry policy, so actual attempts may exceed four.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import multivon_eval
from multivon_eval import (
    AnthropicAdapter, EvalCase, EvalReport, EvalSuite, ExactMatch, JudgeConfig,
    ProviderJournal, capture_provider_events, regrade,
)
from multivon_eval.judge import make_judge_call, make_judge_call_async


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise RuntimeError('ANTHROPIC_API_KEY is required')
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('Freeze a clean checkout before executing this protocol')
    protocol = {
        'purpose': 'Verify native provider evidence against real endpoint, not measure accuracy',
        'revision': revision, 'python': platform.python_version(),
        'module_version': multivon_eval.__version__,
        'packages': {name: importlib.metadata.version(name) for name in ['anthropic', 'openai', 'multivon-eval']},
        'model': 'claude-haiku-4-5', 'temperature': 0.2, 'max_tokens': 16, 'timeout_seconds': 30,
        'intended_calls': {'parallel_targets': 2, 'sync_judge': 1, 'async_judge': 1},
        'retry_policy': 'Native SDK default; every observed HTTP attempt retained',
        'pricing': 'No dollar estimate; raw provider usage is not an invoice',
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    config = JudgeConfig(provider='anthropic', model=protocol['model'], temperature=0.2,
                         max_tokens=16, timeout=30).resolve()
    outcomes = {}
    with ProviderJournal(args.output / 'events.sqlite') as journal:
        with capture_provider_events(journal=journal, labels={'experiment': 'provider-capture-live-v1'}):
            suite = EvalSuite('live evidence smoke').add_cases([
                EvalCase('Reply with exactly Yes.', 'Yes', case_id='target-a'),
                EvalCase('Reply with exactly No.', 'No', case_id='target-b'),
            ]).add_evaluator(ExactMatch())
            report = suite.run(AnthropicAdapter(protocol['model'], temperature=0.2, max_tokens=16,
                                               timeout=30), workers=2, verbose=False)
            report.save_json(str(args.output / 'report.json'))
            restored = EvalReport.from_dict(json.loads(report.to_json()))
            before = sum(e['kind'] == 'http_request' for e in journal.events())
            reviewed = regrade(restored, EvalSuite('saved output check').add_evaluator(ExactMatch()))
            reviewed.save_json(str(args.output / 'regraded.json'))
            outcomes['regrade_new_requests'] = sum(e['kind'] == 'http_request' for e in journal.events()) - before
            for name, call in [
                ('sync_judge', lambda: make_judge_call('Reply with exactly Yes.', config)),
                ('async_judge', lambda: asyncio.run(make_judge_call_async('Reply with exactly Yes.', config))),
            ]:
                try:
                    outcomes[name] = {'reply': call()}
                except Exception as exc:
                    outcomes[name] = {'error_type': type(exc).__name__, 'message': str(exc)}
        events = journal.events()
    (args.output / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
    requests = [e for e in events if e['kind'] == 'http_request']
    responses = [e for e in events if e['kind'] == 'http_response']
    responded = {e['request_id'] for e in responses}
    outcomes.update({
        'http_attempts': len(requests), 'http_responses': len(responses),
        'response_statuses': [e['response']['status_code'] for e in responses],
        'responses_with_usage': sum(e['response']['usage'] is not None for e in responses),
        'requests_without_response': [e['request_id'] for e in requests if e['request_id'] not in responded],
        'request_temperatures': [e['request']['body']['value'].get('temperature') for e in requests],
        'usage': [e['response']['usage'] for e in responses],
        'case_statuses': [r.status.value for r in report.case_results],
    })
    (args.output / 'outcomes.json').write_text(json.dumps(outcomes, indent=2) + '\n')
    secret = os.environ['ANTHROPIC_API_KEY'].encode()
    for path in args.output.iterdir():
        if path.is_file() and secret in path.read_bytes():
            raise RuntimeError('Credential detected in evidence artifact; do not publish')
    print(json.dumps({k: outcomes[k] for k in ['http_attempts', 'http_responses', 'responses_with_usage',
                                             'regrade_new_requests', 'response_statuses']}, indent=2))


if __name__ == '__main__':
    main()
