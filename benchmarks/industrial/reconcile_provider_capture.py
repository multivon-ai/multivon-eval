"""Offline repricing of the frozen four-call capture; makes no provider calls."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tarfile
from pathlib import Path

from multivon_eval import EvalReport, account_provider_events
from multivon_eval.integrations.litellm_pricing import LiteLLMPricer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('Freeze a clean checkout before repricing')
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    # Enforce offline estimation, including accidental upstream network lookups.
    import socket
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError('This experiment prohibits network access')
    socket.socket.connect = forbidden
    import litellm
    litellm.telemetry = False
    source = root / 'benchmarks/industrial/results/provider-capture-2026-09-17/evidence.tar.gz'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    with tarfile.open(source) as archive:
        events = json.load(archive.extractfile('events.json'))
        report = EvalReport.from_dict(json.load(archive.extractfile('report.json')))
    report.costs = account_provider_events(events, price_estimator=LiteLLMPricer(litellm),
        coverage_declaration='Frozen four-call protocol: native target and sync/async judge calls are all journaled')
    assert report.costs.complete and report.costs.total_calls == 4 and report.costs.total_tokens == 68
    assert report.costs.total_cost_usd is not None
    assert not attempts, 'Upstream initialization or pricing attempted network access'
    report.assert_budget(max_total_tokens=68, max_total_cost_usd=0.001)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'accounting.json').write_text(json.dumps(report.costs.to_dict(), indent=2) + '\n')
    summary = {
        'source_archive_sha256': source_sha,
        'reconciliation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'litellm_version': importlib.metadata.version('litellm'),
        'calls': report.costs.total_calls, 'input_tokens': report.costs.total_input_tokens,
        'output_tokens': report.costs.total_output_tokens, 'estimate_usd': report.costs.total_cost_usd,
        'network_access': 'Blocked before LiteLLM import; zero attempted connections',
        'limits': 'Current upstream list-price estimate for archived responses, not historical billed amount',
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
