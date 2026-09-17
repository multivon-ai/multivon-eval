"""Offline native Inspect limits, concurrency and cancellation experiment.

Uses upstream mockllm with declared synthetic usage; no provider accuracy or
billing claim. Run from a clean committed checkout with --output DIR.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import sqlite3

import multivon_eval

from inspect_ai import Task, eval_async
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant, ModelCost, ModelInfo, ModelOutput, ModelUsage, get_model, set_model_info
from inspect_ai.solver import solver

from multivon_eval import AcceptancePolicy, CaseManifest, CheckRequirement, EvalCase, ExactMatch
from multivon_eval.integrations.inspect import as_inspect_scorer, from_inspect_log, to_inspect_dataset
from multivon_eval.evaluators.base import Evaluator


def dataset(count):
    return to_inspect_dataset(CaseManifest('execution control fixtures', [
        EvalCase(f'Post local invoice {i}', 'posted', case_id=f'invoice-{i}', source_id=f'source-{i}')
        for i in range(count)
    ]))


async def limit_probe(root: Path, kind: str):
    work = root / kind
    work.mkdir(parents=True, exist_ok=True)
    database = work / 'ledger.sqlite'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE entries (invoice TEXT PRIMARY KEY)')
    class LedgerCommitted(Evaluator):
        name = 'ledger_committed'
        def evaluate(self, case, output):
            with sqlite3.connect(database) as connection:
                count = connection.execute('SELECT COUNT(*) FROM entries WHERE invoice=?', (case.case_id,)).fetchone()[0]
            return self._result(float(count == 1), reason=f'Independent persisted row count: {count}')
    stats = {'model_calls': 0, 'solver_cleanup': False}
    async def response(*args):
        stats['model_calls'] += 1
        answer = ModelOutput.from_content('controls', 'posted')
        answer.usage = ModelUsage(input_tokens=4, output_tokens=6, total_tokens=10)
        return answer
    @solver
    def incomplete_post():
        async def solve(state, generate):
            # The acknowledgment alone is not proof of a committed transaction.
            state.output = ModelOutput.from_content('fixture', 'posted')
            try:
                if kind in ('time', 'working'):
                    await asyncio.Event().wait()
                elif kind == 'message':
                    for _ in range(5):
                        state.messages.append(ChatMessageAssistant(content='posted'))
                elif kind != 'complete':
                    for _ in range(5):
                        state = await generate(state)
                with sqlite3.connect(database) as connection:
                    connection.execute('INSERT INTO entries VALUES (?)', (state.sample_id,))
                return state
            finally:
                stats['solver_cleanup'] = True
        return solve
    limits = {'time': {'time_limit': 1}, 'working': {'working_limit': 1},
              'message': {'message_limit': 2}, 'token': {'token_limit': 8},
              'cost': {'cost_limit': 0.000008}, 'turn': {'turn_limit': 1}, 'complete': {}}[kind]
    task = Task(dataset=dataset(1), solver=incomplete_post(),
                scorer=[as_inspect_scorer(ExactMatch()), as_inspect_scorer(LedgerCommitted())], **limits)
    set_model_info('mockllm/controls', ModelInfo(context_length=4096))
    model = get_model('mockllm/controls', custom_outputs=response)
    log = (await eval_async(task, model=model, log_dir=str(root / kind),
                            max_samples=1, max_connections=1, log_model_api=True,
                            model_cost_config={'mockllm/controls': ModelCost(
                                input=1, output=1, input_cache_write=1, input_cache_read=1)}))[0]
    native = read_eval_log(log.location)
    assert native.status == 'success'
    assert (native.samples[0].limit.type if native.samples[0].limit else 'complete') == kind
    with sqlite3.connect(database) as connection:
        stats['persisted_entries'] = connection.execute('SELECT COUNT(*) FROM entries').fetchone()[0]
    assert stats['solver_cleanup'] and stats['persisted_entries'] == int(kind == 'complete')
    policy = AcceptancePolicy((CheckRequirement('exact_match'),))
    strict = from_inspect_log(native)
    bounded = from_inspect_log(native, accepted_limits=(kind,) if kind != 'complete' else ())
    assert policy.evaluate(strict).decision == ('accept' if kind == 'complete' else 'indeterminate')
    relaxed_errors = AcceptancePolicy((CheckRequirement('exact_match'),), max_error_rate=1)
    assert relaxed_errors.evaluate(strict).decision == policy.evaluate(strict).decision
    assert policy.evaluate(bounded).decision == 'accept'
    assert bounded.case_results[0].results[0].passed
    outcome = AcceptancePolicy((CheckRequirement('exact_match'), CheckRequirement('ledger_committed', critical=True)))
    assert outcome.evaluate(bounded).decision == ('accept' if kind == 'complete' else 'reject')
    strict.save_json(str(root / kind / 'strict-report.json'))
    bounded.save_json(str(root / kind / 'bounded-report.json'))
    return {'kind': kind, 'native_status': native.status,
            'limit': native.samples[0].limit.model_dump(mode='json') if native.samples[0].limit else None,
            'strict_decision': policy.evaluate(strict).decision,
            'relaxed_error_budget_decision': relaxed_errors.evaluate(strict).decision,
            'declared_boundary_decision': policy.evaluate(bounded).decision,
            'outcome_decision': outcome.evaluate(bounded).decision,
            'native_usage': {k: v.model_dump(mode='json') for k, v in native.samples[0].model_usage.items()},
            **stats, 'log': str(Path(log.location).relative_to(root)),
            'critique': 'The text check alone accepts all outputs; the independent SQLite query distinguishes missing commits'}


async def concurrency_probe(root: Path):
    stats = {'solvers': 0, 'solver_peak': 0, 'models': 0, 'model_peak': 0, 'model_calls': 0}
    async def response(*args):
        stats['models'] += 1
        stats['model_calls'] += 1
        stats['model_peak'] = max(stats['model_peak'], stats['models'])
        try:
            await asyncio.sleep(0.02)
            output = ModelOutput.from_content('controls', 'posted')
            output.usage = ModelUsage(input_tokens=4, output_tokens=6, total_tokens=10)
            return output
        finally:
            stats['models'] -= 1
    @solver
    def bounded_solver():
        async def solve(state, generate):
            stats['solvers'] += 1
            stats['solver_peak'] = max(stats['solver_peak'], stats['solvers'])
            try:
                return await generate(state)
            finally:
                stats['solvers'] -= 1
        return solve
    task = Task(dataset=dataset(6), solver=bounded_solver(), scorer=as_inspect_scorer(ExactMatch()))
    log = (await eval_async(task, model=get_model('mockllm/controls', custom_outputs=response),
                            log_dir=str(root / 'concurrency'),
                            max_samples=2, max_connections=1, log_model_api=True))[0]
    assert log.status == 'success' and len(log.samples) == 6
    assert stats == {'solvers': 0, 'solver_peak': 2, 'models': 0, 'model_peak': 1, 'model_calls': 6}
    return {**stats, 'cases': 6, 'max_samples': 2, 'max_connections': 1,
            'log': str(Path(log.location).relative_to(root))}


async def cancellation_probe(root: Path):
    ready, release = asyncio.Event(), asyncio.Event()
    stats = {'started': 0, 'active': 0, 'cleaned': 0}
    async def response(*args):
        stats['started'] += 1
        stats['active'] += 1
        if stats['started'] == 2:
            ready.set()
        try:
            await release.wait()
            return ModelOutput.from_content('controls', 'posted')
        finally:
            stats['active'] -= 1
            stats['cleaned'] += 1
    @solver
    def cancellable():
        async def solve(state, generate):
            return await generate(state)
        return solve
    task = Task(dataset=dataset(3), solver=cancellable(), scorer=as_inspect_scorer(ExactMatch()))
    run = asyncio.create_task(eval_async(task, model=get_model('mockllm/controls', custom_outputs=response),
                                        log_dir=str(root / 'cancellation'),
                                        max_samples=2, max_connections=2, log_buffer=1))
    propagated = False
    try:
        await asyncio.wait_for(ready.wait(), 10)
        run.cancel()
        try:
            await asyncio.wait_for(run, 10)
        except asyncio.CancelledError:
            propagated = True
        assert stats == {'started': 2, 'active': 0, 'cleaned': 2}
    finally:
        release.set()
        if not run.done():
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)
    files = list((root / 'cancellation').glob('*.eval'))
    assert len(files) == 1
    log = read_eval_log(str(files[0]))
    assert log.status == 'cancelled'
    report = from_inspect_log(log)
    assert AcceptancePolicy((CheckRequirement('exact_match'),)).evaluate(report).decision == 'indeterminate'
    return {**stats, 'cancelled_error_propagated': propagated, 'native_status': log.status,
            'recorded_samples': len(log.samples), 'log': str(files[0].relative_to(root))}


async def experiment(root):
    return {'limits': [await limit_probe(root, kind) for kind in ('complete', 'time', 'working', 'message', 'token', 'cost', 'turn')],
            'concurrency': await concurrency_probe(root), 'cancellation': await cancellation_probe(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repository, text=True).strip():
        raise RuntimeError('Commit a clean checkout before the measured experiment')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = {'schema': 'multivon.execution-controls/v1',
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
                'python': sys.version, 'inspect': importlib.metadata.version('inspect-ai'),
                'core_distribution': importlib.metadata.version('multivon-eval'),
                'core_module': multivon_eval.__version__,
                'api_requests': 0, 'usage': 'Synthetic mockllm usage only; not provider usage or spend',
                'scenarios': ['complete', 'time', 'working', 'message', 'token', 'cost', 'turn', 'concurrency', 'cancellation']}
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    # No SDK provider or remote service is needed, including during evaluation.
    import socket
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError('Network disabled for this offline experiment')
    socket.socket.connect = forbidden
    results = asyncio.run(experiment(root))
    results['attempted_network_connections'] = len(attempts)
    assert not attempts
    (root / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
