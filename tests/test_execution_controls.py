"""Side-effect prevention, actual concurrency and cancellation ownership."""
import asyncio
import threading

import pytest

from multivon_eval import EvalCase, EvalSuite, NotEmpty, ProviderJournal, capture_provider_events
from multivon_eval.evaluators.base import Evaluator


@pytest.mark.parametrize('name', ['runs', 'workers', 'concurrency', 'evaluator_concurrency'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5, '2'])
def test_invalid_counts_fail_before_preparation_or_target(name, value):
    calls = []
    class Prepared(NotEmpty):
        def prepare(self):
            calls.append('prepare')
    suite = EvalSuite('invalid').add_case(EvalCase('x')).add_evaluator(Prepared())
    def target(_):
        calls.append('target')
        return 'yes'
    with pytest.raises(ValueError, match=name):
        if name in ('runs', 'workers'):
            suite.run(target, verbose=False, **{name: value})
        else:
            asyncio.run(suite.run_async(target, verbose=False, **{name: value}))
    assert calls == []


@pytest.mark.parametrize('method', ['run', 'run_async', 'run_on_cases'])
@pytest.mark.parametrize('name', ['fail_threshold', 'max_error_rate'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1, True])
def test_gate_configuration_cannot_bypass_checks(method, name, value):
    suite = EvalSuite('invalid gate').add_case(EvalCase('x'))
    with pytest.raises(ValueError, match=name):
        options = {name: value, 'verbose': False}
        if method == 'run_async':
            asyncio.run(suite.run_async(lambda _: 'yes', **options))
        elif method == 'run_on_cases':
            suite.run_on_cases([(EvalCase('x'), 'yes')], **options)
        else:
            suite.run(lambda _: 'yes', **options)


@pytest.mark.asyncio
@pytest.mark.parametrize('origin', ['child', 'parent'])
async def test_target_cancellation_drains_owned_tasks_and_journal(tmp_path, origin):
    started, release = asyncio.Event(), asyncio.Event()
    active, finished = set(), set()
    async def target(prompt):
        active.add(prompt)
        if len(active) == 2:
            started.set()
        try:
            await started.wait()
            if prompt == 'a' and origin == 'child':
                raise asyncio.CancelledError('target cancelled itself')
            await release.wait()
            return 'late'
        finally:
            await asyncio.sleep(0)
            active.remove(prompt)
            finished.add(prompt)
    suite = EvalSuite('cancel').add_cases([EvalCase('a'), EvalCase('b')]).add_evaluator(NotEmpty())
    unrelated = asyncio.create_task(release.wait())
    try:
        with ProviderJournal(tmp_path / 'cancel.sqlite') as journal, capture_provider_events(journal=journal):
            run = asyncio.create_task(suite.run_async(target, verbose=False, concurrency=2))
            await asyncio.wait_for(started.wait(), 5)
            if origin == 'parent':
                run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run, 5)
            assert active == set() and finished == {'a', 'b'}
            assert not unrelated.done()
            closures = [e for e in journal.events() if e['kind'] == 'capture_finished']
            assert sorted(e['capture_kind'] for e in closures) == ['run', 'trial', 'trial']
            assert all(e['error']['type'] == 'CancelledError' for e in closures)
    finally:
        release.set()
        await unrelated


@pytest.mark.asyncio
async def test_evaluator_cancellation_drains_sibling_evaluators():
    ready, release = asyncio.Event(), asyncio.Event()
    active, finished = set(), set()
    class Child(Evaluator):
        def __init__(self, name):
            self.name = name
        def evaluate(self, *args):
            raise AssertionError('async required')
        async def aevaluate(self, *args):
            active.add(self.name)
            if len(active) == 2:
                ready.set()
            try:
                await ready.wait()
                if self.name == 'cancel':
                    raise asyncio.CancelledError()
                await release.wait()
            finally:
                await asyncio.sleep(0)
                active.remove(self.name)
                finished.add(self.name)
    async def target(_):
        return 'yes'
    suite = EvalSuite('cancel grader').add_case(EvalCase('x')).add_evaluators(Child('cancel'), Child('wait'))
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(suite.run_async(target, verbose=False), 5)
        assert not active and finished == {'cancel', 'wait'}
    finally:
        release.set()


@pytest.mark.asyncio
async def test_case_and_evaluator_limits_are_actual_run_wide_bounds():
    stats = {'targets': 0, 'target_peak': 0, 'graders': 0, 'grader_peak': 0}
    class Grader(Evaluator):
        def evaluate(self, *args):
            raise AssertionError('async required')
        async def aevaluate(self, *args):
            stats['graders'] += 1
            stats['grader_peak'] = max(stats['grader_peak'], stats['graders'])
            try:
                await asyncio.sleep(0.01)
                return self._result(1.0)
            finally:
                stats['graders'] -= 1
    async def target(_):
        stats['targets'] += 1
        stats['target_peak'] = max(stats['target_peak'], stats['targets'])
        try:
            await asyncio.sleep(0.01)
            return 'yes'
        finally:
            stats['targets'] -= 1
    suite = EvalSuite('bounded').add_cases([EvalCase(str(i)) for i in range(5)])
    suite.add_evaluators(Grader(), Grader(), Grader())
    report = await suite.run_async(target, verbose=False, concurrency=2, evaluator_concurrency=1, runs=2)
    assert report.passed == 5 and len(report.case_results[0].trials) == 2
    assert stats == {'targets': 0, 'target_peak': 2, 'graders': 0, 'grader_peak': 1}


@pytest.mark.asyncio
async def test_async_cancellation_does_not_claim_to_terminate_sync_threads():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    class Blocking(Evaluator):
        def evaluate(self, *args):
            started.set()
            try:
                release.wait(5)
                return self._result(1.0)
            finally:
                finished.set()
    async def target(_):
        return 'yes'
    suite = EvalSuite('thread boundary').add_case(EvalCase('x')).add_evaluator(Blocking())
    run = asyncio.create_task(suite.run_async(target, verbose=False))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        assert not finished.is_set()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
