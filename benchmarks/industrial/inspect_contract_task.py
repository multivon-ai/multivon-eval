"""Native retry fixture: durable writes and a separately versioned business rule."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from inspect_ai import Task, task
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput
from inspect_ai.solver import solver

from multivon_eval import CaseManifest, EvalCase, ExactMatch, declare_dependencies
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.integrations.inspect import as_inspect_scorer, bind_inspect_task, to_inspect_dataset


class LedgerRequirement(Evaluator):
    name = 'ledger_requirement'

    def __init__(self, required, database):
        self.required = required
        self.database = str(database)

    def evaluate(self, case, output):
        with sqlite3.connect(self.database) as connection:
            row = connection.execute('SELECT amount FROM entries WHERE id=?', (case.case_id,)).fetchone()
        return self._result(float(row == (self.required[case.case_id],)), reason=f'Persisted amount: {row}')


@solver
def post_invoice(workdir: str):
    async def solve(state, generate):
        root = Path(workdir)
        with (root / 'calls.jsonl').open('a') as output:
            output.write(json.dumps({'case': state.sample_id}) + '\n')
        amount = state.metadata['multivon_case_v1']['case']['metadata']['amount']
        with sqlite3.connect(root / 'ledger.sqlite') as connection:
            connection.execute('INSERT OR IGNORE INTO entries VALUES (?, ?)', (state.sample_id, amount))
        if state.sample_id == 'c' and not (root / 'resume_allowed').exists():
            raise RuntimeError('Injected interruption after durable commit')
        state.output = ModelOutput.from_content('fixture', 'posted')
        return state
    return solve


@task
def contract_probe(workdir: str):
    root = Path(workdir)
    required = json.loads((root / 'policy.json').read_text())
    grader = declare_dependencies(LedgerRequirement(required, root / 'ledger.sqlite'),
        version='ledger-query/v1', dependencies={'schema': 'entries(id,amount)/v1'},
        files={'policy': root / 'policy.json'})
    manifest = CaseManifest('contract retry ledger', [
        EvalCase('Post once', 'posted', case_id=case_id, source_id=case_id, metadata={'amount': amount})
        for case_id, amount in [('a', 100), ('b', 200), ('c', 300)]
    ])
    native = Task(dataset=to_inspect_dataset(manifest), solver=post_invoice(workdir),
                  scorer=[as_inspect_scorer(ExactMatch()), as_inspect_scorer(grader)])
    guard = root / 'expected_log.txt'
    previous = read_eval_log(guard.read_text().strip()) if guard.exists() else None
    return bind_inspect_task(native, version='posting-task/v1',
        dependencies={'state_schema': 'entries(id,amount)/v1', 'replay': 'idempotent-insert/v1'},
        files={'policy': root / 'policy.json', 'task_source': Path(__file__)},
        configuration={'state': 'Runtime ledger rows and resume markers are mutable episode state'},
        previous_log=previous)
