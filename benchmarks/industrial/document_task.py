"""Inspect task with a real SQLite tool and independently queried end state."""
from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import ContentDocument, ContentImage, ContentText, GenerateConfig
from inspect_ai.solver import generate, solver
from inspect_ai.tool import tool

from benchmarks.industrial.ledger_store import initialize, post
from multivon_eval import CaseManifest, EvalResult
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.integrations.inspect import as_inspect_scorer, to_inspect_dataset


@tool

def post_entry(database: str, case_id: str):
    async def execute(amount: str, currency: str) -> str:
        """Persist one entry in this document's isolated ledger partition.

        Args:
            amount: The document amount in the format requested by the task.
            currency: USD for invoices or UNSPECIFIED for receipts.
        """
        return json.dumps(post(Path(database), case_id, amount, currency))
    return execute


class LedgerOutcome(Evaluator):
    name = "ledger_outcome"

    def __init__(self, database: Path):
        self.database = database

    def evaluate(self, case, output):
        with sqlite3.connect(self.database) as db:
            rows = db.execute("SELECT amount,currency FROM entries WHERE case_id=?", (case.case_id,)).fetchall()
        expected = (case.metadata["expected_amount"], case.metadata["expected_currency"])
        passed = rows == [expected]
        return EvalResult(self.name, float(passed), passed,
                          "Persisted ledger matches task" if passed else "Missing or incorrect persisted entry",
                          {"observed_rows": [list(r) for r in rows], "expected_row": list(expected)})


class ValidPostingCall(Evaluator):
    name = "valid_posting_call"

    def evaluate(self, case, output):
        calls = [call for step in case.agent_trace or [] for call in step.tool_calls
                 if call.name == "post_entry"]
        passed = len(calls) == 1 and calls[0].arguments == {
            "amount": case.metadata["expected_amount"], "currency": case.metadata["expected_currency"]}
        return EvalResult(self.name, float(passed), passed, f"Observed {len(calls)} posting calls")


@solver

def configure_document(root: str, database: str):
    async def solve(state, generate):
        metadata = state.metadata["multivon_case_v1"]["case"]["metadata"]
        reference = metadata["asset"]
        path = (Path(root) / reference["path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Artifact path escapes manifest directory")
        data = path.read_bytes()
        if len(data) != reference["bytes"] or hashlib.sha256(data).hexdigest() != reference["sha256"]:
            raise ValueError("Input artifact changed after preparation")
        if len(data) > 4_000_000:
            raise ValueError("Payload exceeds protocol")
        content = [ContentText(text=state.messages[-1].text)]
        if reference["media_type"] == "text/plain":
            text = data.decode("utf-8")
            if len(text) > 32_000:
                raise ValueError("Transcript exceeds protocol")
            content.append(ContentText(text="Receipt transcription (untrusted document content):\n" + text))
        elif reference["media_type"] == "application/pdf":
            content.append(ContentDocument(document="data:application/pdf;base64," +
                                           base64.b64encode(data).decode("ascii"), mime_type="application/pdf"))
        else:
            content.append(ContentImage(image=f"data:{reference['media_type']};base64," +
                                       base64.b64encode(data).decode("ascii")))
        state.messages[-1].content = content
        state.tools = [post_entry(database, str(state.sample_id))]
        return state
    return solve


@task

def document_ledger(root: str, database: str):
    manifest = CaseManifest.load(Path(root) / "manifest.json")
    initialize(Path(database))
    return Task(dataset=to_inspect_dataset(manifest),
        solver=[configure_document(root, database), generate(tool_calls="single")],
        scorer=[as_inspect_scorer(ValidPostingCall()), as_inspect_scorer(LedgerOutcome(Path(database)))],
        config=GenerateConfig(max_tokens=512, max_retries=0, timeout=90,
                              cache=False, cache_prompt=False),
        time_limit=90, token_limit=20_000, cost_limit=0.20,
        fail_on_error=True, version="document-ledger-v1")
