"""Synthetic local ledger task for a real process-crash/retry experiment.

No model API calls. Inspect owns execution and recovery. SQLite supplies an
independent persisted outcome; unique idempotency keys make replay safe.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.solver import solver

from multivon_eval import CaseManifest, EvalCase, EvalResult, ExactMatch
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.integrations.inspect import as_inspect_scorer, to_inspect_dataset


def state_dir() -> Path:
    return Path(os.environ["MULTIVON_RECOVERY_WORKDIR"])


class LedgerInvariant(Evaluator):
    name = "ledger_invariant"

    def evaluate(self, case, output):
        with sqlite3.connect(state_dir() / "ledger.sqlite") as database:
            rows = database.execute("SELECT amount_cents FROM entries WHERE case_id = ?", (case.case_id,)).fetchall()
        valid = rows == [(case.metadata["amount_cents"],)]
        return EvalResult(self.name, float(valid), valid,
                          json.dumps({"entries": len(rows), "amounts_cents": [r[0] for r in rows]}))


@solver
def ledger_writer(mode: str):
    async def solve(state, generate):
        case_id = str(state.sample_id)
        workdir = state_dir()
        with (workdir / "attempts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"case_id": case_id, "pid": os.getpid()}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        key = case_id if mode == "idempotent" else str(uuid.uuid4())
        amount = state.metadata["multivon_case_v1"]["case"]["metadata"]["amount_cents"]
        with sqlite3.connect(workdir / "ledger.sqlite") as database:
            database.execute("INSERT OR IGNORE INTO entries VALUES (?, ?, ?)", (key, case_id, amount))
        # Crash after the side effect commits but before the solver returns.
        if case_id == "invoice-c" and not (workdir / "resume_allowed").exists():
            (workdir / "crash_ready").write_text("committed\n", encoding="utf-8")
            while not (workdir / "resume_allowed").exists():
                await asyncio.sleep(0.05)
        state.output = ModelOutput.from_content("fixture", "posted")
        state.messages.append(ChatMessageAssistant(content="posted"))
        return state
    return solve


@task
def recovery_probe(mode: str = "idempotent"):
    if mode not in {"idempotent", "unsafe"}:
        raise ValueError("mode must be idempotent or unsafe")
    manifest = CaseManifest("synthetic ledger recovery", [
        EvalCase("Post this approved invoice once", "posted", case_id=f"invoice-{name}",
                 source_id=f"document-{name}", metadata={"amount_cents": amount})
        for name, amount in [("a", 100), ("b", 200), ("c", 300)]
    ])
    return Task(dataset=to_inspect_dataset(manifest), solver=ledger_writer(mode),
                scorer=[as_inspect_scorer(ExactMatch()), as_inspect_scorer(LedgerInvariant())])


if __name__ == "__main__":
    import argparse

    from inspect_ai import eval as inspect_eval
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()
    inspect_eval(f"{Path(__file__).resolve()}@recovery_probe", task_args={"mode": args.mode},
                 model="mockllm/model", log_dir=args.log_dir, display="none", max_samples=1,
                 log_buffer=1, log_model_api=True)
