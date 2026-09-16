"""Kill and resume Inspect with an ambiguous committed side effect.

Run in a fresh output directory using Python with multivon-eval[inspect].
Writes native logs, ledger state, trial reports and acceptance decisions.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from inspect_ai import eval_retry
from inspect_ai.log import recover_eval_log

from multivon_eval import AcceptancePolicy, CheckRequirement
from multivon_eval.integrations.inspect import from_inspect_log


def experiment(mode: str, root: Path) -> dict:
    workdir = root / mode
    workdir.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(workdir / "ledger.sqlite") as database:
        database.execute("CREATE TABLE entries (idempotency_key TEXT PRIMARY KEY, case_id TEXT, amount_cents INTEGER)")
    environment = dict(os.environ, MULTIVON_RECOVERY_WORKDIR=str(workdir))
    task_file = Path(__file__).with_name("inspect_recovery_task.py")
    original_logs = workdir / "original"
    with (workdir / "process.log").open("w", encoding="utf-8") as output:
        child = subprocess.Popen([sys.executable, str(task_file), "--mode", mode,
                                  "--log-dir", str(original_logs)], env=environment,
                                 stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while not (workdir / "crash_ready").exists():
                if child.poll() is not None:
                    raise RuntimeError(f"Probe exited before crash point; see {workdir / 'process.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Probe did not reach crash point in 45 seconds")
                time.sleep(0.05)
            child.kill()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    logs = list(original_logs.glob("*.eval"))
    if len(logs) != 1:
        raise RuntimeError(f"Expected one original native log, found {len(logs)}")
    recovered = recover_eval_log(str(logs[0]), cleanup=False, incomplete_action="retry")
    completed = sorted(str(sample.id) for sample in recovered.samples or [] if sample.error is None)
    if completed != ["invoice-a", "invoice-b"]:
        raise AssertionError(f"Unexpected completed samples after crash: {completed}")
    before = from_inspect_log(recovered)
    contract = AcceptancePolicy((CheckRequirement("exact_match"),
                                 CheckRequirement("ledger_invariant", critical=True)),
                                min_cases=3, min_source_groups=3)
    if contract.evaluate(before).decision != "indeterminate":
        raise AssertionError("Interrupted log was not blocked")
    (workdir / "resume_allowed").write_text("allow replay: local fixture only\n", encoding="utf-8")
    previous = os.environ.get("MULTIVON_RECOVERY_WORKDIR")
    os.environ["MULTIVON_RECOVERY_WORKDIR"] = str(workdir)
    try:
        resumed = eval_retry(recovered.location, log_dir=str(workdir / "resumed"),
                             display="none", max_samples=1, log_buffer=1)[0]
    finally:
        if previous is None:
            os.environ.pop("MULTIVON_RECOVERY_WORKDIR", None)
        else:
            os.environ["MULTIVON_RECOVERY_WORKDIR"] = previous
    if resumed.status != "success":
        raise AssertionError(f"Native retry did not finish: {resumed.status}")
    report = from_inspect_log(resumed, previous_logs=[recovered])
    report.save_json(str(workdir / "report.json"))
    all_attempts_decision = contract.evaluate(report)
    if all_attempts_decision.decision not in {"indeterminate", "reject"}:
        raise AssertionError("Interrupted attempt disappeared from the acceptance evidence")
    # This experiment explicitly permits replay. The critical persisted-state
    # assertion must still reject an unsafe duplicate after recovery.
    from dataclasses import replace
    decision = replace(contract, trial_scope="final_attempt").evaluate(report)
    (workdir / "decision.json").write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
    calls = Counter(json.loads(line)["case_id"] for line in (workdir / "attempts.jsonl").read_text().splitlines())
    with sqlite3.connect(workdir / "ledger.sqlite") as database:
        entries = dict(database.execute("SELECT case_id, COUNT(*) FROM entries GROUP BY case_id").fetchall())
    if calls != {"invoice-a": 1, "invoice-b": 1, "invoice-c": 2}:
        raise AssertionError(f"Completed samples were rerun or interrupted sample was lost: {calls}")
    expected_entries = {"invoice-a": 1, "invoice-b": 1, "invoice-c": 1 if mode == "idempotent" else 2}
    if entries != expected_entries:
        raise AssertionError(f"Unexpected ledger state: {entries}")
    expected_decision = "accept" if mode == "idempotent" else "reject"
    if decision.decision != expected_decision:
        raise AssertionError(f"Expected {expected_decision}, got {decision.decision}")
    return {"mode": mode, "samples": 3, "completed_samples_preserved": completed,
            "target_invocations": dict(calls), "persisted_entries": entries,
            "decision": decision.decision, "interrupted_decision": "indeterminate",
            "all_attempts_decision": all_attempts_decision.decision,
            "retained_execution_count": sum(len(row.trials) for row in report.case_results),
            "api_calls": 0, "api_spend_usd": 0,
            "native_log": resumed.location, "note": "Synthetic local fault injection, not customer validation"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    outcomes = [experiment(mode, root) for mode in ("idempotent", "unsafe")]
    artifact = {"schema": "multivon.recovery-experiment/v1", "outcomes": outcomes,
                "claim": "This fixture tests sample preservation and a specific duplicate-write failure mode"}
    (root / "summary.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
