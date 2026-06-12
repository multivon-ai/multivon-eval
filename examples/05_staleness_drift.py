"""Prompt-drift staleness: catch the eval your last prompt edit invalidated.

Purpose:       Show the staleness loop end-to-end — baseline a repo's prompt
               call sites, edit a prompt, and watch the report name exactly
               which case went stale. Everything runs against a throwaway
               mini-repo created in a temp directory.
Runtime:       ~5s. Cost: $0.00 — staleness is pure static analysis, no LLM.
Output shape:  The real `multivon-eval staleness` text report twice: clean
               after baselining, then CHANGED (with old → new fingerprints
               and the bound case) after the edit. Exits 0; the final run
               demonstrates `--fail-on changed` exiting 1 the way CI would.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

APP = '''\
import anthropic

SYSTEM = "You are a careful insurance assistant. Cite the policy section for every claim."

def answer(client: anthropic.Anthropic, question: str) -> str:
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return msg.content[0].text
'''

CASES = (
    '{"input": "What does section 4.2 cover?", '
    '"expected_output": "rental car reimbursement", "metadata": {}}\n'
)


def staleness(repo: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "multivon_eval", "staleness", str(repo),
           "--cases", str(repo / "cases.jsonl"), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="staleness-demo-") as tmp:
        repo = Path(tmp)
        (repo / "app.py").write_text(APP)
        (repo / "cases.jsonl").write_text(CASES)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=demo", "-c",
             "user.email=demo@example.com", "commit", "-q", "--allow-empty",
             "-m", "init"],
            check=True,
        )

        # 1. Baseline: snapshot every prompt call site the scanner can read.
        out = subprocess.run(
            [sys.executable, "-m", "multivon_eval", "staleness", "baseline", str(repo)],
            capture_output=True, text=True,
        )
        print(out.stdout.strip())

        # 2. Bind our case to the call site it exercises, then a clean report.
        stamped = subprocess.run(
            [sys.executable, "-m", "multivon_eval", "staleness", "stamp",
             "--repo", str(repo), "--cases", str(repo / "cases.jsonl"),
             "--site", "app.py::answer.system", "--index", "0"],
            capture_output=True, text=True,
        )
        print(stamped.stdout.strip() or stamped.stderr.strip())
        print("\n── report, before any edit ──")
        print(staleness(repo).stdout)

        # 3. The thing that happens in every real repo: someone edits the prompt.
        (repo / "app.py").write_text(
            APP.replace("Cite the policy section for every claim.",
                        "Be brief; cite sections only when asked.")
        )

        print("── report, after the prompt edit ──")
        print(staleness(repo).stdout)

        # 4. In CI you'd gate on it. Exit code 1 = drift found in a gated category.
        gated = staleness(repo, "--fail-on", "changed")
        print(f"── with --fail-on changed: exit {gated.returncode} "
              f"(this is the CI gate) ──")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
