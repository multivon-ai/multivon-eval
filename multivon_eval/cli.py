"""
multivon-eval CLI — run eval files and view reports.

Usage:
    multivon-eval run eval.py
    multivon-eval report results.json
"""
from __future__ import annotations
import sys
import json
import argparse


def cmd_run(args):
    """Execute a Python eval file."""
    import runpy
    runpy.run_path(args.file, run_name="__main__")


def cmd_report(args):
    """Pretty-print a saved JSON report."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    with open(args.file) as f:
        data = json.load(f)

    console.rule(f"[bold]{data.get('suite', 'Eval Report')}[/]")
    if data.get("model"):
        console.print(f"  Model: [dim]{data['model']}[/]")

    summary = data.get("summary", {})
    console.print(f"\n  Total: {summary.get('total')}  "
                  f"Passed: [green]{summary.get('passed')}[/]  "
                  f"Failed: [red]{summary.get('failed')}[/]  "
                  f"Pass Rate: {summary.get('pass_rate', 0):.1%}\n")

    by_ev = summary.get("by_evaluator", {})
    if by_ev:
        t = Table(box=box.SIMPLE_HEAD, title="By Evaluator", padding=(0, 1))
        t.add_column("Evaluator")
        t.add_column("Avg Score", justify="right")
        for name, score in by_ev.items():
            color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
            t.add_row(name, f"[{color}]{score:.3f}[/]")
        console.print(t)


def main():
    parser = argparse.ArgumentParser(prog="multivon-eval", description="Multivon Eval CLI")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Execute an eval file")
    run_p.add_argument("file", help="Python eval file to run")

    report_p = sub.add_parser("report", help="Display a saved JSON report")
    report_p.add_argument("file", help="JSON results file")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
