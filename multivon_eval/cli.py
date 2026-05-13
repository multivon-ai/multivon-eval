"""
multivon-eval CLI

Usage:
    multivon-eval run eval.py
    multivon-eval report results.json
    multivon-eval experiments list
    multivon-eval experiments history <name>
    multivon-eval experiments compare <name> <run_a> <run_b>
    multivon-eval generate --from docs/faq.md --n 20 --task qa
"""
from __future__ import annotations
import sys
import json
import argparse


def cmd_run(args):
    import os
    import runpy
    if args.html:
        os.environ["MULTIVON_HTML_OUTPUT"] = args.html
    if args.json:
        os.environ["MULTIVON_JSON_OUTPUT"] = args.json
    try:
        runpy.run_path(args.file, run_name="__main__")
    finally:
        os.environ.pop("MULTIVON_HTML_OUTPUT", None)
        os.environ.pop("MULTIVON_JSON_OUTPUT", None)


def cmd_report(args):
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

    if args.html:
        from .result import EvalReport
        report = EvalReport.from_dict(data)
        report.save_html(args.html)
        console.print(f"\n  HTML report saved → [dim]{args.html}[/]")


def cmd_experiments(args):
    from .experiments import Experiment, list_experiments

    if args.exp_cmd == "list":
        names = list_experiments()
        if not names:
            print("No experiments recorded yet.")
            print("Record a run with: exp.record(report, tags={...})")
        else:
            print(f"\n  {len(names)} experiment(s):")
            for n in names:
                exp = Experiment(n)
                runs = exp.history(n=1)
                last = f"  last run: {runs[0].timestamp[:19]}" if runs else ""
                print(f"    {n}{last}")
        print()

    elif args.exp_cmd == "history":
        Experiment(args.name).print_history(n=args.n)

    elif args.exp_cmd == "compare":
        Experiment(args.name).compare(args.run_a, args.run_b)

    else:
        print("Usage: multivon-eval experiments [list|history|compare]")


def cmd_generate(args):
    from dotenv import load_dotenv
    load_dotenv()

    from .generate import generate_from_file, generate_from_text
    import os

    if args.source:
        print(f"Generating {args.n} {args.task} cases from {args.source}...")
        cases = generate_from_file(args.source, n=args.n, task=args.task)
    elif args.text:
        cases = generate_from_text(args.text, n=args.n, task=args.task)
    else:
        print("Provide --from <file> or --text <text>")
        sys.exit(1)

    out = [
        {
            "input": c.input,
            "expected_output": c.expected_output or "",
            "context": c.context or "",
        }
        for c in cases
    ]

    if args.output:
        import json
        with open(args.output, "w") as f:
            for row in out:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {len(cases)} cases to {args.output}")
    else:
        for i, c in enumerate(cases, 1):
            print(f"\n[{i}] input:    {c.input[:120]}")
            if c.expected_output:
                print(f"     expected: {c.expected_output[:120]}")
            if c.context:
                print(f"     context:  {c.context[:80]}...")

    print(f"\n  Generated {len(cases)} cases.")


def main():
    parser = argparse.ArgumentParser(prog="multivon-eval", description="Multivon Eval CLI")
    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Execute an eval file")
    run_p.add_argument("file", help="Python eval file to run")
    run_p.add_argument("--html", metavar="PATH", help="Save HTML report to PATH")
    run_p.add_argument("--json", metavar="PATH", help="Save JSON report to PATH")

    # report
    report_p = sub.add_parser("report", help="Display a saved JSON report")
    report_p.add_argument("file", help="JSON results file")
    report_p.add_argument("--html", metavar="PATH", help="Also save an HTML report to PATH")

    # experiments
    exp_p = sub.add_parser("experiments", help="Manage experiment history")
    exp_sub = exp_p.add_subparsers(dest="exp_cmd")

    exp_sub.add_parser("list", help="List all experiments")

    hist_p = exp_sub.add_parser("history", help="Show run history for an experiment")
    hist_p.add_argument("name", help="Experiment name")
    hist_p.add_argument("--n", type=int, default=10, help="Number of runs to show")

    cmp_p = exp_sub.add_parser("compare", help="Compare two runs")
    cmp_p.add_argument("name", help="Experiment name")
    cmp_p.add_argument("run_a", help="First run ID (baseline)")
    cmp_p.add_argument("run_b", help="Second run ID (new)")

    # generate
    gen_p = sub.add_parser("generate", help="Generate eval cases from text or files")
    gen_p.add_argument("--from", dest="source", help="Source file path")
    gen_p.add_argument("--text", help="Raw text to generate from")
    gen_p.add_argument("--n", type=int, default=10, help="Number of cases to generate")
    gen_p.add_argument("--task", default="qa", choices=["qa", "summarization", "hallucination"],
                       help="Type of eval cases to generate")
    gen_p.add_argument("--output", "-o", help="Save to JSONL file (default: print to stdout)")

    # audit-package — one-shot compliance evidence zip
    audit_p = sub.add_parser(
        "audit-package",
        help="Bundle audit log + calibration + verifier into a single zip for auditors",
    )
    audit_p.add_argument("--logs", required=True,
                         help="Directory containing the audit log NDJSON files")
    audit_p.add_argument("--suite", required=True,
                         help="Suite name (matches the audit log filename)")
    audit_p.add_argument("--framework", required=True,
                         choices=["eu-ai-act", "nist-ai-rmf", "hipaa", "none"],
                         help="Compliance framework that drove the audit log")
    audit_p.add_argument("--out", required=True, help="Output ZIP path")
    audit_p.add_argument("--period", default=None,
                         help='Human label like "2026-Q2" (default: today)')

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "experiments":
        cmd_experiments(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "audit-package":
        from . import audit_package as _ap
        sys.exit(_ap._cli([
            "--logs", args.logs, "--suite", args.suite,
            "--framework", args.framework, "--out", args.out,
            *(["--period", args.period] if args.period else []),
        ]))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
