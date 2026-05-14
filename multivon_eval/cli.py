"""
multivon-eval CLI

Usage:
    multivon-eval init --template rag --dir ./my-eval
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
import os
from pathlib import Path


def cmd_init(args):
    """Scaffold a starter project — runnable eval in under 5 minutes."""
    from .templates import render, list_templates

    target = Path(args.dir).resolve()
    files = render(args.template, with_ci=args.ci)

    # If --dir points at an existing file, fail with a clean error rather
    # than letting `target.iterdir()` raise NotADirectoryError downstream.
    if target.exists() and not target.is_dir():
        print(f"--dir must be a directory, but {target} is a file.", file=sys.stderr)
        return 1
    # Refuse to clobber a non-empty existing directory unless --force.
    if target.exists() and any(target.iterdir()) and not args.force:
        print(f"Target directory is not empty: {target}", file=sys.stderr)
        print(f"Re-run with --force to overwrite.", file=sys.stderr)
        return 1
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for rel_path, content in files.items():
        out_path = target / rel_path
        # Reject any path that tries to escape the target dir (defense in depth;
        # all template keys are author-controlled but the check is cheap).
        if not out_path.resolve().is_relative_to(target):
            raise ValueError(f"Template path escapes target dir: {rel_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        written.append(out_path)

    rel = target.relative_to(Path.cwd()) if target.is_relative_to(Path.cwd()) else target
    print(f"\n  Scaffolded {len(written)} file(s) into {rel}/")
    for p in sorted(written):
        print(f"    {p.relative_to(target)}")

    # Print the 3-command flow the README will also have.
    print(f"\n  Next:")
    print(f"    cd {rel}")
    print(f"    pip install -r requirements.txt")
    if args.template != "quickstart":
        print(f"    cp .env.example .env  # then add your API key")
    print(f"    python eval.py\n")
    return 0


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

    if args.junit:
        from .result import EvalReport
        report = EvalReport.from_dict(data)
        report.save_junit_xml(args.junit)
        console.print(f"  JUnit XML saved → [dim]{args.junit}[/]  "
                      f"(GitHub Actions / GitLab CI will render this as a test panel)")


def cmd_view(args):
    """Open a saved JSON report as HTML, served locally.

    Generates the HTML once into a temp dir, starts a tiny stdlib
    http.server on the requested port, and opens the user's browser.
    Stays alive until Ctrl-C. The temp dir is cleaned up on every exit
    path (success, Ctrl-C, port collision).
    """
    from pathlib import Path
    import http.server
    import signal
    import socketserver
    import tempfile
    import webbrowser
    import threading

    # Translate SIGTERM into a KeyboardInterrupt so the with-block's
    # cleanup (TemporaryDirectory unlink, httpd shutdown) runs on
    # `docker stop`, `kill <pid>`, or pytest's proc.terminate() the
    # same way Ctrl-C does.
    def _term_handler(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term_handler)

    report_path = Path(args.file)
    if not report_path.exists():
        print(f"Report not found: {report_path}", file=sys.stderr)
        return 1

    with open(report_path) as f:
        data = json.load(f)

    from .result import EvalReport
    report = EvalReport.from_dict(data)
    html = report.to_html()

    # TemporaryDirectory removes the dir on context exit — including the
    # Ctrl-C path inside it via the with-block. No orphaned multivon-view-*
    # dirs left behind on bind failure, exception, or normal shutdown.
    with tempfile.TemporaryDirectory(prefix="multivon-view-") as tmp_str:
        tmp_dir = Path(tmp_str)
        (tmp_dir / "index.html").write_text(html, encoding="utf-8")

        port = args.port or 0

        class _Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *posargs, **kw):
                super().__init__(*posargs, directory=str(tmp_dir), **kw)

            def log_message(self, format, *fmtargs):
                # Suppress default access logs — user just wants the URL.
                pass

        class _ReusableServer(socketserver.TCPServer):
            # Quick-restart friendly: skip the kernel's TIME_WAIT timer if
            # the user Ctrl-C'd a moment ago and is now rerunning.
            allow_reuse_address = True

        try:
            httpd = _ReusableServer(("127.0.0.1", port), _Handler)
        except OSError as ex:
            # Print a clean error instead of leaking a traceback when
            # the explicit --port is taken.
            target = f"127.0.0.1:{port}" if port else "127.0.0.1:auto"
            print(f"multivon-eval view: could not bind {target} — {ex}", file=sys.stderr)
            return 1

        with httpd:
            actual_port = httpd.server_address[1]
            url = f"http://127.0.0.1:{actual_port}/"
            print(f"  multivon-eval view  →  {url}")
            print(f"  Source: {report_path}")
            print(f"  Press Ctrl-C to stop.\n")

            if args.no_browser:
                print("  --no-browser was set; not opening browser automatically.")
            else:
                # Delay browser open until AFTER the server is bound so the
                # first request can't race.
                threading.Timer(0.2, lambda: webbrowser.open(url)).start()

            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n  Stopping server.")
    return 0


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

    # init — scaffold a starter project
    init_p = sub.add_parser(
        "init",
        help="Scaffold a starter eval project (runnable in under 5 minutes)",
    )
    init_p.add_argument(
        "--template", "-t",
        default="rag",
        choices=["quickstart", "rag", "agent", "regulated"],
        help="Which starter to generate (default: rag)",
    )
    init_p.add_argument(
        "--dir", "-d",
        default=".",
        help="Target directory (default: current directory)",
    )
    init_p.add_argument(
        "--ci",
        default=None,
        choices=["github"],
        help="Also generate a CI workflow (currently: github)",
    )
    init_p.add_argument(
        "--force", action="store_true",
        help="Overwrite files in --dir even if it's not empty",
    )

    # run
    run_p = sub.add_parser("run", help="Execute an eval file")
    run_p.add_argument("file", help="Python eval file to run")
    run_p.add_argument("--html", metavar="PATH", help="Save HTML report to PATH")
    run_p.add_argument("--json", metavar="PATH", help="Save JSON report to PATH")

    # report
    report_p = sub.add_parser("report", help="Display a saved JSON report")
    report_p.add_argument("file", help="JSON results file")
    report_p.add_argument("--html", metavar="PATH", help="Also save an HTML report to PATH")
    report_p.add_argument("--junit", metavar="PATH",
                          help="Also save a JUnit XML report to PATH (renders natively in GitHub Actions / GitLab CI)")

    # view — local HTML report server
    view_p = sub.add_parser(
        "view",
        help="Open a saved JSON report as HTML in a local web server",
    )
    view_p.add_argument("file", help="JSON results file to render")
    view_p.add_argument(
        "--port", type=int, default=0,
        help="Port to listen on (default: OS picks an open port)",
    )
    view_p.add_argument(
        "--no-browser", action="store_true",
        help="Don't open the browser automatically (useful for SSH / containers)",
    )

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

    if args.command == "init":
        sys.exit(cmd_init(args) or 0)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "view":
        sys.exit(cmd_view(args) or 0)
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
