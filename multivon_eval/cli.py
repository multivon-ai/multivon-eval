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
        choices=[
            "quickstart", "rag",
            "agent", "agent-langgraph", "agent-openai-sdk",
            "conversation", "regulated",
        ],
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

    # compare — diff two eval report JSONs
    cmp_p = sub.add_parser(
        "compare",
        help="Compare two eval report JSONs (pass-rate delta, regressions, McNemar p)",
    )
    cmp_p.add_argument("baseline", help="Baseline report JSON")
    cmp_p.add_argument("proposal", help="Proposal report JSON to compare")
    cmp_p.add_argument("--regressions-only", action="store_true",
                       help="Hide improvements section (good for CI summaries)")
    cmp_p.add_argument("--markdown", action="store_true",
                       help="Emit GitHub-flavored Markdown (PR comment style)")
    cmp_p.add_argument("--json", action="store_true",
                       help="Emit the diff as JSON")
    cmp_p.add_argument("--fail-on-regression", action="store_true",
                       help="Exit 1 if any regressions are detected")

    # discover — emit machine-readable capability catalog as JSON
    disc_p = sub.add_parser(
        "discover",
        help="Emit machine-readable capability catalog as JSON (for agents)",
    )
    disc_p.add_argument("--compact", action="store_true",
                        help="Single-line JSON (no indent)")

    # doctor — pre-flight check for keys, providers, optional deps
    doc_p = sub.add_parser(
        "doctor",
        help="Pre-flight: check API keys, ping providers, surface env issues",
    )
    doc_p.add_argument("--no-ping", action="store_true",
                       help="Skip live provider ping (offline mode)")
    doc_p.add_argument("--json", action="store_true",
                       help="Emit results as JSON for CI consumption")

    # bootstrap — cold-start eval suite generator
    boot_p = sub.add_parser(
        "bootstrap",
        help="Cold-start eval bootstrap: product description + traces → tuned EvalSuite",
    )
    boot_p.add_argument("--product", "-p", required=True,
                        help="Path to product description markdown (free-form, < 5K words)")
    boot_p.add_argument("--traces", "-t", required=True,
                        help="Path to traces JSONL (one trace per line, max 10K rows)")
    boot_p.add_argument("--output", "-o", default="./eval-bootstrap",
                        help="Output directory (default: ./eval-bootstrap)")
    boot_p.add_argument("--judge-model", default="claude-haiku-4-5-20251001",
                        help="Judge model for LLM proposal + calibration "
                             "(default: claude-haiku-4-5-20251001)")
    boot_p.add_argument("--judge-provider", default="anthropic",
                        choices=["anthropic", "openai", "google"],
                        help="Judge provider (default: anthropic)")
    boot_p.add_argument("--n-seed-cases", type=int, default=30,
                        help="Number of adversarial seed cases to generate (default: 30)")
    boot_p.add_argument("--pii-policy", default="redact",
                        choices=["redact", "strict", "allow"],
                        help="PII handling: redact (default), strict (abort on detection), "
                             "allow (send raw — requires confirmation)")
    boot_p.add_argument("--skip-seed-cases", action="store_true",
                        help="Skip adversarial seed-case generation (saves ~$0.02)")
    boot_p.add_argument("--skip-calibration", action="store_true",
                        help="Skip threshold calibration (use proposed thresholds as-is)")
    boot_p.add_argument("--validate", action="store_true",
                        help="N-shot judge-noise filter the generated seed cases against "
                             "a stub-refusal baseline (uses validate_adversarial_cases). "
                             "Drops cases outside hardness band (0.5–1.0). Adds ~$0.03 "
                             "but typically removes 20-40%% of synthetic noise.")
    boot_p.add_argument("--validate-n-shots", type=int, default=3,
                        help="N-shot count for --validate (default: 3)")

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
    elif args.command == "compare":
        from . import compare as _cmp
        sys.exit(_cmp._cli([
            args.baseline, args.proposal,
            *(["--regressions-only"] if args.regressions_only else []),
            *(["--markdown"] if args.markdown else []),
            *(["--json"] if args.json else []),
            *(["--fail-on-regression"] if args.fail_on_regression else []),
        ]))
    elif args.command == "discover":
        sys.exit(cmd_discover(args) or 0)
    elif args.command == "doctor":
        sys.exit(cmd_doctor(args) or 0)
    elif args.command == "bootstrap":
        sys.exit(cmd_bootstrap(args) or 0)
    else:
        parser.print_help()


def cmd_bootstrap(args) -> int:
    """Cold-start eval bootstrap: product + traces → tuned EvalSuite + report."""
    from pathlib import Path
    from .discover import bootstrap
    from .judge import JudgeConfig

    product = Path(args.product)
    traces = Path(args.traces)
    if not product.exists():
        print(f"error: --product file not found: {product}", file=sys.stderr)
        return 2
    if not traces.exists():
        print(f"error: --traces file not found: {traces}", file=sys.stderr)
        return 2

    if args.pii_policy == "allow":
        confirm = input(
            "WARNING: --pii-policy=allow sends raw traces (with any PII / secrets) "
            "to the configured judge. Type 'yes' to continue: "
        ).strip().lower()
        if confirm != "yes":
            print("aborted.", file=sys.stderr)
            return 1

    judge = JudgeConfig(provider=args.judge_provider, model=args.judge_model)

    print(f"\n  bootstrapping eval suite for {product.name}...")
    print(f"  judge: {args.judge_provider}:{args.judge_model}")
    print(f"  pii policy: {args.pii_policy}")
    print()

    result = bootstrap(
        description_path=product,
        traces_path=traces,
        output_dir=args.output,
        judge=judge,
        pii_policy=args.pii_policy,
        skip_seed_cases=args.skip_seed_cases,
        skip_calibration=args.skip_calibration,
        n_seed_cases=args.n_seed_cases,
    )

    print(f"  ✓ inferred shape: {result.shape}")
    print(f"  ✓ traces analyzed: {result.summary.count}")
    if result.summary.pii_label_counts:
        pii_str = ", ".join(f"{k}={v}" for k, v in result.summary.pii_label_counts.items())
        print(f"  ✓ pii redacted before LLM call: {pii_str}")
    print(f"  ✓ recommended {len(result.evaluators)} evaluators")
    print(f"  ✓ generated {len(result.seed_cases)} adversarial seed cases")

    # --validate: optionally N-shot filter the seed cases via the
    # validate_adversarial_cases primitive. The stub-refusal baseline
    # ("I don't know") is intentionally weak — cases the baseline can
    # confidently refuse (failure_rate < 0.5) aren't really stressing the
    # primary evaluator and get dropped. Bumps cost by ~$0.03 typically.
    if args.validate and result.seed_cases:
        from .auto import validate_adversarial_cases
        from pathlib import Path
        import json as _json

        def _stub_refusal(_input: str) -> str:
            return "I don't have specific information about that."

        kept, reports = validate_adversarial_cases(
            result.seed_cases,
            _stub_refusal,
            n_shots=args.validate_n_shots,
            judge=judge,
        )
        dropped = len(result.seed_cases) - len(kept)
        print(f"  ✓ validated seed cases (n_shots={args.validate_n_shots}): "
              f"kept {len(kept)}, dropped {dropped} as noise")

        # Overwrite seed_cases.jsonl with the validated subset.
        seed_path = Path(result.artifacts["seed_cases"])
        with seed_path.open("w") as f:
            for c in kept:
                f.write(_json.dumps({
                    "input": c.input,
                    "expected_output": c.expected_output,
                    "context": c.context,
                    "tags": c.tags,
                    "metadata": c.metadata,
                }) + "\n")
        # Also write a hardness report alongside for transparency.
        hardness_path = seed_path.parent / "hardness_report.jsonl"
        with hardness_path.open("w") as f:
            for r in reports:
                f.write(_json.dumps({
                    "input": r.case.input[:200],
                    "evaluator": r.evaluator_name,
                    "failure_rate": r.failure_rate,
                    "in_hardness_band": r.in_hardness_band,
                    "scores": r.scores,
                }) + "\n")
        print(f"  ✓ hardness report: {hardness_path}")

    print(f"  ✓ estimated cost: ${result.cost_usd:.4f}")
    print()
    print("  artifacts:")
    for label, path in result.artifacts.items():
        print(f"    {label:<14} {path}")
    print()
    print("  next: review DISCOVERY_REPORT.md, then `python eval_suite.py` "
          "to verify the suite loads")
    return 0


def cmd_doctor(args) -> int:
    """Pre-flight check that surfaces every env issue at once.

    What we check:
      - Python version (≥3.10 required; warns on 3.14 for missing wheels).
      - Anthropic / OpenAI / Google API keys present + reachable.
      - multivon-eval version vs latest on PyPI (if reachable).
      - Optional deps: presidio_analyzer (PII NER), opentelemetry, datasets.
      - ~/.multivon writable for experiment history.

    Exit codes: 0 = all green, 1 = at least one ERROR, 2 = WARN only.
    """
    import os
    import sys
    import json
    import platform
    from . import __version__

    checks: list[dict] = []  # {category, name, status, detail}

    def add(category: str, name: str, status: str, detail: str = "") -> None:
        checks.append({"category": category, "name": name, "status": status, "detail": detail})

    # Python
    py = sys.version_info
    py_str = f"{py.major}.{py.minor}.{py.micro}"
    if py >= (3, 10):
        if py >= (3, 14):
            add("env", "Python", "WARN",
                f"{py_str} — some optional deps (presidio, certain ML wheels) may not have prebuilt wheels yet. "
                "Pin to 3.11/3.12 for max compatibility, or expect occasional `pip install` source builds.")
        else:
            add("env", "Python", "OK", py_str)
    else:
        add("env", "Python", "ERROR", f"{py_str} — requires ≥3.10")

    add("env", "Platform", "OK", platform.platform())
    add("env", "multivon-eval", "OK", __version__)

    # API keys
    for env_name, label, prefix in [
        ("ANTHROPIC_API_KEY", "Anthropic", "sk-ant-"),
        ("OPENAI_API_KEY", "OpenAI", "sk-"),
        ("GOOGLE_API_KEY", "Google", ""),
    ]:
        key = os.environ.get(env_name, "")
        if not key:
            add("keys", label, "WARN", f"{env_name} not set — judge calls to {label} will fail")
            continue
        if prefix and not key.startswith(prefix):
            add("keys", label, "WARN", f"{env_name} present but doesn't start with {prefix!r} — possibly malformed")
        else:
            add("keys", label, "OK", f"{env_name} set ({len(key)} chars, prefix={key[:6]}...)")

    # Provider pings (lazy; skip on --no-ping)
    if not args.no_ping:
        for env_name, label, ping_fn in (
            ("ANTHROPIC_API_KEY", "Anthropic", _ping_anthropic),
            ("OPENAI_API_KEY", "OpenAI", _ping_openai),
            ("GOOGLE_API_KEY", "Google", _ping_google),
        ):
            if not os.environ.get(env_name):
                continue
            try:
                msg = ping_fn()
                add("ping", label, "OK", msg)
            except Exception as e:
                add("ping", label, "ERROR", f"ping failed: {type(e).__name__}: {str(e)[:120]}")

    # Optional deps
    for mod, label, why in (
        ("presidio_analyzer", "Presidio NER", "PII evaluator use_ner=True falls back to regex without it"),
        ("opentelemetry", "OpenTelemetry", "tracer integration unavailable"),
        ("datasets", "HuggingFace datasets", "dataset loaders unavailable"),
        ("anthropic", "anthropic SDK", "Anthropic judge calls unavailable"),
        ("openai", "openai SDK", "OpenAI judge calls unavailable"),
    ):
        try:
            __import__(mod)
            add("deps", label, "OK", f"{mod} importable")
        except ImportError:
            add("deps", label, "WARN", f"{mod} not installed — {why}")

    # ~/.multivon writability
    multivon_dir = os.path.expanduser("~/.multivon")
    try:
        os.makedirs(multivon_dir, exist_ok=True)
        probe = os.path.join(multivon_dir, ".doctor-probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        add("env", "~/.multivon", "OK", f"writable at {multivon_dir}")
    except Exception as e:
        add("env", "~/.multivon", "WARN", f"not writable: {e}")

    # Summary
    n_err = sum(1 for c in checks if c["status"] == "ERROR")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_ok = sum(1 for c in checks if c["status"] == "OK")

    if args.json:
        print(json.dumps({
            "summary": {"ok": n_ok, "warn": n_warn, "error": n_err},
            "checks": checks,
        }, indent=2))
    else:
        # Pretty terminal output — group by category, color via ANSI.
        try:
            from rich.console import Console
            from rich.table import Table
            con = Console()
            table = Table(show_header=True, header_style="bold")
            table.add_column("Category", style="dim")
            table.add_column("Check")
            table.add_column("Status", justify="center")
            table.add_column("Detail")
            for c in checks:
                style = {"OK": "green", "WARN": "yellow", "ERROR": "red"}.get(c["status"], "white")
                symbol = {"OK": "✓", "WARN": "⚠", "ERROR": "✗"}.get(c["status"], "?")
                table.add_row(c["category"], c["name"], f"[{style}]{symbol} {c['status']}[/{style}]", c["detail"])
            con.print(table)
            con.print()
            if n_err:
                con.print(f"[red]✗ {n_err} ERROR[/red] · [yellow]⚠ {n_warn} WARN[/yellow] · [green]✓ {n_ok} OK[/green]")
                con.print("  fix the ERROR rows above before running evaluations.")
            elif n_warn:
                con.print(f"[yellow]⚠ {n_warn} WARN[/yellow] · [green]✓ {n_ok} OK[/green]")
                con.print("  evaluations should work; some optional features unavailable.")
            else:
                con.print(f"[green]✓ all {n_ok} checks passed.[/green]")
        except ImportError:
            for c in checks:
                print(f"  [{c['status']:5s}] {c['category']:8s} {c['name']:20s}  {c['detail']}")
            print(f"\n  {n_ok} OK · {n_warn} WARN · {n_err} ERROR")

    return 1 if n_err else (2 if n_warn else 0)


def _ping_anthropic() -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=4,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    text = resp.content[0].text if resp.content else ""
    return f"Claude Haiku reachable; got {text!r}"


def _ping_openai() -> str:
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=4,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    text = resp.choices[0].message.content or ""
    return f"gpt-4o-mini reachable; got {text!r}"


def _ping_google() -> str:
    from google import genai
    client = genai.Client()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with the single word: ok",
    )
    text = (getattr(resp, "text", "") or "")[:20]
    return f"gemini-2.5-flash reachable; got {text!r}"


def cmd_discover(args) -> int:
    """Emit a JSON capability catalog (evaluators, jurisdictions, judges, suites).

    Same shape exposed by multivon-mcp's eval_discover tool — provided as a CLI
    so agents that don't speak MCP (or shell scripts, or CI gates) can pipe
    ``multivon-eval discover --json | jq ...`` to plan a run.
    """
    import inspect
    import multivon_eval
    from .evaluators.base import Evaluator
    from . import __version__

    evaluators: list[dict] = []
    for name in dir(multivon_eval):
        obj = getattr(multivon_eval, name)
        try:
            is_eval = inspect.isclass(obj) and issubclass(obj, Evaluator) and obj is not Evaluator
        except TypeError:
            is_eval = False
        if not is_eval:
            continue
        evaluators.append({
            "name": name,
            "import": f"from multivon_eval import {name}",
            "evaluator_id": getattr(obj, "name", name.lower()),
            "doc": (obj.__doc__ or "").strip().split("\n")[0],
        })
    evaluators.sort(key=lambda e: e["name"])

    catalog = {
        "package": "multivon-eval",
        "version": __version__,
        "evaluators": evaluators,
        "evaluator_count": len(evaluators),
        "pii_jurisdictions": ["gdpr", "ccpa", "pipeda", "hipaa", "dpdp", "all"],
        "compliance_frameworks": ["eu-ai-act", "nist-ai-rmf", "hipaa", "dpdp", "none"],
        "judge_providers": ["anthropic", "openai", "google"],
        "templates": [
            "quickstart", "rag",
            "agent", "agent-langgraph", "agent-openai-sdk",
            "conversation", "regulated",
        ],
    }
    json.dump(catalog, sys.stdout, indent=None if args.compact else 2)
    print()
    return 0


if __name__ == "__main__":
    main()
