from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from ..result import EvalReport

console = Console()


def print_report(report: EvalReport) -> None:
    console.print()
    console.rule(f"[bold white]{report.suite_name}[/]")
    if report.model_id:
        console.print(f"  Model: [dim]{report.model_id}[/]")
    multi_run = report.runs_per_case > 1
    if multi_run:
        console.print(f"  Runs per case: [dim]{report.runs_per_case}[/]")
    console.print()

    # Per-case table
    table = Table(box=box.SIMPLE_HEAD, show_footer=False, padding=(0, 1))
    table.add_column("#", style="dim", width=4)
    table.add_column("Input", max_width=36)
    table.add_column("Output", max_width=36)
    table.add_column("Score", justify="right", width=7)
    if multi_run:
        table.add_column("Pass Rate", justify="right", width=10)
        table.add_column("Stability", width=10)
    table.add_column("Status", width=8)
    table.add_column("Latency", justify="right", width=9)

    for i, cr in enumerate(report.case_results):
        score_color = "green" if cr.score >= 0.7 else "yellow" if cr.score >= 0.5 else "red"

        if multi_run:
            if cr.is_flaky:
                status = "[yellow]FLAKY[/]"
            elif cr.passed:
                status = "[green]PASS[/]"
            else:
                status = "[red]FAIL[/]"

            std_str = f"±{cr.score_std:.2f}" if cr.score_std > 0 else ""
            pr_color = "green" if cr.run_pass_rate >= 0.8 else "yellow" if cr.run_pass_rate >= 0.4 else "red"

            table.add_row(
                str(i + 1),
                cr.case_input[:36],
                cr.actual_output[:36],
                f"[{score_color}]{cr.score:.2f}[/] [dim]{std_str}[/]",
                f"[{pr_color}]{cr.run_pass_rate:.0%}[/]",
                "[yellow]flaky[/]" if cr.is_flaky else "[green]stable[/]",
                status,
                f"{cr.latency_ms:.0f}ms",
            )
        else:
            status = "[green]PASS[/]" if cr.passed else "[red]FAIL[/]"
            table.add_row(
                str(i + 1),
                cr.case_input[:36],
                cr.actual_output[:36],
                f"[{score_color}]{cr.score:.2f}[/]",
                status,
                f"{cr.latency_ms:.0f}ms",
            )

    console.print(table)

    # Flaky cases callout
    if multi_run and report.flaky_count > 0:
        flaky = [cr for cr in report.case_results if cr.is_flaky]
        console.print(f"  [yellow]⚠ {report.flaky_count} flaky case(s) — passed inconsistently across {report.runs_per_case} runs:[/]")
        for cr in flaky:
            console.print(f"    • {cr.case_input[:60]!r}  ({cr.pass_count}/{cr.runs} runs passed)")
        console.print()

    # Per-evaluator breakdown
    ev_scores = report.scores_by_evaluator()
    ev_pass = report.passed_by_evaluator()
    if ev_scores:
        ev_table = Table(box=box.SIMPLE_HEAD, show_footer=False, padding=(0, 1), title="By Evaluator")
        ev_table.add_column("Evaluator")
        ev_table.add_column("Avg Score", justify="right")
        ev_table.add_column("Pass Rate", justify="right")
        for name, score in ev_scores.items():
            pass_rate = ev_pass.get(name, 0.0)
            score_color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
            ev_table.add_row(name, f"[{score_color}]{score:.2f}[/]", f"{pass_rate:.0%}")
        console.print(ev_table)

    # Summary panel
    rate_color = "green" if report.pass_rate >= 0.8 else "yellow" if report.pass_rate >= 0.5 else "red"
    summary = (
        f"[bold]Total:[/] {report.total}   "
        f"[green]Passed:[/] {report.passed}   "
        f"[red]Failed:[/] {report.failed}   "
        f"[bold]Pass Rate:[/] [{rate_color}]{report.pass_rate:.1%}[/]   "
        f"[bold]Avg Score:[/] {report.avg_score:.2f}"
    )
    if multi_run:
        stab_color = "green" if report.stability_score >= 0.9 else "yellow" if report.stability_score >= 0.7 else "red"
        summary += (
            f"   [bold]Stability:[/] [{stab_color}]{report.stability_score:.0%}[/]"
            f"   [bold]Flaky:[/] {report.flaky_count}"
        )
    console.print(Panel(summary, title="Summary", border_style="dim"))
    console.print()
