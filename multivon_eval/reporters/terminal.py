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
    console.print()

    # Per-case table
    table = Table(box=box.SIMPLE_HEAD, show_footer=False, padding=(0, 1))
    table.add_column("#", style="dim", width=4)
    table.add_column("Input", max_width=40)
    table.add_column("Output", max_width=40)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Status", width=8)
    table.add_column("Latency", justify="right", width=9)

    for i, cr in enumerate(report.case_results):
        status = "[green]PASS[/]" if cr.passed else "[red]FAIL[/]"
        score_color = "green" if cr.score >= 0.7 else "yellow" if cr.score >= 0.5 else "red"
        table.add_row(
            str(i + 1),
            cr.case_input[:40],
            cr.actual_output[:40],
            f"[{score_color}]{cr.score:.2f}[/]",
            status,
            f"{cr.latency_ms:.0f}ms",
        )

    console.print(table)

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
    console.print(Panel(summary, title="Summary", border_style="dim"))
    console.print()
