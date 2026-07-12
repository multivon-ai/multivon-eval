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

    # Per-tag breakdown (only when tags are present)
    tag_scores = report.scores_by_tag()
    tag_pass = report.passed_by_tag()
    tag_count = report.count_by_tag()
    if tag_scores:
        tag_table = Table(box=box.SIMPLE_HEAD, show_footer=False, padding=(0, 1), title="By Tag")
        tag_table.add_column("Tag")
        tag_table.add_column("Cases", justify="right")
        tag_table.add_column("Avg Score", justify="right")
        tag_table.add_column("Pass Rate", justify="right")
        for tag, score in sorted(tag_scores.items()):
            pass_rate = tag_pass.get(tag, 0.0)
            n = tag_count.get(tag, 0)
            score_color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
            tag_table.add_row(tag, str(n), f"[{score_color}]{score:.2f}[/]", f"{pass_rate:.0%}")
        console.print(tag_table)

    # Summary panel
    rate_color = "green" if report.pass_rate >= 0.8 else "yellow" if report.pass_rate >= 0.5 else "red"
    ci_lo, ci_hi = report.pass_rate_ci()
    score_lo, score_hi = report.avg_score_ci()
    summary = (
        f"[bold]Total:[/] {report.total}   "
        f"[green]Passed:[/] {report.passed}   "
        f"[red]Failed:[/] {report.failed}   "
        f"[bold]Pass Rate:[/] [{rate_color}]{report.pass_rate:.1%}[/] "
        f"[dim][{ci_lo:.0%}–{ci_hi:.0%} 95% CI][/]   "
        f"[bold]Avg Score:[/] {report.avg_score:.2f} "
        f"[dim][{score_lo:.2f}–{score_hi:.2f}][/]"
    )
    if multi_run:
        stab_color = "green" if report.stability_score >= 0.9 else "yellow" if report.stability_score >= 0.7 else "red"
        summary += (
            f"   [bold]Stability:[/] [{stab_color}]{report.stability_score:.0%}[/]"
            f"   [bold]Flaky:[/] {report.flaky_count}"
        )
    # Score percentiles
    pct = report.score_percentiles()
    if pct and report.total > 2:
        summary += (
            f"\n[dim]Score distribution  p10:{pct.get('p10', 0):.2f}  "
            f"p50:{pct.get('p50', 0):.2f}  p90:{pct.get('p90', 0):.2f}[/]"
        )
    console.print(Panel(summary, title="Summary", border_style="dim"))

    # Reliability block: pass@k (capability) vs pass^k (reliability),
    # k = runs_per_case — the largest k the recorded data can support.
    if multi_run:
        k = report.runs_per_case
        pak = report.pass_at_k(k)
        phk = report.pass_hat_k(k)
        if pak.value is None or phk.value is None:
            # Honest UNKNOWN (e.g. early_stop left some case with < k
            # trials) — say so instead of silently dropping the block.
            reason = phk.unknown_reason or pak.unknown_reason
            console.print(f"  [bold]Reliability ({k} runs/case)[/]")
            console.print(f"    [yellow]pass@{k} / pass^{k}: {reason}[/]")
        else:
            console.print(f"  [bold]Reliability ({k} runs/case)[/]")
            console.print(
                f"    pass@{k} = {pak.value:.0%} [dim][{pak.ci_low:.0%}–{pak.ci_high:.0%}][/] "
                f"— at least one of {k} tries succeeds (capability)"
            )
            console.print(
                f"    pass^{k} = {phk.value:.0%} [dim][{phk.ci_low:.0%}–{phk.ci_high:.0%}][/] "
                f"— all {k} tries succeed; what a user hitting this feature "
                f"{k} times experiences (reliability)"
            )
            for cr in report.lottery_cases(k)[:3]:
                console.print(
                    f"    [yellow]passes sometimes, never reliably:[/] "
                    f"{cr.case_input[:60]!r} ({cr.pass_count}/{cr.runs} runs)"
                )

    # Power warning: flag when dataset is too small to detect meaningful changes
    if report.total > 0:
        from ..experiments import min_detectable_effect, runs_needed
        mde = min_detectable_effect(report.total)
        if mde > 0.20:
            needed = runs_needed(0.10)
            console.print(
                f"  [yellow]⚡ Power warning:[/] {report.total} case(s) — "
                f"minimum detectable change at 80% power is [bold]~{mde:.0%}[/]. "
                f"Add ≥{needed} cases to reliably detect a 10pp shift."
            )

    _print_saturation_monitor(report)
    _print_zero_pass_footer(report)

    # Judge reliability
    if report.judge_reliability is not None:
        rel_color = "green" if report.judge_reliability >= 0.85 else "yellow" if report.judge_reliability >= 0.70 else "red"
        console.print(
            f"  [dim]Judge consistency:[/] [{rel_color}]{report.judge_reliability:.0%}[/] "
            f"[dim]agreement across repeated calls (sampled {report.total} cases)[/]"
        )

    console.print()


def _print_saturation_monitor(report: EvalReport) -> None:
    """Ceiling-side warning: a suite at 100% can't detect improvement.

    Capability suites (purpose != 'regression') get a graduation nudge
    quantified as the minimum detectable regression; regression suites
    invert — any task below ceiling is the news. Always a warning,
    never a gate.
    """
    from ..result import EvalStatus

    if report.purpose == "regression":
        # FAILED_QUALITY already covers pass_count < runs (multi-run below
        # ceiling); error statuses are deliberately excluded — an outage is
        # not a regression.
        below_ceiling = [
            cr for cr in report.case_results
            if cr.status == EvalStatus.FAILED_QUALITY
        ]
        if below_ceiling:
            console.print(
                f"  [yellow]⚠ Regression suite:[/] {len(below_ceiling)} "
                f"previously-passing task(s) below ceiling — something broke; "
                f"triage before shipping."
            )
            for cr in below_ceiling[:5]:
                console.print(f"    • {cr.case_input[:60]!r}")
        return

    # A tiny saturated suite already gets the power warning above; both
    # firing on 2 cases is noise.
    if not report.saturated or report.evaluated < 3:
        return
    wilson_lower = report.pass_rate_ci()[0]
    mde = report.min_detectable_regression
    console.print(
        f"  [yellow]⚠ Saturated:[/] {report.passed}/{report.evaluated} trials "
        f"passed. All this run can claim is a pass rate ≥ {wilson_lower:.1%} "
        f"(95% Wilson). At n={report.evaluated}, the smallest regression this "
        f"suite can detect at 80% power is [bold]~{mde:.0%}[/] — a real 5pp "
        f"quality drop would look like noise. Graduate this suite to a "
        f"regression suite (purpose='regression') and add harder capability "
        f"tasks."
    )


def _print_zero_pass_footer(report: EvalReport) -> None:
    """Floor-side warning, paired with the saturation monitor above:
    0% pass usually indicts the task or grader, not the agent."""
    suspects = report.zero_pass_cases
    if not suspects:
        return
    console.print(
        f"  [yellow]⚠[/] {len(suspects)} task(s) failed every trial. "
        f"0% pass usually means a broken task or grader, not an incapable "
        f"agent — run [bold]multivon-eval validate[/] before blaming the model."
    )


def print_validation(vreport) -> None:
    """Render a :class:`multivon_eval.validate.ValidationReport`."""
    from rich.table import Table as _Table

    console.print()
    console.rule(f"[bold white]Validate: {vreport.suite_name}[/]")
    console.print()

    table = _Table(box=box.SIMPLE_HEAD, show_footer=False, padding=(0, 1))
    table.add_column("#", style="dim", width=4)
    table.add_column("Case", max_width=44)
    table.add_column("Verdict", width=24)
    table.add_column("Detail", max_width=60)
    _COLORS = {
        "OK": "green",
        "BROKEN_TASK_OR_GRADER": "red",
        "NO_DISCRIMINATION": "yellow",
        "UNVALIDATABLE": "dim",
    }
    for cv in vreport.results:
        color = _COLORS.get(cv.status, "white")
        detail = cv.reason
        if cv.failed_graders:
            fg = cv.failed_graders[0]
            detail = f"{fg.evaluator}: {fg.reason}" if fg.reason else fg.evaluator
        table.add_row(
            str(cv.case_index + 1),
            cv.case_input[:44],
            f"[{color}]{cv.status}[/]",
            detail[:60],
        )
    console.print(table)

    ok_count, validated = vreport.effective_informative_cases
    if validated:
        console.print(f"  effective informative cases: {ok_count}/{validated} validated")
    for cv in vreport.broken:
        console.print(
            f"  [red]✗ broken task or grader:[/] {cv.case_input[:60]!r} — the "
            f"reference output fails its own graders; the agent is innocent."
        )
    for cv in vreport.no_discrimination:
        console.print(
            f"  [yellow]⚠ no discrimination:[/] {cv.case_input[:60]!r} — "
            f"grader passes both the reference and its known-bad contrast twin; "
            f"it contributes zero information."
        )
    if vreport.unvalidatable:
        console.print(
            f"  [dim]ℹ {len(vreport.unvalidatable)} case(s) unvalidatable — add "
            f"expected_output or reference_output to validate these tasks.[/]"
        )
    skipped = sorted({g for cv in vreport.results for g in cv.skipped_graders})
    if skipped:
        console.print(
            f"  [dim]judge-backed grader(s) not run (offline default): "
            f"{', '.join(skipped)} — pass --judges to include them.[/]"
        )
    costs = getattr(vreport, "costs", None)
    if costs is not None and getattr(costs, "total_tokens", 0):
        spend = costs.total_cost_usd
        spend_s = f"${spend:.4f}" if spend is not None else "unknown (no pricing data)"
        console.print(f"  [dim]judge spend: {spend_s} · {costs.total_tokens:,} tokens[/]")
    if vreport.passed:
        verdict = "[green]PASSED[/]"
    elif getattr(vreport, "nothing_validated", False):
        verdict = "[red]NOTHING_VALIDATED[/]"
        console.print(
            "  [red]✗ nothing validated:[/] zero graders executed — this run "
            "validated nothing and is NOT a green result."
        )
    else:
        verdict = "[red]FAILED[/]"
    console.print(f"  Validation {verdict} — {len(vreport.broken)} broken, "
                  f"{len(vreport.no_discrimination)} non-discriminating, "
                  f"{len(vreport.unvalidatable)} unvalidatable, "
                  f"{len(vreport.ok)} OK")
    console.print()
