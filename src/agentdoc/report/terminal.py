"""Rich-formatted terminal rendering of a `ReportSummary`.

Reuses the visual style established in `agentdoc.cli` for the `parse`
command (ASCII-boxed panels, role/category color coding) so `parse` and
`diagnose` output feel like the same tool. Rendering never assumes a
Unicode-capable console — callers are expected to have already configured
their `Console`/streams for graceful degradation (see `agentdoc.cli`), since
failure justifications come from an LLM and can contain arbitrary characters.
"""

from __future__ import annotations

from rich.box import ASCII
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from agentdoc.classifier.results import FlaggedFailure
from agentdoc.classifier.taxonomy import CATEGORY_NAMES, FailureCategory, get_failure_mode
from agentdoc.parsers.schema import NormalizedTrace, Turn
from agentdoc.report.summary import ReportSummary

_SEVERITY_STYLE_BY_RANK = ("bold red", "red", "yellow")


def render_report(
    console: Console, summary: ReportSummary, trace: NormalizedTrace | None = None
) -> None:
    """Print a full terminal report: summary panel, then failures by category.

    Args:
        console: The Rich console to print to.
        summary: The report to render.
        trace: The source trace, if available. When given, each flagged
            failure's panel includes the offending turn(s) rendered inline
            rather than just their indices, so the failure is legible without
            cross-referencing a separate `parse` output.
    """
    console.print(_render_summary_panel(summary))
    console.print()

    if summary.total_failures == 0:
        return

    turns_by_step = {turn.step: turn for turn in trace} if trace is not None else {}

    for category in FailureCategory:
        failures = [f for f in summary.flagged_failures if f.category is category]
        if not failures:
            continue
        console.print(f"[bold underline]{CATEGORY_NAMES[category]}[/]")
        for failure in failures:
            console.print(_render_flagged_failure(failure, turns_by_step))
        console.print()


def _render_summary_panel(summary: ReportSummary) -> Panel:
    header = Text()
    header.append("MAST Diagnosis Summary", style="bold")
    if summary.source_framework:
        header.append(f"  |  framework={summary.source_framework}", style="dim")
    header.append(f"  |  turns={summary.trace_turn_count}", style="dim")
    if summary.model:
        header.append(f"  |  model={summary.model}", style="dim")

    body_lines: list[Text] = [Text(summary.narrative)]

    if summary.total_failures > 0:
        counts_line = Text("\n")
        for i, cc in enumerate(summary.category_counts):
            if i > 0:
                counts_line.append("   ")
            style = "bold red" if cc.count > 0 else "dim"
            counts_line.append(f"{CATEGORY_NAMES[cc.category]}: {cc.count}", style=style)
        body_lines.append(counts_line)

        if summary.ranked_failure_modes:
            top_line = Text("\n")
            top_line.append("Most frequent: ", style="dim")
            top_parts = [
                f"{get_failure_mode(fmc.failure_mode.value).name} "
                f"({fmc.failure_mode.value}) x{fmc.count}"
                for fmc in summary.ranked_failure_modes[:3]
            ]
            top_line.append(", ".join(top_parts))
            body_lines.append(top_line)

    style = "bold red" if summary.total_failures > 0 else "green"
    group = Group(header, *body_lines)
    return Panel(group, border_style=style, expand=False, box=ASCII)


def _render_flagged_failure(
    failure: FlaggedFailure, turns_by_step: dict[int, Turn]
) -> Panel:
    mode_info = get_failure_mode(failure.failure_mode.value)

    header = Text()
    header.append(f"{failure.failure_mode.value}", style="bold red")
    header.append(f"  {mode_info.name}", style="bold")
    header.append(f"  |  confidence={failure.confidence:.2f}", style="dim")
    header.append(f"  |  turns={failure.turn_indices}", style="dim")

    body = Text()
    body.append(failure.justification)

    offending_turns = [
        turns_by_step[i] for i in failure.turn_indices if i in turns_by_step
    ]
    if offending_turns:
        body.append("\n")
        for turn in offending_turns:
            body.append(f"\n[step {turn.step}] ", style="bold yellow")
            agent_label = f"{turn.agent} " if turn.agent else ""
            body.append(f"{agent_label}({turn.role.value})", style="yellow")
            if turn.content:
                body.append(f"\n{turn.content}")

    group = Text.assemble(header, "\n", body)
    return Panel(group, border_style="red", expand=False, box=ASCII)
