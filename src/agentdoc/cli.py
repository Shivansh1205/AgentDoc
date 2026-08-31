"""AgentDoc command-line interface."""

import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.box import ASCII
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agentdoc import __version__
from agentdoc.classifier import (
    BACKENDS,
    DEFAULT_BACKEND,
    LLMBackendError,
    MastClassifier,
)
from agentdoc.parsers import PARSERS
from agentdoc.parsers.base import TraceParser
from agentdoc.parsers.schema import NormalizedTrace, Role, Turn
from agentdoc.report import generate_report, render_report, write_report_json

# Load variables from a .env file in the current (or an ancestor) directory,
# e.g. GROQ_API_KEY / ANTHROPIC_API_KEY, without overriding any already set
# in the real environment. Keeps API keys project-local rather than requiring
# a permanent system-wide environment variable. A missing .env is not an
# error — load_dotenv() just no-ops.
load_dotenv()

app = typer.Typer(
    name="agentdoc",
    help=(
        "Diagnose why multi-agent LLM systems fail.\n\n"
        "Classifies execution traces against MAST (Multi-Agent System "
        "Failure Taxonomy, arXiv:2503.13657) - 14 failure modes across "
        "system design, inter-agent coordination, and task verification."
    ),
    epilog=(
        "Examples:\n\n"
        "  agentdoc diagnose trace.json\n\n"
        "  agentdoc diagnose trace.json --backend anthropic --json report.json\n\n"
        "Run 'agentdoc COMMAND --help' for details on a specific command."
    ),
    add_completion=False,
    rich_markup_mode="rich",
)

# Diagnosis output includes text an LLM generated, which can contain any
# Unicode character (smart quotes, em/en dashes, etc.) regardless of what
# characters *our own* code uses. On a legacy Windows console (cp1252), that
# can raise UnicodeEncodeError deep inside Rich's write path. Reconfiguring
# stdout/stderr to replace unencodable characters instead of raising makes
# rendering robust to arbitrary LLM output, not just the specific characters
# we've happened to avoid ourselves. `reconfigure` is a no-op on streams that
# don't support it (e.g. when stdout is captured by a test runner).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

# `legacy_windows=False` makes rich write plain ANSI (or no styling, if the
# terminal doesn't support it) instead of going through the Win32 console API
# that can crash on non-cp1252 characters in older Windows terminals.
console = Console(legacy_windows=False, safe_box=False)

_ROLE_STYLE = {
    Role.SYSTEM: "dim italic",
    Role.HUMAN: "bold cyan",
    Role.AGENT: "bold green",
    Role.TOOL: "yellow",
}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"agentdoc {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the agentdoc version and exit.",
    ),
) -> None:
    """AgentDoc CLI."""


@app.command()
def parse(
    file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a trace file to parse.",
    ),
    framework: str = typer.Option(
        "langgraph",
        "--framework",
        "-f",
        help=f"Source framework of the trace. One of: {', '.join(sorted(PARSERS))}.",
    ),
) -> None:
    """Parse a multi-agent trace file and pretty-print the normalized trace.

    Converts a framework-specific trace into AgentDoc's normalized turn
    sequence and prints it. Useful for sanity-checking a trace before
    diagnosing it, or for inspecting how AgentDoc interpreted it.

    Example:

        agentdoc parse examples/langgraph_trace_example.json
    """
    parser_cls = PARSERS.get(framework)
    if parser_cls is None:
        console.print(
            f"[bold red]Unknown framework[/] {framework!r}. "
            f"Available: {', '.join(sorted(PARSERS))}"
        )
        raise typer.Exit(code=1)

    parser: TraceParser = parser_cls()

    try:
        trace = parser.parse(file)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        console.print(f"[bold red]Failed to parse trace:[/] {exc}")
        raise typer.Exit(code=1) from exc

    _render_trace(trace)


def _render_trace(trace: NormalizedTrace) -> None:
    console.print(
        f"[bold]Normalized trace[/] "
        f"(framework={trace.source_framework!r}, turns={len(trace)})\n"
    )
    for turn in trace:
        console.print(_render_turn(turn))


def _render_turn(turn: Turn) -> Panel:
    style = _ROLE_STYLE.get(turn.role, "white")

    header = Text()
    header.append(f"step {turn.step}", style="bold")
    if turn.agent:
        header.append(f"  |  {turn.agent}", style=style)
    header.append(f"  |  {turn.role.value}", style=style)
    if turn.timestamp:
        header.append(f"  |  {turn.timestamp}", style="dim")

    body = Text()
    if turn.content:
        body.append(turn.content)

    for call in turn.tool_calls:
        if body.plain:
            body.append("\n\n")
        body.append(f"-> tool_call: {call.name}", style="bold yellow")
        if call.call_id:
            body.append(f" (id={call.call_id})", style="dim")
        if call.args:
            body.append(f"\n  args: {call.args}")
        if call.result is not None:
            body.append(f"\n  result: {call.result}", style="green")
        if call.error is not None:
            body.append(f"\n  error: {call.error}", style="red")

    if not body.plain:
        body.append("(no content)", style="dim italic")

    group = Text.assemble(header, "\n", body)
    return Panel(group, border_style=style, expand=False, box=ASCII)


@app.command()
def diagnose(
    file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a trace file to diagnose.",
    ),
    framework: str = typer.Option(
        "langgraph",
        "--framework",
        "-f",
        help=f"Source framework of the trace. One of: {', '.join(sorted(PARSERS))}.",
    ),
    backend: str = typer.Option(
        DEFAULT_BACKEND,
        "--backend",
        "-b",
        help=(
            "LLM backend to classify with. One of: "
            f"{', '.join(sorted(BACKENDS))}. Groq is free-tier; Anthropic is "
            "a paid alternative."
        ),
    ),
    json_path: Path | None = typer.Option(
        None,
        "--json",
        help="Also write the full structured report as JSON to this path.",
    ),
    json_only: bool = typer.Option(
        False,
        "--json-only",
        help="Suppress the terminal report; requires --json.",
    ),
) -> None:
    """Parse a trace, classify it against MAST, and report the diagnosis.

    Requires an API key for the selected --backend: GROQ_API_KEY (default,
    free - get one at https://console.groq.com/keys) or ANTHROPIC_API_KEY
    (--backend anthropic). Set it as an environment variable or in a .env
    file in the current directory.

    Examples:

        agentdoc diagnose trace.json

        agentdoc diagnose trace.json --backend anthropic

        agentdoc diagnose trace.json --json report.json

        agentdoc diagnose trace.json --json report.json --json-only
    """
    if json_only and json_path is None:
        console.print("[bold red]--json-only requires --json <path>.[/]")
        raise typer.Exit(code=1)

    parser_cls = PARSERS.get(framework)
    if parser_cls is None:
        console.print(
            f"[bold red]Unknown framework[/] {framework!r}. "
            f"Available: {', '.join(sorted(PARSERS))}"
        )
        raise typer.Exit(code=1)

    parser: TraceParser = parser_cls()

    try:
        trace = parser.parse(file)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        console.print(f"[bold red]Failed to parse trace:[/] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        classifier = MastClassifier(backend=backend)
    except LLMBackendError as exc:
        console.print(f"[bold red]Cannot run classifier:[/] {exc}")
        raise typer.Exit(code=1) from exc

    with console.status("[bold]Classifying trace against MAST taxonomy..."):
        try:
            result = classifier.classify(trace)
        except LLMBackendError as exc:
            console.print(f"[bold red]Classification failed:[/] {exc}")
            raise typer.Exit(code=1) from exc

    summary = generate_report(trace, result)

    if json_path is not None:
        try:
            write_report_json(summary, json_path)
        except OSError as exc:
            console.print(f"[bold red]Failed to write JSON report:[/] {exc}")
            raise typer.Exit(code=1) from exc
        if not json_only:
            console.print(f"[dim]Wrote JSON report to {json_path}[/]\n")

    if not json_only:
        render_report(console, summary, trace)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
