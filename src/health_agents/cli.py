"""The console loop.

This is the only module that knows stdin exists. The graph asks for input by returning
an interrupt payload describing *what* it needs; deciding *how* to collect it (one line
or a multi-line paste) lives here.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from langgraph.types import Command

from health_agents.config import (
    ROLE_ENV_VARS,
    MissingApiKey,
    model_map,
    require_api_key,
    validate_models,
)

__all__ = ["main", "read_multiline", "models_table"]

BANNER = """[bold]Health agents[/bold] — a LangGraph router demo

Try:
  • [cyan]what causes anemia?[/cyan]                    → Q/A agent
  • [cyan]summarize samples/sample_report.txt[/cyan]    → summarizer
  • [cyan]summarize this report[/cyan]                  → summarizer (asks you to paste)
  • [cyan]create a health report for me[/cyan]          → report builder (interviews you)

Commands: [magenta]/help[/magenta]  [magenta]/models[/magenta]  [magenta]/new[/magenta] (forget the conversation)  [magenta]/quit[/magenta]"""

AGENT_LABELS = {
    "summarize": "summarizer",
    "build_report": "report builder",
    "qa": "Q/A bot",
}

ROLE_NOTES = {
    "router": "one 3-way classification, every turn — wants small and fast",
    "summarizer": "reads dense numeric reports — the hardest job",
    "report_writer": "formats supplied facts — needs restraint, not knowledge",
    "qa": "open-ended questions — wants breadth and good prose",
}


def models_table() -> Table:
    """Which model each agent uses, and the env var that overrides it."""
    table = Table(title="Model per agent", title_style="bold", box=box.SIMPLE)
    table.add_column("agent", style="cyan")
    table.add_column("model")
    table.add_column("override with", style="dim")
    table.add_column("why", style="dim")
    for role, model in model_map().items():
        table.add_row(role, model, ROLE_ENV_VARS[role], ROLE_NOTES[role])
    return table


def read_multiline(console: Console, prompt: str) -> str:
    """Collect lines until a line containing only EOF (or a lone existing file path)."""
    console.print(prompt, markup=False, highlight=False)
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF":
            break
        lines.append(line)
        # Convenience: a single line naming a real file needs no EOF terminator.
        if len(lines) == 1 and Path(line.strip()).expanduser().is_file():
            break
    return "\n".join(lines)


def _collect(console: Console, payload: dict) -> str:
    if payload.get("multiline"):
        return read_multiline(console, payload["prompt"])
    return console.input(f"[yellow]{payload['prompt']}[/yellow] ")


def _stream_until_pause(graph, config: dict, payload):
    """Run until the graph pauses or finishes. Returns the interrupt payload, or None."""
    interrupted = None
    for event in graph.stream(payload, config):
        if "__interrupt__" in event:
            interrupted = event["__interrupt__"][0].value
    return interrupted


def _offer_retry(graph, console: Console, config: dict, exc: Exception) -> bool:
    """After a mid-turn failure, offer to resume from the node that failed.

    The checkpointer still holds everything answered so far, so a rate-limited call at
    the end of an 8-question interview does not have to cost the user those answers.
    """
    console.print(f"[red]{type(exc).__name__}: {exc}[/red]")

    state = graph.get_state(config)
    if not state.next:
        return False  # nothing parked, so nothing to resume

    saved = state.values.get("report_data") or {}
    detail = f" Your {len(saved)} collected answer(s) are safe." if saved else ""
    console.print(f"[yellow]Stopped at '{state.next[0]}'.{detail}[/yellow]")
    reply = console.input("[yellow]Retry from there? [Y/n][/yellow] ").strip().lower()
    return reply in {"", "y", "yes"}


def _run_turn(graph, console: Console, config: dict, query: str) -> None:
    """Stream one turn, servicing interrupts until the graph reaches END."""
    payload: object = {"query": query}
    while True:
        try:
            interrupted = _stream_until_pause(graph, config, payload)
        except Exception as exc:  # noqa: BLE001 - decide with the user, not a traceback
            if not _offer_retry(graph, console, config, exc):
                return
            payload = None  # None resumes the node that failed
            continue
        if interrupted is None:
            break
        payload = Command(resume=_collect(console, interrupted))

    values = graph.get_state(config).values
    label = AGENT_LABELS.get(values.get("route"), "agent")
    console.print(f"[dim]— handled by the {label} —[/dim]")
    console.print(
        Panel(values.get("answer") or "(no answer)", border_style="green", padding=(0, 1)),
        markup=False,
        highlight=False,
    )


def main() -> int:
    console = Console()

    try:
        require_api_key()
    except MissingApiKey as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    # Check the configured model ids up front. Otherwise a typo'd or retired id only
    # surfaces when that one agent happens to run — possibly mid-interview.
    problems, lookup_error = validate_models()
    if problems:
        console.print("[red]Configured models are not usable:[/red]")
        for problem in problems:
            console.print(f"  [red]•[/red] {problem}")
        console.print("\n[dim]Fix these in .env, then run again. List what your key has:[/dim]")
        # markup=False: the list comprehension's brackets would otherwise be parsed as
        # Rich markup tags and silently stripped from the command.
        console.print(
            "  uv run python -c \"import groq,dotenv;dotenv.load_dotenv();"
            "print([m.id for m in groq.Groq().models.list().data])\"",
            markup=False,
            highlight=False,
            style="dim",
        )
        return 1
    if lookup_error:
        console.print(f"[yellow]Could not verify models ({lookup_error}); continuing.[/yellow]")

    from health_agents.graph import build_graph

    graph = build_graph()

    console.print(Panel(BANNER, border_style="cyan", padding=(0, 1)))
    distinct = sorted(set(model_map().values()))
    console.print(
        f"[dim]{len(distinct)} model(s) in use: {', '.join(distinct)}"
        "   ([magenta]/models[/magenta] for the per-agent breakdown)[/dim]"
    )
    thread_id = str(uuid.uuid4())

    while True:
        try:
            query = console.input("\n[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return 0

        if not query:
            continue
        if query in {"/quit", "/exit", "/q"}:
            console.print("[dim]bye[/dim]")
            return 0
        if query == "/help":
            console.print(Panel(BANNER, border_style="cyan", padding=(0, 1)))
            continue
        if query == "/models":
            console.print(models_table())
            continue
        if query == "/new":
            thread_id = str(uuid.uuid4())
            console.print("[dim]started a fresh conversation[/dim]")
            continue

        config = {"configurable": {"thread_id": thread_id}}
        try:
            _run_turn(graph, console, config, query)
        except KeyboardInterrupt:
            # Abandon this turn but keep the session (and the conversation) alive.
            console.print("\n[dim]cancelled[/dim]")
        except Exception as exc:  # noqa: BLE001 - a console app should not traceback
            console.print(f"[red]Something went wrong: {exc}[/red]")
