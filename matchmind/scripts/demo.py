"""
Demo Script — Run MatchMind on a StatsBomb or synthetic snapshot.

Usage:
    python -m matchmind.scripts.demo
    python -m matchmind.scripts.demo --match-id 3788741 --event-idx 150 --player 14
    python -m matchmind.scripts.demo --synthetic  # no API key needed
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = typer.Typer()
console = Console()


@app.command()
def main(
    match_id: int = typer.Option(3788741, "--match-id", help="StatsBomb match ID"),
    event_idx: int = typer.Option(150, "--event-idx", help="Event index in match"),
    player: int = typer.Option(0, "--player", help="Focus player ID (0=auto-detect)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use synthetic data (no StatsBomb needed)"),
    question: str = typer.Option("", "--question", help="Custom question for the agent"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the MatchMind tactical agent and display the result."""

    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)

    console.print(Panel.fit(
        "[bold green]🧠 MatchMind Tactical Agent Demo[/bold green]",
        border_style="green",
    ))

    # ── Load match state ───────────────────────────────────────
    if synthetic:
        from matchmind.tests.fixtures.sample_data import make_sample_match_state
        state, focus_id = make_sample_match_state()
        console.print("✅ Using synthetic match state")
    else:
        try:
            from matchmind.data_sources.statsbomb_adapter import StatsBombAdapter
            console.print(f"Loading StatsBomb match {match_id}, event {event_idx}...")
            adapter = StatsBombAdapter()
            state = adapter.load_snapshot(match_id=match_id, event_index=event_idx)
            focus_id = player if player > 0 else state.players[0].id
            console.print(f"✅ Loaded match state: {len(state.players)} players")
        except Exception as exc:
            console.print(f"[yellow]StatsBomb failed ({exc}), using synthetic data[/yellow]")
            from matchmind.tests.fixtures.sample_data import make_sample_match_state
            state, focus_id = make_sample_match_state()

    if player > 0:
        focus_id = player

    console.print(f"\n[bold]Analysing:[/bold] Match {state.match_id} | Player {focus_id} | Minute {state.minute}'")

    # ── Run agent ──────────────────────────────────────────────
    from matchmind.agent.graph import TacticalAgent
    agent = TacticalAgent()

    q = question or f"What should player {focus_id} do in this situation to create tactical advantage?"

    console.print("\n[bold yellow]Running agent pipeline...[/bold yellow]")

    try:
        advice, trace = agent.analyze(state, focus_id, question=q)
    except Exception as exc:
        console.print(f"[bold red]Agent failed: {exc}[/bold red]")
        raise typer.Exit(1)

    # ── Display results ────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold green]✅ RECOMMENDED ACTION[/bold green]\n\n{advice.recommended_action}",
        border_style="green",
    ))

    console.print(Panel(
        f"[bold]Reasoning:[/bold]\n{advice.reasoning}",
        title="💭 Reasoning",
        border_style="blue",
    ))

    # Evidence table
    table = Table(title="📚 Evidence Used")
    table.add_column("Source", style="cyan", width=10)
    table.add_column("ID", style="yellow", width=25)
    table.add_column("Content", style="white")

    for e in advice.evidence[:6]:
        table.add_row(e.source.upper(), e.id[:24], e.content[:80] + "...")

    console.print(table)

    # Alternatives
    if advice.alternatives:
        alt_table = Table(title="🔄 Alternatives Considered")
        alt_table.add_column("Action", style="cyan")
        alt_table.add_column("Why Not", style="yellow")
        alt_table.add_column("Δ PC", justify="right")
        for alt in advice.alternatives[:3]:
            delta = f"{alt.delta_pitch_control*100:+.1f}%" if alt.delta_pitch_control else "—"
            alt_table.add_row(alt.action[:40], alt.why_not[:60], delta)
        console.print(alt_table)

    # Summary stats
    console.print(f"\n[bold]Confidence:[/bold] {advice.confidence:.2f}")
    console.print(f"[bold]Cited concepts:[/bold] {', '.join(advice.cited_concepts)}")

    if trace:
        total_ms = sum(s.get("latency_ms", 0) for s in trace)
        total_tokens = sum(s.get("tokens", 0) for s in trace)
        console.print(f"[bold]Total latency:[/bold] {total_ms:.0f}ms | [bold]Tokens:[/bold] {total_tokens}")

    console.print("\n[bold green]✅ Demo complete![/bold green]")


if __name__ == "__main__":
    app()
