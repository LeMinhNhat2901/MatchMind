"""
Demo Script - Run MatchMind on a StatsBomb or synthetic snapshot.

Usage:
    python -m matchmind.scripts.demo --synthetic
    python -m matchmind.scripts.demo --match-id 3788741 --event-idx 150 --player 14
"""
from __future__ import annotations

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


def _force_utf8_stdout() -> None:
    """Make Rich box-drawing / non-ASCII output safe on legacy Windows consoles.

    Guarded: some stdout objects (pytest capture, notebooks, already-wrapped
    streams) have no ``.buffer`` — silently skip in that case.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        enc = (getattr(stream, "encoding", "") or "").lower()
        if enc in ("utf-8", "utf8"):
            continue
        try:
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:  # Python 3.7+ text streams
                reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


@app.command()
def main(
    match_id: int = typer.Option(3788741, help="StatsBomb match ID"),
    event_idx: int = typer.Option(150, help="Event index in match"),
    player: int = typer.Option(0, help="Focus player ID (0 = auto-detect)"),
    synthetic: bool = typer.Option(
        False, "--synthetic/--no-synthetic", help="Use synthetic data (no StatsBomb needed)"
    ),
    dual: bool = typer.Option(
        False, "--dual/--no-dual",
        help="Analyse BOTH sides: the team in possession (on-ball) and the opposing "
             "team (how to stop it) — auto-picks a focus player per side. Runs the "
             "agent twice (2x LLM calls).",
    ),
    question: str = typer.Option("", help="Custom question for the agent (single-player mode only)"),
    out: str = typer.Option("", help="Path to save the annotated pitch image (default: output/pitch_images/)"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Enable debug logging"),
) -> None:
    """Run the MatchMind tactical agent and display the result."""
    _force_utf8_stdout()
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)

    console.print(
        Panel.fit("[bold green]MatchMind - Tactical Agent Demo[/bold green]", border_style="green")
    )

    # ── Load match state ───────────────────────────────────────
    state, focus_id = _load_state(synthetic, match_id, event_idx, player)

    if player > 0:
        valid_ids = [p.id for p in state.players]
        if player in valid_ids:
            focus_id = player
        else:
            console.print(
                f"[yellow]Player {player} not in this snapshot "
                f"(available: {valid_ids[:8]}...). Using player {focus_id}.[/yellow]"
            )

    console.print(
        f"\n[bold]Analysing:[/bold] Match {state.match_id} | "
        f"Player {focus_id} | Minute {state.minute}' | "
        f"velocity={'yes' if state.has_velocity else 'no (pitch control degraded)'}"
    )

    # ── Run agent ──────────────────────────────────────────────
    try:
        from matchmind.agent.graph import TacticalAgent
    except ImportError as exc:
        console.print(f"[bold red]Missing dependency: {exc}[/bold red]  → run `poetry install`")
        raise typer.Exit(1)

    console.print("\n[bold yellow]Running agent pipeline...[/bold yellow]")

    if dual:
        try:
            attacking, defending = TacticalAgent().analyze_dual(state)
        except Exception as exc:
            console.print(f"[bold red]Agent failed:[/bold red] {exc}")
            if "api" in str(exc).lower() or "key" in str(exc).lower():
                console.print("[yellow]Check your LLM provider's API key in .env[/yellow]")
            raise typer.Exit(1)

        att_advice, att_trace = attacking
        def_advice, def_trace = defending
        console.print(Panel.fit("[bold]ATTACKING TEAM (in possession)[/bold]", border_style="yellow"))
        _render(att_advice, att_trace)
        console.print(Panel.fit("[bold]DEFENDING TEAM (must stop it)[/bold]", border_style="magenta"))
        _render(def_advice, def_trace)
        _save_dual_image(state, att_advice, def_advice, out)
        return

    q = question or (
        f"What should player {focus_id} do in this situation to create tactical advantage?"
    )
    try:
        advice, trace = TacticalAgent().analyze(state, focus_id, question=q)
    except Exception as exc:
        console.print(f"[bold red]Agent failed:[/bold red] {exc}")
        if "api" in str(exc).lower() or "key" in str(exc).lower():
            console.print("[yellow]Check your LLM provider's API key in .env[/yellow]")
        raise typer.Exit(1)

    _render(advice, trace)
    _save_advice_image(state, advice, out)


def _load_state(synthetic: bool, match_id: int, event_idx: int, player: int):
    """Return (MatchState, focus_id), falling back to synthetic on any failure."""
    from matchmind.tests.fixtures.sample_data import make_sample_match_state

    if not synthetic:
        try:
            from matchmind.data_sources.statsbomb_adapter import StatsBombAdapter

            console.print(f"Loading StatsBomb match {match_id}, event {event_idx}...")
            state = StatsBombAdapter().load_snapshot(match_id=match_id, event_index=event_idx)
            console.print(f"[green]Loaded: {len(state.players)} players[/green]")
            return state, (player if player > 0 else state.players[0].id)
        except Exception as exc:
            console.print(f"[yellow]StatsBomb failed ({exc}); using synthetic data[/yellow]")

    state, fid = make_sample_match_state()
    console.print("[green]Using synthetic match state[/green]")
    return state, fid


def _render(advice, trace: list[dict]) -> None:
    # Situation snapshot (computed features)
    f = advice.situation_features
    if f is not None:
        baseline = next(
            (s.get("baseline_control") for s in trace if s.get("node") == "counterfactual"), None
        )
        lines = [
            f"dist to ball: {f.distance_to_ball:.1f}m   nearest opp: {f.nearest_opponent_distance:.1f}m"
            f"   space ahead: {f.space_ahead:.1f}m",
            f"local numerical adv: {f.local_numerical_advantage:+d}   "
            f"open lanes: {f.open_passing_lane_player_ids or '—'}",
            f"half-space: {f.half_space_occupied}   third-man: {f.third_man_opportunity}   "
            f"switch viable: {f.switch_play_viable}",
        ]
        if baseline is not None:
            lines.append(f"pitch-control baseline (focus team): {baseline * 100:.1f}%")
        console.print(Panel("\n".join(lines), title="Situation (computed)", border_style="magenta"))

    console.print(
        Panel(
            f"[bold green]RECOMMENDED ACTION[/bold green]\n\n{advice.recommended_action}",
            border_style="green",
        )
    )
    console.print(Panel(f"[bold]Reasoning:[/bold]\n{advice.reasoning}", title="Reasoning", border_style="blue"))

    ev = Table(title="Evidence Used")
    ev.add_column("Source", style="cyan", width=8)
    ev.add_column("ID", style="yellow", width=24)
    ev.add_column("Content", style="white")
    for e in advice.evidence[:6]:
        content = e.content if len(e.content) <= 90 else e.content[:87] + "..."
        ev.add_row(e.source.upper(), e.id[:24], content)
    console.print(ev)

    if advice.alternatives:
        alt = Table(title="Alternatives Considered")
        alt.add_column("Action", style="cyan")
        alt.add_column("Why Not", style="yellow")
        alt.add_column("Δ pitch control", justify="right")
        for a in advice.alternatives[:4]:
            d = f"{a.delta_pitch_control:+.1f}%" if a.delta_pitch_control is not None else "—"
            alt.add_row(a.action[:44], a.why_not[:64], d)
        console.print(alt)

    console.print(f"\n[bold]Confidence:[/bold] {advice.confidence:.2f}")
    console.print(f"[bold]Cited concepts:[/bold] {', '.join(advice.cited_concepts) or '(none)'}")
    if trace:
        total_ms = sum(s.get("latency_ms", 0) for s in trace)
        total_tokens = sum(s.get("tokens", 0) for s in trace)
        console.print(f"[bold]Latency:[/bold] {total_ms:.0f}ms | [bold]Tokens:[/bold] {total_tokens}")
    console.print("\n[bold green]Demo complete![/bold green]")


def _default_out_path(suffix: str) -> str:
    import time
    from matchmind.config import settings

    settings.pitch_images_dir.mkdir(parents=True, exist_ok=True)
    return str(settings.pitch_images_dir / f"advice_{suffix}_{int(time.time())}.png")


def _save_advice_image(state, advice, out: str) -> None:
    """Render the recommendation as an arrow (not just text/coordinates) and save it."""
    from matchmind.visualization.snapshot_renderer import render_advice_snapshot

    save_path = out or _default_out_path(str(advice.focus_player_id))
    try:
        render_advice_snapshot(state, advice, save_path)
        console.print(f"[bold]Annotated image:[/bold] {save_path}")
    except Exception as exc:
        console.print(f"[yellow]Could not render annotated image: {exc}[/yellow]")


def _save_dual_image(state, attacking_advice, defending_advice, out: str) -> None:
    from matchmind.visualization.snapshot_renderer import render_dual_advice_snapshot

    save_path = out or _default_out_path("dual")
    try:
        render_dual_advice_snapshot(state, attacking_advice, defending_advice, save_path)
        console.print(f"[bold]Annotated image (attack + defense):[/bold] {save_path}")
    except Exception as exc:
        console.print(f"[yellow]Could not render annotated image: {exc}[/yellow]")


if __name__ == "__main__":
    app()
