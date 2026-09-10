"""
Snapshot Renderer — Draw MatchState on Pitch.

Renders players, ball, and focus player highlight on the pitch image.
Optionally overlays pitch control heatmap.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matchmind.schema.match_state import MatchState
from matchmind.visualization.pitch_renderer import plot_pitch

logger = logging.getLogger(__name__)

# Team colours
HOME_COLOR = "#E74C3C"   # red
AWAY_COLOR = "#3498DB"   # blue
BALL_COLOR = "#F1C40F"   # yellow
FOCUS_RING_COLOR = "#FFFFFF"
FOCUS_RING_SIZE = 600
PLAYER_SIZE = 350


def render_snapshot(
    state: MatchState,
    focus_player_id: int,
    save_path: str,
    show_pitch_control: bool = False,
    pitch_control_grid: np.ndarray | None = None,
    field_color: str = "dark_green",
    dpi: int = 120,
) -> str:
    """
    Render a MatchState as a pitch image PNG.

    Args:
        state: Match snapshot to visualise.
        focus_player_id: Player to highlight with a white ring.
        save_path: File path to save the PNG.
        show_pitch_control: If True and grid provided, overlay heatmap.
        pitch_control_grid: 2D array from spearman_model (optional).
        field_color: Pitch background colour.
        dpi: Image DPI.

    Returns:
        save_path (for convenience).
    """
    fig, ax = plot_pitch(field_color=field_color, figsize=(10.5, 6.8))

    # ── Optional pitch control heatmap ────────────────────────
    if show_pitch_control and pitch_control_grid is not None:
        extent = [0, 105, 0, 68]
        ax.imshow(
            pitch_control_grid,
            extent=extent,
            origin="lower",
            cmap="RdBu",
            alpha=0.35,
            vmin=0, vmax=1,
            zorder=1,
        )

    # ── Draw players ─────────────────────────────────────────
    for player in state.players:
        color = HOME_COLOR if player.team == "home" else AWAY_COLOR
        is_focus = player.id == focus_player_id

        # Highlight ring for focus player
        if is_focus:
            ax.scatter(
                player.x, player.y,
                c=FOCUS_RING_COLOR,
                s=FOCUS_RING_SIZE,
                zorder=3,
                linewidths=0,
            )

        # Player dot
        ax.scatter(
            player.x, player.y,
            c=color,
            s=PLAYER_SIZE,
            zorder=4,
            edgecolors="white" if is_focus else "none",
            linewidths=2 if is_focus else 0,
        )

        # Player ID label
        ax.annotate(
            str(player.id),
            (player.x, player.y),
            ha="center", va="center",
            fontsize=7, fontweight="bold",
            color="white",
            zorder=5,
        )

    # ── Draw ball ─────────────────────────────────────────────
    ax.scatter(
        state.ball.x, state.ball.y,
        c=BALL_COLOR,
        s=150,
        edgecolors="black",
        linewidths=1.5,
        zorder=6,
        marker="o",
    )

    # ── Title ──────────────────────────────────────────────────
    min_str = f"{state.minute}'"
    score_str = f"Home {state.score_home}–{state.score_away} Away"
    ax.set_title(
        f"Match: {state.match_id} | {min_str} | {score_str} | "
        f"Focus: Player {focus_player_id} (●)",
        fontsize=9, pad=8,
        color="white" if field_color == "dark_green" else "black",
    )

    # ── Legend ─────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HOME_COLOR, markersize=10, label="Home"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AWAY_COLOR, markersize=10, label="Away"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BALL_COLOR, markersize=8, label="Ball", markeredgecolor="black"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        facecolor="black",
        labelcolor="white",
        fontsize=7,
        framealpha=0.7,
    )

    # ── Save ───────────────────────────────────────────────────
    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.debug(f"Snapshot saved to {save_path}")
    return save_path
