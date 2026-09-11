"""
Pitch Renderer — adapted from LaurieOnTracking/Metrica_Viz.py.

Draws a proper football pitch with standard markings.
Adapted to use our coordinate system (0–105m × 0–68m, origin bottom-left).

Original: @EightyFivePoint (Laurie Shaw)
Adaptation: MatchMind project
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

# matplotlib is imported lazily inside plot_pitch(). Importing it at module load
# time (before sentence-transformers / torch initialises) triggers a native DLL
# crash on Windows + Python 3.13. See _lazy_pyplot().
if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def _lazy_pyplot():
    """Import matplotlib.pyplot with the Agg backend, on first use only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def plot_pitch(
    field_color: Literal["green", "dark_green", "white"] = "dark_green",
    linewidth: float = 2,
    markersize: float = 20,
    figsize: tuple[float, float] = (10.5, 6.8),
    ax: "Axes | None" = None,
) -> "tuple[Figure, Axes]":
    """
    Plot a full football pitch with proper markings.

    Adapted from LaurieOnTracking/Metrica_Viz.py (Laurie Shaw @EightyFivePoint).
    Coordinate system: 0–105m (x, left→right) × 0–68m (y, bottom→top).

    Args:
        field_color: Background colour.
        linewidth: Line width for pitch markings.
        markersize: Size of penalty spot etc.
        figsize: Figure dimensions in inches.
        ax: Existing axes to draw on. If None, creates new figure.

    Returns:
        (fig, ax) — matplotlib figure and axes.
    """
    if ax is None:
        plt = _lazy_pyplot()
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Colour scheme
    colors = {
        "green": {"bg": "#4CAF50", "line": "white", "spot": "white"},
        "dark_green": {"bg": "#2D6A2D", "line": "#F0F0E8", "spot": "#F0F0E8"},
        "white": {"bg": "white", "line": "black", "spot": "black"},
    }
    c = colors.get(field_color, colors["dark_green"])

    ax.set_facecolor(c["bg"])
    fig.patch.set_facecolor(c["bg"])
    lc = c["line"]  # line colour
    pc = c["spot"]  # spot colour

    # ── Pitch dimensions ──────────────────────────────────────
    hl = PITCH_LENGTH / 2  # half length
    hw = PITCH_WIDTH / 2   # half width

    metres_per_yard = 0.9144
    goal_line_width = 8 * metres_per_yard
    box_width = 20 * metres_per_yard        # 6-yard box width
    box_length = 6 * metres_per_yard        # 6-yard box length
    area_width = 44 * metres_per_yard       # penalty area width
    area_length = 18 * metres_per_yard      # penalty area length
    penalty_spot_dist = 12 * metres_per_yard
    corner_radius = 1 * metres_per_yard
    centre_circle_radius = 10 * metres_per_yard
    D_length = 8 * metres_per_yard
    D_radius = 10 * metres_per_yard
    D_pos = 12 * metres_per_yard

    # ── Centre circle & halfway line ─────────────────────────
    ax.plot([hl, hl], [0, PITCH_WIDTH], color=lc, linewidth=linewidth)
    theta = np.linspace(0, 2 * np.pi, 150)
    ax.plot(
        hl + centre_circle_radius * np.cos(theta),
        hw + centre_circle_radius * np.sin(theta),
        color=lc, linewidth=linewidth,
    )
    ax.scatter(hl, hw, marker="o", facecolor=pc, linewidth=0, s=markersize)

    # ── Pitch boundary ────────────────────────────────────────
    ax.plot([0, PITCH_LENGTH], [0, 0], color=lc, linewidth=linewidth)
    ax.plot([0, PITCH_LENGTH], [PITCH_WIDTH, PITCH_WIDTH], color=lc, linewidth=linewidth)
    ax.plot([0, 0], [0, PITCH_WIDTH], color=lc, linewidth=linewidth)
    ax.plot([PITCH_LENGTH, PITCH_LENGTH], [0, PITCH_WIDTH], color=lc, linewidth=linewidth)

    for sign in [-1, 1]:
        gx = 0 if sign == -1 else PITCH_LENGTH
        gx_inner = box_length if sign == -1 else PITCH_LENGTH - box_length
        gx_area = area_length if sign == -1 else PITCH_LENGTH - area_length

        # Goal
        ax.plot(
            [gx, gx],
            [hw - goal_line_width / 2, hw + goal_line_width / 2],
            color=pc, linewidth=linewidth * 2,
        )

        # 6-yard box
        ax.plot([gx, gx_inner], [hw + box_width / 2, hw + box_width / 2], color=lc, linewidth=linewidth)
        ax.plot([gx, gx_inner], [hw - box_width / 2, hw - box_width / 2], color=lc, linewidth=linewidth)
        ax.plot([gx_inner, gx_inner], [hw - box_width / 2, hw + box_width / 2], color=lc, linewidth=linewidth)

        # Penalty area
        ax.plot([gx, gx_area], [hw + area_width / 2, hw + area_width / 2], color=lc, linewidth=linewidth)
        ax.plot([gx, gx_area], [hw - area_width / 2, hw - area_width / 2], color=lc, linewidth=linewidth)
        ax.plot([gx_area, gx_area], [hw - area_width / 2, hw + area_width / 2], color=lc, linewidth=linewidth)

        # Penalty spot
        pspot_x = penalty_spot_dist if sign == -1 else PITCH_LENGTH - penalty_spot_dist
        ax.scatter(pspot_x, hw, marker="o", facecolor=pc, linewidth=0, s=markersize * 0.5)

        # Penalty D arc
        arc_angles = np.linspace(0, 2 * np.pi, 200)
        arc_x = pspot_x + D_radius * np.cos(arc_angles)
        arc_y = hw + D_radius * np.sin(arc_angles)
        mask = arc_x > gx_area if sign == -1 else arc_x < gx_area
        ax.plot(arc_x[mask], arc_y[mask], color=lc, linewidth=linewidth)

        # Corner arcs
        for cy in [0, PITCH_WIDTH]:
            corner_angles = np.linspace(
                (0 if cy == 0 else -np.pi / 2) + (0 if sign == -1 else np.pi / 2),
                (np.pi / 2 if cy == 0 else 0) + (0 if sign == -1 else np.pi / 2),
                50,
            )
            ax.plot(
                gx + corner_radius * np.cos(corner_angles),
                cy + corner_radius * np.sin(corner_angles),
                color=lc, linewidth=linewidth,
            )

    ax.set_xlim(-2, PITCH_LENGTH + 2)
    ax.set_ylim(-2, PITCH_WIDTH + 2)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax
