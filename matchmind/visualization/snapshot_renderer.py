"""
Snapshot Renderer — Draw MatchState on Pitch.

Renders players, ball, and focus player highlight on the pitch image.
Optionally overlays pitch control heatmap.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from matchmind.schema.match_state import MatchState
from matchmind.visualization.pitch_renderer import _lazy_pyplot, plot_pitch

# matplotlib is NOT imported at module load — doing so before sentence-transformers
# / torch initialises causes a native DLL crash on Windows + Python 3.13.
logger = logging.getLogger(__name__)

# Team colours
HOME_COLOR = "#E74C3C"   # red
AWAY_COLOR = "#3498DB"   # blue
# White filled "+" — a distinct shape AND colour from the round red/blue
# player dots, so the ball never gets lost among them (was a yellow circle).
BALL_COLOR = "#FFFFFF"
BALL_EDGE_COLOR = "#111111"
BALL_MARKER = "P"        # matplotlib "filled plus"
BALL_SIZE = 220
FOCUS_RING_COLOR = "#FFFFFF"
FOCUS_RING_SIZE = 600
PLAYER_SIZE = 350
VISIBLE_AREA_COLOR = "#FFFFFF"

# Recommendation arrows (render_advice_snapshot): solid gold = the ball travels
# there (pass/dribble/shot/switch); dashed cyan = the PLAYER runs there and the
# ball stays with its real carrier (off-ball movement) — same colour coding as
# the moves_ball split in pitch_control/counterfactual_analyzer.py.
ARROW_BALL_COLOR = "#FFD700"
ARROW_RUN_COLOR = "#00E5FF"
ARROW_ALT_COLOR = "#CCCCCC"

# Dual (attack vs defense) rendering: the attacking focus keeps the standard
# white ring + gold/cyan arrow; the defending focus gets its own ring + arrow
# colour so the two recommendations never get confused on one image.
DEFENSE_RING_COLOR = "#FFA500"
DEFENSE_ARROW_COLOR = "#FF33CC"

# Fixed direction (up-right) the ball glyph is nudged toward when it would
# otherwise sit on top of a player dot and hide the jersey number.
_BALL_OFFSET_ANGLE_RAD = math.radians(50)


def _touch_offset_metres(ax, fig, marker_s1: float, marker_s2: float, gap_pts: float = 2.0) -> float:
    """Centre-to-centre data distance for two `scatter` markers to just touch.

    `s` in `ax.scatter` is marker area in points^2, so radius_pts = sqrt(s/pi).
    We measure the axes' actual points-per-data-unit (accounts for figsize,
    dpi, and set_aspect('equal')) rather than assuming pitch geometry.
    """
    r1 = math.sqrt(marker_s1 / math.pi)
    r2 = math.sqrt(marker_s2 / math.pi)
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 0.0))  # 1 data unit (metre) away in x
    px_per_dataunit = abs(p1[0] - p0[0]) or 1.0
    pts_per_dataunit = px_per_dataunit * 72.0 / fig.dpi
    data_per_pt = 1.0 / pts_per_dataunit if pts_per_dataunit > 0 else 1.0
    return (r1 + r2 + gap_pts) * data_per_pt


def _draw_scene(
    ax,
    fig,
    state: MatchState,
    focus_player_id: int | None,
    show_pitch_control: bool,
    pitch_control_grid: np.ndarray | None,
) -> None:
    """Draw the shared base layer: heatmap, 360 visible area, players, ball.

    Shared by `render_snapshot` (pre-advice, fed to the VLM) and
    `render_advice_snapshot` (post-advice, annotated with arrows) so the two
    never drift apart visually.
    """
    # ── Optional pitch control heatmap ────────────────────────
    if show_pitch_control and pitch_control_grid is not None:
        ax.imshow(
            pitch_control_grid,
            extent=[0, 105, 0, 68],
            origin="lower",
            cmap="RdBu",
            alpha=0.35,
            vmin=0, vmax=1,
            zorder=1,
        )

    # ── 360 camera field-of-view (StatsBomb only) ─────────────
    # Drawn first (low zorder) so it reads as pitch context, not a player.
    # Explains at a glance why fewer than 22 dots are shown.
    if state.visible_area and len(state.visible_area) >= 3:
        from matplotlib.patches import Polygon

        ax.add_patch(
            Polygon(
                state.visible_area,
                closed=True,
                fill=False,
                edgecolor=VISIBLE_AREA_COLOR,
                linestyle="--",
                linewidth=1.2,
                alpha=0.55,
                zorder=2,
            )
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
    # Highest zorder — must never be hidden behind a player dot. If the ball
    # sits on/inside the nearest player's dot (almost always true for the
    # actual ball carrier, since event location ≈ actor position), nudge the
    # ball glyph so the two markers are adjacent/touching instead of stacked —
    # otherwise the ball glyph covers that player's jersey-number label.
    draw_x, draw_y = state.ball.x, state.ball.y
    nearest = min(
        state.players,
        key=lambda p: (p.x - draw_x) ** 2 + (p.y - draw_y) ** 2,
        default=None,
    )
    if nearest is not None:
        dist = math.hypot(nearest.x - draw_x, nearest.y - draw_y)
        touch_dist = _touch_offset_metres(ax, fig, PLAYER_SIZE, BALL_SIZE)
        if dist < touch_dist:
            draw_x = nearest.x + touch_dist * math.cos(_BALL_OFFSET_ANGLE_RAD)
            draw_y = nearest.y + touch_dist * math.sin(_BALL_OFFSET_ANGLE_RAD)
            draw_x = max(0.0, min(105.0, draw_x))
            draw_y = max(0.0, min(68.0, draw_y))

    ax.scatter(
        draw_x, draw_y,
        c=BALL_COLOR,
        s=BALL_SIZE,
        edgecolors=BALL_EDGE_COLOR,
        linewidths=1.4,
        zorder=10,
        marker=BALL_MARKER,
    )


def _base_legend(extra: list | None = None) -> list:
    from matplotlib.lines import Line2D

    elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HOME_COLOR, markersize=10, label="Home"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AWAY_COLOR, markersize=10, label="Away"),
        Line2D(
            [0], [0], marker=BALL_MARKER, color="w", markerfacecolor=BALL_COLOR,
            markersize=10, label="Ball", markeredgecolor=BALL_EDGE_COLOR,
        ),
    ]
    return elements + (extra or [])


def render_snapshot(
    state: MatchState,
    focus_player_id: int | None,
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
    plt = _lazy_pyplot()
    fig, ax = plot_pitch(field_color=field_color, figsize=(10.5, 6.8))

    _draw_scene(ax, fig, state, focus_player_id, show_pitch_control, pitch_control_grid)

    # ── Title ──────────────────────────────────────────────────
    min_str = f"{state.minute}'"
    score_str = f"Home {state.score_home}–{state.score_away} Away"
    title = f"Match: {state.match_id} | {min_str} | {score_str}"
    if focus_player_id is not None:
        title += f" | Focus: Player {focus_player_id} (●)"
    if state.visible_area:
        # StatsBomb 360 only tracks players inside the broadcast camera view —
        # real games run ~4-21 of 22 players, never all 22. Not a data bug.
        title += f" | {len(state.players)} players in 360 view (dashed outline)"
    ax.set_title(
        title,
        fontsize=9, pad=8,
        color="white" if field_color == "dark_green" else "black",
    )

    # ── Legend ─────────────────────────────────────────────────
    ax.legend(
        handles=_base_legend(),
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


# ──────────────────────────────────────────────────────────────────────────────
# Post-advice rendering — draws the recommendation as an arrow, not just text.
# ──────────────────────────────────────────────────────────────────────────────

def _draw_action_arrow(
    ax,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    zorder: int = 8,
) -> None:
    ax.annotate(
        "",
        xy=end_xy,
        xytext=start_xy,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            shrinkA=10,
            shrinkB=10,
            mutation_scale=16,
        ),
        zorder=zorder,
    )


def _arrow_legend_entry(label: str, color: str, linestyle: str):
    from matplotlib.lines import Line2D

    return Line2D([0], [0], color=color, lw=2.5, linestyle=linestyle, label=label)


def render_advice_snapshot(
    state: MatchState,
    advice,  # TacticalAdvice — typed loosely to avoid a hard import-order dependency
    save_path: str,
    show_pitch_control: bool = False,
    pitch_control_grid: np.ndarray | None = None,
    field_color: str = "dark_green",
    dpi: int = 120,
    show_alternatives: bool = True,
) -> str:
    """
    Render the pitch AFTER the agent has produced advice, drawing the
    recommendation as an arrow instead of leaving the reader to infer
    movement from raw coordinates.

    - Solid gold arrow  = the BALL travels there (pass / dribble / shot / switch).
    - Dashed cyan arrow = the PLAYER runs there; the ball stays with its real
      carrier (off-ball movement — see situation_engine/candidate_actions.py).
    - Up to 2 alternatives are drawn faint/thin in grey if `show_alternatives`.

    Requires `advice.recommended_action_id` (and `.candidate_actions`) to be
    set — see agent/nodes/validator_node.py. If the LLM's recommendation
    didn't match a shortlist entry, no arrow is drawn and the title says so.

    Returns save_path (for convenience).
    """
    plt = _lazy_pyplot()
    fig, ax = plot_pitch(field_color=field_color, figsize=(10.5, 6.8))

    focus_id = advice.focus_player_id
    _draw_scene(ax, fig, state, focus_id, show_pitch_control, pitch_control_grid)

    focus = state.get_player(focus_id) if focus_id is not None else None
    legend_extra: list = []
    drew_recommended = False

    if focus is not None:
        rec = advice.resolve_action(advice.recommended_action_id)
        if rec is not None and rec.target_xy is not None:
            color = ARROW_BALL_COLOR if rec.moves_ball else ARROW_RUN_COLOR
            style = "solid" if rec.moves_ball else "dashed"
            _draw_action_arrow(ax, (focus.x, focus.y), rec.target_xy, color, style, 3.2, 1.0, zorder=8)
            legend_extra.append(
                _arrow_legend_entry("Recommended (ball)" if rec.moves_ball else "Recommended (run)", color, style)
            )
            drew_recommended = True

        if show_alternatives:
            drew_alt = False
            for alt in advice.alternatives[:2]:
                alt_cand = advice.resolve_action(alt.action_id)
                if alt_cand is None or alt_cand.target_xy is None:
                    continue
                style = "solid" if alt_cand.moves_ball else "dashed"
                _draw_action_arrow(ax, (focus.x, focus.y), alt_cand.target_xy, ARROW_ALT_COLOR, style, 1.6, 0.55, zorder=7)
                drew_alt = True
            if drew_alt:
                legend_extra.append(_arrow_legend_entry("Alternative", ARROW_ALT_COLOR, "dotted"))

    # ── Title: the recommendation itself, not just match metadata ─────
    action_text = advice.recommended_action
    if len(action_text) > 80:
        action_text = action_text[:77] + "..."
    title = f"Player {focus_id}: {action_text} (confidence {advice.confidence:.0%})"
    if not drew_recommended:
        title += "  [no geometric match — arrow not drawn]"
    ax.set_title(
        title,
        fontsize=9, pad=8,
        color="white" if field_color == "dark_green" else "black",
    )

    ax.legend(
        handles=_base_legend(legend_extra),
        loc="upper right",
        facecolor="black",
        labelcolor="white",
        fontsize=7,
        framealpha=0.7,
    )

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.debug(f"Advice snapshot saved to {save_path}")
    return save_path


def render_dual_advice_snapshot(
    state: MatchState,
    attacking_advice,  # TacticalAdvice for the team in possession
    defending_advice,  # TacticalAdvice for the opposing team
    save_path: str,
    show_pitch_control: bool = False,
    pitch_control_grid: np.ndarray | None = None,
    field_color: str = "dark_green",
    dpi: int = 120,
) -> str:
    """
    Render ONE pitch image with both sides' recommendations at once — the
    attacking player's arrow (gold/cyan, white ring) and the defending
    player's arrow (magenta, orange ring) — so "what should the team in
    possession do" and "how should the other team stop it" read together.

    See agent.graph.TacticalAgent.analyze_dual for producing the two advices.
    """
    plt = _lazy_pyplot()
    fig, ax = plot_pitch(field_color=field_color, figsize=(10.5, 6.8))

    att_id = attacking_advice.focus_player_id
    def_id = defending_advice.focus_player_id

    # Base scene with the attacking player's ring (standard white); the
    # defending player's ring is added separately below in a different colour.
    _draw_scene(ax, fig, state, att_id, show_pitch_control, pitch_control_grid)

    def_focus = state.get_player(def_id) if def_id is not None else None
    if def_focus is not None:
        ax.scatter(def_focus.x, def_focus.y, c=DEFENSE_RING_COLOR, s=FOCUS_RING_SIZE, zorder=3, linewidths=0)

    legend_extra: list = []

    att_focus = state.get_player(att_id) if att_id is not None else None
    if att_focus is not None:
        rec = attacking_advice.resolve_action(attacking_advice.recommended_action_id)
        if rec is not None and rec.target_xy is not None:
            color = ARROW_BALL_COLOR if rec.moves_ball else ARROW_RUN_COLOR
            style = "solid" if rec.moves_ball else "dashed"
            _draw_action_arrow(ax, (att_focus.x, att_focus.y), rec.target_xy, color, style, 3.2, 1.0, zorder=8)
            legend_extra.append(_arrow_legend_entry(f"Attack (P{att_id})", color, style))

    if def_focus is not None:
        rec2 = defending_advice.resolve_action(defending_advice.recommended_action_id)
        if rec2 is not None and rec2.target_xy is not None:
            style2 = "solid" if rec2.moves_ball else "dashed"
            _draw_action_arrow(ax, (def_focus.x, def_focus.y), rec2.target_xy, DEFENSE_ARROW_COLOR, style2, 3.2, 1.0, zorder=8)
            legend_extra.append(_arrow_legend_entry(f"Defense (P{def_id})", DEFENSE_ARROW_COLOR, style2))

    # ── Two-line title: one recommendation per side ──
    att_text = attacking_advice.recommended_action
    def_text = defending_advice.recommended_action
    att_text = att_text[:62] + ("..." if len(att_text) > 62 else "")
    def_text = def_text[:62] + ("..." if len(def_text) > 62 else "")
    title = f"ATTACK P{att_id}: {att_text}\nDEFENSE P{def_id}: {def_text}"
    ax.set_title(
        title,
        fontsize=8, pad=12,
        color="white" if field_color == "dark_green" else "black",
    )

    ax.legend(
        handles=_base_legend(legend_extra),
        loc="upper right",
        facecolor="black",
        labelcolor="white",
        fontsize=6.5,
        framealpha=0.7,
    )

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.debug(f"Dual advice snapshot saved to {save_path}")
    return save_path
