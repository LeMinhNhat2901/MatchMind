"""
Pitch View Component — broadcast-style minimap card.

Header, pitch image and legend render as ONE HTML block: the image is inlined
as base64 so it sits inside the card. (Wrapping st.image between two separate
st.markdown calls doesn't work — Streamlit closes each markdown element on its
own, so the frame never actually contained the image.)
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import streamlit as st

from matchmind.dashboard.components.ui import icon, render_html
from matchmind.schema.match_state import MatchState, TacticalAdvice


def _legend_items(advice: TacticalAdvice | None, dual: bool) -> list[str]:
    """Only the marks actually drawn on this image — no dead legend entries."""
    items = [
        '<span class="legend-item"><span class="legend-dot red-dot"></span>Home</span>',
        '<span class="legend-item"><span class="legend-dot blue-dot"></span>Away</span>',
        '<span class="legend-item"><span class="legend-dot ball-dot"></span>Ball</span>',
        '<span class="legend-item"><span class="legend-dot ring-dot"></span>Focus player</span>',
    ]
    if advice is not None or dual:
        items += [
            '<span class="legend-item"><span class="legend-dot gold-line"></span>Ball action</span>',
            '<span class="legend-item"><span class="legend-dot cyan-dash"></span>Off-ball run</span>',
        ]
    if advice is not None and not dual:
        items.append('<span class="legend-item"><span class="legend-dot grey-line"></span>Alternative</span>')
    if dual:
        items.append('<span class="legend-item"><span class="legend-dot magenta-line"></span>Defensive action</span>')
    return items


def render_pitch_panel(
    state: MatchState,
    focus_player_id: int | None,
    show_pitch_control: bool = False,
    advice: TacticalAdvice | None = None,
    dual_advice: tuple[TacticalAdvice, TacticalAdvice] | None = None,
) -> None:
    """
    Render the minimap card.

    - Base snapshot: player dots, ball, focus ring.
    - Advice active: recommendation arrow (gold = ball moves, cyan = player runs).
    - Dual mode: attack (gold/cyan) + defense (magenta) arrows on one image.
    """
    if dual_advice is not None:
        title = "Attack vs Defense"
        mode_badge = f'<span class="tactical-badge badge-amber">{icon("swords", 13)}Dual phase</span>'
    elif advice is not None:
        title = "Tactical Directive"
        mode_badge = f'<span class="tactical-badge badge-green">{icon("target", 13)}Directive</span>'
    else:
        title = "Match Snapshot"
        mode_badge = '<span class="tactical-badge badge-blue"><span class="status-dot"></span>Live feed</span>'

    pc_team = "home"
    if dual_advice is not None and dual_advice[0].situation_features:
        pc_team = dual_advice[0].situation_features.focus_player_team
    elif advice is not None and advice.situation_features:
        pc_team = advice.situation_features.focus_player_team
    elif focus_player_id is not None:
        fp = state.get_player(focus_player_id)
        if fp:
            pc_team = fp.team

    pc_grid = None
    if show_pitch_control:
        with st.spinner("Computing pitch control surface..."):
            try:
                from matchmind.pitch_control.spearman_model import generate_pitch_control_for_match_state

                _, _, pc_grid = generate_pitch_control_for_match_state(state, pc_team)
            except Exception as exc:
                st.caption(f"Pitch control unavailable: {exc}")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        common = dict(
            save_path=tmp_path,
            show_pitch_control=show_pitch_control and pc_grid is not None,
            pitch_control_grid=pc_grid,
            annotate=False,
        )
        if dual_advice is not None:
            from matchmind.visualization.snapshot_renderer import render_dual_advice_snapshot

            render_dual_advice_snapshot(
                state=state, attacking_advice=dual_advice[0], defending_advice=dual_advice[1], **common
            )
        elif advice is not None:
            from matchmind.visualization.snapshot_renderer import render_advice_snapshot

            render_advice_snapshot(state=state, advice=advice, **common)
        else:
            from matchmind.visualization.snapshot_renderer import render_snapshot

            render_snapshot(state=state, focus_player_id=focus_player_id, **common)

        img_b64 = base64.b64encode(Path(tmp_path).read_bytes()).decode("ascii")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    focus_txt = f"#{focus_player_id}" if focus_player_id is not None else "—"
    alt_text = f"Pitch minimap, {title.lower()}, focus player {focus_txt}"
    legend = "".join(_legend_items(advice, dual_advice is not None))

    render_html(
        f"""
        <div class="pitch-card">
            <div class="pitch-monitor-header">
                <div class="pitch-monitor-title">
                    {icon("pitch", 18)}
                    <span>{title}</span>
                    {mode_badge}
                </div>
                <div class="pitch-meta-specs">
                    <span>105 × 68 m</span>
                    <span>·</span>
                    <span>Attack</span>
                    {icon("arrow-right", 14)}
                </div>
            </div>
            <img class="pitch-image" src="data:image/png;base64,{img_b64}" alt="{alt_text}">
            <div class="tactical-legend-bar">{legend}</div>
        </div>
        """
    )
