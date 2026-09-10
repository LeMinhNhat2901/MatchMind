"""
Pitch View Component — renders the match snapshot as an interactive Streamlit component.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from matchmind.schema.match_state import MatchState


def render_pitch_panel(
    state: MatchState,
    focus_player_id: int,
    show_pitch_control: bool = False,
) -> None:
    """Render the pitch visualization panel."""
    st.markdown("### 🏟️ Match Snapshot")

    # Render pitch image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        import numpy as np
        pc_grid = None

        if show_pitch_control:
            with st.spinner("Computing pitch control..."):
                try:
                    from matchmind.pitch_control.spearman_model import (
                        generate_pitch_control_for_match_state,
                    )
                    focus_player = state.get_player(focus_player_id)
                    team = focus_player.team if focus_player else "home"
                    _, _, pc_grid = generate_pitch_control_for_match_state(state, team)
                except Exception as exc:
                    st.caption(f"⚠️ Pitch control unavailable: {exc}")

        from matchmind.visualization.snapshot_renderer import render_snapshot
        render_snapshot(
            state=state,
            focus_player_id=focus_player_id,
            save_path=tmp_path,
            show_pitch_control=show_pitch_control and pc_grid is not None,
            pitch_control_grid=pc_grid,
        )
        st.image(tmp_path, use_container_width=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Match info pills
    score_col, min_col, phase_col = st.columns(3)
    with score_col:
        st.markdown(
            f'<span class="pill pill-blue">⚽ {state.score_home}–{state.score_away}</span>',
            unsafe_allow_html=True,
        )
    with min_col:
        st.markdown(
            f'<span class="pill pill-amber">⏱️ {state.minute}\'</span>',
            unsafe_allow_html=True,
        )
    with phase_col:
        phase = state.phase_of_play or "open play"
        st.markdown(
            f'<span class="pill pill-green">📋 {phase.replace("_", " ").title()}</span>',
            unsafe_allow_html=True,
        )
