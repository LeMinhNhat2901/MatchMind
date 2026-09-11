"""
Pitch View Component — renders the match snapshot as an interactive Streamlit component.

Shows the BASE snapshot before the agent has run. Once advice is available it
switches to the annotated recommendation image (arrow drawn ON the minimap,
not just described in text) — single-player or dual attack/defense.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from matchmind.schema.match_state import MatchState, TacticalAdvice


def render_pitch_panel(
    state: MatchState,
    focus_player_id: int | None,
    show_pitch_control: bool = False,
    advice: TacticalAdvice | None = None,
    dual_advice: tuple[TacticalAdvice, TacticalAdvice] | None = None,
) -> None:
    """
    Render the pitch visualization panel.

    - No advice yet: base snapshot, ring on `focus_player_id`.
    - `advice` set: recommendation drawn as an arrow (gold=pass/dribble/shot,
      cyan=off-ball run) — this IS "the agent's answer, on the minimap".
    - `dual_advice` set: both sides' recommendations on one image (attack:
      white ring/gold-cyan arrow, defense: orange ring/magenta arrow).
    """
    if dual_advice is not None:
        st.markdown("### 🏟️ Match Snapshot — Attack vs Defense")
    elif advice is not None:
        st.markdown("### 🏟️ Match Snapshot — Recommendation")
    else:
        st.markdown("### 🏟️ Match Snapshot")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pc_grid = None

        # Pitch control is computed for whichever team the shown recommendation
        # (or the currently selected player) belongs to.
        pc_team = "home"
        if dual_advice is not None and dual_advice[0].situation_features:
            pc_team = dual_advice[0].situation_features.focus_player_team
        elif advice is not None and advice.situation_features:
            pc_team = advice.situation_features.focus_player_team
        elif focus_player_id is not None:
            fp = state.get_player(focus_player_id)
            if fp:
                pc_team = fp.team

        if show_pitch_control:
            with st.spinner("Computing pitch control..."):
                try:
                    from matchmind.pitch_control.spearman_model import (
                        generate_pitch_control_for_match_state,
                    )
                    _, _, pc_grid = generate_pitch_control_for_match_state(state, pc_team)
                except Exception as exc:
                    st.caption(f"⚠️ Pitch control unavailable: {exc}")

        if dual_advice is not None:
            from matchmind.visualization.snapshot_renderer import render_dual_advice_snapshot

            att_advice, def_advice = dual_advice
            render_dual_advice_snapshot(
                state=state,
                attacking_advice=att_advice,
                defending_advice=def_advice,
                save_path=tmp_path,
                show_pitch_control=show_pitch_control and pc_grid is not None,
                pitch_control_grid=pc_grid,
            )
        elif advice is not None:
            from matchmind.visualization.snapshot_renderer import render_advice_snapshot

            render_advice_snapshot(
                state=state,
                advice=advice,
                save_path=tmp_path,
                show_pitch_control=show_pitch_control and pc_grid is not None,
                pitch_control_grid=pc_grid,
            )
        else:
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
