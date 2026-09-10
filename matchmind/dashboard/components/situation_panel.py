"""
Situation Panel — displays computed situation features.
"""
from __future__ import annotations

import streamlit as st

from matchmind.schema.match_state import MatchState


def render_situation_panel(state: MatchState, focus_player_id: int) -> None:
    """Render the computed situation features panel."""
    st.markdown("### 📊 Situation Analysis")

    focus = state.get_player(focus_player_id)
    if focus is None:
        st.warning(f"Player {focus_player_id} not found in this snapshot.")
        return

    # Compute features
    with st.spinner("Computing situation features..."):
        try:
            from matchmind.situation_engine.encoder import encode_situation
            features = encode_situation(state, focus_player_id)
        except Exception as exc:
            st.error(f"Situation engine error: {exc}")
            return

    # Player info
    role = focus.role or "Unknown"
    st.markdown(
        f"**Player {focus_player_id}** | {role.replace('_', ' ').title()} | "
        f"Team: {focus.team.title()} | Position: ({focus.x:.1f}m, {focus.y:.1f}m)"
    )

    # Spatial metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📏 Dist to Ball", f"{features.distance_to_ball:.1f}m")
        st.metric("👥 Nearest Opp", f"{features.nearest_opponent_distance:.1f}m",
                  delta=None, delta_color="off")
    with col2:
        st.metric("🚀 Space Ahead", f"{features.space_ahead:.1f}m")
        st.metric("🤝 Nearest TM", f"{features.nearest_teammate_distance:.1f}m")
    with col3:
        adv = features.local_numerical_advantage
        st.metric("⚡ Local Advantage", f"{adv:+d}",
                  delta=f"{'Overload' if adv > 0 else 'Underload' if adv < 0 else 'Equal'}",
                  delta_color="normal" if adv >= 0 else "inverse")

    st.divider()

    # Tactical flags
    st.markdown("**Tactical Flags**")
    flags = {
        "🔓 Passing Lane Open": features.passing_lane_open,
        "🎯 Progressive Passes": features.progressive_passes_available > 0,
        "👨‍👧‍👦 Third-Man Opportunity": features.third_man_opportunity,
        "↔️ Switch Play Viable": features.switch_play_viable,
        "🌓 Half-Space": features.half_space_occupied,
        "⚔️ Attacking Third": features.is_in_attacking_third,
        "🛡️ Defensive Third": features.is_in_defensive_third,
        "📦 Overload Left": features.overload_left,
        "📦 Overload Right": features.overload_right,
    }

    flag_cols = st.columns(2)
    for i, (label, value) in enumerate(flags.items()):
        with flag_cols[i % 2]:
            colour = "🟢" if value else "⚪"
            st.markdown(f"{colour} {label}")

    st.divider()

    # Team structure
    st.markdown("**Team Structure**")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.metric("🏰 Def Line", f"{features.defensive_line_height:.0f}m")
    with tc2:
        st.metric("📐 Compactness", f"{features.team_compactness:.1f}m")
    with tc3:
        st.metric("↔️ Attack Width", f"{features.attacking_width:.1f}m")

    # NL description expander
    with st.expander("💬 Natural Language Description (RAG Query)"):
        st.caption(features.natural_language_description)
