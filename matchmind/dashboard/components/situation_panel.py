"""
Situation Panel Component — player dossier & spatial telemetry.

Displays the Situation Engine's computed features as scouting-style cards.
"""
from __future__ import annotations

import html

import streamlit as st

from matchmind.dashboard.components.ui import icon, render_html
from matchmind.schema.match_state import MatchState


def _metric(label: str, value: str, unit: str, footer: str, tone: str = "") -> str:
    unit_html = f'<span class="metric-unit">{unit}</span>' if unit else ""
    return (
        f'<div class="metric-box"><div class="metric-label" title="{label}">{label}</div>'
        f'<div class="metric-value {tone}">{value}{unit_html}</div>'
        f'<div class="metric-footer">{footer}</div></div>'
    )


def render_situation_panel(state: MatchState, focus_player_id: int) -> None:
    """Render the tactical situation dossier for the selected focus player."""
    focus = state.get_player(focus_player_id)
    if focus is None:
        st.warning(f"Player #{focus_player_id} is not in this snapshot.")
        return

    with st.spinner("Computing spatial metrics..."):
        try:
            from matchmind.situation_engine.encoder import encode_situation

            features = encode_situation(state, focus_player_id)
        except Exception as exc:
            st.error(f"Situation Engine error: {exc}")
            return

    third_str = "Attacking third" if features.is_in_attacking_third else (
        "Defensive third" if features.is_in_defensive_third else "Middle third"
    )
    lane_str = "Half-space" if features.half_space_occupied else "Central channel"

    opp_dist = features.nearest_opponent_distance
    if opp_dist < 2.0:
        pressure_tag = '<span class="pressure-pill pressure-high"><span class="status-dot"></span>High pressure</span>'
        opp_tone = "bad"
    elif opp_dist < 4.0:
        pressure_tag = '<span class="pressure-pill pressure-med"><span class="status-dot"></span>Contested</span>'
        opp_tone = "warn"
    else:
        pressure_tag = '<span class="pressure-pill pressure-low"><span class="status-dot"></span>Unmarked</span>'
        opp_tone = "good"

    is_carrier = features.is_ball_carrier
    carrier_badge = (
        f'<span class="tactical-badge badge-carrier">{icon("ball", 13)}On-ball</span>'
        if is_carrier
        else f'<span class="tactical-badge badge-offball">{icon("footprints", 13)}Off-ball</span>'
    )

    team_color_class = "team-home-badge" if focus.team == "home" else "team-away-badge"
    role_label = html.escape((focus.role or "Player").replace("_", " "))

    render_html(
        f"""
        <div class="scout-card">
            <div class="scout-header">
                <div class="scout-identity">
                    <div class="jersey-circle {team_color_class}">{focus.id}</div>
                    <div>
                        <div class="scout-role">{role_label}</div>
                        <div class="scout-team-sub">{focus.team.title()} team · {third_str} · {lane_str}</div>
                    </div>
                </div>
                <div class="scout-badges">{carrier_badge}{pressure_tag}</div>
            </div>
            <div class="coords-strip">
                <span>X {focus.x:.1f} m</span><span>Y {focus.y:.1f} m</span><span>Speed {focus.speed:.1f} m/s</span>
            </div>
        </div>
        """
    )

    adv = features.local_numerical_advantage
    if adv > 0:
        adv_tone, adv_text = "good", "Overload"
    elif adv < 0:
        adv_tone, adv_text = "bad", "Underload"
    else:
        adv_tone, adv_text = "", "Balanced"
    carrier_info = "In possession" if is_carrier else f"Carrier #{features.ball_carrier_id or '?'}"

    render_html('<div class="mm-kicker">Spatial telemetry</div>')
    render_html(
        '<div class="metric-grid">'
        + _metric("Ball distance", f"{features.distance_to_ball:.1f}", "m", carrier_info)
        + _metric("Nearest opponent", f"{opp_dist:.1f}", "m",
                  f"Teammate {features.nearest_teammate_distance:.1f}&nbsp;m", opp_tone)
        + _metric("Local balance · 15 m", f"{adv:+d}", "", adv_text, adv_tone)
        + "</div>"
    )

    flags = [
        ("Passing lane", "Open" if features.passing_lane_open else "Blocked", features.passing_lane_open),
        ("Progressive route", f"{features.progressive_passes_available} available",
         features.progressive_passes_available > 0),
        ("Third-man option", "Viable" if features.third_man_opportunity else "None", features.third_man_opportunity),
        ("Switch of play", "Feasible" if features.switch_play_viable else "Not viable", features.switch_play_viable),
        ("Half-space", "Occupied" if features.half_space_occupied else "Outside", features.half_space_occupied),
        ("Flank overload", "Left" if features.overload_left else ("Right" if features.overload_right else "Central"),
         features.overload_left or features.overload_right),
    ]
    rows = "".join(
        f'<div class="tactical-flag-row{" on" if on else ""}">'
        f'<span class="flag-label">{label}</span>'
        f'<span class="flag-val {"val-active" if on else "val-dim"}">{val}</span></div>'
        for label, val, on in flags
    )
    render_html('<div class="mm-kicker">Lanes &amp; options</div>')
    render_html(f'<div class="flag-grid">{rows}</div>')

    def_height = features.defensive_line_height
    line_label = "High line" if def_height > 60 else ("Mid block" if def_height > 40 else "Low block")
    render_html('<div class="mm-kicker">Team shape</div>')
    render_html(
        '<div class="metric-grid">'
        + _metric("Defensive line", f"{def_height:.1f}", "m", line_label)
        + _metric("Compactness", f"{features.team_compactness:.1f}", "m", "Block spread")
        + _metric("Attack width", f"{features.attacking_width:.1f}", "m", "Pitch coverage")
        + "</div>"
    )

    with st.expander("Situation briefing", icon=":material/description:"):
        render_html(f'<div class="analyst-memo">{html.escape(features.natural_language_description)}</div>')
