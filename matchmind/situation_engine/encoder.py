"""
Situation Encoder — Main Entry Point for Situation Engine.

Aggregates spatial_features + team_structure + passing_analyzer into a
unified SituationFeatures object, plus a natural-language description
for use as the RAG query and prompt context.

This is the last purely-computational step before the LangGraph agent.
NO LLM is used here.
"""
from __future__ import annotations

import logging

from matchmind.schema.match_state import MatchState, SituationFeatures
from matchmind.situation_engine.orientation import orient_for_team
from matchmind.situation_engine.spatial_features import compute_spatial_features
from matchmind.situation_engine.team_structure import compute_team_structure
from matchmind.situation_engine.passing_analyzer import compute_passing_opportunities

logger = logging.getLogger(__name__)

# Event types where event_player_id is the actual ball carrier at the snapshot.
_ON_BALL_EVENT_TYPES = {"Pass", "Carry", "Dribble", "Shot", "Dispossessed", "Miscontrol", "Ball Receipt*"}


def _determine_ball_carrier(state: MatchState) -> int | None:
    """Best-effort id of the player who currently has the ball.

    Prefers StatsBomb's event_player_id (ground truth for the acting player on
    on-ball event types) and falls back to "nearest player to the ball" for
    sources without that metadata (Metrica, synthetic, off-ball StatsBomb
    events like Pressure).
    """
    if (
        state.event_player_id is not None
        and (state.event_type in _ON_BALL_EVENT_TYPES if state.event_type else False)
        and state.get_player(state.event_player_id) is not None
    ):
        return state.event_player_id

    if not state.players:
        return None
    nearest = min(
        state.players,
        key=lambda p: (p.x - state.ball.x) ** 2 + (p.y - state.ball.y) ** 2,
    )
    return nearest.id


# Public alias — other modules (team_focus, dashboard, API) shouldn't reach
# into a "private" name just because this lives in the same package.
determine_ball_carrier = _determine_ball_carrier


def encode_situation(
    state: MatchState,
    focus_player_id: int,
) -> SituationFeatures:
    """
    Full situation encoding pipeline.

    Steps:
    1. Compute spatial features (distances, zones, numerical advantage)
    2. Compute team structure (defensive line, compactness, width)
    3. Compute passing opportunities (open lanes, third-man, switch play)
    4. Build natural-language description (rule-based)
    5. Assemble → SituationFeatures

    Args:
        state: Unified MatchState.
        focus_player_id: Player to analyse.

    Returns:
        SituationFeatures — structured representation for RAG + agent.
    """
    focus = state.get_player(focus_player_id)
    if focus is None:
        raise ValueError(
            f"Player {focus_player_id} not found in match {state.match_id}. "
            f"Available IDs: {[p.id for p in state.players]}"
        )

    logger.debug(f"Encoding situation for player {focus_player_id} in match {state.match_id}")

    # Every calculation below assumes the focus team attacks toward +x; turn the
    # pitch round when it doesn't (no-op for an already-oriented state).
    state, _ = orient_for_team(state, focus.team)
    focus = state.get_player(focus_player_id, focus.team)

    # ── Step 1: Spatial ──────────────────────────────────────
    spatial = compute_spatial_features(state, focus_player_id)
    # Extract pre-computed lists for reuse
    teammates = spatial.pop("_teammates")
    opponents = spatial.pop("_opponents")
    focus_player = spatial.pop("_focus")

    # ── Step 2: Team structure ───────────────────────────────
    team_struct = compute_team_structure(state, focus.team)

    # ── Step 3: Passing opportunities ────────────────────────
    passing = compute_passing_opportunities(
        state, focus_player_id,
        teammates=teammates,
        opponents=opponents,
    )

    # ── Step 3b: Who has the ball ─────────────────────────────
    carrier_id = _determine_ball_carrier(state)
    is_ball_carrier = carrier_id == focus_player_id

    # ── Step 4: Natural language description ─────────────────
    nl_description = _build_nl_description(
        state=state,
        focus=focus_player,
        spatial=spatial,
        team_struct=team_struct,
        passing=passing,
        carrier_id=carrier_id,
        is_ball_carrier=is_ball_carrier,
    )

    # ── Step 5: Computed stats (for evidence assembly) ────────
    computed_stats = {
        **{k: v for k, v in spatial.items() if not k.startswith("angles")},
        **team_struct,
        **passing,
        "ball_x": state.ball.x,
        "ball_y": state.ball.y,
        "focus_player_x": focus.x,
        "focus_player_y": focus.y,
        "minute": state.minute,
        # From the focus player's team (positive = their team leads).
        "score_diff": state.score_for(focus.team)[0] - state.score_for(focus.team)[1],
        "open_lanes_count": len(passing.get("open_passing_lane_player_ids", [])),
        "progressive_passes_available": passing.get("progressive_passes_available", 0),
    }

    # ── Step 6: Assemble SituationFeatures ───────────────────
    return SituationFeatures(
        focus_player_id=focus_player_id,
        focus_player_team=focus.team,
        focus_player_role=focus.role,
        # Spatial
        distance_to_ball=spatial["distance_to_ball"],
        nearest_opponent_distance=spatial["nearest_opponent_distance"],
        nearest_teammate_distance=spatial["nearest_teammate_distance"],
        space_ahead=spatial["space_ahead"],
        # Passing
        passing_lane_open=passing["passing_lane_open"],
        progressive_passes_available=passing["progressive_passes_available"],
        third_man_opportunity=passing["third_man_opportunity"],
        switch_play_viable=passing["switch_play_viable"],
        open_passing_lane_player_ids=passing["open_passing_lane_player_ids"],
        # Local numerical
        local_numerical_advantage=spatial["local_numerical_advantage"],
        overload_left=spatial["overload_left"],
        overload_right=spatial["overload_right"],
        # Half-space & zones
        half_space_occupied=spatial["half_space_occupied"],
        is_in_attacking_third=spatial["is_in_attacking_third"],
        is_in_defensive_third=spatial["is_in_defensive_third"],
        # Team structure
        defensive_line_height=team_struct["defensive_line_height"],
        defensive_line_visible=team_struct["defensive_line_visible"],
        team_compactness=team_struct["team_compactness"],
        attacking_width=team_struct["attacking_width"],
        numerical_superiority_zone=team_struct["numerical_superiority_zone"],
        # Possession
        is_ball_carrier=is_ball_carrier,
        ball_carrier_id=carrier_id,
        # NL description
        natural_language_description=nl_description,
        # Computed stats dict
        computed_stats=computed_stats,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Natural language description builder (rule-based, no LLM)
# ──────────────────────────────────────────────────────────────────────────────

def _build_nl_description(
    state: MatchState,
    focus,
    spatial: dict,
    team_struct: dict,
    passing: dict,
    carrier_id: int | None,
    is_ball_carrier: bool,
) -> str:
    """
    Build a human-readable situation description from computed features.

    This string serves two purposes:
    1. Query for ChromaDB vector retrieval
    2. Context block in the LLM prompt

    Rule-based — no LLM involved.
    """
    lines = []

    # Player context — the real team name when the source provides it (StatsBomb
    # "home" is the acting team, not the match's home side).
    details = [d for d in (focus.name, focus.role) if d]
    role_str = f" ({', '.join(details)})" if details else ""
    named = state.home_team_name if focus.team == "home" else state.away_team_name
    team_desc = f"({named})" if named else f"[{focus.team} team]"
    lines.append(
        f"Player {focus.id}{role_str} {team_desc} at position "
        f"({focus.x:.1f}m, {focus.y:.1f}m)."
    )
    # Partial camera views (StatsBomb 360, video) only show players near the ball;
    # say so, so the LLM doesn't treat absent players as absent from the pitch.
    if state.visible_area or len(state.players) < 22:
        lines.append(f"Camera view shows {len(state.players)} of 22 players.")
    if state.source == "statsbomb":
        # 360 data identifies only the event's player; the rest are anonymous.
        lines.append(
            "Only the player making the event is identified in this data; the other players' "
            "numbers (teammates 51+, opponents 101+) are placeholders, not shirt numbers."
        )

    # Possession — stated up front so retrieval + the LLM never confuse an
    # off-ball player for the ball carrier.
    if is_ball_carrier:
        lines.append(f"Player {focus.id} CURRENTLY HAS THE BALL.")
    else:
        carrier = state.get_player(carrier_id) if carrier_id is not None else None
        if carrier is not None:
            same_team = "teammate" if carrier.team == focus.team else "opponent"
            lines.append(
                f"Player {focus.id} does NOT have the ball — Player {carrier.id} "
                f"({same_team}) currently has it. Analyse Player {focus.id}'s "
                f"OFF-THE-BALL movement/positioning, not a pass/dribble/shot decision."
            )
        else:
            lines.append(f"Player {focus.id} does NOT have the ball.")

    # Minute and score
    # Told from the focus player's side ("home trailing" was misread as "our team
    # is trailing" when the focus player was on the away team).
    lines.append(
        f"Match situation: minute {state.minute}, {state.score_line_for(focus.id, focus.team)}. "
        f"Phase: {state.phase_of_play or 'open play'}."
    )

    # Distance to ball
    d_ball = spatial["distance_to_ball"]
    lines.append(f"Distance to ball: {d_ball:.1f}m.")

    # Pressure
    d_opp = spatial["nearest_opponent_distance"]
    if is_ball_carrier:
        pressure_desc = (
            "under high pressure" if d_opp < 3 else
            "with a nearby opponent" if d_opp < 7 else
            "with space from nearest opponent"
        )
        lines.append(f"Player is {pressure_desc} ({d_opp:.1f}m away).")
    else:
        # Off the ball "pressure" is ambiguous — a defender 2.7 m from the carrier
        # is the one APPLYING pressure. State the fact only.
        lines.append(f"Nearest opponent: {d_opp:.1f}m away.")

    # Space ahead
    space = spatial["space_ahead"]
    lines.append(
        f"Space ahead toward goal: {space:.1f}m "
        f"({'large space' if space > 15 else 'limited space'})."
    )

    # Numerical situation
    num_adv = spatial["local_numerical_advantage"]
    if num_adv > 0:
        lines.append(f"Local numerical overload: +{num_adv} teammates nearby (within 15m).")
    elif num_adv < 0:
        lines.append(f"Local numerical disadvantage: {num_adv} (outnumbered by opponents).")
    else:
        lines.append("Local numerical balance (equal numbers nearby).")

    # Passing — these are computed FROM the focus player's position, i.e. they are
    # only this player's options if they hold the ball. For an off-ball player
    # they read like the carrier's options and got misattributed ("the keeper's
    # run opens 5 lanes for #10" — those were the keeper's own lanes).
    if is_ball_carrier:
        if passing["passing_lane_open"]:
            lane_ids = passing["open_passing_lane_player_ids"]
            prog = passing["progressive_passes_available"]
            lines.append(
                f"Open passing lanes to {len(lane_ids)} teammate(s) "
                f"(IDs: {lane_ids}), including {prog} progressive option(s)."
            )
        else:
            lines.append("No clear passing lanes — under heavy press or congested area.")

        # Advanced opportunities
        if passing["third_man_opportunity"]:
            lines.append("Third-man run opportunity detected (combination play feasible).")
        if passing["switch_play_viable"]:
            lines.append("Switch of play viable — wide player free on opposite flank.")

    # Zone context
    if spatial["half_space_occupied"]:
        lines.append("Player is operating in the half-space (high-value attacking zone).")
    if spatial["is_in_attacking_third"]:
        lines.append("Player is in the attacking third.")
    elif spatial["is_in_defensive_third"]:
        lines.append("Player is in the defensive third.")

    # Team structure
    if team_struct["defensive_line_visible"]:
        lines.append(f"Opponent defensive line at x={team_struct['defensive_line_height']:.1f}m.")
    else:
        lines.append("Opponent back line is not in the camera view (only players near the ball are visible).")
    if spatial.get("overload_left"):
        lines.append("Numerical overload on left flank.")
    if spatial.get("overload_right"):
        lines.append("Numerical overload on right flank.")

    return " ".join(lines)
