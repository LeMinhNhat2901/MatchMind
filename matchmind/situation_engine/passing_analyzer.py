"""
Passing Opportunity Analyzer.

Detects open passing lanes, progressive pass options, third-man opportunities,
and switch-play viability from raw player positions.

All computation is geometric — pure Python/NumPy, no LLM.
"""
from __future__ import annotations

import math

import numpy as np

from matchmind.schema.match_state import MatchState, PlayerState
from matchmind.situation_engine.spatial_features import euclidean, PASSING_LANE_RADIUS

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
SWITCH_PLAY_MIN_DIST = 30.0  # metres — minimum distance for a switch of play
PROGRESSIVE_PASS_MIN_GAIN = 10.0  # metres of x-advance = "progressive"
MAX_PASS_DISTANCE = 40.0  # metres — beyond this a pass is unrealistic in open play
THIRD_MAN_MAX_DIST = 25.0  # metres — max distance between 3 players for third-man


def compute_passing_opportunities(
    state: MatchState,
    focus_player_id: int,
    teammates: list[PlayerState] | None = None,
    opponents: list[PlayerState] | None = None,
) -> dict:
    """
    Compute all passing opportunity features for a focus player.

    Args:
        state: Current match snapshot.
        focus_player_id: The player we're advising.
        teammates: Pre-computed list (avoid re-filtering). If None, computed here.
        opponents: Pre-computed list. If None, computed here.

    Returns:
        dict with passing opportunity features.
    """
    focus = state.get_player(focus_player_id)
    if focus is None:
        raise ValueError(f"Player {focus_player_id} not found in MatchState")

    if teammates is None:
        teammates = [p for p in state.get_team_players(focus.team) if p.id != focus_player_id]
    if opponents is None:
        opponents = state.get_opponents(focus.team)

    open_lanes: list[int] = []
    progressive_ids: list[int] = []

    for tm in teammates:
        dist = euclidean(focus.position, tm.position)
        if dist > MAX_PASS_DISTANCE:
            continue

        if _is_lane_open(focus.position, tm.position, opponents):
            open_lanes.append(tm.id)

            # Progressive: teammate is significantly closer to opponent goal
            x_gain = tm.x - focus.x
            if x_gain >= PROGRESSIVE_PASS_MIN_GAIN:
                progressive_ids.append(tm.id)

    # Third-man opportunity
    third_man_possible = _detect_third_man(focus, teammates, opponents)

    # Switch of play viability
    switch_play_viable = _detect_switch_play(focus, teammates, opponents)

    return {
        "passing_lane_open": len(open_lanes) > 0,
        "open_passing_lane_player_ids": open_lanes,
        "progressive_passes_available": len(progressive_ids),
        "third_man_opportunity": third_man_possible,
        "switch_play_viable": switch_play_viable,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────


def _is_lane_open(
    origin: tuple[float, float],
    target: tuple[float, float],
    blockers: list[PlayerState],
    lane_width: float = PASSING_LANE_RADIUS,
) -> bool:
    """
    Check if a passing lane from origin → target is free of blockers.

    Uses perpendicular distance from each blocker to the line segment.
    If any blocker is within `lane_width` metres AND between origin and target, lane is blocked.
    """
    ox, oy = origin
    tx, ty = target
    dx = tx - ox
    dy = ty - oy
    lane_length = math.sqrt(dx**2 + dy**2)

    if lane_length < 1e-6:
        return True  # same position

    # Unit vector along lane
    ux, uy = dx / lane_length, dy / lane_length

    for blocker in blockers:
        bx, by = blocker.position

        # Vector from origin to blocker
        vx = bx - ox
        vy = by - oy

        # Project onto lane direction (t = scalar projection)
        t = vx * ux + vy * uy

        # Only consider blockers between origin and target (with small margin)
        if t < 1.0 or t > lane_length - 1.0:
            continue

        # Perpendicular distance from blocker to lane
        perp_x = vx - t * ux
        perp_y = vy - t * uy
        perp_dist = math.sqrt(perp_x**2 + perp_y**2)

        if perp_dist < lane_width:
            return False  # lane is blocked

    return True


def _detect_third_man(
    focus: PlayerState,
    teammates: list[PlayerState],
    opponents: list[PlayerState],
) -> bool:
    """
    Detect if a third-man combination is geometrically feasible.

    Third-man: A passes to B who lays off to C who is arriving into space.
    Conditions:
    1. At least 2 teammates within THIRD_MAN_MAX_DIST of focus.
    2. A chain exists: focus→tm1→tm2 where tm2 has an open lane and space ahead.
    3. tm2 is further forward than tm1.
    """
    close_teammates = [
        t for t in teammates
        if euclidean(focus.position, t.position) <= THIRD_MAN_MAX_DIST
    ]

    if len(close_teammates) < 2:
        return False

    for i, tm1 in enumerate(close_teammates):
        for tm2 in close_teammates[i + 1:]:
            # tm2 should be further forward
            if tm2.x <= tm1.x:
                continue

            # Lane from focus to tm1
            lane1_ok = _is_lane_open(focus.position, tm1.position, opponents)
            # Lane from tm1 to tm2
            lane2_ok = _is_lane_open(tm1.position, tm2.position, opponents)

            if lane1_ok and lane2_ok:
                # Check tm2 has some free space ahead
                opp_ahead = [o for o in opponents if o.x > tm2.x and abs(o.y - tm2.y) < 8.0]
                if not opp_ahead or min(euclidean(tm2.position, o.position) for o in opp_ahead) > 5.0:
                    return True

    return False


def _detect_switch_play(
    focus: PlayerState,
    teammates: list[PlayerState],
    opponents: list[PlayerState],
) -> bool:
    """
    Check if switching play (long cross-field pass) is viable.

    Conditions:
    1. A teammate is on the opposite flank (y distance > 30m from focus).
    2. The lane to that teammate is open.
    3. That teammate has space (nearest opponent > 5m away).
    """
    for tm in teammates:
        lateral_dist = abs(tm.y - focus.y)
        pass_dist = euclidean(focus.position, tm.position)

        if lateral_dist >= 30.0 and pass_dist <= MAX_PASS_DISTANCE:
            lane_ok = _is_lane_open(focus.position, tm.position, opponents, lane_width=4.0)
            if lane_ok:
                tm_nearest_opp = min(
                    (euclidean(tm.position, o.position) for o in opponents),
                    default=999.0,
                )
                if tm_nearest_opp >= 5.0:
                    return True

    return False
