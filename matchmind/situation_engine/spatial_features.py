"""
Spatial Feature Computation Engine.

Computes geometric and spatial features from raw player coordinates.
This is PURE Python/NumPy — no LLM, no external API calls.

All distances in metres. Coordinate system: 0–105m (x) × 0–68m (y).
Attacking direction: increasing x (left → right).
"""
from __future__ import annotations

import logging
import math

import numpy as np

from matchmind.schema.match_state import MatchState, PlayerState

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
LOCAL_RADIUS = 15.0        # metres for "local" numerical advantage
PASSING_LANE_RADIUS = 3.0  # metres — corridor width that must be clear for a passing lane
HALF_SPACE_X_MIN = 35.0    # half-space: x > 35 (between centre and wide)
HALF_SPACE_X_MAX = 70.0
HALF_SPACE_Y_RANGE = (10.0, 25.0)  # and 43 to 58 (symmetric)
ATTACKING_THIRD_X = 70.0   # x > 70 = attacking third
DEFENSIVE_THIRD_X = 35.0   # x < 35 = defensive third
SWITCH_PLAY_MIN_DIST = 30.0  # minimum pass distance to count as switch of play


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Fast 2D Euclidean distance."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def compute_spatial_features(state: MatchState, focus_player_id: int) -> dict:
    """
    Compute all spatial features for a focus player in a MatchState.

    Returns a flat dict of named features — these feed into SituationFeatures.

    Args:
        state: The current match snapshot.
        focus_player_id: The player we're analysing.

    Returns:
        dict with keys matching SituationFeatures spatial fields.
    """
    focus = state.get_player(focus_player_id)
    if focus is None:
        raise ValueError(f"Player {focus_player_id} not found in MatchState")

    ball_pos = (state.ball.x, state.ball.y)
    focus_pos = focus.position
    teammates = [p for p in state.get_team_players(focus.team) if p.id != focus_player_id]
    opponents = state.get_opponents(focus.team)

    # ── 1. Distance to ball ──────────────────────────────────
    distance_to_ball = euclidean(focus_pos, ball_pos)

    # ── 2. Nearest opponent ──────────────────────────────────
    if opponents:
        opp_dists = [euclidean(focus_pos, o.position) for o in opponents]
        nearest_opponent_distance = min(opp_dists)
        nearest_opp_idx = int(np.argmin(opp_dists))
    else:
        nearest_opponent_distance = 999.0
        nearest_opp_idx = 0

    # ── 3. Nearest teammate ──────────────────────────────────
    if teammates:
        tm_dists = [euclidean(focus_pos, t.position) for t in teammates]
        nearest_teammate_distance = min(tm_dists)
    else:
        nearest_teammate_distance = 999.0

    # ── 4. Space ahead (toward goal) ────────────────────────
    # "Ahead" = increasing x toward opponent's goal
    # Measure free space by checking how close the nearest opponent is directly ahead
    space_ahead = _compute_space_ahead(focus, opponents)

    # ── 5. Local numerical advantage ────────────────────────
    local_teammates = sum(
        1 for t in teammates if euclidean(focus_pos, t.position) <= LOCAL_RADIUS
    )
    local_opponents = sum(
        1 for o in opponents if euclidean(focus_pos, o.position) <= LOCAL_RADIUS
    )
    local_numerical_advantage = local_teammates - local_opponents

    # ── 6. Overload left / right ─────────────────────────────
    overload_left, overload_right = _compute_flank_overloads(state, focus)

    # ── 7. Half-space ────────────────────────────────────────
    half_space_occupied = _is_in_half_space(focus)

    # ── 8. Attacking / defensive third ──────────────────────
    is_in_attacking_third = focus.x > ATTACKING_THIRD_X
    is_in_defensive_third = focus.x < DEFENSIVE_THIRD_X

    # ── 9. Angles to teammates ───────────────────────────────
    angles_to_teammates = _compute_angles(focus_pos, [t.position for t in teammates])

    return {
        "distance_to_ball": round(distance_to_ball, 2),
        "nearest_opponent_distance": round(nearest_opponent_distance, 2),
        "nearest_teammate_distance": round(nearest_teammate_distance, 2),
        "space_ahead": round(space_ahead, 2),
        "local_numerical_advantage": local_numerical_advantage,
        "overload_left": overload_left,
        "overload_right": overload_right,
        "half_space_occupied": half_space_occupied,
        "is_in_attacking_third": is_in_attacking_third,
        "is_in_defensive_third": is_in_defensive_third,
        "angles_to_teammates": angles_to_teammates,
        "_teammates": teammates,  # passed to passing analyzer
        "_opponents": opponents,  # passed to passing analyzer
        "_focus": focus,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Private computation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _compute_space_ahead(
    focus: PlayerState,
    opponents: list[PlayerState],
) -> float:
    """
    Estimate free space directly ahead of the focus player (toward goal).

    Method: find the nearest opponent who is (a) ahead of the focus player
    (higher x) and (b) within 10m laterally. The distance to that opponent
    is the 'space ahead'. If none, return distance to goal line.
    """
    opponents_ahead = [
        o for o in opponents
        if o.x > focus.x and abs(o.y - focus.y) < 10.0
    ]
    if not opponents_ahead:
        return round(PITCH_LENGTH - focus.x, 2)

    return round(
        min(euclidean(focus.position, o.position) for o in opponents_ahead),
        2,
    )


def _compute_flank_overloads(
    state: MatchState,
    focus: PlayerState,
) -> tuple[bool, bool]:
    """
    Detect numerical overloads on left / right flanks.

    Left flank: y < 22 (bottom third of pitch width).
    Right flank: y > 46 (top third of pitch width).
    Counts players within x±20m of focus player.
    """
    team = focus.team
    x_range = (max(0, focus.x - 20), min(PITCH_LENGTH, focus.x + 20))

    def in_x_range(p: PlayerState) -> bool:
        return x_range[0] <= p.x <= x_range[1]

    def count(players: list[PlayerState], y_min: float, y_max: float) -> int:
        return sum(1 for p in players if in_x_range(p) and y_min <= p.y <= y_max)

    teammates = [p for p in state.get_team_players(team) if p.id != focus.id]
    opponents = state.get_opponents(team)

    # Left flank (y 0–22)
    left_adv = count(teammates, 0, 22) > count(opponents, 0, 22)
    # Right flank (y 46–68)
    right_adv = count(teammates, 46, 68) > count(opponents, 46, 68)

    return left_adv, right_adv


def _is_in_half_space(focus: PlayerState) -> bool:
    """
    Check if player is in a half-space zone.

    Half-space = the area between the centre lane and the wide lanes.
    Defined as: x > 35 AND (10 < y < 25 OR 43 < y < 58).
    """
    x_ok = HALF_SPACE_X_MIN < focus.x < HALF_SPACE_X_MAX
    y_ok = (HALF_SPACE_Y_RANGE[0] < focus.y < HALF_SPACE_Y_RANGE[1]) or \
           (PITCH_WIDTH - HALF_SPACE_Y_RANGE[1] < focus.y < PITCH_WIDTH - HALF_SPACE_Y_RANGE[0])
    return x_ok and y_ok


def _compute_angles(
    origin: tuple[float, float],
    targets: list[tuple[float, float]],
) -> list[float]:
    """Compute angles (degrees) from origin to each target position."""
    angles = []
    for t in targets:
        dx = t[0] - origin[0]
        dy = t[1] - origin[1]
        angle_rad = math.atan2(dy, dx)
        angles.append(round(math.degrees(angle_rad), 1))
    return angles
