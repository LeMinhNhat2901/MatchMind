"""
Team Structure Analysis.

Computes macro-level tactical structure features for both teams:
defensive line height, compactness, attacking width, press intensity.

These are team-level features, distinct from individual spatial features.
"""
from __future__ import annotations

import math

import numpy as np

from matchmind.schema.match_state import MatchState, PlayerState
from matchmind.situation_engine.spatial_features import euclidean

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Goalkeepers typically stay behind x=15 (defending) or x=90 (attacking)
GK_THRESHOLD_X = 15.0


def compute_team_structure(state: MatchState, focus_team: str) -> dict:
    """
    Compute team-structure features for both the focus team and their opponents.

    Args:
        state: Current match snapshot.
        focus_team: "home" or "away" — the team we're advising.

    Returns:
        dict with team structure features.
    """
    opp_team = "away" if focus_team == "home" else "home"

    focus_players = _exclude_gk(state.get_team_players(focus_team))
    opp_players = _exclude_gk(state.get_opponents(focus_team))

    return {
        # Defensive line = y-position of the defensive team's deepest outfield line
        "defensive_line_height": _defensive_line_height(opp_players, focus_team),

        # Compactness = average inter-player distance of the focus team
        "team_compactness": _team_compactness(focus_players),

        # Attacking width = lateral spread (y range) of focus team in final third
        "attacking_width": _attacking_width(focus_players),

        # Numerical superiority in the key zone (around ball)
        "numerical_superiority_zone": _numerical_superiority_around_ball(
            state, focus_team, radius=20.0
        ),

        # Press intensity = how tightly opposing team marks focus players
        "press_intensity": _press_intensity(focus_players, opp_players),

        # Formation shape (simplified)
        "formation_shape": _estimate_formation(focus_players),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────


def _exclude_gk(players: list[PlayerState]) -> list[PlayerState]:
    """Exclude goalkeepers (role='goalkeeper' OR position at edge of pitch x<5 or x>100)."""
    filtered = [
        p for p in players
        if p.role != "goalkeeper" and 5.0 < p.x < 100.0
    ]
    return filtered if filtered else players  # fallback: keep all


def _defensive_line_height(
    defending_players: list[PlayerState],
    attacking_team: str,
) -> float:
    """
    The x-coordinate of the defensive line (deepest outfield defenders).

    For 'home' attacking (left→right): defenders are at high x, so line = min(x) of back 4.
    For 'away' attacking: defenders are at low x, so line = max(x) of back 4.

    Returns x in metres (0–105). Higher x = higher defensive line.
    """
    if not defending_players:
        return 52.5  # default: halfway line

    x_values = sorted([p.x for p in defending_players])
    # Defensive line = the 4 deepest defenders (lowest x for home attack)
    if attacking_team == "home":
        # Defenders at low x (their own half)
        line_x = float(np.mean(x_values[:4]))
    else:
        # Defenders at high x
        line_x = float(np.mean(x_values[-4:]))

    return round(line_x, 2)


def _team_compactness(players: list[PlayerState]) -> float:
    """
    Average pairwise distance between all outfield players (metres).

    Lower = more compact shape.
    """
    if len(players) < 2:
        return 0.0

    positions = [(p.x, p.y) for p in players]
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distances.append(euclidean(positions[i], positions[j]))

    return round(float(np.mean(distances)), 2) if distances else 0.0


def _attacking_width(players: list[PlayerState]) -> float:
    """
    Lateral spread (y range) of attacking team players in the final third (x > 70m).

    Measures how wide the team is stretching the opponent.
    """
    final_third = [p for p in players if p.x > 70.0]
    if len(final_third) < 2:
        final_third = players  # fall back to all players if nobody in final third

    if not final_third:
        return 0.0

    y_values = [p.y for p in final_third]
    return round(max(y_values) - min(y_values), 2)


def _numerical_superiority_around_ball(
    state: MatchState,
    focus_team: str,
    radius: float = 20.0,
) -> bool:
    """Check if focus team has numerical superiority within `radius` metres of ball."""
    ball = (state.ball.x, state.ball.y)
    focus_near = sum(
        1 for p in state.get_team_players(focus_team)
        if euclidean(p.position, ball) <= radius
    )
    opp_near = sum(
        1 for p in state.get_opponents(focus_team)
        if euclidean(p.position, ball) <= radius
    )
    return focus_near > opp_near


def _press_intensity(
    focus_players: list[PlayerState],
    opp_players: list[PlayerState],
) -> float:
    """
    Estimate press intensity on the focus team.

    Computed as: average of the minimum opponent distance for each focus player.
    Lower value = higher press intensity (opponents very close).
    """
    if not opp_players or not focus_players:
        return 99.0

    min_dists = []
    for fp in focus_players:
        d = min(euclidean(fp.position, o.position) for o in opp_players)
        min_dists.append(d)

    return round(float(np.mean(min_dists)), 2)


def _estimate_formation(players: list[PlayerState]) -> str:
    """
    Very rough formation estimation based on player x-distribution.

    Assigns players to defensive / midfield / attacking bands and counts lines.
    Not production-grade — for informational context only.
    """
    if len(players) < 8:
        return "unknown"

    # Sort by x
    x_vals = sorted([p.x for p in players])
    n = len(x_vals)

    # Split into thirds of the team's own distribution
    third = n // 3
    defenders = x_vals[:third]
    midfielders = x_vals[third: 2 * third]
    attackers = x_vals[2 * third:]

    return f"{len(defenders)}-{len(midfielders)}-{len(attackers)}"
