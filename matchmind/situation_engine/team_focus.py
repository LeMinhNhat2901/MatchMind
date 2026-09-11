"""
Team Focus Picker — auto-select one player per team for a combined
attack/defense analysis (see agent.graph.TacticalAgent.analyze_dual).

Pure computation, no LLM.
"""
from __future__ import annotations

from matchmind.schema.match_state import MatchState
from matchmind.situation_engine.encoder import determine_ball_carrier


class DualFocus:
    """Which player represents each side of a combined attack/defense analysis."""

    def __init__(self, attacking_player_id: int, defending_player_id: int, attacking_team: str) -> None:
        self.attacking_player_id = attacking_player_id
        self.defending_player_id = defending_player_id
        self.attacking_team = attacking_team
        self.defending_team = "away" if attacking_team == "home" else "home"


def pick_dual_focus_players(state: MatchState) -> DualFocus:
    """
    Auto-pick a sensible focus player for each side of the ball.

    - Attacking focus: the ball carrier (their team is asked "what to do WITH
      the ball" — an on-ball question).
    - Defending focus: the nearest opponent to the ball carrier — the player
      most immediately responsible for winning it back (asked "how do you
      stop this" — an off-ball/defensive question).

    Falls back to the first player of each team if the ball carrier can't be
    determined (e.g. an empty snapshot slipped through).
    """
    carrier_id = determine_ball_carrier(state)
    carrier = state.get_player(carrier_id) if carrier_id is not None else None

    if carrier is None:
        home = next((p for p in state.players if p.team == "home"), None)
        away = next((p for p in state.players if p.team == "away"), None)
        if home is None or away is None:
            raise ValueError("MatchState needs at least one player per team to pick dual focus players")
        return DualFocus(home.id, away.id, attacking_team="home")

    opponents = state.get_opponents(carrier.team)
    defender = min(
        opponents,
        key=lambda p: (p.x - carrier.x) ** 2 + (p.y - carrier.y) ** 2,
        default=None,
    )
    if defender is None:
        # No opponents in this snapshot (shouldn't happen — MatchState requires
        # both teams) — fall back to the carrier itself so callers don't crash.
        defender = carrier

    return DualFocus(carrier.id, defender.id, attacking_team=carrier.team)
