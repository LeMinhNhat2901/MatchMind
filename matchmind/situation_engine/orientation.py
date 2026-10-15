"""
Attacking-direction normalisation.

The situation engine, candidate actions and pitch-control simulation all reason
as if the focus team attacks toward x=105 ("ahead" = +x, own goal at x=0). That
holds for StatsBomb data (the acting team is normalised to attack +x) but not
for a team attacking the other way — e.g. the away side in the synthetic demo,
or either team after half-time in Metrica. Without this, an away centre-back
standing in front of his own goal was described as having "23 m of free space
ahead" and as being "in the attacking third".

Rather than teaching every calculation about direction, the pitch is turned
180° once at the entry point when the focus team attacks -x (x -> 105-x,
y -> 68-y, velocities negated — a rotation, not a mirror, so a left winger
stays on his team's left), and target coordinates are turned back once at the
exit so arrows are drawn on the real pitch.
"""
from __future__ import annotations

from typing import Literal

from matchmind.schema.match_state import MatchState, TacticalAdvice

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def rotate_xy(x: float, y: float) -> tuple[float, float]:
    return PITCH_LENGTH - x, PITCH_WIDTH - y


def rotate_state(state: MatchState) -> MatchState:
    """The same snapshot seen from the other end of the pitch. Applying it twice
    gives back the original."""
    players = [
        p.model_copy(update={"x": PITCH_LENGTH - p.x, "y": PITCH_WIDTH - p.y, "vx": -p.vx, "vy": -p.vy})
        for p in state.players
    ]
    ball_x, ball_y = rotate_xy(state.ball.x, state.ball.y)
    visible = [rotate_xy(x, y) for x, y in state.visible_area] if state.visible_area else state.visible_area
    return state.model_copy(update={
        "players": players,
        "ball": state.ball.model_copy(update={"x": ball_x, "y": ball_y}),
        "visible_area": visible,
        "home_attacks_right": not state.attacks_right("home"),
    })


def orient_for_team(state: MatchState, team: Literal["home", "away"]) -> tuple[MatchState, bool]:
    """Return (state in which `team` attacks +x, whether it was rotated).

    Idempotent: the returned state records its direction explicitly, so orienting
    it again for the same team is a no-op.
    """
    if state.attacks_right(team):
        if state.home_attacks_right is None:
            state = state.model_copy(update={"home_attacks_right": state.attacks_right("home")})
        return state, False
    return rotate_state(state), True


def unorient_advice(advice: TacticalAdvice, flipped: bool) -> TacticalAdvice:
    """Map candidate-action targets from the oriented frame back onto the real
    pitch (they drive the arrows and the follow-up projection)."""
    if not flipped:
        return advice
    actions = [
        c.model_copy(update={"target_xy": rotate_xy(*c.target_xy)}) if c.target_xy is not None else c
        for c in advice.candidate_actions
    ]
    return advice.model_copy(update={"candidate_actions": actions})
