"""
Estimated team reactions for the "what happens next?" projection.

When the user projects a recommendation forward, the other players should not
stay frozen where they were. Running the agent for every player would cost
~5k LLM tokens each (~90k per click, about half of Groq's free daily quota),
and scoring every player's options with pitch control takes ~30 s. Football
teams react in a few well-known ways, though, so this module moves everyone
with cheap geometric rules (zero tokens, < 1 ms):

  after a pass   defending side: the nearest outfielder presses the receiver,
                 goal-side; the next one covers behind him; the rest of the block
                 slides toward the ball; the keeper shuffles across his line.
                 attacking side: teammates move up with the ball.
  after a run    the nearest defender tracks the runner; nobody else reacts
                 (the ball has not moved).

Every move is capped by what a player can physically cover in the elapsed time,
with the pitch-control model's own kinematics (Spearman 2018: 5 m/s top speed,
0.7 s reaction, 15 m/s ball). A 10 m pass therefore barely moves anyone, while
a 30 m switch lets the block slide ~9 m. These are estimates, labelled as such
on the pitch; V2/V3 tracking data can replace the rules with learnt movement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from matchmind.schema.match_state import MatchState, PlayerState

MAX_SPEED_MS = 5.0          # same as pitch_control.spearman_model
REACTION_S = 0.7
BALL_SPEED_MS = 15.0
CONTROL_TIME_S = 0.5        # receiver's first touch before the next decision

PRESS_DISTANCE_M = 1.5      # presser stops this far from the ball, goal-side
COVER_DISTANCE_M = 8.0      # second defender sits this far behind the ball, goal-side
DEF_SLIDE_X, DEF_SLIDE_Y = 0.5, 0.6     # share of the ball's displacement the block follows
ATT_SLIDE_X, ATT_SLIDE_Y = 0.4, 0.4
GK_TRACK_Y = 0.3            # keeper moves this share of the ball's offset from the goal centre
GOAL_HALF_WIDTH = 3.66
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
_MIN_MOVE_M = 0.1


@dataclass(frozen=True)
class Move:
    team: str
    player_id: int
    from_xy: tuple[float, float]
    to_xy: tuple[float, float]


def max_displacement(elapsed_s: float) -> float:
    """How far a player starting from rest can get in `elapsed_s` seconds."""
    return max(0.0, MAX_SPEED_MS * (elapsed_s - REACTION_S))


def elapsed_for_pass(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> float:
    """Ball travel + the receiver's first touch."""
    return math.dist(from_xy, to_xy) / BALL_SPEED_MS + CONTROL_TIME_S


def elapsed_for_run(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> float:
    """The runner chose to go, so no reaction delay for them — only for the defence."""
    return math.dist(from_xy, to_xy) / MAX_SPEED_MS


def _toward(p: PlayerState, target: tuple[float, float], cap: float) -> tuple[float, float]:
    dx, dy = target[0] - p.x, target[1] - p.y
    d = math.hypot(dx, dy)
    if d > cap:
        dx, dy = dx * cap / d, dy * cap / d
    return (min(PITCH_LENGTH, max(0.0, p.x + dx)), min(PITCH_WIDTH, max(0.0, p.y + dy)))


def _goal_side(ball: tuple[float, float], goal: tuple[float, float], dist: float) -> tuple[float, float]:
    dx, dy = goal[0] - ball[0], goal[1] - ball[1]
    d = math.hypot(dx, dy) or 1.0
    return ball[0] + dx * dist / d, ball[1] + dy * dist / d


def _is_keeper(p: PlayerState) -> bool:
    return (p.role or "").lower() in {"goalkeeper", "gk", "keeper"}


def _apply(state: MatchState, targets: dict[tuple[str, int], tuple[float, float]], cap: float):
    moves: list[Move] = []
    players = []
    for p in state.players:
        target = targets.get((p.team, p.id))
        if target is None or cap <= 0:
            players.append(p)
            continue
        nx, ny = _toward(p, target, cap)
        if math.hypot(nx - p.x, ny - p.y) < _MIN_MOVE_M:
            players.append(p)
            continue
        moves.append(Move(p.team, p.id, (p.x, p.y), (nx, ny)))
        players.append(p.model_copy(update={"x": nx, "y": ny}))
    return state.model_copy(update={"players": players}), moves


def shift_after_pass(
    before: MatchState, after: MatchState, *, attacking_team: str, carrier_id: int, elapsed_s: float,
) -> tuple[MatchState, list[Move]]:
    """`after` already has the ball and the receiver at the target (projection.py);
    move everyone else the way teams react to that pass."""
    defending = "away" if attacking_team == "home" else "home"
    bx0, by0 = before.ball.x, before.ball.y
    ball = (after.ball.x, after.ball.y)
    dx, dy = ball[0] - bx0, ball[1] - by0
    goal = (PITCH_LENGTH if after.attacks_right(attacking_team) else 0.0, PITCH_WIDTH / 2)

    targets: dict[tuple[str, int], tuple[float, float]] = {}
    defenders = [p for p in after.players if p.team == defending]
    outfield = sorted((p for p in defenders if not _is_keeper(p)), key=lambda p: math.dist((p.x, p.y), ball))
    for rank, p in enumerate(outfield):
        if rank == 0:
            targets[(p.team, p.id)] = _goal_side(ball, goal, PRESS_DISTANCE_M)
        elif rank == 1:
            targets[(p.team, p.id)] = _goal_side(ball, goal, COVER_DISTANCE_M)
        else:
            targets[(p.team, p.id)] = (p.x + DEF_SLIDE_X * dx, p.y + DEF_SLIDE_Y * dy)
    for p in defenders:
        if _is_keeper(p):
            gy = goal[1] + GK_TRACK_Y * (ball[1] - goal[1])
            targets[(p.team, p.id)] = (p.x, max(goal[1] - GOAL_HALF_WIDTH, min(goal[1] + GOAL_HALF_WIDTH, gy)))
    for p in after.players:
        if p.team == attacking_team and p.id != carrier_id and not _is_keeper(p):
            targets[(p.team, p.id)] = (p.x + ATT_SLIDE_X * dx, p.y + ATT_SLIDE_Y * dy)

    return _apply(after, targets, max_displacement(elapsed_s))


def shift_after_run(
    before: MatchState, after: MatchState, *, attacking_team: str, runner_id: int, elapsed_s: float,
) -> tuple[MatchState, list[Move]]:
    """`after` already has the runner at the end of the run; the nearest defender
    to that spot tracks them (goal-side). The ball has not moved, so nobody else does."""
    runner = after.get_player(runner_id, attacking_team)
    if runner is None:
        return after, []
    defending = "away" if attacking_team == "home" else "home"
    end = (runner.x, runner.y)
    goal = (PITCH_LENGTH if after.attacks_right(attacking_team) else 0.0, PITCH_WIDTH / 2)
    outfield = [p for p in after.players if p.team == defending and not _is_keeper(p)]
    if not outfield:
        return after, []
    tracker = min(outfield, key=lambda p: math.dist((p.x, p.y), end))
    return _apply(after, {(tracker.team, tracker.id): _goal_side(end, goal, PRESS_DISTANCE_M)},
                  max_displacement(elapsed_s))
