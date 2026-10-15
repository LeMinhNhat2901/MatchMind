"""Estimated team reactions for the "what happens next?" projection.

Zero LLM tokens: geometry + the same kinematics as the pitch-control model
(Spearman 2018: 5 m/s max speed, 0.7 s reaction, 15 m/s ball). Nobody may move
further than they physically could while the ball travels and is controlled.
"""
from __future__ import annotations

import math

from matchmind.schema.match_state import BallState, MatchState, PlayerState
from matchmind.situation_engine.team_shift import (
    CONTROL_TIME_S,
    elapsed_for_pass,
    elapsed_for_run,
    max_displacement,
    shift_after_pass,
    shift_after_run,
)


def _state() -> MatchState:
    # Home attacks right (+x); away defends the goal at x=105.
    return MatchState(
        match_id="t", timestamp=0.0, possession="home", source="synthetic", home_attacks_right=True,
        ball=BallState(x=50.0, y=34.0),
        players=[
            PlayerState(id=5, team="home", x=50.0, y=34.0),            # passer
            PlayerState(id=19, team="home", x=70.0, y=15.0),           # receiver
            PlayerState(id=8, team="home", x=55.0, y=50.0),
            PlayerState(id=1, team="away", role="goalkeeper", x=102.0, y=34.0),
            PlayerState(id=4, team="away", x=75.0, y=20.0),            # nearest to the receiver
            PlayerState(id=6, team="away", x=78.0, y=30.0),
            PlayerState(id=10, team="away", x=60.0, y=50.0),
        ],
    )


def _after_pass(state: MatchState) -> MatchState:
    """What projection.py already does: receiver + ball at the target."""
    players = [p.model_copy(update={"x": 70.0, "y": 15.0}) if p.id == 19 else p for p in state.players]
    return state.model_copy(update={"players": players, "ball": BallState(x=70.0, y=15.0)})


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_kinematics():
    assert max_displacement(0.5) == 0.0                     # still reacting
    assert math.isclose(max_displacement(1.7), 5.0)          # 1 s of running at 5 m/s
    t = elapsed_for_pass((50, 34), (80, 34))                 # 30 m ball
    assert math.isclose(t, 2.0 + CONTROL_TIME_S)
    assert math.isclose(elapsed_for_run((50, 34), (60, 34)), 2.0)


def test_nobody_moves_further_than_physically_possible():
    before = _state()
    t = elapsed_for_pass((50, 34), (70, 15))
    after, moves = shift_after_pass(before, _after_pass(before), attacking_team="home", carrier_id=19, elapsed_s=t)
    cap = max_displacement(t) + 1e-6
    assert moves
    for m in moves:
        assert _dist(m.from_xy, m.to_xy) <= cap


def test_nearest_defender_presses_goal_side_of_the_receiver():
    before = _state()
    after, _ = shift_after_pass(before, _after_pass(before), attacking_team="home", carrier_id=19, elapsed_s=10.0)
    presser = after.get_player(4, "away")
    # Plenty of time: ends ~2 m from the ball, between ball and the goal it defends (x=105).
    assert _dist((presser.x, presser.y), (70.0, 15.0)) < 2.5
    assert presser.x > 70.0


def test_block_slides_toward_the_ball_and_receiver_and_ball_stay_put():
    before = _state()
    after, _ = shift_after_pass(before, _after_pass(before), attacking_team="home", carrier_id=19, elapsed_s=3.0)
    far = after.get_player(10, "away")
    assert far.y < 50.0                                       # ball went to low y; the block follows it
    assert (after.ball.x, after.ball.y) == (70.0, 15.0)
    rec = after.get_player(19, "home")
    assert (rec.x, rec.y) == (70.0, 15.0)
    gk = after.get_player(1, "away")
    assert gk.y < 34.0 and gk.x == 102.0                      # keeper shuffles across his line only


def test_short_pass_freezes_everyone():
    before = _state()
    after, moves = shift_after_pass(before, _after_pass(before), attacking_team="home", carrier_id=19, elapsed_s=0.6)
    assert moves == []


def test_run_is_tracked_by_the_nearest_defender_only():
    before = _state()
    runner_to = (80.0, 25.0)
    moved = before.model_copy(update={"players": [
        p.model_copy(update={"x": runner_to[0], "y": runner_to[1]}) if (p.id, p.team) == (8, "home") else p
        for p in before.players]})
    t = elapsed_for_run((55.0, 50.0), runner_to)
    after, moves = shift_after_run(before, moved, attacking_team="home", runner_id=8, elapsed_s=t)
    assert [(m.team, m.player_id) for m in moves] == [("away", 6)]
    tracker = after.get_player(6, "away")
    assert _dist((tracker.x, tracker.y), runner_to) < _dist((78.0, 30.0), runner_to)


def test_attackers_keep_their_width_while_the_defence_compacts():
    before = _state()
    after, _ = shift_after_pass(before, _after_pass(before), attacking_team="home", carrier_id=19, elapsed_s=10.0)
    # Ball moved 19 m toward low y. Home #8 (far side) barely narrows; away #10 slides much more.
    home_shift = 50.0 - after.get_player(8, "home").y
    away_shift = 50.0 - after.get_player(10, "away").y
    assert home_shift < 4.0 < away_shift
