"""Tests for the "what happens next?" state-projection helper."""
from __future__ import annotations

from matchmind.schema.match_state import BallState, CandidateAction, MatchState, PlayerState
from matchmind.situation_engine.projection import project_after_action


def _state() -> MatchState:
    return MatchState(
        match_id="proj_test",
        timestamp=0.0,
        ball=BallState(x=50.0, y=34.0),
        source="synthetic",
        players=[
            PlayerState(id=10, team="home", role="attacking_midfielder", x=50.0, y=34.0),
            PlayerState(id=8, team="home", role="central_midfielder", x=45.0, y=20.0),
            PlayerState(id=3, team="away", role="center_back", x=48.0, y=33.0),
            PlayerState(id=4, team="away", role="center_back", x=60.0, y=45.0),
        ],
    )


def test_shoot_action_returns_none():
    """A shot ends the sequence — there's no meaningful 'next' to project."""
    state = _state()
    action = CandidateAction(action="shoot", label="Shoot at goal", target_xy=(104.0, 34.0))
    assert project_after_action(state, 10, "home", action) is None


def test_action_without_target_returns_none():
    state = _state()
    action = CandidateAction(action="hold_position", label="Hold current position", target_xy=None)
    assert project_after_action(state, 10, "home", action) is None


def test_pass_moves_ball_and_receiver_to_target():
    state = _state()
    action = CandidateAction(
        action="pass_to_8",
        label="Pass to player 8",
        target_xy=(55.0, 25.0),
        target_player_id=8,
        moves_ball=True,
    )
    projection = project_after_action(state, 10, "home", action)
    assert projection is not None
    assert projection.focus_player_id == 8
    receiver = projection.state.get_player(8, team="home")
    assert (receiver.x, receiver.y) == (55.0, 25.0)
    assert (projection.state.ball.x, projection.state.ball.y) == (55.0, 25.0)
    assert "Player 8" in projection.question
    assert "received the ball" in projection.question
    # The passer follows the play, but only as far as a player can run while a
    # ~10 m pass travels and is controlled (team_shift kinematics).
    from matchmind.situation_engine.team_shift import elapsed_for_pass, max_displacement

    passer = projection.state.get_player(10, team="home")
    cap = max_displacement(elapsed_for_pass((50.0, 34.0), (55.0, 25.0)))
    assert ((passer.x - 50.0) ** 2 + (passer.y - 34.0) ** 2) ** 0.5 <= cap + 1e-6


def test_dribble_moves_ball_with_same_focus_player():
    state = _state()
    action = CandidateAction(
        action="dribble_half_space",
        label="Dribble into the half-space",
        target_xy=(60.0, 30.0),
        moves_ball=True,
    )
    projection = project_after_action(state, 10, "home", action)
    assert projection is not None
    assert projection.focus_player_id == 10
    focus = projection.state.get_player(10, team="home")
    assert (focus.x, focus.y) == (60.0, 30.0)
    assert (projection.state.ball.x, projection.state.ball.y) == (60.0, 30.0)


def test_off_ball_run_moves_only_the_player_not_the_ball():
    state = _state()
    action = CandidateAction(
        action="run_in_behind",
        label="Run in behind the defensive line",
        target_xy=(70.0, 40.0),
        moves_ball=False,
    )
    projection = project_after_action(state, 10, "home", action)
    assert projection is not None
    assert projection.focus_player_id == 10
    focus = projection.state.get_player(10, team="home")
    assert (focus.x, focus.y) == (70.0, 40.0)
    # Ball is untouched — this is an off-ball movement.
    assert (projection.state.ball.x, projection.state.ball.y) == (50.0, 34.0)


def test_off_ball_run_notes_the_marker_pull_when_it_changes():
    state = _state()
    # Player 10 starts at (50, 34), nearest opponent is #3 at (48, 33).
    # Target (70, 40) is much closer to opponent #4 at (60, 45).
    action = CandidateAction(
        action="run_in_behind",
        label="Run in behind the defensive line",
        target_xy=(70.0, 40.0),
        moves_ball=False,
    )
    projection = project_after_action(state, 10, "home", action)
    assert projection is not None
    assert "#4" in projection.question
    assert "pulls opponent #4" in projection.question


def test_off_ball_run_no_pull_note_when_marker_unchanged():
    state = _state()
    # A tiny move that stays closest to the same opponent (#3).
    action = CandidateAction(
        action="check_to_receive",
        label="Check to the ball",
        target_xy=(49.0, 33.5),
        moves_ball=False,
    )
    projection = project_after_action(state, 10, "home", action)
    assert projection is not None
    assert "pulls opponent" not in projection.question
    assert "#3 is now the nearest defender" in projection.question


def test_pass_makes_the_receiver_the_carrier_even_for_statsbomb_snapshots():
    """User report: after "#5 passes to #19" the projected situation still
    treated #5 as the ball carrier. StatsBomb snapshots carry event_player_id
    (the passer), which the encoder prefers over ball proximity."""
    from matchmind.situation_engine.encoder import _determine_ball_carrier, encode_situation

    state = _state().model_copy(update={"source": "statsbomb", "event_player_id": 10, "event_type": "Pass"})
    action = CandidateAction(action="pass_to_8", label="Pass to player 8", target_xy=(55.0, 25.0),
                             target_player_id=8, moves_ball=True)
    projected = project_after_action(state, 10, "home", action).state
    assert projected.event_player_id == 8
    assert _determine_ball_carrier(projected) == 8
    assert "Player 8 CURRENTLY HAS THE BALL" in encode_situation(projected, 8).natural_language_description


def test_pass_projection_moves_the_other_players_and_records_trails():
    state = _state().model_copy(update={"home_attacks_right": True})
    action = CandidateAction(action="pass_to_8", label="Pass to player 8", target_xy=(80.0, 10.0),
                             target_player_id=8, moves_ball=True)
    proj = project_after_action(state, 10, "home", action)
    moved = {(m.team, m.player_id) for m in proj.moves}
    assert ("home", 8) in moved                       # the receiver's own move is a trail too
    assert any(team == "away" for team, _ in moved)   # the defence reacted
    assert "estimated" in proj.question


def test_run_projection_records_the_runner_and_the_tracker():
    state = _state().model_copy(update={"home_attacks_right": True})
    action = CandidateAction(action="run_in_behind", label="Run in behind", target_xy=(70.0, 34.0), moves_ball=False)
    proj = project_after_action(state, 10, "home", action)
    assert [(m.team, m.player_id) for m in proj.moves][0] == ("home", 10)


def test_dribble_uses_running_time_not_ball_speed():
    """A 10 m dribble takes ~2 s at 5 m/s (defenders get ~6.5 m), not 0.7+0.5 s."""
    from matchmind.situation_engine.team_shift import elapsed_for_run, max_displacement

    state = _state().model_copy(update={"home_attacks_right": True})
    action = CandidateAction(action="dribble", label="Dribble", target_xy=(60.0, 34.0), moves_ball=True)
    proj = project_after_action(state, 10, "home", action)
    cb = next(m for m in proj.moves if (m.team, m.player_id) == ("away", 3))
    dist = ((cb.to_xy[0] - cb.from_xy[0]) ** 2 + (cb.to_xy[1] - cb.from_xy[1]) ** 2) ** 0.5
    assert dist > max_displacement(1.2)            # more than a pass-timed reaction would allow
    assert dist <= max_displacement(elapsed_for_run((50.0, 34.0), (60.0, 34.0))) + 1e-6
