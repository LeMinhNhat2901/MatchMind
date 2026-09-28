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
    # The passer (10) is untouched.
    passer = projection.state.get_player(10, team="home")
    assert (passer.x, passer.y) == (50.0, 34.0)


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
