"""
Tests for the shirt-number collision fix: MatchState.get_player(id) is NOT
globally unique across teams (Metrica reuses 1-11 on both sides), so
get_player(team=) / has_ambiguous_id() / agent.graph._disambiguate_by_team
must resolve it correctly.
"""
from __future__ import annotations

from matchmind.agent.graph import _disambiguate_by_team
from matchmind.schema.match_state import BallState, MatchState, PlayerState


def _colliding_state() -> MatchState:
    return MatchState(
        match_id="collision_test",
        timestamp=0.0,
        ball=BallState(x=50.0, y=30.0),
        source="synthetic",
        players=[
            PlayerState(id=10, team="home", role="striker", x=50.0, y=30.0),
            PlayerState(id=10, team="away", role="center_back", x=60.0, y=30.0),
            PlayerState(id=2, team="home", x=20.0, y=20.0),
            PlayerState(id=3, team="away", x=80.0, y=20.0),
        ],
    )


def test_has_ambiguous_id():
    state = _colliding_state()
    assert state.has_ambiguous_id(10) is True
    assert state.has_ambiguous_id(2) is False
    assert state.has_ambiguous_id(999) is False


def test_get_player_team_scoped():
    state = _colliding_state()
    assert state.get_player(10, team="home").team == "home"
    assert state.get_player(10, team="away").team == "away"
    assert state.get_player(10, team="home").role == "striker"
    assert state.get_player(10, team="away").role == "center_back"


def test_get_player_without_team_is_first_match_not_none():
    """Documents the pre-existing (still supported) ambiguous behaviour when
    the caller doesn't disambiguate — must not crash, just isn't guaranteed."""
    state = _colliding_state()
    assert state.get_player(10) is not None


def test_disambiguate_by_team_removes_collision():
    state = _colliding_state()
    fixed = _disambiguate_by_team(state, 10, "away")
    assert fixed.has_ambiguous_id(10) is False
    assert fixed.get_player(10).team == "away"
    # the other team's player is still present, just relabelled
    assert any(p.team == "home" and p.id != 10 and p.role == "striker" for p in fixed.players)


def test_disambiguate_by_team_noop_when_no_collision():
    state = _colliding_state()
    fixed = _disambiguate_by_team(state, 2, "home")
    assert fixed is state  # unchanged — returns the same object, not a copy
