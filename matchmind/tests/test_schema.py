"""
Tests: Unified Match Schema
"""
import pytest
from pydantic import ValidationError

from matchmind.schema.match_state import (
    BallState,
    MatchState,
    PlayerState,
    SituationFeatures,
    TacticalAdvice,
    EvidenceItem,
    AlternativeAction,
)
from matchmind.tests.fixtures.sample_data import make_sample_match_state


def test_ball_state_valid():
    b = BallState(x=52.5, y=34.0)
    assert b.x == 52.5
    assert b.z == 0.0


def test_ball_state_invalid():
    with pytest.raises(ValidationError):
        BallState(x=200.0, y=34.0)  # x > 105


def test_player_state_speed():
    p = PlayerState(id=1, team="home", x=50.0, y=34.0, vx=3.0, vy=4.0)
    assert p.speed == 5.0  # 3-4-5 triangle


def test_match_state_requires_both_teams():
    with pytest.raises(ValidationError):
        MatchState(
            match_id="test",
            timestamp=0.0,
            ball=BallState(x=52.5, y=34.0),
            players=[
                PlayerState(id=1, team="home", x=50.0, y=34.0),
            ],
        )


def test_match_state_valid():
    state, focus_id = make_sample_match_state()
    assert len(state.players) == 22
    assert state.minute == 62
    assert state.get_player(focus_id) is not None
    assert len(state.get_team_players("home")) == 11
    assert len(state.get_team_players("away")) == 11


def test_match_state_score_diff():
    state, _ = make_sample_match_state(score_home=0, score_away=1)
    assert state.score_diff_from_home == -1
    assert state.away_winning is True
    assert state.home_winning is False


def test_match_state_get_opponents():
    state, _ = make_sample_match_state()
    opponents = state.get_opponents("home")
    assert all(p.team == "away" for p in opponents)


def test_tactical_advice_structure():
    advice = TacticalAdvice(
        recommended_action="Pass to player 11 in the channel",
        reasoning="Player has space ahead and an open passing lane",
        confidence=0.82,
        evidence=[
            EvidenceItem(id="third_man_run", source="rag", content="Third man run applicable"),
            EvidenceItem(id="computed_features", source="stats", content="Space ahead: 15m"),
        ],
        cited_concepts=["Third-Man Run (Combination Play)"],
        alternatives=[
            AlternativeAction(
                action="Dribble forward",
                why_not="Nearest opponent only 3m away, high press risk",
                delta_pitch_control=-5.0,  # percentage points
            )
        ],
    )
    assert advice.confidence == 0.82
    assert len(advice.alternatives) == 1
    assert advice.alternatives[0].delta_pitch_control == -5.0


def test_alternative_action_delta_coercion():
    """LLMs return '+2.1%' / bare fractions — normalise to percentage points."""
    assert AlternativeAction(action="a", why_not="b", delta_pitch_control="+2.1%").delta_pitch_control == 2.1
    assert AlternativeAction(action="a", why_not="b", delta_pitch_control="-3%").delta_pitch_control == -3.0
    assert AlternativeAction(action="a", why_not="b", delta_pitch_control=-0.05).delta_pitch_control == -5.0
    assert AlternativeAction(action="a", why_not="b", delta_pitch_control="null").delta_pitch_control is None
    assert AlternativeAction(action="a", why_not="b", delta_pitch_control=12.0).delta_pitch_control == 12.0


def test_match_state_property_minute():
    state, _ = make_sample_match_state(minute=72)
    assert state.minute == 72
