"""
Tests: Situation Engine (pure computation, no mocks needed)
"""
import pytest

from matchmind.situation_engine.encoder import encode_situation
from matchmind.situation_engine.spatial_features import (
    compute_spatial_features,
    euclidean,
)
from matchmind.situation_engine.team_structure import compute_team_structure
from matchmind.situation_engine.passing_analyzer import (
    compute_passing_opportunities,
    _is_lane_open,
)
from matchmind.tests.fixtures.sample_data import make_sample_match_state, make_counter_attack_state


def test_euclidean_distance():
    assert euclidean((0, 0), (3, 4)) == 5.0
    assert euclidean((0, 0), (0, 0)) == 0.0


def test_spatial_features_distances():
    state, focus_id = make_sample_match_state()
    features = compute_spatial_features(state, focus_id)

    assert features["distance_to_ball"] >= 0.0
    assert features["nearest_opponent_distance"] > 0.0
    assert isinstance(features["local_numerical_advantage"], int)
    assert isinstance(features["half_space_occupied"], bool)
    assert isinstance(features["is_in_attacking_third"], bool)


def test_spatial_features_focus_player_not_found():
    state, _ = make_sample_match_state()
    with pytest.raises(ValueError, match="not found"):
        compute_spatial_features(state, 9999)


def test_team_structure_both_teams():
    state, _ = make_sample_match_state()
    struct = compute_team_structure(state, "home")

    assert struct["defensive_line_height"] >= 0.0
    assert struct["defensive_line_height"] <= 105.0
    assert struct["team_compactness"] >= 0.0
    assert struct["attacking_width"] >= 0.0
    assert isinstance(struct["numerical_superiority_zone"], bool)


def test_passing_lane_open_no_blockers():
    from matchmind.schema.match_state import PlayerState
    blockers = []
    assert _is_lane_open((0, 0), (20, 0), blockers) is True


def test_passing_lane_blocked():
    from matchmind.schema.match_state import PlayerState
    # Blocker directly in the middle of the lane
    blockers = [PlayerState(id=1, team="away", x=10.0, y=0.0)]
    assert _is_lane_open((0, 0), (20, 0), blockers) is False


def test_passing_opportunities_structure():
    state, focus_id = make_sample_match_state()
    result = compute_passing_opportunities(state, focus_id)

    assert "passing_lane_open" in result
    assert "progressive_passes_available" in result
    assert "third_man_opportunity" in result
    assert "switch_play_viable" in result
    assert isinstance(result["open_passing_lane_player_ids"], list)


def test_encode_situation_returns_features():
    state, focus_id = make_sample_match_state()
    features = encode_situation(state, focus_id)

    assert features.focus_player_id == focus_id
    assert features.focus_player_team == "home"
    assert features.distance_to_ball >= 0.0
    assert len(features.natural_language_description) > 50
    assert "Player 10" in features.natural_language_description


def test_encode_situation_counter_attack():
    state, focus_id = make_counter_attack_state()
    features = encode_situation(state, focus_id)

    assert features.space_ahead > 0
    # Counter-attack state: player at 55m x, large space ahead
    assert features.space_ahead >= 5.0


def test_encode_situation_nl_description_contains_key_info():
    state, focus_id = make_sample_match_state(minute=62, score_home=0, score_away=1)
    features = encode_situation(state, focus_id)

    desc = features.natural_language_description
    assert "minute 62" in desc.lower() or "62" in desc
    assert "home" in desc.lower()
