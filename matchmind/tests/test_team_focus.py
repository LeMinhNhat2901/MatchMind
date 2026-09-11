"""Tests for situation_engine.team_focus (dual attack/defense focus picking)."""
from __future__ import annotations

from matchmind.situation_engine.team_focus import pick_dual_focus_players
from matchmind.tests.fixtures.sample_data import make_counter_attack_state, make_sample_match_state


def test_attacking_focus_is_the_ball_carrier():
    state, carrier_id = make_sample_match_state()
    focus = pick_dual_focus_players(state)
    assert focus.attacking_player_id == carrier_id
    assert focus.attacking_team == state.get_player(carrier_id).team


def test_defending_focus_is_an_opponent():
    state, carrier_id = make_sample_match_state()
    focus = pick_dual_focus_players(state)
    carrier_team = state.get_player(carrier_id).team
    defender = state.get_player(focus.defending_player_id)
    assert defender is not None
    assert defender.team != carrier_team
    assert focus.defending_team != focus.attacking_team


def test_defending_focus_is_nearest_opponent_to_carrier():
    state, carrier_id = make_sample_match_state()
    carrier = state.get_player(carrier_id)
    focus = pick_dual_focus_players(state)
    opponents = state.get_opponents(carrier.team)
    expected = min(opponents, key=lambda p: (p.x - carrier.x) ** 2 + (p.y - carrier.y) ** 2)
    assert focus.defending_player_id == expected.id


def test_works_on_counter_attack_fixture_too():
    state, carrier_id = make_counter_attack_state()
    focus = pick_dual_focus_players(state)
    assert focus.attacking_player_id == carrier_id
    assert focus.defending_player_id != carrier_id
