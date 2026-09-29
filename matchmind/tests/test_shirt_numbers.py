"""StatsBomb actors get their real shirt number and name.

User report (2026-09-29): answers called the player "số 23558" — the StatsBomb
player id. The lineups give the jersey number; the other players in a 360 frame
are anonymous in the data, so they keep placeholder numbers that cannot be
mistaken for real ones (teammates 51+, opponents 101+) and the description says so.
"""
from __future__ import annotations

import pandas as pd

from matchmind.data_sources.statsbomb_local import StatsBombLocalLoader
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.statsbomb_mini import MATCH_ID, write_mini_lineups, write_mini_open_data


def _first_state(tmp_path, with_lineups: bool):
    root = write_mini_open_data(tmp_path / "d")
    if with_lineups:
        write_mini_lineups(root)
    return next(StatsBombLocalLoader(root).iter_snapshots(MATCH_ID))[2]


def test_actor_gets_jersey_number_and_name(tmp_path):
    state = _first_state(tmp_path, with_lineups=True)
    actor = state.get_player(8, "home")
    assert actor is not None and actor.name == "Pirlo"
    assert state.event_player_id == 8


def test_anonymous_players_get_placeholder_numbers_that_never_collide(tmp_path):
    state = _first_state(tmp_path, with_lineups=True)
    mates = [p.id for p in state.players if p.team == "home" and p.name is None]
    opps = [p.id for p in state.players if p.team == "away"]
    assert mates and all(51 <= i < 100 for i in mates)
    assert 51 not in mates or 8 != 51          # placeholder never reuses the actor's number
    assert all(i >= 101 for i in opps)
    ids = [p.id for p in state.players]
    assert len(ids) == len(set(ids))


def test_without_lineups_actor_keeps_statsbomb_id(tmp_path):
    state = _first_state(tmp_path, with_lineups=False)
    assert state.get_player(500, "home") is not None


def test_description_names_the_actor_and_flags_placeholders(tmp_path):
    state = _first_state(tmp_path, with_lineups=True)
    text = encode_situation(state, 8).natural_language_description
    assert "Player 8 (Pirlo" in text
    assert "placeholder" in text


def test_statsbombpy_adapter_uses_jersey_numbers():
    from matchmind.data_sources.statsbomb_adapter import StatsBombAdapter

    rows = pd.DataFrame([
        {"id": "u", "teammate": True, "actor": True, "keeper": False, "location": [60.0, 40.0]},
        {"id": "u", "teammate": True, "actor": False, "keeper": False, "location": [50.0, 40.0]},
        {"id": "u", "teammate": False, "actor": False, "keeper": False, "location": [65.0, 40.0]},
    ])
    event = pd.Series({"id": "u", "type": "Pass", "minute": 1, "second": 0, "team": "Italy",
                       "location": [60.0, 40.0], "player_id": 12072, "player": "Giorgio Chiellini"})
    state = StatsBombAdapter()._event_to_match_state(
        event, rows, 99, ["Italy", "Turkey"], roster={12072: (3, "Chiellini")})
    actor = state.get_player(3, "home")
    assert actor is not None and actor.name == "Chiellini"
    assert state.event_player_id == 3
    assert all(p.id >= 51 for p in state.players if p is not actor)
