"""StatsBombLocalLoader reads open-data from disk and keeps the focus player in the snapshot."""
from __future__ import annotations

import pandas as pd
import pytest

from matchmind.data_sources.statsbomb_local import StatsBombLocalLoader
from matchmind.tests.fixtures.statsbomb_mini import MATCH_ID, write_mini_open_data


@pytest.fixture
def loader(tmp_path):
    return StatsBombLocalLoader(write_mini_open_data(tmp_path / "data"))


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        StatsBombLocalLoader(tmp_path / "nope")


def test_match_ids(loader):
    assert loader.match_ids_with_360() == [MATCH_ID]


def test_actor_keeps_statsbomb_player_id(loader):
    snaps = list(loader.iter_snapshots(MATCH_ID))
    idx, event, state = snaps[0]
    assert idx == 0
    actor = state.get_player(500)
    assert actor is not None and actor.team == "home"
    assert state.event_player_id == 500
    assert state.possession == "home"
    assert state.home_attacks_right is True       # set, not inferred (frames often lack keepers)


def test_frame_without_actor_is_skipped(loader):
    indices = [i for i, _, _ in loader.iter_snapshots(MATCH_ID)]
    assert 1 not in indices          # carry whose frame has no actor
    assert 5 not in indices          # no 360 frame at all


def test_corner_is_set_piece_phase(loader):
    states = {i: s for i, _, s in loader.iter_snapshots(MATCH_ID)}
    assert states[2].phase_of_play == "set_piece"
    assert states[0].phase_of_play == "open_play"


def test_running_score_counts_goals_before_event(loader):
    states = {i: s for i, _, s in loader.iter_snapshots(MATCH_ID, frozenset({"Pass", "Shot"}))}
    assert (states[0].score_home, states[0].score_away) == (0, 0)
    assert (states[4].score_home, states[4].score_away) == (1, 0)   # goal at index 3


def test_single_team_frame_gets_padded_opponent(loader):
    states = {i: s for i, _, s in loader.iter_snapshots(MATCH_ID)}
    assert {p.team for p in states[4].players} == {"home", "away"}


def test_player_ids_do_not_collide(loader):
    _, _, state = next(loader.iter_snapshots(MATCH_ID))
    ids = [p.id for p in state.players]
    assert len(ids) == len(set(ids))


def test_load_test_cases_focus_player_exists(loader):
    cases = loader.load_test_cases(n=5, per_match=5)
    assert cases, "expected at least one case"
    for c in cases:
        assert c["snapshot"].get_player(c["focus_player_id"]) is not None
        assert c["ground_truth_event_type"] in {"Pass", "Carry", "Dribble", "Shot"}
        assert c["snapshot"].phase_of_play != "set_piece"


def test_statsbombpy_adapter_maps_actor_to_event_player():
    """Regression: the network adapter numbered the actor 1..n, so the case's
    focus_player_id (StatsBomb player id) was never in the snapshot."""
    from matchmind.data_sources.statsbomb_adapter import StatsBombAdapter

    rows = pd.DataFrame([
        {"id": "u", "teammate": True, "actor": True, "keeper": False, "location": [60.0, 40.0]},
        {"id": "u", "teammate": False, "actor": False, "keeper": False, "location": [65.0, 40.0]},
    ])
    event = pd.Series({"id": "u", "type": "Pass", "minute": 1, "second": 0,
                       "location": [60.0, 40.0], "player_id": 12072})
    state = StatsBombAdapter()._event_to_match_state(event, rows, 99)
    assert state.get_player(12072) is not None
