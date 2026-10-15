"""Real team names, neutral pressure wording, and partial camera views.

User feedback (2026-09-29) on StatsBomb match 3788741, event 20:
- show real team names (StatsBomb "home" = the acting team, not the match's home side);
- having the ball does not make a team "the attacking team" — state facts only;
- a partial camera view is the normal case (StatsBomb 360 now, V2/V3 later): the
  opponent's back line must not be invented from whoever happens to be in frame.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from matchmind.retrieval.triggers import TRIGGER_PREDICATES
from matchmind.schema.match_state import BallState, MatchState, PlayerState
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import make_sample_match_state
from matchmind.tests.fixtures.statsbomb_mini import MATCH_ID, write_mini_open_data

OPEN_DATA = Path(__file__).resolve().parents[3] / "open-data" / "data"


# ── team names ────────────────────────────────────────────────────────────────

def test_team_label_falls_back_to_side():
    s, _ = make_sample_match_state()
    assert s.team_label("home") == "home"
    named = s.model_copy(update={"home_team_name": "Italy", "away_team_name": "Turkey"})
    assert named.team_label("away") == "Turkey"
    assert named.score_line_for(23, "away") == "Player 23's team (Turkey) is leading 1–0 against Italy"


def test_local_loader_carries_team_names(tmp_path):
    from matchmind.data_sources.statsbomb_local import StatsBombLocalLoader

    _, _, state = next(StatsBombLocalLoader(write_mini_open_data(tmp_path / "d")).iter_snapshots(MATCH_ID))
    assert state.home_team_name == "Acting FC"          # the event's own team


@pytest.mark.skipif(not (OPEN_DATA / "events" / "3788741.json").exists(), reason="open-data clone not present")
def test_real_statsbomb_event_is_described_with_real_team_names():
    from matchmind.data_sources.statsbomb_local import StatsBombLocalLoader

    snaps = {i: s for i, _, s in StatsBombLocalLoader(OPEN_DATA).iter_snapshots(3788741)}
    state = snaps[min(i for i in snaps if i >= 20)]
    assert {state.home_team_name, state.away_team_name} == {"Italy", "Turkey"}
    text = encode_situation(state, state.event_player_id).natural_language_description
    assert f"({state.home_team_name})" in text and "home team" not in text


# ── pressure wording ──────────────────────────────────────────────────────────

def test_off_ball_player_is_not_described_as_under_pressure():
    s, _ = make_sample_match_state()
    carrier = s.get_player(10, "home")
    presser = s.model_copy(update={"players": [
        p.model_copy(update={"x": carrier.x + 1.5, "y": carrier.y}) if p.id == 26 else p for p in s.players]})
    text = encode_situation(presser, 26).natural_language_description
    assert "under high pressure" not in text
    assert "Nearest opponent" in text


# ── opponent back line ────────────────────────────────────────────────────────

def test_opponent_back_line_is_its_deepest_players():
    """Bug: the 'line' was the mean of the opponent's 4 most ADVANCED players
    (x≈54.8 in the sample) instead of the 4 deepest (the back four, x≈80.5)."""
    s, _ = make_sample_match_state()
    f = encode_situation(s, 10)
    assert f.defensive_line_height == pytest.approx((80 + 82 + 82 + 78) / 4, abs=0.01)
    assert f.defensive_line_visible


def _partial_state(camera_max_x: float) -> MatchState:
    players = [PlayerState(id=1, team="home", role="goalkeeper", x=5, y=34),
               PlayerState(id=9, team="home", x=40, y=30),
               PlayerState(id=10, team="home", x=45, y=34),
               PlayerState(id=21, team="away", x=50, y=36),
               PlayerState(id=22, team="away", x=55, y=30)]
    area = [(20.0, 5.0), (camera_max_x, 5.0), (camera_max_x, 63.0), (20.0, 63.0)]
    return MatchState(match_id="m", timestamp=60.0, ball=BallState(x=45.5, y=34), possession="home",
                      players=players, visible_area=area, home_attacks_right=True)


def test_back_line_outside_camera_view_is_reported_as_unknown():
    f = encode_situation(_partial_state(camera_max_x=58.0), 10)   # camera ends right behind #22
    assert not f.defensive_line_visible
    assert "not in the camera view" in f.natural_language_description
    assert "Camera view shows 5 of 22 players" in f.natural_language_description
    assert not TRIGGER_PREDICATES["defensive_line_high"](f, _partial_state(58.0))


def test_back_line_is_trusted_when_the_camera_sees_behind_it():
    f = encode_situation(_partial_state(camera_max_x=75.0), 10)   # 20 m of empty pitch behind #22
    assert f.defensive_line_visible
    assert f.defensive_line_height == pytest.approx(52.5, abs=0.01)


# ── dual analysis wording ─────────────────────────────────────────────────────

def test_dual_questions_do_not_assume_the_ball_holder_is_attacking():
    import inspect

    from matchmind.agent.graph import TacticalAgent

    src = inspect.getsource(TacticalAgent.analyze_dual)
    assert "stop the attack" not in src
