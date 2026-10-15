"""Jersey inference for anonymous 360 freeze-frame players.

A dot gets a real identity only when that player touched the ball within a few
seconds of the frame AND the assignment is clearly better than the runner-up;
everything else stays anonymous (spike: ~93% right on that subset, ~55% on
players with no recent touch — too unreliable to show).
"""
from __future__ import annotations

from matchmind.data_sources.jersey_inference import EventRecord, FrameDot, JerseyInference

HOME, AWAY = "Reds", "Blues"
ACTOR, PASSER, FAR, GK = 1, 2, 3, 9
PRESSER, OPP_OTHER, OPP_GK = 21, 22, 29


def _lineup(ids, gk):
    return [(pid, "Goalkeeper" if pid == gk else "Center Midfield") for pid in ids]


def _events() -> list[EventRecord]:
    ev = [
        EventRecord("xi-h", "Starting XI", HOME, None, None, 1, 0.0, lineup=_lineup([ACTOR, PASSER, FAR, GK], GK)),
        EventRecord("xi-a", "Starting XI", AWAY, None, None, 1, 0.0,
                    lineup=_lineup([PRESSER, OPP_OTHER, OPP_GK], OPP_GK)),
        # FAR's last touch is a minute before the frame, on the other flank.
        EventRecord("far", "Pass", HOME, FAR, (30.0, 10.0), 1, 540.0),
        EventRecord("opp-old", "Pass", AWAY, OPP_OTHER, (60.0, 20.0), 1, 500.0),
        EventRecord("gk", "Pass", HOME, GK, (5.0, 40.0), 1, 400.0),
        EventRecord("ogk", "Goal Keeper", AWAY, OPP_GK, (3.0, 40.0), 1, 300.0),
        EventRecord("pass", "Pass", HOME, PASSER, (50.0, 30.0), 1, 599.0),
        EventRecord("rec", "Ball Receipt*", HOME, ACTOR, (62.0, 40.0), 1, 600.0),
        EventRecord("target", "Carry", HOME, ACTOR, (62.0, 40.0), 1, 600.5),
        # Pressure is in the AWAY team's frame: (120-65, 80-42) in the home frame.
        EventRecord("press", "Pressure", AWAY, PRESSER, (55.0, 38.0), 1, 601.0),
    ]
    return ev


def _dots() -> list[FrameDot]:
    return [
        FrameDot(teammate=True, actor=True, keeper=False, x=62.0, y=40.0),
        FrameDot(teammate=True, actor=False, keeper=False, x=52.0, y=31.0),   # the passer, 1.5 s ago
        FrameDot(teammate=True, actor=False, keeper=False, x=35.0, y=12.0),   # FAR — no recent touch
        FrameDot(teammate=False, actor=False, keeper=False, x=65.0, y=41.0),  # the presser
        FrameDot(teammate=False, actor=False, keeper=False, x=80.0, y=60.0),  # nobody touched lately
    ]


def test_recent_touchers_get_identities_others_stay_anonymous():
    inf = JerseyInference(_events())
    got = inf.identify("target", _dots(), actor_id=ACTOR)
    assert got == {1: PASSER, 3: PRESSER}


def test_actor_is_never_reassigned():
    inf = JerseyInference(_events())
    dots = _dots()
    got = inf.identify("target", dots, actor_id=ACTOR)
    assert 0 not in got and ACTOR not in got.values()


def test_substituted_off_player_is_not_a_candidate():
    ev = _events() + [EventRecord("sub", "Substitution", HOME, PASSER, None, 1, 599.5, replacement_id=77)]
    got = JerseyInference(ev).identify("target", _dots(), actor_id=ACTOR)
    assert PASSER not in got.values()


def test_from_open_data_json_parses_raw_events():
    raw = [
        {"id": "a", "type": {"name": "Starting XI"}, "team": {"name": HOME}, "period": 1, "timestamp": "00:00:00.000",
         "tactics": {"lineup": [{"player": {"id": 5}, "position": {"name": "Goalkeeper"}}]}},
        {"id": "b", "type": {"name": "Pass"}, "team": {"name": HOME}, "player": {"id": 5}, "period": 2,
         "timestamp": "00:10:02.500", "location": [10, 20]},
        {"id": "c", "type": {"name": "Substitution"}, "team": {"name": HOME}, "player": {"id": 5}, "period": 2,
         "timestamp": "00:20:00.000", "substitution": {"replacement": {"id": 6}}},
    ]
    recs = [EventRecord.from_open_data(e) for e in raw]
    assert recs[0].lineup == [(5, "Goalkeeper")]
    assert (recs[1].player_id, recs[1].period, recs[1].t, recs[1].location) == (5, 2, 602.5, (10.0, 20.0))
    assert recs[2].replacement_id == 6


def test_local_loader_names_a_recent_passer_and_placeholders_skip_real_numbers(tmp_path):
    """End to end on a tiny open-data tree: #51 (player 501) passed one second
    before the frame, so their dot shows #51 — and the placeholder for the
    remaining anonymous teammate must not also be 51."""
    import json

    from matchmind.data_sources.statsbomb_local import StatsBombLocalLoader
    from matchmind.tests.fixtures.statsbomb_mini import MATCH_ID, _ev, _frame, write_mini_lineups

    root = tmp_path / "d"
    (root / "events").mkdir(parents=True)
    (root / "three-sixty").mkdir(parents=True)
    xi = lambda team, ids, gk: {  # noqa: E731
        "id": f"xi{team}", "index": 0, "type": {"name": "Starting XI"}, "team": {"id": team, "name": "Acting FC" if team == 10 else "Other FC"},
        "period": 1, "timestamp": "00:00:00.000", "minute": 0, "second": 0,
        "tactics": {"lineup": [{"player": {"id": i}, "position": {"name": "Goalkeeper" if i == gk else "Wing"}}
                               for i in ids]}}
    events = [
        xi(10, [500, 501, 502], 502), xi(20, [600, 601], 601),
        _ev(1, "prev", "Pass", 0, 59, [70.0, 30.0], player_id=501, period=1, timestamp="00:00:59.000"),
        _ev(2, "e0", "Pass", 1, 0, [60.0, 40.0], player_id=500, period=1, timestamp="00:01:00.000"),
    ]
    (root / "events" / f"{MATCH_ID}.json").write_text(json.dumps(events), encoding="utf-8")
    (root / "three-sixty" / f"{MATCH_ID}.json").write_text(json.dumps([_frame("e0")]), encoding="utf-8")
    write_mini_lineups(root)

    state = next(StatsBombLocalLoader(root).iter_snapshots(MATCH_ID))[2]
    named = state.get_player(51, "home")
    assert named is not None and named.name == "Other Mate" and (round(named.x), round(named.y)) == (61, 26)
    home_ids = [p.id for p in state.players if p.team == "home"]
    assert len(home_ids) == len(set(home_ids)) and 52 in home_ids


def test_description_lists_identified_players_and_calls_the_rest_placeholders():
    from matchmind.schema.match_state import BallState, MatchState, PlayerState
    from matchmind.situation_engine.encoder import encode_situation

    state = MatchState(
        match_id="1", timestamp=60.0, possession="home", phase_of_play="open_play", source="statsbomb",
        ball=BallState(x=52.0, y=34.0), home_attacks_right=True, event_player_id=3,
        players=[PlayerState(id=3, team="home", name="Chiellini", x=52.0, y=34.0),
                 PlayerState(id=5, team="home", name="Locatelli", x=45.0, y=30.0),
                 PlayerState(id=51, team="home", x=60.0, y=20.0),
                 PlayerState(id=101, team="away", x=58.0, y=36.0)],
    )
    text = encode_situation(state, 3).natural_language_description
    assert "Identified players: #3 Chiellini, #5 Locatelli" in text
    assert "every other number is a placeholder" in text
