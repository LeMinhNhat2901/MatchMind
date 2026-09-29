"""A tiny on-disk StatsBomb open-data tree for loader/example-builder tests."""
from __future__ import annotations

import json
from pathlib import Path

MATCH_ID = 1


def _ev(idx, uuid, type_name, minute, second, loc, team_id=10, player_id=500,
        possession=1, possession_team=10, **extra):
    e = {
        "id": uuid, "index": idx, "type": {"name": type_name},
        "minute": minute, "second": second, "location": loc,
        "team": {"id": team_id, "name": "Acting FC" if team_id == 10 else "Other FC"},
        "player": {"id": player_id, "name": f"P{player_id}"},
        "position": {"name": "Right Wing"},
        "possession": possession, "possession_team": {"id": possession_team},
        "play_pattern": {"name": "Regular Play"},
    }
    e.update(extra)
    return e


def _frame(uuid, with_actor=True, opponents=True):
    ff = [{"teammate": True, "actor": with_actor, "keeper": False, "location": [60.0, 40.0]},
          {"teammate": True, "actor": False, "keeper": False, "location": [70.0, 30.0]},
          {"teammate": True, "actor": False, "keeper": True, "location": [5.0, 40.0]}]
    if opponents:
        ff += [{"teammate": False, "actor": False, "keeper": False, "location": [65.0, 42.0]},
               {"teammate": False, "actor": False, "keeper": True, "location": [118.0, 40.0]}]
    return {"event_uuid": uuid, "visible_area": [0, 0, 120, 0, 120, 80, 0, 80, 0, 0],
            "freeze_frame": ff}


def write_mini_lineups(root: Path) -> Path:
    """Optional lineups/<match>.json: the actor (player 500) wears #8."""
    (root / "lineups").mkdir(parents=True, exist_ok=True)
    lineups = [{"team_id": 10, "team_name": "Acting FC", "lineup": [
        {"player_id": 500, "player_name": "Andrea Pirlo Full", "player_nickname": "Pirlo", "jersey_number": 8},
        {"player_id": 501, "player_name": "Other Mate", "player_nickname": None, "jersey_number": 51},
    ]}]
    (root / "lineups" / f"{MATCH_ID}.json").write_text(json.dumps(lineups), encoding="utf-8")
    return root


def write_mini_open_data(root: Path) -> Path:
    """Events: 0 pass (360, actor) · 1 carry (360, NO actor) · 2 corner pass (360)
    · 3 shot=goal by team 10 · 4 pass (360, only one team visible) · 5 pass w/o 360."""
    (root / "events").mkdir(parents=True)
    (root / "three-sixty").mkdir(parents=True)
    events = [
        _ev(0, "e0", "Pass", 10, 0, [60.0, 40.0],
            **{"pass": {"end_location": [80.0, 30.0], "height": {"name": "Ground Pass"}}}),
        _ev(1, "e1", "Carry", 10, 3, [80.0, 30.0], player_id=501),
        _ev(2, "e2", "Pass", 11, 0, [120.0, 0.0],
            **{"pass": {"type": {"name": "Corner"}, "end_location": [110.0, 40.0]}}),
        _ev(3, "e3", "Shot", 11, 5, [110.0, 40.0], **{"shot": {"outcome": {"name": "Goal"}}}),
        _ev(4, "e4", "Pass", 20, 0, [50.0, 40.0], possession=2),
        _ev(5, "e5", "Pass", 20, 4, [55.0, 40.0], possession=2),
    ]
    frames = [_frame("e0"), _frame("e1", with_actor=False), _frame("e2"),
              _frame("e4", opponents=False)]
    (root / "events" / f"{MATCH_ID}.json").write_text(json.dumps(events), encoding="utf-8")
    (root / "three-sixty" / f"{MATCH_ID}.json").write_text(json.dumps(frames), encoding="utf-8")
    return root
