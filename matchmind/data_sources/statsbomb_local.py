"""
StatsBomb Open Data — local reader.

Reads a local clone of https://github.com/statsbomb/open-data (default
../open-data/data, see settings.statsbomb_open_data_dir) instead of going
through statsbombpy's network calls. Used by evaluation test-case caching and
by the Example Library builder, which both need thousands of events.

360 freeze frames carry NO player ids — only teammate/actor/keeper flags. The
actor is the event's player, so it gets the real StatsBomb player id (this is
what keeps a test case's focus_player_id inside its own snapshot). Everyone
else gets a placeholder id (teammates 51+, opponents 101+) unless
jersey_inference can name them from their touches moments earlier. The acting team is
always "home" (same convention as StatsBombAdapter).

Licence: StatsBomb Open Data — free for non-commercial use with attribution.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterator

from matchmind.data_sources.jersey_inference import EventRecord, FrameDot, JerseyInference
from matchmind.data_sources.statsbomb_adapter import SB_POSITION_MAP, _sb_to_metres
from matchmind.schema.match_state import BallState, MatchState, PlayerState

logger = logging.getLogger(__name__)

ON_BALL_TYPES: frozenset[str] = frozenset({"Pass", "Carry", "Dribble", "Shot"})
SET_PIECE_PASS_TYPES: frozenset[str] = frozenset(
    {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
)
# 360 frames identify only the actor. Everyone else is anonymous in the data, so
# they get placeholder numbers that can't pass for real shirt numbers.
_TEAMMATE_ID_BASE = 50
_OPPONENT_ID_BASE = 100
_PADDING_OPPONENT_ID = 999

Roster = dict[int, tuple[int, str]]   # StatsBomb player_id -> (jersey number, display name)


def _next_placeholder(current: int, reserved: int | set[int]) -> int:
    reserved = {reserved} if isinstance(reserved, int) else reserved
    nxt = current + 1
    while nxt in reserved:
        nxt += 1
    return nxt


def _seconds(event: dict) -> float:
    return float(event.get("minute", 0)) * 60.0 + float(event.get("second", 0))


def _phase(event: dict) -> str:
    type_name = event["type"]["name"]
    if type_name == "Pass":
        pass_type = ((event.get("pass") or {}).get("type") or {}).get("name")
        if pass_type in SET_PIECE_PASS_TYPES:
            return "set_piece"
    if (event.get("play_pattern") or {}).get("name") == "From Counter":
        return "attacking_transition"
    return "open_play"


def _is_goal(event: dict) -> bool:
    return (
        event["type"]["name"] == "Shot"
        and ((event.get("shot") or {}).get("outcome") or {}).get("name") == "Goal"
    )


class StatsBombLocalLoader:
    """Read events + 360 frames for one match at a time from a local open-data tree."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not (self.root / "events").is_dir() or not (self.root / "three-sixty").is_dir():
            raise FileNotFoundError(
                f"{self.root} is not a StatsBomb open-data 'data' dir "
                "(needs events/ and three-sixty/)"
            )

    def match_ids_with_360(self) -> list[int]:
        return sorted(int(p.stem) for p in (self.root / "three-sixty").glob("*.json"))

    def events(self, match_id: int) -> list[dict]:
        path = self.root / "events" / f"{match_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def frames(self, match_id: int) -> dict[str, dict]:
        path = self.root / "three-sixty" / f"{match_id}.json"
        return {f["event_uuid"]: f for f in json.loads(path.read_text(encoding="utf-8"))}

    def roster(self, match_id: int) -> Roster:
        """StatsBomb player_id -> (jersey number, nickname or name), from lineups/."""
        path = self.root / "lineups" / f"{match_id}.json"
        if not path.exists():
            return {}
        out: Roster = {}
        for team in json.loads(path.read_text(encoding="utf-8")):
            for p in team.get("lineup", []):
                if p.get("jersey_number") is not None:
                    out[int(p["player_id"])] = (int(p["jersey_number"]),
                                                p.get("player_nickname") or p.get("player_name"))
        return out

    def to_match_state(
        self,
        event: dict,
        frame: dict,
        match_id: int,
        score: tuple[int, int] = (0, 0),
        opponent_name: str | None = None,
        roster: Roster | None = None,
        inference: JerseyInference | None = None,
    ) -> MatchState | None:
        """One on-ball event + its freeze frame -> MatchState, or None if the
        frame has no actor (the focus player would not exist).

        The actor is labelled with their real shirt number and name when the
        roster knows them (else the StatsBomb player id)."""
        sb_id = int(event["player"]["id"])
        actor_id, actor_name = (roster or {}).get(sb_id, (sb_id, (event.get("player") or {}).get("name")))
        role_name = (event.get("position") or {}).get("name", "")
        players: list[PlayerState] = []
        rows = [r for r in frame.get("freeze_frame", [])
                if isinstance(r.get("location"), (list, tuple)) and len(r["location"]) >= 2]
        # Anonymous players who touched the ball moments ago get their real
        # number too (see jersey_inference); the rest keep placeholders.
        known: dict[int, tuple[int, str]] = {}
        if inference is not None and roster:
            dots = [FrameDot(bool(r.get("teammate")), bool(r.get("actor")), bool(r.get("keeper")),
                             float(r["location"][0]), float(r["location"][1])) for r in rows]
            known = {i: roster[pid] for i, pid in inference.identify(event["id"], dots, sb_id).items()
                     if pid in roster}
        reserved = {actor_id} | {num for num, _ in known.values()}
        mate_id = _next_placeholder(_TEAMMATE_ID_BASE, reserved)
        opp_id, has_actor = _next_placeholder(_OPPONENT_ID_BASE, reserved), False

        for i, row in enumerate(rows):
            loc = row["location"]
            x, y = _sb_to_metres(float(loc[0]), float(loc[1]))
            if i in known and not row.get("actor"):
                num, name = known[i]
                players.append(PlayerState(
                    id=num, team="home" if row.get("teammate") else "away", name=name,
                    role="goalkeeper" if row.get("keeper") else None, x=x, y=y,
                ))
            elif row.get("actor"):
                has_actor = True
                players.append(PlayerState(
                    id=actor_id, team="home", name=actor_name,
                    role=SB_POSITION_MAP.get(role_name, role_name.lower().replace(" ", "_") or None),
                    x=x, y=y,
                ))
            elif row.get("teammate"):
                players.append(PlayerState(
                    id=mate_id, team="home",
                    role="goalkeeper" if row.get("keeper") else None, x=x, y=y,
                ))
                mate_id = _next_placeholder(mate_id, reserved)
            else:
                players.append(PlayerState(
                    id=opp_id, team="away",
                    role="goalkeeper" if row.get("keeper") else None, x=x, y=y,
                ))
                opp_id = _next_placeholder(opp_id, reserved)

        if not has_actor:
            return None
        if not any(p.team == "away" for p in players):
            # MatchState requires both teams; a camera that sees no opponent is
            # rare but real — pad one far away so it barely affects features.
            players.append(PlayerState(id=_PADDING_OPPONENT_ID, team="away", x=100.0, y=34.0))

        raw_area = frame.get("visible_area") or []
        visible = (
            [_sb_to_metres(float(raw_area[i]), float(raw_area[i + 1])) for i in range(0, len(raw_area), 2)]
            if len(raw_area) >= 6 and len(raw_area) % 2 == 0 else None
        )
        bx, by = _sb_to_metres(float(event["location"][0]), float(event["location"][1]))
        return MatchState(
            match_id=str(match_id),
            timestamp=_seconds(event),
            score_home=score[0],
            score_away=score[1],
            possession="home",
            phase_of_play=_phase(event),
            ball=BallState(x=bx, y=by),
            players=players,
            visible_area=visible,
            has_velocity=False,
            # StatsBomb coordinates put the acting team ("home") attacking +x;
            # 360 frames often miss the keepers, so don't leave this to inference.
            home_attacks_right=True,
            # "home" here is the acting team, often the match's AWAY side.
            home_team_name=(event.get("team") or {}).get("name"),
            away_team_name=opponent_name,
            source="statsbomb",
            event_type=event["type"]["name"],
            event_player_id=actor_id,
        )

    def iter_snapshots(
        self,
        match_id: int,
        event_types: frozenset[str] = ON_BALL_TYPES,
    ) -> Iterator[tuple[int, dict, MatchState]]:
        """Yield (index, raw event, snapshot) for every usable on-ball event.

        Score is tracked from the acting team's point of view: goals by the
        event's team count as score_home, everyone else's as score_away.
        """
        events = self.events(match_id)
        frames = self.frames(match_id)
        roster = self.roster(match_id)
        inference = JerseyInference(EventRecord.from_open_data(e) for e in events)
        team_names = {
            (e.get("team") or {}).get("id"): (e.get("team") or {}).get("name") for e in events if e.get("team")
        }
        goals: dict[int, int] = {}
        for idx, event in enumerate(events):
            team_id = (event.get("team") or {}).get("id")
            if event["type"]["name"] in event_types and event["id"] in frames and event.get("location"):
                own = goals.get(team_id, 0)
                other = sum(v for k, v in goals.items() if k != team_id)
                opponent = next((n for tid, n in team_names.items() if tid != team_id), None)
                try:
                    state = self.to_match_state(event, frames[event["id"]], match_id, (own, other), opponent, roster,
                                               inference)
                except Exception as exc:  # noqa: BLE001 — one malformed frame must not stop a scan
                    logger.debug(f"skip {match_id}:{idx}: {exc}")
                    state = None
                if state is not None:
                    yield idx, event, state
            if _is_goal(event):
                goals[team_id] = goals.get(team_id, 0) + 1

    def load_test_cases(self, n: int = 30, per_match: int = 3, seed: int = 7) -> list[dict]:
        """Open-play on-ball cases, spread across matches (at most `per_match`
        each, sampled with a fixed seed so the cache is reproducible)."""
        rng = random.Random(seed)
        cases: list[dict] = []
        for match_id in self.match_ids_with_360():
            if len(cases) >= n:
                break
            usable = [(i, e, s) for i, e, s in self.iter_snapshots(match_id) if s.phase_of_play != "set_piece"]
            for _idx, event, state in rng.sample(usable, min(per_match, len(usable))):
                focus = state.event_player_id   # shirt number when the lineup knows it
                cases.append({
                    "snapshot": state,
                    "focus_player_id": focus,
                    "question": f"What should player {focus} do in this situation?",
                    "ground_truth_action": event["type"]["name"].lower(),
                    "ground_truth_event_type": event["type"]["name"],
                })
                if len(cases) >= n:
                    break
        return cases
