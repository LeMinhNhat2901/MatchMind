"""
StatsBomb Open Data Adapter.

Converts StatsBomb 360 freeze-frame data → MatchState.

StatsBomb 360 provides visible player positions (freeze_frame) at the moment
of each event. This is the primary data source for Phase 1 MVP.

Reference data: statsbomb/open-data repository
Usage:
    adapter = StatsBombAdapter()
    snapshot = adapter.load_snapshot(match_id=3788741, event_index=150)
    test_cases = adapter.load_test_cases(n=25)
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from matchmind.data_sources.base_adapter import BaseAdapter
from matchmind.schema.match_state import BallState, MatchState, PlayerState

logger = logging.getLogger(__name__)

# ── Coordinate systems ──────────────────────────────────────────────────────
# StatsBomb's own pitch unit is 120 (x) x 80 (y) — NOT our schema's 105x68
# metres. Every raw location coming out of statsbombpy (events, 360 freeze
# frames, visible_area) must be rescaled before it goes into PlayerState/
# BallState, or it silently distorts every distance-based feature and can
# exceed the schema's [0,105]x[0,68] bounds (very common near either box).
SB_PITCH_LENGTH = 120.0
SB_PITCH_WIDTH = 80.0
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def _sb_to_metres(x: float, y: float) -> tuple[float, float]:
    """Rescale a raw StatsBomb (0-120, 0-80) coordinate to our (0-105, 0-68) metres.

    360 freeze-frame points are camera-estimated and occasionally fall a
    little outside the nominal 120x80 box — clamp after scaling.
    """
    mx = x * (PITCH_LENGTH / SB_PITCH_LENGTH)
    my = y * (PITCH_WIDTH / SB_PITCH_WIDTH)
    return max(0.0, min(PITCH_LENGTH, mx)), max(0.0, min(PITCH_WIDTH, my))


# Known StatsBomb Open Data (competition_id, season_id) pairs that actually
# ship 360 freeze-frame data. La Liga 2015/16 (11, 37) — the season previously
# hardcoded here — predates 360 entirely and returns 404 for every match.
KNOWN_360_COMPETITIONS: list[tuple[int, int]] = [
    (43, 106),   # FIFA World Cup 2022 — most complete public 360 dataset
    (11, 90),    # La Liga 2020/2021
    (55, 43),    # UEFA Euro 2020
    (55, 282),   # UEFA Euro 2024
    (53, 106),   # UEFA Women's Euro 2022
    (53, 315),   # UEFA Women's Euro 2025
    (9, 281),    # 1. Bundesliga 2023/2024
    (7, 235),    # Ligue 1 2022/2023
    (7, 108),    # Ligue 1 2021/2022
    (44, 107),   # Major League Soccer 2023
    (72, 107),   # Women's World Cup 2023
    (1267, 107), # Africa Cup of Nations 2023
]

# Mapping from StatsBomb position names to our role strings
SB_POSITION_MAP: dict[str, str] = {
    "Goalkeeper": "goalkeeper",
    "Right Back": "right_back",
    "Right Center Back": "center_back",
    "Center Back": "center_back",
    "Left Center Back": "center_back",
    "Left Back": "left_back",
    "Right Wing Back": "right_wingback",
    "Left Wing Back": "left_wingback",
    "Right Defensive Midfield": "defensive_midfielder",
    "Center Defensive Midfield": "defensive_midfielder",
    "Left Defensive Midfield": "defensive_midfielder",
    "Right Center Midfield": "central_midfielder",
    "Center Midfield": "central_midfielder",
    "Left Center Midfield": "central_midfielder",
    "Right Midfield": "right_midfielder",
    "Left Midfield": "left_midfielder",
    "Right Attacking Midfield": "attacking_midfielder",
    "Center Attacking Midfield": "attacking_midfielder",
    "Left Attacking Midfield": "attacking_midfielder",
    "Right Wing": "right_winger",
    "Left Wing": "left_winger",
    "Right Center Forward": "striker",
    "Center Forward": "striker",
    "Left Center Forward": "striker",
    "Secondary Striker": "striker",
}

# Phase of play mapping from StatsBomb event types
PHASE_MAP: dict[str, str] = {
    "Pass": "open_play",
    "Carry": "open_play",
    "Shot": "attacking_transition",
    "Dribble": "attacking_transition",
    "Ball Receipt*": "open_play",
    "Pressure": "defensive_transition",
    "Tackle": "defensive_transition",
    "Interception": "defensive_transition",
    "Goal Keeper": "open_play",
    "Clearance": "defensive_transition",
    "Foul Committed": "open_play",
    "Free Kick": "set_piece",
    "Corner": "set_piece",
    "Throw-in": "set_piece",
    "Goal": "open_play",
}


class StatsBombAdapter(BaseAdapter):
    """
    Adapter for StatsBomb Open Data (including 360 freeze-frames).

    Requires: `pip install statsbombpy`
    Data: https://github.com/statsbomb/open-data
    """

    def __init__(self) -> None:
        try:
            import statsbombpy  # noqa: F401
        except ImportError:
            raise ImportError(
                "statsbombpy not installed. Run: pip install statsbombpy"
            )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def load_snapshot(
        self,
        match_id: int,
        event_index: int,
        home_team_id: int | None = None,
    ) -> MatchState:
        """
        Load a single snapshot from StatsBomb 360 data.

        Args:
            match_id: StatsBomb match ID.
            event_index: Row index in the events DataFrame.
            home_team_id: If None, determined automatically.

        Returns:
            MatchState — ready for Situation Engine and Agent.
        """
        from statsbombpy import sb

        events = sb.events(match_id=match_id, fmt="dataframe")
        frames = sb.frames(match_id=match_id)

        event_row = events.iloc[event_index]
        return self._event_to_match_state(event_row, frames, match_id)

    def load_test_cases(self, n: int = 25) -> list[dict]:
        """
        Load N test cases from StatsBomb 360 for evaluation.

        Selects events that have 360 freeze-frame data AND a clear following action.
        Prioritises attacking events (Pass, Carry, Shot, Dribble).
        """
        from statsbombpy import sb

        test_cases: list[dict] = []
        target_event_types = ["Pass", "Shot", "Dribble", "Carry"]

        for comp_id, season_id in KNOWN_360_COMPETITIONS:
            if len(test_cases) >= n:
                break
            try:
                matches = sb.matches(competition_id=comp_id, season_id=season_id)
            except Exception as exc:
                logger.debug(f"No matches for competition={comp_id} season={season_id}: {exc}")
                continue
            if matches.empty:
                continue

            for _, match in matches.iterrows():
                if len(test_cases) >= n:
                    break
                mid = match["match_id"]
                try:
                    cases = self._extract_test_cases_from_match(mid, target_event_types, limit=5)
                    test_cases.extend(cases)
                except Exception as exc:
                    logger.warning(f"Skipping match {mid}: {exc}")
                    continue

        return test_cases[:n]

    def list_available_matches(self) -> pd.DataFrame:
        """List all StatsBomb Open Data matches with 360 data."""
        from statsbombpy import sb

        all_competitions = sb.competitions()
        dfs = []
        for comp_id, season_id in KNOWN_360_COMPETITIONS:
            try:
                m = sb.matches(competition_id=comp_id, season_id=season_id)
                m["competition_id"] = comp_id
                m["season_id"] = season_id
                dfs.append(m)
            except Exception:
                pass
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _event_to_match_state(
        self,
        event: pd.Series,
        frames: pd.DataFrame,
        match_id: int,
    ) -> MatchState:
        """Convert a StatsBomb event row + 360 frames → MatchState."""
        event_id = event["id"]

        # Get freeze frame for this event
        frame_rows = frames[frames["id"] == event_id]
        players = self._parse_freeze_frame(frame_rows, event)
        visible_area = self._parse_visible_area(frame_rows)

        # Ball position — StatsBomb units (0-120, 0-80) rescaled to metres
        location = event.get("location", [60.0, 40.0])
        bx, by = _sb_to_metres(float(location[0]), float(location[1]))
        ball = BallState(x=bx, y=by)

        # Phase of play
        event_type = str(event.get("type", "Pass"))
        phase = PHASE_MAP.get(event_type, "open_play")

        # In freeze frames we map the acting team's players to "home", so for
        # on-ball events (Pass/Carry/Dribble/Shot) "home" has possession.
        possession = "home" if event_type in ("Pass", "Carry", "Dribble", "Shot") else None

        return MatchState(
            match_id=str(match_id),
            timestamp=float(event.get("minute", 0)) * 60 + float(event.get("second", 0)),
            score_home=int(event.get("score_home", 0)) if "score_home" in event else 0,
            score_away=int(event.get("score_away", 0)) if "score_away" in event else 0,
            possession=possession,
            phase_of_play=phase,
            ball=ball,
            players=players,
            visible_area=visible_area,
            # StatsBomb 360 is a single freeze-frame → no velocity → degraded pitch control.
            has_velocity=False,
            source="statsbomb",
            event_type=event_type,
            event_player_id=int(event.get("player_id", -1)) if pd.notna(event.get("player_id")) else None,
        )

    def _parse_freeze_frame(
        self,
        frame_rows: pd.DataFrame,
        event: pd.Series,
    ) -> list[PlayerState]:
        """Parse StatsBomb 360 freeze_frame → list[PlayerState].

        A 360 freeze frame only contains players inside the broadcast camera's
        field of view (see _parse_visible_area) — real matches range from ~4
        to 21 tracked players, NEVER all 22. That is expected, not a bug.
        """
        players: list[PlayerState] = []

        if frame_rows.empty:
            # Fallback: create synthetic players if no 360 data
            logger.debug(f"No freeze frame for event {event.get('id')} — using synthetic data")
            return self._make_synthetic_players(event)

        for _, row in frame_rows.iterrows():
            teammate = row.get("teammate", True)
            team: str = "home" if teammate else "away"

            loc = row.get("location")
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue
            px, py = _sb_to_metres(float(loc[0]), float(loc[1]))

            player_id = int(row.get("player_id", len(players) + 1)) if pd.notna(row.get("player_id", None)) else len(players) + 1
            position_name = str(row.get("position", ""))
            role = SB_POSITION_MAP.get(position_name, position_name.lower().replace(" ", "_") or None)

            players.append(
                PlayerState(
                    id=player_id,
                    team=team,
                    role=role,
                    x=px,
                    y=py,
                )
            )

        # Ensure we have both teams
        teams = {p.team for p in players}
        if len(teams) < 2 and players:
            # Add at least one opponent
            players.append(
                PlayerState(id=999, team="away" if "home" in teams else "home", x=70.0, y=34.0)
            )

        return players or self._make_synthetic_players(event)

    def _parse_visible_area(self, frame_rows: pd.DataFrame) -> list[tuple[float, float]] | None:
        """Extract + rescale the 360 camera field-of-view polygon for this event.

        Same polygon is repeated on every row of the freeze frame — take it
        from the first row. Stored as [x0,y0,x1,y1,...] flat SB-unit pairs.
        """
        if frame_rows.empty:
            return None
        raw = frame_rows.iloc[0].get("visible_area")
        if not isinstance(raw, (list, tuple)) or len(raw) < 6 or len(raw) % 2 != 0:
            return None
        points = [
            _sb_to_metres(float(raw[i]), float(raw[i + 1]))
            for i in range(0, len(raw), 2)
        ]
        return points

    def _make_synthetic_players(self, event: pd.Series) -> list[PlayerState]:
        """Create minimal synthetic player list when freeze frame is missing."""
        loc = event.get("location", [60.0, 40.0])
        x, y = _sb_to_metres(float(loc[0]), float(loc[1]))
        return [
            PlayerState(id=1, team="home", role="attacking_midfielder", x=x, y=y),
            PlayerState(id=2, team="home", role="striker", x=min(x + 10, PITCH_LENGTH), y=min(y + 5, PITCH_WIDTH)),
            PlayerState(id=3, team="away", role="center_back", x=min(x + 15, PITCH_LENGTH), y=y),
            PlayerState(id=4, team="away", role="center_back", x=min(x + 15, PITCH_LENGTH), y=max(y - 5, 0.0)),
        ]

    def _extract_test_cases_from_match(
        self,
        match_id: int,
        target_event_types: list[str],
        limit: int = 5,
    ) -> list[dict]:
        """Extract test cases with ground truth from a single match."""
        from statsbombpy import sb

        events = sb.events(match_id=match_id, fmt="dataframe")
        frames = sb.frames(match_id=match_id)

        cases = []
        for idx, row in events.iterrows():
            if len(cases) >= limit:
                break
            event_type = str(row.get("type", ""))
            if event_type not in target_event_types:
                continue

            # Need a freeze frame
            frame_rows = frames[frames["id"] == row["id"]]
            if frame_rows.empty:
                continue

            try:
                snapshot = self._event_to_match_state(row, frames, match_id)
            except Exception as exc:
                logger.debug(f"Skipping event {idx}: {exc}")
                continue

            # Focus on the player who performed the event
            focus_id = int(row.get("player_id", snapshot.players[0].id)) if pd.notna(row.get("player_id", None)) else snapshot.players[0].id

            # Ground truth = the event_type itself
            gt_action = event_type.lower()

            cases.append(
                {
                    "snapshot": snapshot,
                    "focus_player_id": focus_id,
                    "question": f"What should player {focus_id} do in this situation?",
                    "ground_truth_action": gt_action,
                    "ground_truth_event_type": event_type,
                }
            )

        return cases
