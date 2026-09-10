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

        # Use La Liga 2015/16 season — has good 360 data
        matches = sb.matches(competition_id=11, season_id=37)  # La Liga 2015/16
        if matches.empty:
            # Fallback: Women's Champions League
            matches = sb.matches(competition_id=37, season_id=42)

        test_cases: list[dict] = []
        target_event_types = ["Pass", "Shot", "Dribble", "Carry"]

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
        # Competitions known to have 360 data
        comp_seasons = [(11, 37), (37, 42), (53, 106)]
        for comp_id, season_id in comp_seasons:
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

        # Ball position
        location = event.get("location", [52.5, 34.0])
        ball = BallState(x=float(location[0]), y=float(location[1]))

        # Determine possession from event's team
        event_team = event.get("team", "")
        # StatsBomb uses home/away via match lineup — simplified heuristic here
        possession: str | None = None  # set downstream if needed

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
        """Parse StatsBomb 360 freeze_frame → list[PlayerState]."""
        players: list[PlayerState] = []

        if frame_rows.empty:
            # Fallback: create synthetic players if no 360 data
            logger.debug(f"No freeze frame for event {event.get('id')} — using synthetic data")
            return self._make_synthetic_players(event)

        for _, row in frame_rows.iterrows():
            teammate = row.get("teammate", True)
            team: str = "home" if teammate else "away"

            loc = row.get("location", [52.5, 34.0])
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue

            player_id = int(row.get("player_id", len(players) + 1)) if pd.notna(row.get("player_id", None)) else len(players) + 1
            position_name = str(row.get("position", ""))
            role = SB_POSITION_MAP.get(position_name, position_name.lower().replace(" ", "_") or None)

            players.append(
                PlayerState(
                    id=player_id,
                    team=team,
                    role=role,
                    x=float(loc[0]),
                    y=float(loc[1]),
                )
            )

        # Ensure we have both teams
        teams = {p.team for p in players}
        if len(teams) < 2 and players:
            # Add at least one opponent
            players.append(
                PlayerState(id=999, team="away" if "home" in teams else "home", x=80.0, y=40.0)
            )

        return players or self._make_synthetic_players(event)

    def _make_synthetic_players(self, event: pd.Series) -> list[PlayerState]:
        """Create minimal synthetic player list when freeze frame is missing."""
        loc = event.get("location", [52.5, 34.0])
        x, y = float(loc[0]), float(loc[1])
        return [
            PlayerState(id=1, team="home", role="attacking_midfielder", x=x, y=y),
            PlayerState(id=2, team="home", role="striker", x=x + 10, y=y + 5),
            PlayerState(id=3, team="away", role="center_back", x=x + 15, y=y),
            PlayerState(id=4, team="away", role="center_back", x=x + 15, y=y - 5),
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
