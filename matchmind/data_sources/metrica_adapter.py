"""
Metrica Sports Tracking Data Adapter.

Converts Metrica CSV tracking files → MatchState.

Metrica provides raw player tracking data at 25fps with x,y coordinates
normalised to [0,1]. This adapter denormalises to [0,105] × [0,68].

Reference: https://github.com/metrica-sports/sample-data
Coordinate system: Metrica uses [0,1] normalised — we convert to metres.

Usage:
    adapter = MetricaAdapter()
    # Load from CSV files (from LaurieOnTracking format)
    home_df, away_df = adapter.load_tracking_csvs(
        home_path="sample-data/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        away_path="sample-data/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    )
    snapshot = adapter.load_snapshot(home_df, away_df, frame_idx=1000)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from matchmind.data_sources.base_adapter import BaseAdapter
from matchmind.schema.match_state import BallState, MatchState, PlayerState

logger = logging.getLogger(__name__)

PITCH_LENGTH = 105.0  # metres
PITCH_WIDTH = 68.0    # metres

# Metrica CSV expected column patterns
# Home: "Home_1_x", "Home_1_y", "Home_2_x", ...
# Away: "Away_1_x", "Away_1_y", ...
# Ball: "ball_x", "ball_y"


class MetricaAdapter(BaseAdapter):
    """
    Adapter for Metrica Sports sample tracking data.

    Metrica tracking CSVs have:
    - Row 0: team player IDs
    - Row 1: attribute names (x, y)
    - Row 2+: frame data

    Coordinates are normalised [0,1] → we scale to metres.
    """

    def __init__(
        self,
        match_id: str = "metrica_game1",
        fps: float = 25.0,
    ) -> None:
        self.match_id = match_id
        self.fps = fps

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def load_tracking_csvs(
        self,
        home_path: str | Path,
        away_path: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load Metrica CSV tracking files (LaurieOnTracking format).

        Returns (home_df, away_df) with proper column names and NaN handling.
        """
        home_df = self._read_metrica_csv(Path(home_path), "Home")
        away_df = self._read_metrica_csv(Path(away_path), "Away")
        logger.info(f"Loaded Metrica CSVs: {len(home_df)} frames")
        return home_df, away_df

    def load_snapshot(
        self,
        home_df: pd.DataFrame,
        away_df: pd.DataFrame,
        frame_idx: int,
    ) -> MatchState:
        """
        Convert a single frame (row) from Metrica tracking data → MatchState.

        Args:
            home_df: Home team tracking DataFrame (LaurieOnTracking format).
            away_df: Away team tracking DataFrame.
            frame_idx: Row index (frame number).

        Returns:
            MatchState ready for Situation Engine.
        """
        home_row = home_df.iloc[frame_idx]
        away_row = away_df.iloc[frame_idx]

        timestamp = float(frame_idx) / self.fps

        players: list[PlayerState] = []
        players.extend(self._parse_team_row(home_row, "Home", "home"))
        players.extend(self._parse_team_row(away_row, "Away", "away"))

        # Velocities from the previous frame (central-ish difference at 1 frame lag).
        has_velocity = False
        if frame_idx > 0:
            prev = {
                p.id: p
                for p in (
                    self._parse_team_row(home_df.iloc[frame_idx - 1], "Home", "home")
                    + self._parse_team_row(away_df.iloc[frame_idx - 1], "Away", "away")
                )
            }
            for p in players:
                q = prev.get(p.id)
                if q is not None:
                    p.vx = round((p.x - q.x) * self.fps, 3)
                    p.vy = round((p.y - q.y) * self.fps, 3)
            has_velocity = True

        # Ball position — in home_df columns
        ball = self._parse_ball(home_row)

        if len({p.team for p in players}) < 2:
            raise ValueError(f"Frame {frame_idx}: could not find players from both teams")

        return MatchState(
            match_id=self.match_id,
            timestamp=timestamp,
            ball=ball,
            players=players,
            source="metrica",
            phase_of_play="open_play",
            has_velocity=has_velocity,
        )

    def load_test_cases(self, n: int = 25) -> list[dict]:
        """
        Metrica data doesn't have labelled events, so test cases use synthetic questions.
        For real evaluation, use StatsBombAdapter.load_test_cases() instead.
        """
        logger.warning(
            "MetricaAdapter.load_test_cases() returns synthetic test cases — "
            "no ground truth actions. Use StatsBombAdapter for evaluation."
        )
        return []

    def frame_range_to_snapshots(
        self,
        home_df: pd.DataFrame,
        away_df: pd.DataFrame,
        start: int,
        end: int,
        step: int = 25,
    ) -> list[MatchState]:
        """Convert a range of frames to MatchState list (e.g. for animation)."""
        snapshots = []
        for idx in range(start, min(end, len(home_df)), step):
            try:
                snapshots.append(self.load_snapshot(home_df, away_df, idx))
            except Exception as exc:
                logger.debug(f"Skipping frame {idx}: {exc}")
        return snapshots

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _read_metrica_csv(self, path: Path, team: str) -> pd.DataFrame:
        """
        Read Metrica tracking CSV in LaurieOnTracking format.

        LaurieOnTracking's Metrica_IO.read_tracking_csv() outputs a DataFrame
        with column names like "Home_1_x", "Home_1_y", etc. This method
        replicates that loading logic without requiring the full Metrica_IO module.
        """
        # Row 0: player IDs, Row 1: attributes (x/y), Row 2+: data
        df = pd.read_csv(path, header=[0, 1], index_col=0)

        # Flatten multi-index columns: ('1', 'x') → 'Home_1_x'
        new_cols = []
        for col in df.columns:
            player_id, attr = col
            if str(player_id).strip().lower() == "ball":
                new_cols.append(f"ball_{attr.strip()}")
            else:
                new_cols.append(f"{team}_{player_id.strip()}_{attr.strip()}")
        df.columns = new_cols

        # Convert to float, NaN where missing
        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    def _parse_team_row(
        self,
        row: pd.Series,
        team_prefix: str,  # "Home" or "Away"
        team_label: str,   # "home" or "away"
    ) -> list[PlayerState]:
        """Extract all players for one team from a tracking row."""
        players = []
        x_cols = [c for c in row.index if c.startswith(f"{team_prefix}_") and c.endswith("_x")]

        for x_col in x_cols:
            player_id_str = x_col.split("_")[1]
            y_col = x_col.replace("_x", "_y")

            if y_col not in row.index:
                continue

            raw_x = row[x_col]
            raw_y = row[y_col]

            # Skip if player not tracked in this frame
            if pd.isna(raw_x) or pd.isna(raw_y):
                continue

            # Denormalise [0,1] → metres
            x = float(raw_x) * PITCH_LENGTH
            y = float(raw_y) * PITCH_WIDTH

            # Clamp to pitch boundaries
            x = max(0.0, min(PITCH_LENGTH, x))
            y = max(0.0, min(PITCH_WIDTH, y))

            try:
                pid = int(player_id_str)
            except ValueError:
                pid = hash(player_id_str) % 1000

            players.append(
                PlayerState(id=pid, team=team_label, x=x, y=y)
            )

        return players

    def _parse_ball(self, row: pd.Series) -> BallState:
        """Extract ball position from tracking row."""
        ball_x_col = next((c for c in row.index if "ball" in c.lower() and c.endswith("_x")), None)
        ball_y_col = next((c for c in row.index if "ball" in c.lower() and c.endswith("_y")), None)

        if ball_x_col and ball_y_col:
            bx = row.get(ball_x_col)
            by = row.get(ball_y_col)
            if pd.notna(bx) and pd.notna(by):
                return BallState(
                    x=max(0.0, min(PITCH_LENGTH, float(bx) * PITCH_LENGTH)),
                    y=max(0.0, min(PITCH_WIDTH, float(by) * PITCH_WIDTH)),
                )

        # Default: centre of pitch
        logger.debug("Ball position not found in frame — defaulting to centre")
        return BallState(x=52.5, y=34.0)
