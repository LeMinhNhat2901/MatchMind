"""Build SituationExamples from StatsBomb open data (local clone)."""
from __future__ import annotations

import logging

from matchmind.data_sources.statsbomb_local import StatsBombLocalLoader
from matchmind.examples.labeling import SB_X_TO_M, label_outcome
from matchmind.examples.schema import SituationExample
from matchmind.schema.match_state import MatchState, SituationFeatures
from matchmind.situation_engine.encoder import encode_situation

logger = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = (
    "nearest_opponent_distance", "nearest_teammate_distance", "space_ahead",
    "local_numerical_advantage", "progressive_passes_available", "passing_lane_open",
    "third_man_opportunity", "switch_play_viable", "half_space_occupied",
    "is_in_attacking_third", "defensive_line_height", "ball_x", "ball_y_from_centre",
)
_EVENTS_URL = "https://github.com/statsbomb/open-data/blob/master/data/events/{match_id}.json"
_OUTCOME_TEXT = {"shot": "the move ended in a shot", "progressed": "the ball was carried/played forward",
                 "retained": "the team kept the ball without progressing", "lost": "possession was lost"}


def feature_vector(f: SituationFeatures, s: MatchState) -> list[float]:
    return [
        float(f.nearest_opponent_distance), float(f.nearest_teammate_distance), float(f.space_ahead),
        float(f.local_numerical_advantage), float(f.progressive_passes_available),
        float(f.passing_lane_open), float(f.third_man_opportunity), float(f.switch_play_viable),
        float(f.half_space_occupied), float(f.is_in_attacking_third),
        # A back line outside the camera view is unknown, not "wherever the deepest
        # visible presser stands" — use a neutral value so it doesn't skew the kNN.
        float(f.defensive_line_height if f.defensive_line_visible else 52.5),
        float(s.ball.x), float(abs(s.ball.y - 34.0)),
    ]


def describe_action(event: dict) -> tuple[str, str]:
    kind = event["type"]["name"]
    if kind == "Pass":
        p = event.get("pass") or {}
        end = p.get("end_location") or event["location"]
        dx = (float(end[0]) - float(event["location"][0])) * SB_X_TO_M
        label = "progressive pass" if dx >= 10 else ("backward pass" if dx <= -5 else "square pass")
        if p.get("cross"):
            label = "cross"
        if p.get("through_ball"):
            label = "through ball"
        if p.get("switch"):
            label = "switch of play"
        height = ((p.get("height") or {}).get("name") or "").lower()
        return "pass", f"{label} ({height})" if height else label
    if kind == "Carry":
        end = (event.get("carry") or {}).get("end_location") or event["location"]
        dx = (float(end[0]) - float(event["location"][0])) * SB_X_TO_M
        return "carry", "forward carry" if dx >= 5 else "carry"
    if kind == "Dribble":
        return "dribble", "take-on"
    return "shot", "shot"


def _summary(f: SituationFeatures, action_detail: str, outcome: str) -> str:
    adv = f.local_numerical_advantage
    numbers = "outnumbered" if adv < 0 else ("with a numerical advantage" if adv > 0 else "level on numbers")
    zone = "in the attacking third" if f.is_in_attacking_third else "outside the attacking third"
    pressure = "under close pressure" if f.nearest_opponent_distance < 3.0 else "with time on the ball"
    return f"Carrier {pressure}, {numbers}, {zone}: played a {action_detail}; {_OUTCOME_TEXT[outcome]}."


def build_examples(
    loader: StatsBombLocalLoader, match_ids: list[int], max_per_match: int = 200
) -> list[SituationExample]:
    out: list[SituationExample] = []
    for match_id in match_ids:
        events = loader.events(match_id)
        n = 0
        for idx, event, state in loader.iter_snapshots(match_id):
            if n >= max_per_match:
                break
            if state.phase_of_play == "set_piece":
                continue
            try:
                feats = encode_situation(state, state.event_player_id)
            except Exception as exc:  # noqa: BLE001 — skip unusable frames, keep scanning
                logger.debug(f"skip example {match_id}:{idx}: {exc}")
                continue
            action_type, detail = describe_action(event)
            outcome = label_outcome(events, idx)
            out.append(SituationExample(
                example_id=f"sb_{match_id}_{idx}", match_id=str(match_id), event_index=idx,
                minute=int(event.get("minute", 0)), action_type=action_type, action_detail=detail,
                outcome=outcome, summary=_summary(feats, detail, outcome),
                vector=feature_vector(feats, state),
                source_url=_EVENTS_URL.format(match_id=match_id),
            ))
            n += 1
    return out
