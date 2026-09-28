"""
SituationFeatures + user question -> RetrievalQuery.

The phrases are deliberately written in the KB's own vocabulary (the words
concept titles/definitions use), never numbers: a concept like "Third-Man Run"
is found by "third-man combination", not by "Distance to ball: 0.0m".
Thresholds mirror the trigger predicates in retrieval/triggers.py so the two
signals agree on what a situation "is".
"""
from __future__ import annotations

import re

from matchmind.retrieval.types import RetrievalQuery
from matchmind.schema.match_state import MatchState, SituationFeatures


def feature_phrases(
    f: SituationFeatures, s: MatchState, in_possession: bool | None
) -> list[str]:
    phrases: list[str] = []
    phase = s.phase_of_play
    if (f.focus_player_role or "").lower() in {"goalkeeper", "gk", "keeper"}:
        # A keeper is never "running in behind" — see candidate_actions._goalkeeper_actions.
        if f.is_ball_carrier:
            phrases += ["goalkeeper distribution", "build-up from the goalkeeper"]
        elif in_possession is False:
            phrases += ["goalkeeper positioning", "narrow the shooting angle", "defending the box"]
        else:
            phrases += ["goalkeeper positioning while the team attacks",
                        "sweeper keeper covering space behind the defence", "rest defence"]
        if phase == "set_piece":
            phrases.append("set piece routine")
        return phrases
    if in_possession is False:
        phrases.append("defending out of possession")
        if phase == "defensive_transition":
            phrases.append("counter-press immediately after losing the ball")
        if f.distance_to_ball < 10.0:
            phrases.append("press the ball carrier")
        else:
            phrases.append("defensive shape and compactness")
        if f.defensive_line_height > 55.0:
            phrases.append("high defensive line")
    else:
        if not f.is_ball_carrier:
            phrases.append("off-ball movement to receive or create space")
        if f.is_ball_carrier and f.nearest_opponent_distance < 3.0:
            phrases.append("under pressure, escape the press")
        if f.third_man_opportunity:
            phrases.append("third-man combination")
        if f.switch_play_viable:
            phrases.append("switch of play to the far side")
        if not f.half_space_occupied and f.is_in_attacking_third:
            phrases.append("attack the half-space")
        if f.space_ahead > 15.0:
            phrases.append("space behind the defensive line, run in behind")
        elif f.space_ahead < 8.0:
            phrases.append("tight space, combine with short passes")
        if f.local_numerical_advantage > 0:
            phrases.append("numerical superiority near the ball")
        elif f.local_numerical_advantage < 0:
            phrases.append("outnumbered near the ball, create an overload")
        if f.overload_left or f.overload_right:
            phrases.append("flank overload")
        if f.progressive_passes_available > 0:
            phrases.append("progressive pass to break lines")
        if f.is_in_attacking_third:
            phrases.append("final third chance creation")
        if phase == "attacking_transition":
            phrases.append("counter-attack after winning the ball")
    if phase == "set_piece":
        phrases.append("set piece routine")
    return phrases


def _clean_question(question: str) -> str:
    # "player 11" -> "player": shirt numbers only add noise to BM25/embeddings.
    return re.sub(r"\s+", " ", re.sub(r"[#\d]+", " ", question or "")).strip()


def build_retrieval_query(
    features: SituationFeatures,
    match_state: MatchState,
    question: str = "",
) -> RetrievalQuery:
    in_possession = (
        None if match_state.possession is None
        else match_state.possession == features.focus_player_team
    )
    return RetrievalQuery(
        question=_clean_question(question),
        tactical_phrases=feature_phrases(features, match_state, in_possession),
        situation_text=features.natural_language_description,
        in_possession=in_possession,
        phase_of_play=match_state.phase_of_play,
    )
