"""
Evaluation — Dimension 0: Retrieval Quality.

precision@k / recall@k for the RAG retriever, using each concept's
``trigger_conditions`` as a proxy ground truth:

    a concept is "relevant" to a situation when >= RELEVANCE_MIN_FRACTION of its
    machine-checkable trigger tokens are satisfied by the SituationFeatures /
    MatchState.

This is a proxy (trigger tokens are hand-written and only partly checkable),
so treat the numbers as directional — exactly as bonus.md §6 / revise.md §9
intend. It is still the single most important RAG health metric.
"""
from __future__ import annotations

import logging
from typing import Callable

from matchmind.schema.match_state import MatchState, SituationFeatures, TacticalConcept

logger = logging.getLogger(__name__)

RELEVANCE_MIN_FRACTION = 0.5


# ── Trigger token → predicate(feats, state) -> bool ───────────────────────────
# Only tokens present here are "checkable". Unknown tokens are ignored when
# scoring a concept (they neither help nor hurt its relevance).

def _press_intensity(feats: SituationFeatures) -> float:
    return float(feats.computed_stats.get("press_intensity", 99.0))


def _players_near_ball(state: MatchState, radius: float = 15.0) -> int:
    bx, by = state.ball.x, state.ball.y
    return sum(1 for p in state.players if ((p.x - bx) ** 2 + (p.y - by) ** 2) ** 0.5 <= radius)


_TRIGGERS: dict[str, Callable[[SituationFeatures, MatchState], bool]] = {
    # spatial / zones
    "ball_near_flank": lambda f, s: s.ball.y < 20.0 or s.ball.y > 48.0,
    "half_space_occupied": lambda f, s: f.half_space_occupied,
    "half_space_free": lambda f, s: not f.half_space_occupied,
    "half_space_occupied_or_available": lambda f, s: True,
    "is_in_attacking_third": lambda f, s: f.is_in_attacking_third,
    "is_in_defensive_third": lambda f, s: f.is_in_defensive_third,
    "space_ahead_large": lambda f, s: f.space_ahead > 15.0,
    "space_ahead_large_for_opponent": lambda f, s: f.space_ahead > 15.0,
    "space_ahead_moderate": lambda f, s: 8.0 <= f.space_ahead <= 15.0,
    "space_ahead_small": lambda f, s: f.space_ahead < 8.0,
    "space_behind_def_line": lambda f, s: f.space_ahead > 12.0,
    "space_behind_defensive_line": lambda f, s: f.space_ahead > 12.0,
    "space_between_lines": lambda f, s: f.space_ahead > 10.0,
    "nearest_opponent_distance_low": lambda f, s: f.nearest_opponent_distance < 4.0,
    # numerical
    "local_numerical_advantage": lambda f, s: f.local_numerical_advantage > 0,
    "local_numerical_advantage_positive": lambda f, s: f.local_numerical_advantage > 0,
    "local_numerical_advantage_negative": lambda f, s: f.local_numerical_advantage < 0,
    "numerical_superiority_zone": lambda f, s: f.numerical_superiority_zone,
    "overload_flank": lambda f, s: f.overload_left or f.overload_right,
    "overload_left_or_right": lambda f, s: f.overload_left or f.overload_right,
    # passing / combination
    "passing_lane_open": lambda f, s: f.passing_lane_open,
    "progressive_pass_available": lambda f, s: f.progressive_passes_available > 0,
    "progressive_pass_not_available": lambda f, s: f.progressive_passes_available == 0,
    "third_man_opportunity": lambda f, s: f.third_man_opportunity,
    "switch_play_viable": lambda f, s: f.switch_play_viable,
    "switch_play_not_viable": lambda f, s: not f.switch_play_viable,
    # team structure
    "attacking_width_high": lambda f, s: f.attacking_width > 30.0,
    "team_compactness_high": lambda f, s: f.team_compactness < 20.0,
    "team_compactness_low": lambda f, s: f.team_compactness > 25.0,
    "defensive_line_height_high": lambda f, s: f.defensive_line_height > 55.0,
    "defensive_line_high": lambda f, s: f.defensive_line_height > 55.0,
    "opponent_defensiveline_high": lambda f, s: f.defensive_line_height > 55.0,
    "defensive_line_height_low": lambda f, s: f.defensive_line_height < 45.0,
    "press_intensity_high": lambda f, s: _press_intensity(f) < 8.0,
    "press_intensity_capable": lambda f, s: _press_intensity(f) < 12.0,
    "opponent_high_press": lambda f, s: _press_intensity(f) < 8.0,
    "players_near_ball": lambda f, s: _players_near_ball(s) >= 3,
    # phase
    "phase_of_play_attacking_transition": lambda f, s: s.phase_of_play == "attacking_transition",
    "phase_of_play_defensive_transition": lambda f, s: s.phase_of_play == "defensive_transition",
    "phase_of_play_defensive": lambda f, s: s.phase_of_play in ("defensive_transition", "settled_defense"),
    "phase_of_play_transition": lambda f, s: (s.phase_of_play or "").endswith("transition"),
    # role
    "striker_role": lambda f, s: (s.get_player(f.focus_player_id) or _NOROLE).role == "striker",
}


class _NoRole:
    role = None


_NOROLE = _NoRole()


def concept_relevance_score(
    concept: TacticalConcept,
    feats: SituationFeatures,
    state: MatchState,
) -> float | None:
    """Fraction of the concept's *checkable* trigger tokens that are satisfied.

    Returns ``None`` when none of the concept's triggers are checkable.
    """
    checkable = [t for t in concept.trigger_conditions if t in _TRIGGERS]
    if not checkable:
        return None
    hits = 0
    for t in checkable:
        try:
            if _TRIGGERS[t](feats, state):
                hits += 1
        except Exception:  # noqa: BLE001 — a broken predicate must not kill eval
            pass
    return hits / len(checkable)


def relevant_concept_ids(
    concepts: list[TacticalConcept],
    feats: SituationFeatures,
    state: MatchState,
) -> set[str]:
    """Proxy ground-truth relevant set for one situation."""
    out: set[str] = set()
    for c in concepts:
        score = concept_relevance_score(c, feats, state)
        if score is not None and score >= RELEVANCE_MIN_FRACTION:
            out.add(c.concept_id)
    return out


def evaluate_retrieval(
    retrieved_ids: list[str],
    concepts: list[TacticalConcept],
    feats: SituationFeatures,
    state: MatchState,
    k: int = 3,
) -> dict:
    """Precision@k / recall@k for one retrieval against the proxy ground truth."""
    relevant = relevant_concept_ids(concepts, feats, state)
    top_k = list(retrieved_ids[:k])
    hits = len(set(top_k) & relevant)

    precision = hits / max(len(top_k), 1)
    recall = hits / len(relevant) if relevant else None  # undefined when no relevant concept

    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4) if recall is not None else None,
        "n_relevant": len(relevant),
        "n_retrieved": len(top_k),
        "n_hits": hits,
        "relevant_ids": sorted(relevant),
        "retrieved_ids": top_k,
    }


def aggregate_retrieval_quality(per_case: list[dict], k: int) -> dict:
    """Mean precision/recall across cases (cases with no relevant concept are
    excluded from the recall mean, not counted as 0)."""
    if not per_case:
        return {"precision_at_k": None, "recall_at_k": None, "retrieval_k": k, "n_cases": 0}
    prec = [c["precision_at_k"] for c in per_case if c.get("precision_at_k") is not None]
    rec = [c["recall_at_k"] for c in per_case if c.get("recall_at_k") is not None]
    return {
        "precision_at_k": round(sum(prec) / len(prec), 4) if prec else None,
        "recall_at_k": round(sum(rec) / len(rec), 4) if rec else None,
        "retrieval_k": k,
        "n_cases": len(per_case),
        "n_cases_with_relevant": len(rec),
    }
