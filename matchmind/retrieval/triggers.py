"""
Trigger-condition predicates: machine-checkable meanings of the tokens in
TacticalConcept.trigger_conditions.

Two consumers:
- retrieval (condition_ranking) — a third ranked list for RRF: concepts whose
  preconditions are actually true in this situation.
- evaluation (retrieval_quality.py) — the old Dim-0 proxy ground truth. Once
  condition matching is a ranking signal that proxy is partly circular, so the
  hand-labelled golden set (evaluation/retrieval_bench.py) is the primary
  retrieval metric from V1 on.
"""
from __future__ import annotations

from typing import Callable, Iterable

from matchmind.schema.match_state import MatchState, SituationFeatures, TacticalConcept


# ── Trigger token → predicate(feats, state) -> bool ───────────────────────────
# Only tokens present here are "checkable". Unknown tokens are ignored when
# scoring a concept (they neither help nor hurt its relevance).

def _press_intensity(feats: SituationFeatures) -> float:
    return float(feats.computed_stats.get("press_intensity", 99.0))


def _players_near_ball(state: MatchState, radius: float = 15.0) -> int:
    bx, by = state.ball.x, state.ball.y
    return sum(1 for p in state.players if ((p.x - bx) ** 2 + (p.y - by) ** 2) ** 0.5 <= radius)


TRIGGER_PREDICATES: dict[str, Callable[[SituationFeatures, MatchState], bool]] = {
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


def trigger_match_fraction(
    concept: TacticalConcept,
    feats: SituationFeatures,
    state: MatchState,
) -> float | None:
    """Fraction of the concept's *checkable* trigger tokens that are satisfied.

    Returns ``None`` when none of the concept's triggers are checkable.
    """
    checkable = [t for t in concept.trigger_conditions if t in TRIGGER_PREDICATES]
    if not checkable:
        return None
    hits = 0
    for t in checkable:
        try:
            if TRIGGER_PREDICATES[t](feats, state):
                hits += 1
        except Exception:  # noqa: BLE001 — a broken predicate must not kill eval
            pass
    return hits / len(checkable)


def condition_ranking(
    concepts: Iterable[TacticalConcept],
    feats: SituationFeatures,
    state: MatchState,
    *,
    min_fraction: float = 0.5,
    min_checkable: int = 2,
) -> list[str]:
    """Concepts whose checkable triggers are mostly satisfied, best first.

    min_checkable=2: a concept with a single (often generic) checkable trigger
    would match almost every situation and flood the list.
    """
    scored: list[tuple[float, int, str]] = []
    for c in concepts:
        n_checkable = sum(1 for t in c.trigger_conditions if t in TRIGGER_PREDICATES)
        if n_checkable < min_checkable:
            continue
        fraction = trigger_match_fraction(c, feats, state)
        if fraction is not None and fraction >= min_fraction:
            scored.append((fraction, n_checkable, c.concept_id))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [cid for _, _, cid in scored]
