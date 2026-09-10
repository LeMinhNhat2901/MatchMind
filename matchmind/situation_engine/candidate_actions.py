"""
Candidate Action Generator.

Rule-based mapping: SituationFeatures -> a small set of tactical options.
Pure computation, no LLM. The output feeds BOTH:
  * counterfactual_node  (pitch-control delta per action)
  * reasoner_node        (the LLM chooses / justifies among them)

Keeping this in one place stops the candidate list from being re-derived
inconsistently in different nodes.
"""
from __future__ import annotations

import logging

from matchmind.schema.match_state import CandidateAction, MatchState, SituationFeatures

logger = logging.getLogger(__name__)

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Always-evaluated baseline options
_DEFAULT_ACTIONS: list[tuple[str, str]] = [
    ("hold_and_recycle", "Hold the ball and recycle possession"),
    ("dribble_forward", "Dribble forward into the space ahead"),
]


def generate_candidate_actions(
    state: MatchState,
    feats: SituationFeatures,
    max_actions: int = 6,
) -> list[CandidateAction]:
    """Build 2-6 candidate actions for the focus player.

    Args:
        state: current snapshot.
        feats: computed SituationFeatures for the focus player.
        max_actions: hard cap on how many options to return.
    """
    focus = state.get_player(feats.focus_player_id)
    if focus is None:
        raise ValueError(f"Focus player {feats.focus_player_id} not in match state")

    actions: list[CandidateAction] = []
    seen: set[str] = set()

    def add(ca: CandidateAction) -> None:
        if ca.action in seen:
            return
        seen.add(ca.action)
        actions.append(ca)

    # 1. One pass_to_<id> per open passing lane (highest signal first: progressive lanes)
    lane_ids = list(feats.open_passing_lane_player_ids)
    # progressive_passes_available is a count in the current schema; if we ever expose
    # the ids we prefer those — fall back to plain lane order.
    for pid in lane_ids:
        tm = state.get_player(pid)
        if tm is None:
            continue
        add(
            CandidateAction(
                action=f"pass_to_{pid}",
                label=f"Pass to player {pid} at ({tm.x:.0f}, {tm.y:.0f})",
                target_xy=(tm.x, tm.y),
                target_player_id=pid,
                origin="passing_lane",
            )
        )

    # 2. Situation-driven options
    if feats.half_space_occupied or feats.space_ahead > 8.0:
        add(
            CandidateAction(
                action="dribble_half_space",
                label="Dribble / carry into the half-space",
                origin="situation",
            )
        )
    if feats.switch_play_viable:
        add(
            CandidateAction(
                action="switch_play",
                label="Switch play to the opposite flank",
                origin="situation",
            )
        )
    if feats.third_man_opportunity:
        add(
            CandidateAction(
                action="third_man_combination",
                label="Third-man combination through a nearby teammate",
                origin="situation",
            )
        )
    if feats.is_in_attacking_third and focus.x > 80.0:
        add(
            CandidateAction(
                action="shoot",
                label="Shoot at goal",
                target_xy=(104.0, 34.0),
                origin="situation",
            )
        )
    if feats.overload_left:
        add(CandidateAction(action="exploit_left_overload", label="Attack the left-flank overload", origin="situation"))
    if feats.overload_right:
        add(CandidateAction(action="exploit_right_overload", label="Attack the right-flank overload", origin="situation"))

    # 3. Baseline fallbacks so there is always something to compare against
    for name, label in _DEFAULT_ACTIONS:
        add(CandidateAction(action=name, label=label, origin="default"))

    result = actions[:max_actions]
    logger.debug(
        "[candidate_actions] player=%s -> %s",
        feats.focus_player_id,
        [a.action for a in result],
    )
    return result
