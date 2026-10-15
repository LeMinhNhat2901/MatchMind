"""
Post-Recommendation State Projection — "what happens next?"

Projects a MatchState forward according to one resolved CandidateAction, so
the agent can be asked a genuine follow-up question about the RESULTING
situation instead of re-analysing the original one. Optional and
user-triggered (see dashboard/app.py's "what happens next?" button /
TacticalAgent.analyze_followup) — never runs automatically, since it's a
second full agent pass (LLM cost) on top of the initial recommendation.

Two distinct follow-up shapes, matching what a coach would actually ask after
a recommendation:
  - on-ball action (pass/dribble/shot): "they've just done that — what now?"
    -> move the ball + receiver to the target, re-run the full agent focused
       on whoever now has the ball.
  Everyone else then reacts (team_shift: zero-token geometric rules capped by
  real running speeds), so the next situation isn't a frozen copy of this one.
  - off-ball action (a run): the ball doesn't move, but the run has a marking
    consequence — compute which opponent was nearest at the run's start vs
    its end, so "who does this pull out of position" is a real geometric
    fact handed to the agent, not something the LLM has to invent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from matchmind.schema.match_state import CandidateAction, MatchState, PlayerState
from matchmind.situation_engine.team_shift import (
    Move,
    elapsed_for_pass,
    elapsed_for_run,
    shift_after_pass,
    shift_after_run,
)


@dataclass
class Projection:
    state: MatchState
    focus_player_id: int
    question: str
    # Every player that moved, for the pitch's "estimated" trails (team_shift).
    moves: list[Move] = field(default_factory=list)


def _estimate_note(elapsed_s: float) -> str:
    return (f" Other players' positions are estimated reactions over the ~{elapsed_s:.1f} s this takes, "
            "not observed data.")


def _nearest_opponent(state: MatchState, team: str, x: float, y: float) -> PlayerState | None:
    opponents = state.get_opponents(team)
    if not opponents:
        return None
    return min(opponents, key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)


def project_after_action(
    state: MatchState,
    focus_player_id: int,
    focus_team: str,
    action: CandidateAction,
) -> Projection | None:
    """Build the hypothetical next-state + follow-up question for one resolved
    action. None when there's nothing meaningful to project: no target
    coordinates, or a sequence-ending action (a shot has no "next").
    """
    if action.action == "shoot" or action.target_xy is None:
        return None

    tx, ty = action.target_xy

    if action.moves_ball:
        new_carrier_id = action.target_player_id or focus_player_id
        receiver = state.get_player(new_carrier_id, team=focus_team)
        new_players = [
            p.model_copy(update={"x": tx, "y": ty})
            if p.id == new_carrier_id and p.team == focus_team
            else p
            for p in state.players
        ]
        # Move the ball to the receiver too — situation_engine's carrier
        # detection falls back to "nearest player to the ball" for any source
        # without StatsBomb on-ball event metadata (synthetic, Metrica, and
        # even StatsBomb once we're projecting a hypothetical), so this is
        # what makes the new carrier actually get recognised as such.
        # The receiver is now the acting player: StatsBomb snapshots carry the
        # PASSER as event_player_id, which the encoder trusts over ball
        # proximity, so leaving it made the passer "still have the ball".
        new_state = state.model_copy(update={
            "players": new_players,
            "ball": state.ball.model_copy(update={"x": tx, "y": ty}),
            "event_player_id": new_carrier_id,
            "event_type": "Ball Receipt*",
            "possession": focus_team,
        })
        # A pass travels at ball speed; a dribble only as fast as the carrier runs.
        timing = elapsed_for_pass if action.target_player_id else elapsed_for_run
        elapsed = timing((state.ball.x, state.ball.y), (tx, ty))
        new_state, moves = shift_after_pass(state, new_state, attacking_team=focus_team,
                                            carrier_id=new_carrier_id, elapsed_s=elapsed)
        if receiver is not None and (receiver.x, receiver.y) != (tx, ty):
            moves.insert(0, Move(focus_team, new_carrier_id, (receiver.x, receiver.y), (tx, ty)))
        question = (
            f"Player {new_carrier_id} has just received the ball at ({tx:.1f}, {ty:.1f}) as a "
            f"direct result of the previous recommendation ({action.label})."
            f"{_estimate_note(elapsed)} What should they do now?"
        )
        return Projection(state=new_state, focus_player_id=new_carrier_id, question=question, moves=moves)

    # Off-ball run: only the focus player's position changes, the ball stays put.
    focus = state.get_player(focus_player_id, team=focus_team)
    if focus is None:
        return None

    before_marker = _nearest_opponent(state, focus_team, focus.x, focus.y)
    after_marker = _nearest_opponent(state, focus_team, tx, ty)

    new_players = [
        p.model_copy(update={"x": tx, "y": ty})
        if p.id == focus_player_id and p.team == focus_team
        else p
        for p in state.players
    ]
    new_state = state.model_copy(update={"players": new_players})
    elapsed = elapsed_for_run((focus.x, focus.y), (tx, ty))
    new_state, moves = shift_after_run(state, new_state, attacking_team=focus_team,
                                       runner_id=focus_player_id, elapsed_s=elapsed)
    moves.insert(0, Move(focus_team, focus_player_id, (focus.x, focus.y), (tx, ty)))

    if before_marker and after_marker and before_marker.id != after_marker.id:
        pull_note = (
            f" This run pulls opponent #{after_marker.id} across to cover the new position "
            f"(previously #{before_marker.id} was the nearest defender to the old spot)."
        )
    elif after_marker:
        pull_note = f" Opponent #{after_marker.id} is now the nearest defender to this position."
    else:
        pull_note = ""

    question = (
        f"Player {focus_player_id} has just completed this off-ball run to "
        f"({tx:.1f}, {ty:.1f}): {action.label}.{pull_note}{_estimate_note(elapsed)} Given this new "
        f"position and its effect on the defensive shape, what should happen next?"
    )
    return Projection(state=new_state, focus_player_id=focus_player_id, question=question, moves=moves)
