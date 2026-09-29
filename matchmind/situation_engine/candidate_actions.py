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
_ON_BALL_DEFAULTS: list[tuple[str, str]] = [
    ("hold_and_recycle", "Hold the ball and recycle possession"),
    ("dribble_forward", "Dribble forward into the space ahead"),
]
_OFF_BALL_DEFAULTS: list[tuple[str, str]] = [
    ("hold_position", "Hold current position and stay available as an out-ball"),
]


def generate_candidate_actions(
    state: MatchState,
    feats: SituationFeatures,
    max_actions: int = 6,
) -> list[CandidateAction]:
    """Build 2-6 candidate actions for the focus player.

    Branches on ``feats.is_ball_carrier``: a player without the ball can't
    pass/dribble/shoot — generating those anyway is exactly the "off-ball
    player analysed as if they had the ball" bug this function must avoid.

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

    if _is_goalkeeper(focus):
        # Keepers get keeper options only. Before this branch the outfield
        # generators sent the home keeper (x=5) "into the half-space pocket
        # near (35, 50)" while #10 attacked — a real user-reported answer.
        for ca in _goalkeeper_actions(state, feats, focus):
            add(ca)
        if not feats.is_ball_carrier:
            for name, label in _OFF_BALL_DEFAULTS:
                add(CandidateAction(action=name, label=label, target_xy=(focus.x, focus.y), moves_ball=False, origin="default"))
    elif feats.is_ball_carrier:
        for ca in _on_ball_actions(state, feats, focus):
            add(ca)
        for name, label in _ON_BALL_DEFAULTS:
            add(CandidateAction(action=name, label=label, origin="default"))
    else:
        carrier = state.get_player(feats.ball_carrier_id) if feats.ball_carrier_id is not None else None
        if carrier is not None and carrier.team == focus.team:
            for ca in _off_ball_attacking_actions(feats, focus, carrier):
                add(ca)
        elif carrier is not None:
            for ca in _off_ball_defensive_actions(focus, carrier):
                add(ca)
        for name, label in _OFF_BALL_DEFAULTS:
            # target_xy = the player's OWN current spot, not None — without an
            # explicit target this would fall through to the string-matched
            # on-ball simulation (_simulate_hold), which wrongly teleports the
            # ball to the focus player. Pinning target_xy keeps it off-ball.
            add(CandidateAction(action=name, label=label, target_xy=(focus.x, focus.y), moves_ball=False, origin="default"))

    result = actions[:max_actions]
    logger.debug(
        "[candidate_actions] player=%s is_ball_carrier=%s -> %s",
        feats.focus_player_id,
        feats.is_ball_carrier,
        [a.action for a in result],
    )
    return result


def _clamp(x: float, y: float) -> tuple[float, float]:
    return max(0.0, min(PITCH_LENGTH, x)), max(0.0, min(PITCH_WIDTH, y))


# ── On-ball: focus player has the ball ────────────────────────────────────

def _on_ball_actions(state: MatchState, feats: SituationFeatures, focus) -> list[CandidateAction]:
    out: list[CandidateAction] = []

    # One pass_to_<id> per open passing lane (highest signal: progressive lanes first)
    for pid in feats.open_passing_lane_player_ids:
        tm = state.get_player(pid)
        if tm is None:
            continue
        out.append(
            CandidateAction(
                action=f"pass_to_{pid}",
                label=f"Pass to player {pid} at ({tm.x:.0f}, {tm.y:.0f})",
                target_xy=(tm.x, tm.y),
                target_player_id=pid,
                origin="passing_lane",
            )
        )

    if feats.half_space_occupied or feats.space_ahead > 8.0:
        out.append(CandidateAction(action="dribble_half_space", label="Dribble / carry into the half-space", origin="situation"))
    if feats.switch_play_viable:
        out.append(CandidateAction(action="switch_play", label="Switch play to the opposite flank", origin="situation"))
    if feats.third_man_opportunity:
        out.append(CandidateAction(action="third_man_combination", label="Third-man combination through a nearby teammate", origin="situation"))
    if feats.is_in_attacking_third and focus.x > 80.0:
        out.append(CandidateAction(action="shoot", label="Shoot at goal", target_xy=(104.0, 34.0), origin="situation"))
    if feats.overload_left:
        out.append(CandidateAction(action="exploit_left_overload", label="Attack the left-flank overload", origin="situation"))
    if feats.overload_right:
        out.append(CandidateAction(action="exploit_right_overload", label="Attack the right-flank overload", origin="situation"))
    return out


# ── Off-ball, own team in possession: movement to help the attack ─────────

def _off_ball_attacking_actions(feats: SituationFeatures, focus, carrier) -> list[CandidateAction]:
    out: list[CandidateAction] = []

    # Run in behind: sprint into the channel if there's room ahead
    if feats.space_ahead > 8.0:
        tx, ty = _clamp(focus.x + 12.0, focus.y)
        out.append(CandidateAction(
            action="run_in_behind",
            label=f"Sprint into the channel behind the defensive line toward ({tx:.0f}, {ty:.0f})",
            target_xy=(tx, ty), moves_ball=False, origin="situation",
        ))

    # Check to the ball: short drop to offer an out-ball for the carrier
    dx, dy = carrier.x - focus.x, carrier.y - focus.y
    dist = (dx**2 + dy**2) ** 0.5
    if dist > 1e-6:
        step = min(8.0, dist * 0.5)
        cx, cy = _clamp(focus.x + dx / dist * step, focus.y + dy / dist * step)
        out.append(CandidateAction(
            action="check_to_receive",
            label=f"Check towards the ball to offer a short passing option near ({cx:.0f}, {cy:.0f})",
            target_xy=(cx, cy), moves_ball=False, origin="situation",
        ))

    # Provide width: stretch the defense if currently narrow
    if abs(focus.y - PITCH_WIDTH / 2) < 18.0:
        wide_y = 6.0 if focus.y <= PITCH_WIDTH / 2 else PITCH_WIDTH - 6.0
        tx, ty = _clamp(focus.x, wide_y)
        out.append(CandidateAction(
            action="provide_width",
            label=f"Hold width near the touchline ({tx:.0f}, {ty:.0f}) to stretch the defense",
            target_xy=(tx, ty), moves_ball=False, origin="situation",
        ))

    # Drop into the half-space pocket if not already occupying one
    if not feats.half_space_occupied:
        pocket_y = 17.5 if focus.y < PITCH_WIDTH / 2 else 50.5
        px, py = _clamp(max(35.0, min(70.0, focus.x + 5.0)), pocket_y)
        out.append(CandidateAction(
            action="drop_into_pocket",
            label=f"Drop into the half-space pocket near ({px:.0f}, {py:.0f}) to open a passing lane",
            target_xy=(px, py), moves_ball=False, origin="situation",
        ))

    # Overlapping run past the ball carrier (decoy / creates a 2v1 out wide)
    if abs(carrier.x - focus.x) < 20.0:
        overlap_y = focus.y + (10.0 if focus.y >= PITCH_WIDTH / 2 else -10.0)
        tx, ty = _clamp(focus.x + 10.0, overlap_y)
        out.append(CandidateAction(
            action="overlapping_run",
            label=f"Make an overlapping run wide of the ball carrier toward ({tx:.0f}, {ty:.0f})",
            target_xy=(tx, ty), moves_ball=False, origin="situation",
        ))

    return out


# ── Goalkeeper ─────────────────────────────────────────────────────────────

# The state reaching here is oriented so the focus team attacks +x
# (situation_engine/orientation.py), so its own goal is always at x=0.
_OWN_GOAL = (0.0, PITCH_WIDTH / 2)
_OWN_BOX_DEPTH = 16.5


def _is_goalkeeper(player) -> bool:
    return (player.role or "").lower() in {"goalkeeper", "gk", "keeper"}


def _goalkeeper_actions(state: MatchState, feats: SituationFeatures, focus) -> list[CandidateAction]:
    """Keeper options. The team's aim is still the opponent's goal: with the ball
    the keeper starts attacks (distribution); while the team attacks, the keeper
    supports it by covering the space behind a high back line; without the ball
    the keeper protects the goal."""
    out: list[CandidateAction] = []

    if feats.is_ball_carrier:
        for ca in _on_ball_actions(state, feats, focus):
            if ca.action.startswith("pass_to_") or ca.action == "switch_play":
                out.append(ca)
        mates = [p for p in state.get_team_players(focus.team) if p.id != focus.id]
        if mates:
            far = max(mates, key=lambda p: p.x)
            out.append(CandidateAction(
                action="gk_long_distribution",
                label=f"Long distribution to the most advanced teammate (Player {far.id}) at ({far.x:.0f}, {far.y:.0f})",
                target_xy=(far.x, far.y), target_player_id=far.id, origin="situation",
            ))
        out.append(CandidateAction(action="hold_and_recycle", label="Keep the ball and reset the build-up", origin="default"))
        return out

    carrier = state.get_player(feats.ball_carrier_id) if feats.ball_carrier_id is not None else None
    if carrier is None or carrier.team == focus.team:
        # Own team attacking: sweeper-keeper behind the back line, ready for the
        # ball played in behind if possession is lost.
        outfield = [p for p in state.get_team_players(focus.team) if p.id != focus.id and not _is_goalkeeper(p)]
        back_line = min((p.x for p in outfield), default=30.0)
        sx = max(focus.x, min(back_line - 10.0, 25.0))
        sy = PITCH_WIDTH / 2 + (state.ball.y - PITCH_WIDTH / 2) * 0.3
        sx, sy = _clamp(sx, sy)
        out.append(CandidateAction(
            action="gk_sweeper_position",
            label=(f"Step up to ({sx:.0f}, {sy:.0f}), just behind our own back line (x≈{back_line:.0f}), "
                   "as a sweeper-keeper: cover the space between our defenders and our goal, ready to "
                   "clear a long ball if we lose possession while attacking"),
            target_xy=(sx, sy), moves_ball=False, origin="situation",
        ))
        if carrier is not None and carrier.x < PITCH_LENGTH / 2:
            bx, by = _clamp(max(focus.x, 8.0), focus.y + (carrier.y - focus.y) * 0.3)
            out.append(CandidateAction(
                action="gk_offer_back_pass",
                label=f"Offer a safe back-pass angle near ({bx:.0f}, {by:.0f}) to help the team keep the ball",
                target_xy=(bx, by), moves_ball=False, origin="situation",
            ))
        return out

    # Opponent has the ball: stand between the ball and the centre of the goal.
    gx, gy = _OWN_GOAL
    dx, dy = carrier.x - gx, carrier.y - gy
    dist = max((dx**2 + dy**2) ** 0.5, 1e-6)
    advance = min(6.0, dist * 0.15)
    tx, ty = _clamp(max(gx + dx / dist * advance, 0.5), gy + dy / dist * advance)
    out.append(CandidateAction(
        action="gk_set_position",
        label=f"Set position on the line between the ball and the centre of the goal at ({tx:.0f}, {ty:.0f})",
        target_xy=(tx, ty), moves_ball=False, origin="situation",
    ))
    if dist < 25.0:
        step = min(dist * 0.35, _OWN_BOX_DEPTH)
        cx, cy = _clamp(gx + dx / dist * step, gy + dy / dist * step)
        out.append(CandidateAction(
            action="gk_close_down",
            label=f"Come off the line toward ({cx:.0f}, {cy:.0f}) to narrow the shooting angle",
            target_xy=(cx, cy), moves_ball=False, origin="situation",
        ))
    return out


# ── Off-ball, opponent in possession: defensive positioning ───────────────

def _off_ball_defensive_actions(focus, carrier) -> list[CandidateAction]:
    out: list[CandidateAction] = []
    dist = ((focus.x - carrier.x) ** 2 + (focus.y - carrier.y) ** 2) ** 0.5
    if dist < 20.0:
        out.append(CandidateAction(
            action="press_ball_carrier",
            label=f"Press the ball carrier (Player {carrier.id}) to force a mistake",
            target_xy=(carrier.x, carrier.y), moves_ball=False, origin="situation",
        ))
    out.append(CandidateAction(
        action="hold_defensive_shape",
        label="Hold defensive shape and cover the nearest passing lane",
        moves_ball=False, origin="default",
    ))
    return out
