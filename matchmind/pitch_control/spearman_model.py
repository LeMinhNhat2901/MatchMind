"""
Spearman Pitch Control Model — adapted from LaurieOnTracking.

Original implementation by Laurie Shaw (@EightyFivePoint), based on:
"Off the Ball Scoring Opportunities" — William Spearman (2018)
http://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf

This module adapts Metrica_PitchControl.py to accept MatchState instead of
Metrica DataFrames — the only change needed to integrate into our pipeline.

Pitch control at position (x,y) = probability that the home/focus team
gains possession if the ball is moved to (x,y) right now.

Key insight for counterfactual reasoning:
- Compute pitch_control BEFORE action
- Simulate action → compute pitch_control AFTER action
- delta = change in controlled area → quantitative evidence for "why action A > B"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from matchmind.schema.match_state import MatchState, PlayerState

logger = logging.getLogger(__name__)

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Grid resolution for full pitch computation
GRID_COLS = 50   # number of x cells
GRID_ROWS = 32   # number of y cells

# Match ids we've already warned about running degraded (no-velocity) pitch control.
_DEGRADED_WARNED: set[str] = set()


# ──────────────────────────────────────────────────────────────────────────────
# Model parameters (Spearman 2018 defaults)
# ──────────────────────────────────────────────────────────────────────────────

def default_model_params() -> dict:
    """
    Default model parameters from Spearman (2018).
    These are the same as in LaurieOnTracking/Metrica_PitchControl.py.
    """
    params = {}
    # Time to control a ball — inverse of player max speed ratio
    params["max_player_speed"] = 5.0        # m/s
    params["average_ball_speed"] = 15.0     # m/s
    params["reaction_time"] = 0.7           # seconds
    params["tti_sigma"] = 0.45              # standard deviation for time-to-intercept
    params["kappa_def"] = 1.0              # weighting for defending team
    params["lambda_att"] = 3.99            # rate parameter (attacker influence)
    params["lambda_def"] = 3.99 * params["kappa_def"]
    params["average_ball_speed"] = 15.0    # m/s (ball speed)
    params["int_dt"] = 0.04               # time step for integration
    params["max_int_time"] = 10           # maximum time (s) for TTI calculation
    params["model_converge_tol"] = 0.01   # convergence tolerance
    return params


# ──────────────────────────────────────────────────────────────────────────────
# Player class (adapted from LaurieOnTracking)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PCPlayer:
    """Player object for pitch control computation."""

    player_id: int
    team: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    params: dict = field(default_factory=default_model_params)

    # Cached time-to-intercept
    _PPCF: np.ndarray | None = field(default=None, repr=False, init=False)
    _tti_grid: np.ndarray | None = field(default=None, repr=False, init=False)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def simple_time_to_intercept(self, r_final: np.ndarray) -> float:
        """
        Estimate time for this player to reach position r_final.
        Uses reaction time + travel time at max speed.
        """
        r = r_final - self.position
        dist = float(np.linalg.norm(r))
        reaction = self.params["reaction_time"]
        # Time = reaction + distance / max_speed (minus velocity component toward target)
        v_proj = float(np.dot(self.velocity, r / (dist + 1e-8)))
        t = reaction + max(0.0, dist - v_proj * reaction) / self.params["max_player_speed"]
        return t

    def probability_intercept_ball(self, T: float) -> float:
        """
        Probability this player intercepts a ball arriving at time T.
        Uses logistic function centred on player's time-to-intercept.
        """
        tti = self.simple_time_to_intercept(self.position)  # placeholder
        f = 1.0 / (1.0 + math.exp(-math.pi / math.sqrt(3.0) / self.params["tti_sigma"] * (T - tti)))
        return f


def _players_from_match_state(state: MatchState, team: str, params: dict) -> list[PCPlayer]:
    """Convert MatchState players for one team → list[PCPlayer]."""
    return [
        PCPlayer(
            player_id=p.id,
            team=p.team,
            x=p.x,
            y=p.y,
            vx=p.vx,
            vy=p.vy,
            params=params,
        )
        for p in state.players
        if p.team == team
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Core pitch control computation
# ──────────────────────────────────────────────────────────────────────────────

def calculate_pitch_control_at_target(
    target_position: np.ndarray,
    attacking_players: list[PCPlayer],
    defending_players: list[PCPlayer],
    ball_start_pos: np.ndarray | None = None,
    params: dict | None = None,
) -> tuple[float, float]:
    """
    Calculate pitch control probability at a single target position.

    Returns:
        (attacking_prob, defending_prob) — should sum to ~1.0.

    Adapted from LaurieOnTracking/Metrica_PitchControl.py.
    """
    if params is None:
        params = default_model_params()

    if ball_start_pos is None:
        ball_start_pos = target_position

    # Ball travel time to target
    ball_dist = float(np.linalg.norm(target_position - ball_start_pos))
    ball_travel_time = ball_dist / params["average_ball_speed"]

    # Time-to-intercept per player — computed ONCE (not inside the integration loop)
    tau_att = np.array(
        [p.simple_time_to_intercept(target_position) for p in attacking_players],
        dtype=float,
    ) if attacking_players else np.array([999.0])
    tau_def = np.array(
        [p.simple_time_to_intercept(target_position) for p in defending_players],
        dtype=float,
    ) if defending_players else np.array([999.0])

    # If neither team can realistically reach: 50/50
    if tau_att.min() >= params["max_int_time"] and tau_def.min() >= params["max_int_time"]:
        return 0.5, 0.5

    dt = params["int_dt"]
    lam_a = params["lambda_att"]
    lam_d = params["lambda_def"]

    # Time grid (vectorised — no Python loop over T)
    T_array = np.arange(
        max(0.0, ball_travel_time - dt),
        ball_travel_time + params["max_int_time"],
        dt,
    )
    if T_array.size == 0:
        return 0.5, 0.5

    # Instantaneous arrival rate for each team at every time step: (nT,)
    diff_a = T_array[None, :] - tau_att[:, None]        # (n_att, nT)
    diff_d = T_array[None, :] - tau_def[:, None]        # (n_def, nT)
    rate_a = np.where(diff_a >= 0.0, lam_a * np.exp(-lam_a * diff_a), 0.0).sum(axis=0)
    rate_d = np.where(diff_d >= 0.0, lam_d * np.exp(-lam_d * diff_d), 0.0).sum(axis=0)

    # Coupled Euler integration solved in closed form:
    #   remaining(t) = prod_{s<t} (1 - (rate_a+rate_d)*dt)
    #   PPCF_x = sum_t remaining_before(t) * rate_x(t) * dt
    total_rate = np.clip((rate_a + rate_d) * dt, 0.0, 1.0)
    surv = np.cumprod(1.0 - total_rate)
    rem_before = np.concatenate(([1.0], surv[:-1]))

    PPCFatt = float(np.sum(rem_before * rate_a * dt))
    PPCFdef = float(np.sum(rem_before * rate_d * dt))

    PPCFatt = max(0.0, min(1.0, PPCFatt))
    PPCFdef = max(0.0, min(1.0, PPCFdef))

    return PPCFatt, PPCFdef


def generate_pitch_control_for_match_state(
    state: MatchState,
    focus_team: str,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pitch control surface over the whole pitch for a MatchState.

    Returns:
        (xgrid, ygrid, PPCFatt) where PPCFatt[i,j] = prob focus team controls (x,y).

    This is the main entry point for pitch control computation.
    """
    if params is None:
        params = default_model_params()

    if not state.has_velocity and state.match_id not in _DEGRADED_WARNED:
        _DEGRADED_WARNED.add(state.match_id)
        logger.warning(
            "Pitch control on '%s' data without velocities (has_velocity=False) — "
            "running degraded (static-player) approximation for match %s.",
            state.source,
            state.match_id,
        )

    opp_team = "away" if focus_team == "home" else "home"

    att_players = _players_from_match_state(state, focus_team, params)
    def_players = _players_from_match_state(state, opp_team, params)

    ball_pos = np.array([state.ball.x, state.ball.y])

    # Build grid
    xgrid = np.linspace(0, PITCH_LENGTH, GRID_COLS)
    ygrid = np.linspace(0, PITCH_WIDTH, GRID_ROWS)
    PPCFatt = np.zeros((GRID_ROWS, GRID_COLS))

    for i, x in enumerate(xgrid):
        for j, y in enumerate(ygrid):
            target = np.array([x, y])
            att_prob, _ = calculate_pitch_control_at_target(
                target_position=target,
                attacking_players=att_players,
                defending_players=def_players,
                ball_start_pos=ball_pos,
                params=params,
            )
            PPCFatt[j, i] = att_prob

    logger.debug(f"Pitch control computed: mean={PPCFatt.mean():.3f}")
    return xgrid, ygrid, PPCFatt


def compute_control_at_point(
    state: MatchState,
    target_x: float,
    target_y: float,
    focus_team: str,
    params: dict | None = None,
) -> float:
    """
    Compute pitch control probability for focus team at a single (x, y) point.

    Fast single-point query — no full grid needed.
    """
    if params is None:
        params = default_model_params()

    opp_team = "away" if focus_team == "home" else "home"
    att_players = _players_from_match_state(state, focus_team, params)
    def_players = _players_from_match_state(state, opp_team, params)

    target = np.array([target_x, target_y])
    ball_pos = np.array([state.ball.x, state.ball.y])

    att_prob, _ = calculate_pitch_control_at_target(
        target_position=target,
        attacking_players=att_players,
        defending_players=def_players,
        ball_start_pos=ball_pos,
        params=params,
    )
    return att_prob


def compute_controlled_area(
    PPCFatt: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Compute the fraction of pitch area controlled by the attacking team.

    Returns a value in [0, 1].
    """
    return float(np.mean(PPCFatt >= threshold))
