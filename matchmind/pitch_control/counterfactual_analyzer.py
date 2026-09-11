"""
Counterfactual Action Analyzer.

Uses pitch control to provide quantitative grounding for counterfactual reasoning.

Instead of: "Action A is better than B" (LLM hallucination)
We compute: "Action A creates +18.3% more controlled area than Action B" (fact)

This is what makes the counterfactual reasoning credible in interviews.

Usage:
    analyzer = CounterfactualAnalyzer(state, focus_player_id, focus_team)
    results = analyzer.analyze(candidate_actions=["pass_to_9", "dribble_half_space", "switch_play"])
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from matchmind.pitch_control.spearman_model import (
    compute_controlled_area,
    generate_pitch_control_for_match_state,
    compute_control_at_point,
)
from matchmind.schema.match_state import (
    BallState,
    CandidateAction,
    MatchState,
    PlayerState,
)

logger = logging.getLogger(__name__)


@dataclass
class ActionAnalysis:
    """Result of analysing one candidate action."""

    action_name: str
    description: str
    simulated_ball_x: float | None
    simulated_ball_y: float | None
    pitch_control_after: float | None    # controlled area fraction [0,1]
    pitch_control_delta: float | None    # change vs baseline
    pitch_control_at_target: float | None  # control at target point specifically
    feasibility_score: float             # 0–1 (geometric feasibility)
    notes: str = ""


class CounterfactualAnalyzer:
    """
    Computes pitch-control-backed counterfactual analysis for candidate actions.

    For each candidate action:
    1. Simulate a simplified resulting MatchState (ball moved, player moved)
    2. Compute pitch control on the resulting state
    3. Return delta vs baseline (current state)

    Limitations acknowledged: simulation is simplified (no velocity changes for
    all players) — for MVP this is acceptable; full simulation would require
    physics-based trajectory modelling.
    """

    def __init__(
        self,
        state: MatchState,
        focus_player_id: int,
        focus_team: str,
    ) -> None:
        self.state = state
        self.focus_player_id = focus_player_id
        self.focus_team = focus_team
        self._baseline_control: float | None = None

    def baseline_controlled_area(self) -> float:
        """Controlled area fraction in the current state (before any action)."""
        if self._baseline_control is None:
            _, _, ppcf = generate_pitch_control_for_match_state(
                self.state, self.focus_team
            )
            self._baseline_control = compute_controlled_area(ppcf)
        return self._baseline_control

    def analyze(
        self,
        candidate_actions: list[str] | list[CandidateAction],
    ) -> list[ActionAnalysis]:
        """
        Analyse all candidate actions.

        Accepts either plain strings ("pass_to_9", "switch_play", ...) or
        CandidateAction objects from situation_engine.candidate_actions
        (preferred — carries an explicit target_xy so the simulation is exact).
        """
        baseline = self.baseline_controlled_area()
        results = []

        for raw in candidate_actions:
            action_str = raw.action if isinstance(raw, CandidateAction) else str(raw)
            explicit_target = raw.target_xy if isinstance(raw, CandidateAction) else None
            moves_ball = raw.moves_ball if isinstance(raw, CandidateAction) else True
            try:
                analysis = self._analyze_single(action_str, baseline, explicit_target, moves_ball)
                results.append(analysis)
            except Exception as exc:
                logger.warning(f"Could not analyze action '{action_str}': {exc}")
                results.append(
                    ActionAnalysis(
                        action_name=action_str,
                        description=action_str,
                        simulated_ball_x=None,
                        simulated_ball_y=None,
                        pitch_control_after=None,
                        pitch_control_delta=None,
                        pitch_control_at_target=None,
                        feasibility_score=0.5,
                        notes=f"Analysis failed: {exc}",
                    )
                )

        # Sort: best (highest delta) first
        results.sort(key=lambda r: r.pitch_control_delta or -999.0, reverse=True)
        return results

    # ──────────────────────────────────────────────────────────
    # Private: per-action simulation and analysis
    # ──────────────────────────────────────────────────────────

    def _analyze_single(
        self,
        action_str: str,
        baseline: float,
        explicit_target: tuple[float, float] | None = None,
        moves_ball: bool = True,
    ) -> ActionAnalysis:
        """Simulate one action and compute its pitch control effect."""
        focus = self.state.get_player(self.focus_player_id)
        if focus is None:
            raise ValueError(f"Focus player {self.focus_player_id} not found")

        teammates = [p for p in self.state.get_team_players(self.focus_team)
                     if p.id != self.focus_player_id]
        opponents = self.state.get_opponents(self.focus_team)

        action_lower = action_str.lower()

        # ── Resolve action → simulated target position ─────
        result: dict[str, Any]
        if explicit_target is not None:
            tx, ty = explicit_target
            if moves_ball:
                # On-ball: the ball travels to the target (pass/dribble/shot/switch).
                result = {
                    "target_x": tx,
                    "target_y": ty,
                    "sim_ball_x": tx,
                    "sim_ball_y": ty,
                    "description": f"{action_str} -> ({tx:.1f}, {ty:.1f})",
                    "feasibility": 0.75,
                }
                if "dribble" in action_lower or "carry" in action_lower:
                    result["focus_new_x"] = tx
                    result["focus_new_y"] = ty
            else:
                # Off-ball movement: the focus player runs to (tx,ty) but does NOT
                # have the ball — it stays with the real carrier. "target_x/y" is
                # still the point we score control at (is this run into good space?).
                result = {
                    "target_x": tx,
                    "target_y": ty,
                    "sim_ball_x": self.state.ball.x,
                    "sim_ball_y": self.state.ball.y,
                    "focus_new_x": tx,
                    "focus_new_y": ty,
                    "description": f"{action_str} (off-ball run) -> ({tx:.1f}, {ty:.1f}), ball stays with carrier",
                    "feasibility": 0.7,
                }
        elif "pass_to" in action_lower or "pass" in action_lower:
            result = self._simulate_pass(action_str, focus, teammates, opponents)
        elif "dribble" in action_lower or "carry" in action_lower:
            result = self._simulate_dribble(action_str, focus, opponents)
        elif "switch" in action_lower:
            result = self._simulate_switch(action_str, focus, teammates, opponents)
        elif "shoot" in action_lower or "shot" in action_lower:
            result = self._simulate_shot(action_str, focus, opponents)
        elif "hold" in action_lower or "recycle" in action_lower:
            result = self._simulate_hold(action_str, focus)
        else:
            result = self._simulate_generic(action_str, focus)

        # ── Compute pitch control after simulated action ────
        # sim_ball_x/y: where the BALL ends up (defaults to target_x/y — the
        # historical behaviour for pass/dribble/shot/switch). Off-ball actions
        # set this explicitly to the ball's real position instead.
        sim_ball_x = result.get("sim_ball_x", result.get("target_x"))
        sim_ball_y = result.get("sim_ball_y", result.get("target_y"))

        if sim_ball_x is not None:
            sim_state = self._build_simulated_state(
                sim_ball_x,
                sim_ball_y,
                result.get("focus_new_x", focus.x),
                result.get("focus_new_y", focus.y),
            )
            _, _, ppcf = generate_pitch_control_for_match_state(sim_state, self.focus_team)
            pc_after = compute_controlled_area(ppcf)
            pc_target = compute_control_at_point(
                sim_state,
                result["target_x"],
                result["target_y"],
                self.focus_team,
            )
            delta = pc_after - baseline
        else:
            pc_after = None
            pc_target = None
            delta = None

        return ActionAnalysis(
            action_name=action_str,
            description=result["description"],
            # Where the ball actually ends up in the simulation — NOT the same
            # as target_x/y for an off-ball action (that's the run destination).
            simulated_ball_x=sim_ball_x,
            simulated_ball_y=sim_ball_y,
            pitch_control_after=pc_after,
            pitch_control_delta=delta,
            pitch_control_at_target=pc_target,
            feasibility_score=result.get("feasibility", 0.7),
            notes=result.get("notes", ""),
        )

    def _simulate_pass(self, action_str: str, focus: PlayerState,
                       teammates: list[PlayerState], opponents: list[PlayerState]) -> dict:
        """Simulate passing to a teammate or general forward area."""
        # Try to extract player ID from action string (e.g., "pass_to_9")
        target_player = None
        for part in action_str.split("_"):
            try:
                pid = int(part)
                target_player = next((t for t in teammates if t.id == pid), None)
                break
            except ValueError:
                pass

        if target_player:
            tx, ty = target_player.x, target_player.y
            desc = f"Pass to player {target_player.id} at ({tx:.1f}, {ty:.1f})"
        else:
            # Progressive pass: aim forward 15m
            tx = min(focus.x + 15.0, 105.0)
            ty = focus.y
            desc = f"Progressive forward pass to ({tx:.1f}, {ty:.1f})"

        return {"target_x": tx, "target_y": ty, "description": desc, "feasibility": 0.8}

    def _simulate_dribble(self, action_str: str, focus: PlayerState,
                          opponents: list[PlayerState]) -> dict:
        """Simulate dribbling forward into space."""
        # Move focus player 10m forward
        new_x = min(focus.x + 10.0, 105.0)
        new_y = focus.y
        # Check if half-space is implied
        if "half_space" in action_str or "half-space" in action_str:
            # Move into the nearest half-space
            if focus.y > 34.0:
                new_y = min(focus.y + 8.0, 58.0)
            else:
                new_y = max(focus.y - 8.0, 10.0)

        desc = f"Dribble to ({new_x:.1f}, {new_y:.1f})"
        return {
            "target_x": new_x, "target_y": new_y,
            "focus_new_x": new_x, "focus_new_y": new_y,
            "description": desc, "feasibility": 0.65,
        }

    def _simulate_switch(self, action_str: str, focus: PlayerState,
                         teammates: list[PlayerState], opponents: list[PlayerState]) -> dict:
        """Simulate switching play to the opposite flank."""
        # Find the widest player on the opposite flank
        opp_flank_y = 68.0 - focus.y
        wide_tm = min(
            [t for t in teammates if abs(t.y - opp_flank_y) < 25.0],
            key=lambda t: abs(t.y - opp_flank_y),
            default=None,
        )
        if wide_tm:
            tx, ty = wide_tm.x, wide_tm.y
            desc = f"Switch play to player {wide_tm.id} on opposite flank"
        else:
            tx, ty = focus.x, opp_flank_y
            desc = f"Switch play to opposite flank ({tx:.1f}, {ty:.1f})"

        return {"target_x": tx, "target_y": ty, "description": desc, "feasibility": 0.6}

    def _simulate_shot(self, action_str: str, focus: PlayerState,
                       opponents: list[PlayerState]) -> dict:
        """Simulate a shot — target is the goal (x=105, y=34)."""
        return {
            "target_x": 104.0, "target_y": 34.0,
            "description": "Shoot at goal",
            "feasibility": 0.9 if focus.x > 80.0 else 0.4,
        }

    def _simulate_hold(self, action_str: str, focus: PlayerState) -> dict:
        """Hold and recycle — ball stays near current position."""
        return {
            "target_x": focus.x, "target_y": focus.y,
            "description": "Hold ball and recycle possession",
            "feasibility": 0.9,
        }

    def _simulate_generic(self, action_str: str, focus: PlayerState) -> dict:
        """Generic forward movement for unrecognised actions."""
        return {
            "target_x": min(focus.x + 8.0, 105.0), "target_y": focus.y,
            "description": action_str,
            "feasibility": 0.5,
            "notes": "Generic simulation (unrecognised action type)",
        }

    def _build_simulated_state(
        self,
        ball_x: float,
        ball_y: float,
        focus_new_x: float | None = None,
        focus_new_y: float | None = None,
    ) -> MatchState:
        """
        Build a simplified simulated MatchState after an action.

        Only the ball position (and optionally focus player position) changes.
        All other players remain in their current positions.
        This is a simplified simulation — opponent reactions not modelled.
        """
        new_players = []
        for p in self.state.players:
            if p.id == self.focus_player_id and focus_new_x is not None:
                new_players.append(
                    PlayerState(
                        id=p.id, team=p.team, role=p.role,
                        x=focus_new_x, y=focus_new_y or p.y,
                        vx=0.0, vy=0.0,
                    )
                )
            else:
                new_players.append(p)

        return MatchState(
            match_id=self.state.match_id,
            timestamp=self.state.timestamp,
            score_home=self.state.score_home,
            score_away=self.state.score_away,
            possession=self.state.possession,
            possession_pct=self.state.possession_pct,
            phase_of_play=self.state.phase_of_play,
            ball=BallState(x=ball_x, y=ball_y),
            players=new_players,
            has_velocity=self.state.has_velocity,
            source=self.state.source,
        )

    def format_for_prompt(self, results: list[ActionAnalysis]) -> str:
        """Format analysis results as a text block for LLM prompt."""
        lines = ["=== Counterfactual Action Analysis (Pitch Control Model) ==="]
        baseline = self.baseline_controlled_area()
        lines.append(f"Baseline controlled area: {baseline*100:.1f}%\n")

        for i, r in enumerate(results):
            rank = "RECOMMENDED" if i == 0 else f"Alternative {i}"
            lines.append(f"[{rank}] {r.action_name}")
            lines.append(f"  Description: {r.description}")
            if r.pitch_control_after is not None:
                lines.append(f"  Controlled area after: {r.pitch_control_after*100:.1f}%")
                if r.pitch_control_delta is not None:
                    sign = "+" if r.pitch_control_delta > 0 else ""
                    lines.append(f"  Delta vs baseline: {sign}{r.pitch_control_delta*100:.1f}%")
            if r.pitch_control_at_target is not None:
                lines.append(f"  Control at target point: {r.pitch_control_at_target*100:.1f}%")
            lines.append(f"  Feasibility: {r.feasibility_score:.0%}")
            if r.notes:
                lines.append(f"  Notes: {r.notes}")
            lines.append("")

        return "\n".join(lines)
