"""
Counterfactual Analysis Node.

Uses pitch control model to compute quantitative action alternatives.
Adds delta_pitch_control to each candidate action — the key quantitative evidence.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict

from matchmind.agent.state import AgentState
from matchmind.pitch_control.counterfactual_analyzer import CounterfactualAnalyzer
from matchmind.situation_engine.candidate_actions import generate_candidate_actions

logger = logging.getLogger(__name__)


def counterfactual_node(state: AgentState) -> AgentState:
    """
    Node: Compute counterfactual action analysis with pitch control.

    Candidate actions come from situation_engine.candidate_actions
    (single source of truth, shared with the reasoner). Produces a ranked
    list with delta_pitch_control for each action.

    Input:  state.match_state, state.focus_player_id, state.situation_features
    Output: state.candidate_actions, state.counterfactual_analysis,
            state.counterfactual_text, state.pitch_control_baseline
    """
    t0 = time.time()

    match_state = state["match_state"]
    focus_id = state["focus_player_id"]
    features = state["situation_features"]

    focus_player = match_state.get_player(focus_id)
    if focus_player is None:
        logger.warning(f"[counterfactual] Player {focus_id} not found — skipping")
        return {**state, "candidate_actions": [], "counterfactual_analysis": [], "counterfactual_text": ""}

    focus_team = focus_player.team

    # Single source of truth for candidate actions (also fed to the reasoner)
    candidates = generate_candidate_actions(match_state, features)
    candidate_dicts = [c.model_dump() for c in candidates]

    # Run pitch control analysis
    try:
        analyzer = CounterfactualAnalyzer(match_state, focus_id, focus_team)
        baseline = analyzer.baseline_controlled_area()
        results = analyzer.analyze(candidates)
        cf_text = analyzer.format_for_prompt(results[:4])  # top 4 for prompt

        # Convert dataclasses to dicts for JSON serialisation in state
        results_dicts = [
            {
                "action_name": r.action_name,
                "description": r.description,
                "pitch_control_after": r.pitch_control_after,
                "pitch_control_delta": r.pitch_control_delta,
                "pitch_control_at_target": r.pitch_control_at_target,
                "feasibility_score": r.feasibility_score,
                "notes": r.notes,
            }
            for r in results
        ]

    except Exception as exc:
        logger.warning(f"[counterfactual] Pitch control failed: {exc} — continuing without it")
        baseline = None
        results_dicts = []
        cf_text = "(Pitch control analysis unavailable)"

    elapsed = (time.time() - t0) * 1000
    logger.debug(f"[counterfactual] {elapsed:.1f}ms | {len(results_dicts)} actions evaluated")

    trace = state.get("trace") or []
    trace.append({
        "node": "counterfactual",
        "latency_ms": round(elapsed, 1),
        "n_actions": len(results_dicts),
        "baseline_control": round(baseline or 0.0, 4),
    })

    return {
        **state,
        "candidate_actions": candidate_dicts,
        "pitch_control_baseline": baseline,
        "counterfactual_analysis": results_dicts,
        "counterfactual_text": cf_text,
        "trace": trace,
    }


def assemble_evidence_node(state: AgentState) -> AgentState:
    """
    Node: Assemble all evidence into structured buckets.

    Three evidence types (from revise.md):
    - RAG: retrieved tactical concepts (static knowledge)
    - Stats: computed features from situation engine + pitch control (quantitative)
    - Context: dynamic match state (score, minute, phase)
    """
    features = state["situation_features"]
    match_state = state["match_state"]
    retrieved = state.get("retrieved_tactics") or []
    cf_analysis = state.get("counterfactual_analysis") or []

    # ── RAG Evidence ──────────────────────────────────────────
    rag_evidence = [
        {"id": r["concept_id"], "source": "rag", "content": f"{r['title']}: {r['content'][:200]}..."}
        for r in retrieved
    ]

    # ── Stats Evidence ────────────────────────────────────────
    possession_fact = (
        "Player HAS THE BALL"
        if features.is_ball_carrier
        else f"Player is OFF THE BALL (Player {features.ball_carrier_id} has it)"
    )
    stats_facts = [
        possession_fact,
        f"Distance to ball: {features.distance_to_ball:.1f}m",
        f"Nearest opponent: {features.nearest_opponent_distance:.1f}m",
        f"Space ahead: {features.space_ahead:.1f}m",
        f"Local advantage: {features.local_numerical_advantage:+d} players",
        f"Open passing lanes: {len(features.open_passing_lane_player_ids)} (IDs: {features.open_passing_lane_player_ids})",
        f"Progressive passes available: {features.progressive_passes_available}",
        f"Half-space occupied: {features.half_space_occupied}",
        f"Third-man opportunity: {features.third_man_opportunity}",
        f"Defensive line at x={features.defensive_line_height:.1f}m",
    ]
    if cf_analysis:
        best = cf_analysis[0]
        if best.get("pitch_control_delta") is not None:
            stats_facts.append(
                f"Best action '{best['action_name']}' creates "
                f"{best['pitch_control_delta']*100:+.1f}% change in controlled area"
            )
    stats_evidence = [{"id": "computed_features", "source": "stats", "content": " | ".join(stats_facts)}]

    # ── Context Evidence ──────────────────────────────────────
    score_diff = match_state.score_diff_from_home
    urgency = "high urgency" if (
        (score_diff < 0 and match_state.minute > 70) or
        (score_diff > 0 and match_state.minute > 80)
    ) else "normal game state"

    context_facts = [
        f"Match minute: {match_state.minute}'",
        f"Score: Home {match_state.score_home}–{match_state.score_away} Away",
        f"Phase of play: {match_state.phase_of_play or 'open play'}",
        f"Possession: {match_state.possession or 'unknown'}",
        f"Game state: {urgency}",
    ]
    context_evidence = [{"id": "match_context", "source": "context", "content": " | ".join(context_facts)}]

    evidence_set = {
        "rag": rag_evidence,
        "stats": stats_evidence,
        "context": context_evidence,
    }

    return {**state, "evidence_set": evidence_set}
