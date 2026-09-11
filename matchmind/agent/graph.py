"""
LangGraph Tactical Agent Graph.

Defines the full agent StateGraph with real conditional branching:

    START
      ↓
    encode_situation        (pure computation)
      ↓
    render_pitch            (matplotlib → PNG)
      ↓
    retrieve_tactics        (ChromaDB RAG)
      ↓
    [check_retrieval_quality]  ← CONDITIONAL EDGE #1
      │
      ├── confidence < threshold → expand_query → retrieve_tactics (loop, max 2)
      │
      └── confidence ok → counterfactual_analysis (pitch control)
                               ↓
                          assemble_evidence
                               ↓
                          generate_reasoning  (LLM call)
                               ↓
                          validate_output
                               ↓
                    [check_validation]  ← CONDITIONAL EDGE #2
                               │
                    ├── errors → increment_retry → generate_reasoning (loop, max 2)
                    │
                    └── valid → END
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from matchmind.agent.state import AgentState
from matchmind.agent.nodes.encode_situation_node import encode_situation_node
from matchmind.agent.nodes.render_pitch_node import render_pitch_node
from matchmind.agent.nodes.retrieve_node import (
    retrieve_node,
    build_expanded_query,
    check_retrieval_quality,
)
from matchmind.agent.nodes.counterfactual_node import (
    counterfactual_node,
    assemble_evidence_node,
)
from matchmind.agent.nodes.reasoner_node import reasoner_node
from matchmind.agent.nodes.validator_node import (
    validator_node,
    check_validation,
    increment_generation_retry,
)
from matchmind.schema.match_state import MatchState, TacticalAdvice

logger = logging.getLogger(__name__)

# Bump applied to the OTHER team's colliding id so downstream `get_player(id)`
# calls (which don't all thread `team` through) resolve unambiguously.
_DISAMBIGUATION_OFFSET = 1_000_000


def _disambiguate_by_team(state: MatchState, focus_player_id: int, focus_team: str) -> MatchState:
    """
    Resolve a shirt-number collision (both teams have a player with this id)
    by relabelling the OTHER team's colliding player(s) to a synthetic id.

    `focus_player_id` itself is left untouched, so every existing "Player N"
    label, arrow, and candidate action stays correct for the player the
    caller actually meant — only the id that would otherwise collide moves.
    """
    changed = False
    new_players = []
    for p in state.players:
        if p.id == focus_player_id and p.team != focus_team:
            new_players.append(p.model_copy(update={"id": p.id + _DISAMBIGUATION_OFFSET}))
            changed = True
        else:
            new_players.append(p)
    if not changed:
        return state
    logger.info(
        f"Disambiguated player id {focus_player_id}: kept for team={focus_team}, "
        f"relabelled the other team's same-numbered player."
    )
    return state.model_copy(update={"players": new_players})


def build_tactical_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Returns a compiled graph ready for invocation.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────
    builder.add_node("encode_situation", encode_situation_node)
    builder.add_node("render_pitch", render_pitch_node)
    builder.add_node("retrieve_tactics", retrieve_node)
    builder.add_node("expand_query", build_expanded_query)
    builder.add_node("counterfactual_analysis", counterfactual_node)
    builder.add_node("assemble_evidence", assemble_evidence_node)
    builder.add_node("generate_reasoning", reasoner_node)
    builder.add_node("validate_output", validator_node)
    builder.add_node("increment_retry", increment_generation_retry)

    # ── Linear edges ───────────────────────────────────────
    builder.set_entry_point("encode_situation")
    builder.add_edge("encode_situation", "render_pitch")
    builder.add_edge("render_pitch", "retrieve_tactics")

    # ── Conditional edge #1: retrieval quality check ────────
    builder.add_conditional_edges(
        "retrieve_tactics",
        check_retrieval_quality,
        {
            "re_retrieve": "expand_query",
            "assemble_evidence": "counterfactual_analysis",
        },
    )
    builder.add_edge("expand_query", "retrieve_tactics")  # loop back

    # ── Linear edges after retrieval ───────────────────────
    builder.add_edge("counterfactual_analysis", "assemble_evidence")
    builder.add_edge("assemble_evidence", "generate_reasoning")
    builder.add_edge("generate_reasoning", "validate_output")

    # ── Conditional edge #2: validation check ──────────────
    builder.add_conditional_edges(
        "validate_output",
        check_validation,
        {
            "regenerate": "increment_retry",
            "end": END,
        },
    )
    builder.add_edge("increment_retry", "generate_reasoning")  # loop back

    return builder.compile()


# ──────────────────────────────────────────────────────────────────────────────
# High-level runner function
# ──────────────────────────────────────────────────────────────────────────────

class TacticalAgent:
    """
    High-level interface for the MatchMind tactical agent.

    Usage:
        agent = TacticalAgent()
        advice = agent.analyze(
            match_state=state,
            focus_player_id=14,
            question="What should player 14 do?",
        )
    """

    def __init__(self) -> None:
        # ── CRITICAL ORDER: load sentence-transformers / torch BEFORE matplotlib.
        # On Windows + Python 3.13, importing matplotlib first corrupts torch's
        # native DLL init and the next model load hard-crashes the process (no
        # traceback). The renderers now import matplotlib lazily, and this
        # pre-warm forces the retrieval model to load first, in the main thread.
        logger.info("Pre-warming retrieval store (loads sentence-transformers first)...")
        try:
            from matchmind.agent.nodes.retrieve_node import _get_vector_store

            store = _get_vector_store()
            _ = store.retrieve_tactics(query_text="warm up tactical retrieval", k=1)
            logger.info(f"Retrieval store ready: {store.count()} concepts")
        except Exception as exc:
            logger.warning(f"Store pre-warm failed (will retry at runtime): {exc}")

        # Safe now — building the graph may import matplotlib via render_pitch_node.
        self._graph = build_tactical_graph()
        logger.info("TacticalAgent initialised")


    def analyze(
        self,
        match_state: MatchState,
        focus_player_id: int,
        focus_team: str | None = None,
        question: str | None = None,
    ) -> TacticalAdvice:
        """
        Run the full agent pipeline on a match snapshot.

        Args:
            match_state: Unified MatchState from any adapter.
            focus_player_id: Which player to analyse.
            focus_team: "home" or "away", if known. `player_id` is NOT
                guaranteed globally unique — some sources (Metrica shirt
                numbers) reuse 1-11 on both teams. Passing focus_team resolves
                that ambiguity for this run; omit it only when you're sure
                `focus_player_id` can't collide (e.g. StatsBomb's global ids).
            question: Natural language question. Defaults to generic question.

        Returns:
            TacticalAdvice — structured, validated output.

        Raises:
            RuntimeError: If agent cannot produce valid advice after max retries.
        """
        if focus_team is not None and match_state.has_ambiguous_id(focus_player_id):
            match_state = _disambiguate_by_team(match_state, focus_player_id, focus_team)

        if question is None:
            player = match_state.get_player(focus_player_id)
            role = f" ({player.role})" if player and player.role else ""
            question = (
                f"What should player {focus_player_id}{role} do right now "
                f"to create tactical advantage for their team?"
            )

        initial_state: AgentState = {
            "match_state": match_state,
            "focus_player_id": focus_player_id,
            "question": question,
            "situation_features": None,
            "pitch_image_path": None,
            "candidate_actions": None,
            "pitch_control_grid": None,
            "pitch_control_baseline": None,
            "counterfactual_analysis": None,
            "counterfactual_text": None,
            "retrieved_tactics": None,
            "retrieval_confidence": None,
            "retrieval_retry_count": 0,
            "expanded_query": None,
            "evidence_set": None,
            "raw_llm_output": None,
            "tactical_advice": None,
            "validation_errors": [],
            "generation_retry_count": 0,
            "validation_passed": False,
            "trace": [],
            "total_tokens_used": 0,
        }

        logger.info(
            f"Running TacticalAgent: match={match_state.match_id}, "
            f"player={focus_player_id}, minute={match_state.minute}"
        )

        final_state = self._graph.invoke(initial_state)

        if final_state.get("tactical_advice") is None:
            errors = final_state.get("validation_errors", [])
            raise RuntimeError(
                f"TacticalAgent failed to produce valid advice. Errors: {errors}"
            )

        advice = final_state["tactical_advice"]
        logger.info(
            f"Agent completed: action='{advice.recommended_action[:60]}', "
            f"confidence={advice.confidence:.2f}, tokens={final_state.get('total_tokens_used', 0)}"
        )

        return advice, final_state.get("trace", [])

    def analyze_dual(
        self,
        match_state: MatchState,
    ) -> tuple[tuple[TacticalAdvice, list[dict]], tuple[TacticalAdvice, list[dict]]]:
        """
        Run TWO independent analyses on the same snapshot: one for the team in
        possession (on-ball question) and one for the opposing team (defensive
        question) — e.g. "home has the ball, what should home's carrier do?"
        alongside "what should away do to stop it?".

        This runs the full graph twice (2x LLM calls) — it is not a cheaper
        shortcut, it is two full `analyze()` calls with auto-picked focus
        players. See situation_engine.team_focus.pick_dual_focus_players for
        how the two players are chosen.

        Returns:
            ((attacking_advice, attacking_trace), (defending_advice, defending_trace))
        """
        from matchmind.situation_engine.team_focus import pick_dual_focus_players

        focus = pick_dual_focus_players(match_state)

        # Attacking and defending ids CAN collide with each other (e.g. both
        # #10) — always resolve with team here, not a bare id lookup.
        att_player = match_state.get_player(focus.attacking_player_id, team=focus.attacking_team)
        att_role = f" ({att_player.role})" if att_player and att_player.role else ""
        attacking_question = (
            f"Player {focus.attacking_player_id}{att_role} ({focus.attacking_team}) has the ball. "
            f"What should they do right now to create a tactical advantage for their team?"
        )

        def_player = match_state.get_player(focus.defending_player_id, team=focus.defending_team)
        def_role = f" ({def_player.role})" if def_player and def_player.role else ""
        defending_question = (
            f"Player {focus.defending_player_id}{def_role} ({focus.defending_team}) does not have "
            f"the ball — the opponent does. What should they do right now to stop the attack and "
            f"win the ball back?"
        )

        logger.info(
            f"analyze_dual: attack=player {focus.attacking_player_id} ({focus.attacking_team}), "
            f"defense=player {focus.defending_player_id} ({focus.defending_team})"
        )

        attacking = self.analyze(
            match_state, focus.attacking_player_id, focus_team=focus.attacking_team, question=attacking_question
        )
        defending = self.analyze(
            match_state, focus.defending_player_id, focus_team=focus.defending_team, question=defending_question
        )
        return attacking, defending
