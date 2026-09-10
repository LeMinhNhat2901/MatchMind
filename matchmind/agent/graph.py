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
        self._graph = build_tactical_graph()
        logger.info("TacticalAgent initialised with LangGraph graph")

    def analyze(
        self,
        match_state: MatchState,
        focus_player_id: int,
        question: str | None = None,
    ) -> TacticalAdvice:
        """
        Run the full agent pipeline on a match snapshot.

        Args:
            match_state: Unified MatchState from any adapter.
            focus_player_id: Which player to analyse.
            question: Natural language question. Defaults to generic question.

        Returns:
            TacticalAdvice — structured, validated output.

        Raises:
            RuntimeError: If agent cannot produce valid advice after max retries.
        """
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
