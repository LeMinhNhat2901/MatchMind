"""
Encode Situation Node.

Computes SituationFeatures from MatchState using the Situation Engine.
This is pure computation — no LLM, no I/O.
"""
from __future__ import annotations

import logging
import time

from matchmind.agent.state import AgentState
from matchmind.situation_engine.encoder import encode_situation

logger = logging.getLogger(__name__)


def encode_situation_node(state: AgentState) -> AgentState:
    """
    Node: Encode situation.

    Input:  state.match_state, state.focus_player_id
    Output: state.situation_features
    """
    t0 = time.time()

    features = encode_situation(
        state["match_state"],
        state["focus_player_id"],
    )

    elapsed = (time.time() - t0) * 1000
    logger.debug(f"[encode_situation] {elapsed:.1f}ms | player={state['focus_player_id']}")

    trace = state.get("trace") or []
    trace.append({"node": "encode_situation", "latency_ms": round(elapsed, 1)})

    return {
        **state,
        "situation_features": features,
        "trace": trace,
    }
