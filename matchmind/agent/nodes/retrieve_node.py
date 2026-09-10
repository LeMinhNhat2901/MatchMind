"""
RAG Retrieval Node.

Queries ChromaDB with the situation description embedding.
Computes retrieval confidence for the conditional edge decision.
"""
from __future__ import annotations

import logging
import time

from matchmind.agent.state import AgentState
from matchmind.config import settings
from matchmind.knowledge_base.embedder import TacticalEmbedder
from matchmind.knowledge_base.vector_store import TacticalVectorStore

logger = logging.getLogger(__name__)

# Module-level singletons (loaded once per process)
_embedder: TacticalEmbedder | None = None
_vector_store: TacticalVectorStore | None = None


def _get_embedder() -> TacticalEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = TacticalEmbedder()
    return _embedder


def _get_vector_store() -> TacticalVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = TacticalVectorStore()
    return _vector_store


def retrieve_node(state: AgentState) -> AgentState:
    """
    Node: Retrieve tactical concepts from ChromaDB.

    Uses the situation description from SituationFeatures as the RAG query.
    If an expanded_query is set (from a re-retrieve), uses that instead.

    Input:  state.situation_features, state.expanded_query (optional)
    Output: state.retrieved_tactics, state.retrieval_confidence
    """
    t0 = time.time()

    features = state["situation_features"]
    match_state = state["match_state"]
    query_text = state.get("expanded_query") or features.natural_language_description

    embedder = _get_embedder()
    store = _get_vector_store()

    # Phase pre-filter narrows the small KB; only on the first attempt so a
    # low-confidence re-retrieve can widen the search.
    is_retry = (state.get("retrieval_retry_count", 0) or 0) > 0
    phase = (
        None
        if (is_retry or not settings.retrieval_phase_filter)
        else (match_state.phase_of_play or "open_play")
    )

    query_emb = embedder.embed_query(query_text)
    results, confidence = store.retrieve_tactics(
        query_embedding=query_emb,
        k=settings.retrieval_top_k,
        filter_phase=phase,
    )

    elapsed = (time.time() - t0) * 1000
    logger.debug(
        f"[retrieve] {elapsed:.1f}ms | k={len(results)}, confidence={confidence:.3f}, "
        f"retry={state.get('retrieval_retry_count', 0)}"
    )

    trace = state.get("trace") or []
    trace.append({
        "node": "retrieve",
        "latency_ms": round(elapsed, 1),
        "confidence": round(confidence, 4),
        "n_results": len(results),
        "retry": state.get("retrieval_retry_count", 0),
        "concept_ids": [r["concept_id"] for r in results],
        "titles": [r["title"] for r in results],
    })

    return {
        **state,
        "retrieved_tactics": results,
        "retrieval_confidence": confidence,
        "trace": trace,
    }


def build_expanded_query(state: AgentState) -> AgentState:
    """
    Build an expanded query for re-retrieval when confidence is low.

    Adds phase_of_play and key features to the query to broaden the search.
    """
    features = state["situation_features"]
    match_state = state["match_state"]

    expanded = (
        f"{features.natural_language_description} "
        f"Phase: {match_state.phase_of_play or 'open play'}. "
        f"Looking for tactical concepts related to: "
        f"{'overlapping run, ' if features.overload_left or features.overload_right else ''}"
        f"{'half space exploitation, ' if features.half_space_occupied else ''}"
        f"{'third man run, ' if features.third_man_opportunity else ''}"
        f"{'counter attack, ' if match_state.phase_of_play == 'attacking_transition' else ''}"
        f"{'space creation, ' if features.local_numerical_advantage < 0 else ''}"
        f"numerical superiority, possession play."
    )
    logger.debug(f"[expand_query] Expanded query: {expanded[:100]}...")

    trace = state.get("trace") or []
    trace.append({"node": "expand_query", "retry": state.get("retrieval_retry_count", 0) + 1})

    return {
        **state,
        "expanded_query": expanded,
        "retrieval_retry_count": state.get("retrieval_retry_count", 0) + 1,
        "trace": trace,
    }


def check_retrieval_quality(state: AgentState) -> str:
    """
    Conditional edge function.

    Returns:
        "re_retrieve" if confidence is below threshold AND retries not exhausted
        "assemble_evidence" otherwise
    """
    confidence = state.get("retrieval_confidence", 0.0) or 0.0
    retry_count = state.get("retrieval_retry_count", 0) or 0
    threshold = settings.effective_retrieval_threshold
    max_retries = settings.agent_max_retries

    if confidence < threshold and retry_count < max_retries:
        logger.info(
            f"[check_retrieval] Confidence {confidence:.3f} < {threshold} "
            f"(retry {retry_count}/{max_retries}) → re-retrieve"
        )
        return "re_retrieve"

    if confidence < threshold:
        logger.warning(
            f"[check_retrieval] Confidence {confidence:.3f} still low after {retry_count} retries — proceeding anyway"
        )

    return "assemble_evidence"
