"""
RAG Retrieval Node.

Uses ChromaDB's built-in SentenceTransformerEmbeddingFunction (thread-safe on Windows).
Does NOT import sentence-transformers directly — avoids ONNX Runtime crash in
LangGraph thread context on Python 3.13.
"""
from __future__ import annotations

import logging
import time

from matchmind.agent.state import AgentState
from matchmind.config import settings
from matchmind.knowledge_base.vector_store import TacticalVectorStore

logger = logging.getLogger(__name__)

# Singleton — one store per process
_vector_store: TacticalVectorStore | None = None


def _get_vector_store() -> TacticalVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = TacticalVectorStore(use_chroma_ef=True)
    return _vector_store


def retrieve_node(state: AgentState) -> AgentState:
    """
    Node: Retrieve tactical concepts from ChromaDB using query text.

    ChromaDB's built-in EF embeds the query string internally — no manual
    sentence-transformers call inside this thread.

    Input:  state.situation_features, state.expanded_query (optional)
    Output: state.retrieved_tactics, state.retrieval_confidence
    """
    t0 = time.time()

    features = state["situation_features"]
    query_text = state.get("expanded_query") or features.natural_language_description

    store = _get_vector_store()
    results, confidence = store.retrieve_tactics(
        query_text=query_text,
        k=settings.retrieval_top_k,
    )

    elapsed = (time.time() - t0) * 1000
    logger.debug(
        f"[retrieve] {elapsed:.1f}ms | k={len(results)}, "
        f"confidence={confidence:.3f}, retry={state.get('retrieval_retry_count', 0)}"
    )

    trace = state.get("trace") or []
    trace.append({
        "node": "retrieve",
        "latency_ms": round(elapsed, 1),
        "confidence": round(confidence, 4),
        "n_results": len(results),
        "retry": state.get("retrieval_retry_count", 0),
        # consumed by evaluation.retrieval_quality (Dim 0)
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
    """Build an expanded query for re-retrieval when confidence is low."""
    features = state["situation_features"]
    match_state = state["match_state"]

    expanded = (
        f"{features.natural_language_description} "
        f"Phase: {match_state.phase_of_play or 'open play'}. "
        f"Tactical concepts related to: "
        f"{'overlapping run, ' if features.overload_left or features.overload_right else ''}"
        f"{'half space exploitation, ' if features.half_space_occupied else ''}"
        f"{'third man run, ' if features.third_man_opportunity else ''}"
        f"{'counter attack, ' if match_state.phase_of_play == 'attacking_transition' else ''}"
        f"{'space creation, ' if features.local_numerical_advantage < 0 else ''}"
        f"numerical superiority, possession play."
    )
    logger.debug(f"[expand_query] {expanded[:100]}...")

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
    Conditional edge: re-retrieve if confidence low and retries not exhausted.

    Returns "re_retrieve" or "assemble_evidence".
    """
    confidence = state.get("retrieval_confidence") or 0.0
    retry_count = state.get("retrieval_retry_count") or 0
    threshold = settings.retrieval_confidence_threshold
    max_retries = settings.agent_max_retries

    if confidence < threshold and retry_count < max_retries:
        logger.info(
            f"[check_retrieval] conf={confidence:.3f} < {threshold}, "
            f"retry {retry_count}/{max_retries} → re-retrieve"
        )
        return "re_retrieve"

    if confidence < threshold:
        logger.warning(
            f"[check_retrieval] conf={confidence:.3f} still low after "
            f"{retry_count} retries — proceeding"
        )
    return "assemble_evidence"
