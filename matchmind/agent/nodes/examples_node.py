"""
Example retrieval node — nearest real situations from the Example Library.

Evidence of a different kind from RAG concepts: not "what the textbook says"
but "what teams actually did in situations that looked like this, and what
happened". Optional: if the library hasn't been built
(scripts/build_example_library.py) the node is a no-op and the agent runs
exactly as before.
"""
from __future__ import annotations

import logging
import time

from matchmind.agent.state import AgentState
from matchmind.config import settings
from matchmind.examples.builder import feature_vector
from matchmind.examples.index import ExampleIndex, summarize_neighbours

logger = logging.getLogger(__name__)

_index: ExampleIndex | None = None
_loaded = False


def get_example_index() -> ExampleIndex | None:
    global _index, _loaded
    if not _loaded:
        _loaded = True
        try:
            if (settings.example_library_dir / "index.npz").exists():
                _index = ExampleIndex.load(settings.example_library_dir)
                logger.info(f"Example library: {len(_index)} examples")
        except Exception as exc:  # noqa: BLE001 — optional evidence, never fatal
            logger.warning(f"Could not load example library: {exc}")
    return _index


def retrieve_examples_node(state: AgentState) -> AgentState:
    t0 = time.time()
    features, match_state = state["situation_features"], state["match_state"]
    step: dict = {"node": "retrieve_examples"}
    examples: list[dict] = []
    summary = ""
    # The library only holds on-ball events of the team in possession, so its
    # examples are only evidence for a ball carrier — never for a defender or an
    # off-ball runner (dual mode, follow-up projections).
    if not features.is_ball_carrier or match_state.possession != features.focus_player_team:
        step["skipped"] = "focus player is not the ball carrier"
    else:
        index = get_example_index()
        if index is not None and len(index):
            try:
                vec = feature_vector(features, match_state)
                neighbours = index.query(vec, k=settings.example_top_k, exclude_match_id=match_state.match_id)
            except Exception as exc:  # noqa: BLE001 — e.g. a stale library with another feature layout
                logger.warning(f"[retrieve_examples] query failed, continuing without examples: {exc}")
                step["error"] = str(exc)[:200]
                neighbours = []
            examples = [
                {"example_id": e.example_id, "summary": e.summary, "action_type": e.action_type,
                 "action_detail": e.action_detail, "outcome": e.outcome, "similarity": sim,
                 "source_url": e.source_url}
                for e, sim in neighbours
            ]
            summary = summarize_neighbours(neighbours)
    trace = state.get("trace") or []
    trace.append({**step, "latency_ms": round((time.time() - t0) * 1000, 1),
                  "n_examples": len(examples), "example_ids": [e["example_id"] for e in examples]})
    return {**state, "similar_examples": examples, "similar_examples_summary": summary, "trace": trace}
