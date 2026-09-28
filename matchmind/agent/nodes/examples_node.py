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
    index = get_example_index()
    examples: list[dict] = []
    summary = ""
    if index is not None and len(index):
        vec = feature_vector(state["situation_features"], state["match_state"])
        neighbours = index.query(vec, k=settings.example_top_k, exclude_match_id=state["match_state"].match_id)
        examples = [
            {"example_id": e.example_id, "summary": e.summary, "action_type": e.action_type,
             "action_detail": e.action_detail, "outcome": e.outcome, "similarity": sim,
             "source_url": e.source_url}
            for e, sim in neighbours
        ]
        summary = summarize_neighbours(neighbours)
    trace = state.get("trace") or []
    trace.append({"node": "retrieve_examples", "latency_ms": round((time.time() - t0) * 1000, 1),
                  "n_examples": len(examples), "example_ids": [e["example_id"] for e in examples]})
    return {**state, "similar_examples": examples, "similar_examples_summary": summary, "trace": trace}
