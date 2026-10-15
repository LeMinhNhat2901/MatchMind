"""dense_rank is exact and deterministic.

Chroma's HNSW search is approximate: on the 331-concept KB it sometimes dropped
a concept that was the 6th most similar (blind_side_run for "runs off the
centre-back's shoulder"), and which one it dropped changed between processes,
so the golden bench flipped between hit@3 0.889 and 0.926 on identical data.
A KB this size is scored exactly instead (cosine against every stored vector).
"""
from __future__ import annotations

import numpy as np

from matchmind.knowledge_base.vector_store import TacticalVectorStore
from matchmind.tests.fixtures.concepts import make_concept


def _store(tmp_path, vectors: dict[str, list[float]]) -> TacticalVectorStore:
    store = TacticalVectorStore(persist_dir=tmp_path / "db", collection_name="test_dense", use_chroma_ef=False)
    store.upsert_concepts([make_concept(cid) for cid in vectors], np.array(list(vectors.values()), dtype=float))
    store._ef = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]      # every query points along x
    return store


def test_exact_ranking_and_similarity_scale(tmp_path):
    store = _store(tmp_path, {"far": [0.0, 1.0, 0.0], "near": [0.9, 0.1, 0.0], "same": [2.0, 0.0, 0.0]})
    [ranked] = store.dense_rank(["q"], n=3)
    assert [cid for cid, _ in ranked] == ["same", "near", "far"]
    sims = dict(ranked)
    assert abs(sims["same"] - 1.0) < 1e-6           # cos 1 -> similarity 1 (1 - distance/2)
    assert abs(sims["far"] - 0.5) < 1e-6            # cos 0 -> 0.5


def test_ties_break_by_id_and_n_caps(tmp_path):
    store = _store(tmp_path, {"b": [1.0, 0.0, 0.0], "a": [1.0, 0.0, 0.0], "c": [0.0, 1.0, 0.0]})
    [ranked] = store.dense_rank(["q"], n=2)
    assert [cid for cid, _ in ranked] == ["a", "b"]


def test_cache_refreshes_after_upsert(tmp_path):
    store = _store(tmp_path, {"a": [0.0, 1.0, 0.0]})
    store.dense_rank(["q"], n=5)
    store.upsert_concepts([make_concept("z")], np.array([[1.0, 0.0, 0.0]]))
    [ranked] = store.dense_rank(["q"], n=5)
    assert ranked[0][0] == "z"
