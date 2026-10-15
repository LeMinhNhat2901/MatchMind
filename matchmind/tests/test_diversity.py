import numpy as np

from matchmind.retrieval.diversity import select_diverse
from matchmind.retrieval.hybrid import HybridRetriever
from matchmind.retrieval.types import RetrievalQuery
from matchmind.tests.test_hybrid_retriever import CONCEPTS

E = {
    "a": np.array([1.0, 0.0]), "a_dup": np.array([0.99, 0.141]),
    "b": np.array([0.0, 1.0]), "c": np.array([0.7, 0.7]),
}


def test_skips_near_duplicates():
    assert select_diverse(["a", "a_dup", "b", "c"], E, k=2, max_sim=0.95) == ["a", "b"]


def test_fills_back_when_not_enough_diverse_items():
    assert select_diverse(["a", "a_dup"], E, k=2, max_sim=0.95) == ["a", "a_dup"]


def test_items_without_embeddings_are_kept():
    assert select_diverse(["x", "a"], E, k=2, max_sim=0.95) == ["x", "a"]


def test_retriever_diversifies_top_k():
    class DupStore:
        order = ["pocket_exploitation", "overlapping_run", "third_man_run", "low_block"]

        def dense_rank(self, qs, n):
            return [[(c, 0.9) for c in self.order] for _ in qs]

        def get_embeddings(self, ids):
            vec = {"pocket_exploitation": [1, 0], "overlapping_run": [1, 0.01],
                   "third_man_run": [0, 1], "low_block": [0.5, 0.5]}
            return {i: np.array(vec[i], dtype=float) for i in ids if i in vec}

    r = HybridRetriever(concepts=CONCEPTS, store=DupStore(), use_reranker=False, diversity_max_sim=0.95)
    ids = r.retrieve(RetrievalQuery(question="receive between the lines"), k=2).concept_ids
    assert not {"pocket_exploitation", "overlapping_run"} <= set(ids)
