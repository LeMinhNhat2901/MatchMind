import numpy as np

from matchmind.retrieval.fusion import reciprocal_rank_fusion
from matchmind.retrieval.hybrid import HybridRetriever
from matchmind.retrieval.types import RetrievalQuery
from matchmind.tests.fixtures.concepts import make_concept

CONCEPTS = [
    make_concept("third_man_run", title="Third-Man Run", content="third player run lay-off combination"),
    make_concept("overlapping_run", title="Overlapping Run", content="full-back overlaps outside the winger"),
    make_concept("low_block", title="Low Block", category="defending", phase="out_of_possession",
                 content="sit deep compact protect the box"),
    make_concept("fifa_front_post_corner_strategy", title="Front Post Corner Strategy",
                 content="corner kick front post runs overload"),
    make_concept("pocket_exploitation", title="Pocket Between Lines", content="receive between the lines"),
]


class FakeStore:
    """Dense ranking = fixed order per query; embeddings = one-hot per id."""

    def __init__(self, order, stale_ids=()):
        self.order = list(stale_ids) + order
        self.calls = []

    def dense_rank(self, query_texts, n):
        self.calls.append(list(query_texts))
        return [[(cid, 0.9 - 0.05 * i) for i, cid in enumerate(self.order[:n])] for _ in query_texts]

    def get_embeddings(self, ids):
        return {cid: np.eye(len(self.order))[self.order.index(cid)] for cid in ids if cid in self.order}


def _retriever(order=None, **kw):
    order = order or ["pocket_exploitation", "overlapping_run", "third_man_run", "low_block",
                      "fifa_front_post_corner_strategy"]
    return HybridRetriever(concepts=CONCEPTS, store=FakeStore(order, kw.pop("stale_ids", ())),
                           use_reranker=False, **kw)


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([(["a", "b", "c"], 1.0), (["b", "c", "a"], 1.0)])
    assert [cid for cid, _ in fused][0] == "b"


def test_rrf_weights():
    fused = dict(reciprocal_rank_fusion([(["a"], 1.0), (["b"], 0.5)]))
    assert fused["a"] > fused["b"]


def test_bm25_can_lift_exact_term_match():
    r = _retriever()
    res = r.retrieve(RetrievalQuery(question="third-man combination", phase_of_play="open_play"), k=3)
    assert "third_man_run" in res.concept_ids[:2]


def test_set_piece_never_returned_in_open_play():
    r = _retriever()
    res = r.retrieve(RetrievalQuery(question="front post corner overload", phase_of_play="open_play"), k=3)
    assert "fifa_front_post_corner_strategy" not in res.concept_ids


def test_possession_gating_applied():
    res = _retriever().retrieve(RetrievalQuery(question="sit deep", in_possession=True), k=3)
    assert "low_block" not in res.concept_ids


def test_stale_index_ids_are_dropped():
    res = _retriever(stale_ids=("deleted_concept",)).retrieve(RetrievalQuery(question="run"), k=3)
    assert "deleted_concept" not in res.concept_ids
    assert len(res.concept_ids) == 3


def test_vietnamese_question_still_returns_k():
    q = RetrievalQuery(question="cầu thủ nên làm gì", tactical_phrases=["third-man combination"],
                       phase_of_play="open_play", in_possession=True)
    assert len(_retriever().retrieve(q, k=3).concept_ids) == 3


def test_relax_adds_situation_text_subquery():
    store = FakeStore(["overlapping_run", "third_man_run", "pocket_exploitation", "low_block"])
    r = HybridRetriever(concepts=CONCEPTS, store=store, use_reranker=False)
    r.retrieve(RetrievalQuery(question="q", situation_text="numeric description"), k=3, relax=True)
    assert "numeric description" in store.calls[-1]


def test_confidence_is_top_dense_similarity_without_reranker():
    res = _retriever().retrieve(RetrievalQuery(question="receive between the lines"), k=3)
    assert 0.0 < res.confidence <= 1.0


def test_legacy_dicts_have_expected_keys():
    r = _retriever()
    dicts = r.to_legacy_dicts(r.retrieve(RetrievalQuery(question="run"), k=2))
    expected = {"concept_id", "title", "content", "category", "subcategory", "phase", "definition",
                "conditions", "possible_actions", "contraindications", "similarity",
                "source_name", "source_url"}
    assert all(expected <= set(d) for d in dicts)


def test_empty_kb_returns_empty_result():
    r = HybridRetriever(concepts=[], store=FakeStore([]), use_reranker=False)
    res = r.retrieve(RetrievalQuery(question="anything"), k=3)
    assert res.items == [] and res.confidence == 0.0
