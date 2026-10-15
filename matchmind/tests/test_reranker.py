from matchmind.retrieval.hybrid import HybridRetriever
from matchmind.retrieval.reranker import CrossEncoderReranker, build_reranker
from matchmind.retrieval.types import RetrievalQuery
from matchmind.tests.test_hybrid_retriever import CONCEPTS, FakeStore


class KeywordReranker:
    """Scores 0.9 for docs containing the keyword, 0.1 otherwise."""

    def __init__(self, keyword):
        self.keyword = keyword

    def score(self, query, docs):
        return [0.9 if self.keyword in d.lower() else 0.1 for d in docs]


class BrokenReranker:
    def score(self, query, docs):
        raise OSError("model download failed")


def _store():
    return FakeStore(["pocket_exploitation", "overlapping_run", "third_man_run", "low_block"])


def test_cross_encoder_logits_are_squashed_into_unit_interval():
    """sentence-transformers 3.4 returns raw logits for ms-marco (seen: -3.07, -11.38);
    confidence thresholds need [0, 1] with the order preserved."""
    import numpy as np

    class LogitModel:
        def predict(self, pairs, convert_to_numpy=True):
            return np.array([-3.07, -11.38, 4.0])

    r = CrossEncoderReranker("fake")
    r._model = LogitModel()
    scores = r.score("q", ["a", "b", "c"])
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[2] > scores[0] > scores[1]


def test_build_reranker_disabled_by_empty_name():
    assert build_reranker("") is None
    assert isinstance(build_reranker("some/model"), CrossEncoderReranker)


def test_reranker_reorders_and_sets_confidence():
    r = HybridRetriever(concepts=CONCEPTS, store=_store(), reranker=KeywordReranker("overlap"))
    res = r.retrieve(RetrievalQuery(question="run"), k=3)
    assert res.concept_ids[0] == "overlapping_run"
    assert res.confidence == 0.9
    assert res.strategy.startswith("hybrid+rerank")


def test_rerank_is_blended_with_fusion_score():
    """A concept every ranker agreed on should not be overturned by a slightly
    higher cross-encoder score (golden bench: blending beat pure rerank)."""
    from matchmind.retrieval.types import ScoredConcept

    class FlatReranker:
        def score(self, query, docs):
            return [0.5, 0.6]

    r = HybridRetriever(concepts=CONCEPTS, store=_store(), reranker=FlatReranker())
    r.rerank_alpha = 0.3
    items = [ScoredConcept(concept_id="overlapping_run", score=1.0),
             ScoredConcept(concept_id="low_block", score=0.1)]
    out = r._rerank(RetrievalQuery(question="q"), items)
    assert [i.concept_id for i in out] == ["overlapping_run", "low_block"]
    assert out[0].rerank_score == 0.5          # confidence still reads the raw rerank score


def test_broken_reranker_falls_back_to_fusion_order():
    r = HybridRetriever(concepts=CONCEPTS, store=_store(), reranker=BrokenReranker())
    res = r.retrieve(RetrievalQuery(question="receive between the lines"), k=3)
    assert len(res.concept_ids) == 3
    assert all(i.rerank_score is None for i in res.items)
