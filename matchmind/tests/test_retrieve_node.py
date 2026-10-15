import json

from matchmind.agent.nodes import retrieve_node as rn
from matchmind.retrieval.hybrid import HybridRetriever
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import make_sample_match_state
from matchmind.tests.test_hybrid_retriever import CONCEPTS, FakeStore


def _state(**extra):
    ms, focus = make_sample_match_state()
    return {"match_state": ms, "focus_player_id": focus, "question": "How does player 10 escape the press?",
            "situation_features": encode_situation(ms, focus), "trace": [], "retrieval_retry_count": 0, **extra}


def _fake(monkeypatch):
    fake = HybridRetriever(concepts=CONCEPTS, store=FakeStore(
        ["pocket_exploitation", "overlapping_run", "third_man_run", "low_block"]), use_reranker=False)
    monkeypatch.setattr(rn, "get_hybrid_retriever", lambda: fake)
    monkeypatch.setattr(rn.settings, "retrieval_strategy", "hybrid")
    return fake


def test_hybrid_node_returns_legacy_dicts_and_trace(monkeypatch):
    _fake(monkeypatch)
    out = rn.retrieve_node(_state())
    assert len(out["retrieved_tactics"]) == rn.settings.retrieval_top_k
    assert {"concept_id", "title", "content"} <= set(out["retrieved_tactics"][0])
    step = out["trace"][-1]
    assert step["node"] == "retrieve" and step["strategy"].startswith("hybrid")
    assert step["concept_ids"] == [r["concept_id"] for r in out["retrieved_tactics"]]
    assert any("escape the press" in q for q in step["subqueries"])


def test_retry_uses_expanded_query_and_relax(monkeypatch):
    _fake(monkeypatch)
    out = rn.retrieve_node(_state(retrieval_retry_count=1, expanded_query="expanded words"))
    step = out["trace"][-1]
    assert "expanded words" in step["subqueries"]
    assert "relaxed" in step["strategy"]


def test_check_uses_calibrated_threshold(monkeypatch, tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"threshold": 0.9}))
    monkeypatch.setattr(rn.settings, "retrieval_calibration_file", cal)
    state = {"retrieval_confidence": 0.8, "retrieval_retry_count": 0}
    assert rn.check_retrieval_quality(state) == "re_retrieve"      # 0.8 < calibrated 0.9


def test_dense_strategy_still_available(monkeypatch):
    class LegacyStore:
        def retrieve_tactics(self, query_text=None, k=3, **kw):
            return [{"concept_id": "a", "title": "A", "content": "x"}], 0.5

    monkeypatch.setattr(rn.settings, "retrieval_strategy", "dense")
    monkeypatch.setattr(rn, "_get_vector_store", lambda: LegacyStore())
    out = rn.retrieve_node(_state())
    assert out["retrieved_tactics"][0]["concept_id"] == "a"
