"""Regression tests for the V1 whole-branch review findings (Important tier)."""
from __future__ import annotations

import numpy as np

from matchmind.agent.nodes import examples_node as en
from matchmind.evaluation.evaluator import compute_groundedness
from matchmind.examples.index import ExampleIndex
from matchmind.examples.schema import SituationExample
from matchmind.retrieval.scope import is_allowed
from matchmind.retrieval.types import RetrievalQuery
from matchmind.schema.match_state import EvidenceItem, TacticalAdvice
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.concepts import make_concept
from matchmind.tests.fixtures.sample_data import make_sample_match_state


# ── #1 example evidence counts as a valid source ──────────────────────────────

def test_example_evidence_does_not_lower_groundedness():
    advice = TacticalAdvice(
        recommended_action="Pass to 8", reasoning="r", confidence=0.8, cited_concepts=[], alternatives=[],
        evidence=[EvidenceItem(id="t", source="rag", content="x"),
                  EvidenceItem(id="sb_1_2", source="example", content="y")],
    )
    assert compute_groundedness(advice, [])["evidence_coverage"] == 1.0


# ── #2 / #3 example node: carriers only, never fatal ──────────────────────────

def _node_state(**state_overrides):
    ms, focus = make_sample_match_state()
    if state_overrides:
        ms = ms.model_copy(update=state_overrides)
    return {"match_state": ms, "focus_player_id": focus, "situation_features": encode_situation(ms, focus), "trace": []}


def _library(dim: int) -> ExampleIndex:
    ex = [SituationExample(example_id=f"e{i}", match_id="other", event_index=i, minute=1, action_type="pass",
                           action_detail="progressive pass", outcome="shot", summary="s",
                           vector=list(np.arange(dim, dtype=float) + i), source_url="u") for i in range(3)]
    return ExampleIndex.build(ex)


def test_examples_skipped_for_off_ball_or_defending_focus(monkeypatch):
    monkeypatch.setattr(en, "get_example_index", lambda: _library(13))
    out = en.retrieve_examples_node(_node_state(possession="away"))   # focus team (home) defending
    assert out["similar_examples"] == []
    assert out["trace"][-1]["skipped"]


def test_examples_node_survives_dimension_mismatch(monkeypatch):
    monkeypatch.setattr(en, "get_example_index", lambda: _library(7))   # stale library on disk
    out = en.retrieve_examples_node(_node_state())
    assert out["similar_examples"] == []
    assert "error" in out["trace"][-1]


# ── #4 technique scope needs the technique named, not a substring ─────────────

STEP_OVER = make_concept("wiki_step_over_dribbling_feint", title="Step Over Dribbling Feint", category="individual")
VOLLEY = make_concept("wiki_volley_and_half_volley_technique", title="Volley and Half-Volley Technique",
                      category="individual")


def test_tactical_words_do_not_unlock_techniques():
    for q in ("Should the full-back overlap?", "Create an overload on the left",
              "The back line should step up", "Attack the half-space"):
        assert not is_allowed(STEP_OVER, RetrievalQuery(question=q)), q
        assert not is_allowed(VOLLEY, RetrievalQuery(question=q)), q


def test_named_technique_still_unlocks():
    assert is_allowed(STEP_OVER, RetrievalQuery(question="Should he do a step over here?"))
    assert is_allowed(VOLLEY, RetrievalQuery(question="Can he volley it first time?"))


# ── #13 invented example ids are dropped ──────────────────────────────────────

def test_validator_drops_example_evidence_not_retrieved():
    from matchmind.agent.nodes.validator_node import drop_unretrieved_examples

    evidence = [EvidenceItem(id="sb_1_2", source="example", content="real"),
                EvidenceItem(id="sb_9_9", source="example", content="invented"),
                EvidenceItem(id="t", source="rag", content="x")]
    kept = drop_unretrieved_examples(evidence, [{"example_id": "sb_1_2"}])
    assert [e.id for e in kept] == ["sb_1_2", "t"]


# ── #5 one vector store per process ───────────────────────────────────────────

def test_hybrid_retriever_reuses_the_node_vector_store(monkeypatch):
    from matchmind.agent.nodes import retrieve_node as rn
    from matchmind.retrieval import hybrid

    sentinel = object()
    monkeypatch.setattr(rn, "_vector_store", sentinel)
    monkeypatch.setattr(hybrid, "_retriever", None)
    monkeypatch.setattr(hybrid, "load_all_concepts", lambda: [make_concept("a")], raising=False)
    r = hybrid.get_hybrid_retriever(use_reranker=False)
    assert r.store is sentinel
