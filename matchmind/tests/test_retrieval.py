"""
Tests: RAG Retrieval (requires built knowledge base)
Skips if ChromaDB has no concepts (KB not yet built).
"""
import pytest

from matchmind.knowledge_base.embedder import TacticalEmbedder, load_all_concepts
from matchmind.knowledge_base.vector_store import TacticalVectorStore
from matchmind.evaluation.evaluator import (
    compute_agreement,
    compute_groundedness,
    compute_actionability,
)
from matchmind.schema.match_state import TacticalAdvice, EvidenceItem, AlternativeAction
from matchmind.tests.fixtures.sample_data import make_sample_match_state


@pytest.fixture(scope="module")
def embedder():
    return TacticalEmbedder()


@pytest.fixture(scope="module")
def vector_store():
    return TacticalVectorStore()


def test_concepts_loadable():
    concepts = load_all_concepts()
    assert len(concepts) >= 30, f"Expected 30+ concepts, got {len(concepts)}"


def test_embedder_query(embedder):
    emb = embedder.embed_query("player in half-space with open passing lane")
    assert emb.shape[0] == 384  # all-MiniLM-L6-v2 dim
    assert abs(sum(emb ** 2) - 1.0) < 0.01  # normalized


def test_retrieval_returns_results(vector_store, embedder):
    count = vector_store.count()
    if count == 0:
        pytest.skip("Knowledge base not built — run `matchmind-build-kb` first")

    state, focus_id = make_sample_match_state()
    from matchmind.situation_engine.encoder import encode_situation
    features = encode_situation(state, focus_id)

    query_emb = embedder.embed_query(features.natural_language_description)
    results, confidence = vector_store.retrieve_tactics(query_emb, k=3)

    assert len(results) >= 1
    assert 0.0 <= confidence <= 1.0
    assert all("concept_id" in r for r in results)
    assert all("title" in r for r in results)


def test_all_titles_non_empty(vector_store):
    if vector_store.count() == 0:
        pytest.skip("Knowledge base not built")
    titles = vector_store.get_all_titles()
    assert all(t for t in titles), "All titles should be non-empty strings"


# ── Evaluation function unit tests ────────────────────────────────────────────

def _make_advice(action: str = "Pass to player 11 in the channel") -> TacticalAdvice:
    return TacticalAdvice(
        recommended_action=action,
        reasoning="Because there is space and an open lane",
        confidence=0.75,
        evidence=[
            EvidenceItem(id="t1", source="rag", content="Third man run"),
            EvidenceItem(id="cf", source="stats", content="Space: 18m"),
            EvidenceItem(id="ctx", source="context", content="Trailing 0-1"),
        ],
        cited_concepts=["Third-Man Run (Combination Play)"],
        alternatives=[
            AlternativeAction(action="Dribble", why_not="Under pressure", delta_pitch_control=-0.03)
        ],
    )


def test_agreement_pass_event():
    advice = _make_advice("Pass to player 11 in the channel")
    assert compute_agreement(advice, "Pass") is True


def test_agreement_mismatch():
    advice = _make_advice("Shoot from distance")
    assert compute_agreement(advice, "Clearance") is False


def test_groundedness_valid_citations():
    advice = _make_advice()
    retrieved_titles = ["Third-Man Run (Combination Play)", "Half-Space Exploitation"]
    g = compute_groundedness(advice, retrieved_titles)
    assert g["citation_validity_rate"] == 1.0
    assert g["no_invented_concepts"] is True


def test_groundedness_hallucinated_citation():
    advice = _make_advice()
    advice = advice.model_copy(update={"cited_concepts": ["Invented Concept XYZ"]})
    g = compute_groundedness(advice, ["Third-Man Run (Combination Play)"])
    assert g["citation_validity_rate"] == 0.0
    assert g["no_invented_concepts"] is False


def test_actionability_specific_action():
    advice = _make_advice("Pass to player 11 in the right channel — overlapping run into half-space")
    score = compute_actionability(advice, use_llm=False)
    assert score >= 3.0


def test_actionability_vague_action():
    advice = _make_advice("play better")
    score = compute_actionability(advice, use_llm=False)
    assert score <= 2.0
