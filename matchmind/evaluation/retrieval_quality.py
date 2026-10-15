"""
Evaluation — Dimension 0: Retrieval Quality.

precision@k / recall@k for the RAG retriever, using each concept's
``trigger_conditions`` as a proxy ground truth:

    a concept is "relevant" to a situation when >= RELEVANCE_MIN_FRACTION of its
    machine-checkable trigger tokens are satisfied by the SituationFeatures /
    MatchState.

This is a proxy (trigger tokens are hand-written and only partly checkable),
so treat the numbers as directional — exactly as bonus.md §6 / revise.md §9
intend. It is still the single most important RAG health metric.
"""
from __future__ import annotations

import logging

from matchmind.schema.match_state import MatchState, SituationFeatures, TacticalConcept

logger = logging.getLogger(__name__)

RELEVANCE_MIN_FRACTION = 0.5


from matchmind.retrieval.triggers import TRIGGER_PREDICATES as _TRIGGERS  # noqa: F401 — re-export
from matchmind.retrieval.triggers import trigger_match_fraction as concept_relevance_score


def relevant_concept_ids(
    concepts: list[TacticalConcept],
    feats: SituationFeatures,
    state: MatchState,
) -> set[str]:
    """Proxy ground-truth relevant set for one situation."""
    out: set[str] = set()
    for c in concepts:
        score = concept_relevance_score(c, feats, state)
        if score is not None and score >= RELEVANCE_MIN_FRACTION:
            out.add(c.concept_id)
    return out


def evaluate_retrieval(
    retrieved_ids: list[str],
    concepts: list[TacticalConcept],
    feats: SituationFeatures,
    state: MatchState,
    k: int = 3,
) -> dict:
    """Precision@k / recall@k for one retrieval against the proxy ground truth."""
    relevant = relevant_concept_ids(concepts, feats, state)
    top_k = list(retrieved_ids[:k])
    hits = len(set(top_k) & relevant)

    precision = hits / max(len(top_k), 1)
    recall = hits / len(relevant) if relevant else None  # undefined when no relevant concept

    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4) if recall is not None else None,
        "n_relevant": len(relevant),
        "n_retrieved": len(top_k),
        "n_hits": hits,
        "relevant_ids": sorted(relevant),
        "retrieved_ids": top_k,
    }


def aggregate_retrieval_quality(per_case: list[dict], k: int) -> dict:
    """Mean precision/recall across cases (cases with no relevant concept are
    excluded from the recall mean, not counted as 0)."""
    if not per_case:
        return {"precision_at_k": None, "recall_at_k": None, "retrieval_k": k, "n_cases": 0}
    prec = [c["precision_at_k"] for c in per_case if c.get("precision_at_k") is not None]
    rec = [c["recall_at_k"] for c in per_case if c.get("recall_at_k") is not None]
    return {
        "precision_at_k": round(sum(prec) / len(prec), 4) if prec else None,
        "recall_at_k": round(sum(rec) / len(rec), 4) if rec else None,
        "retrieval_k": k,
        "n_cases": len(per_case),
        "n_cases_with_relevant": len(rec),
    }
