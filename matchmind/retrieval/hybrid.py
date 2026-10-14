"""
HybridRetriever — the V1 retrieval pipeline (see implementation_plan.md Part A §3).

    scope gating -> [dense x subqueries] + [BM25 x subqueries] + [example questions]
                  + [condition match]
                 -> RRF -> (cross-encoder rerank) -> (diversity) -> top-k

Every stage degrades instead of failing: no Chroma EF -> BM25 + conditions
only; reranker unavailable -> fusion order; no features -> no condition list.
Ids that Chroma returns but the JSON KB no longer has (stale index) are
dropped, so a concept that doesn't exist can never be cited.
"""
from __future__ import annotations

import logging
from typing import Any

from matchmind.config import settings
from matchmind.knowledge_base.embedder import load_all_concepts
from matchmind.retrieval.diversity import select_diverse
from matchmind.retrieval.fusion import reciprocal_rank_fusion
from matchmind.retrieval.lexical import BM25Index
from matchmind.retrieval.reranker import rerank_document
from matchmind.retrieval.scope import allowed_ids
from matchmind.retrieval.triggers import condition_ranking
from matchmind.retrieval.types import RetrievalQuery, RetrievalResult, ScoredConcept
from matchmind.schema.match_state import MatchState, SituationFeatures, TacticalConcept

logger = logging.getLogger(__name__)

DENSE_WEIGHT = 1.0
BM25_WEIGHT = 1.0
CONDITION_WEIGHT = 0.5
QUESTION_WEIGHT = 1.0      # example-question ranker (doc2query), see knowledge_base/example_questions.py


class HybridRetriever:
    def __init__(
        self,
        concepts: list[TacticalConcept] | None = None,
        store: Any | None = None,
        reranker: Any | None = None,
        pool_size: int | None = None,
        diversity_max_sim: float | None = None,
        use_reranker: bool = True,
        question_index: Any | None = None,
    ) -> None:
        if concepts is None:
            concepts = load_all_concepts()
        if store is None:
            from matchmind.knowledge_base.vector_store import TacticalVectorStore

            store = TacticalVectorStore(use_chroma_ef=True)
        self.concepts: dict[str, TacticalConcept] = {c.concept_id: c for c in concepts}
        self.store = store
        self.bm25 = BM25Index(list(self.concepts.values())) if self.concepts else None
        self.pool_size = pool_size or settings.retrieval_pool_size
        self.diversity_max_sim = (
            settings.retrieval_diversity_max_sim if diversity_max_sim is None else diversity_max_sim
        )
        if reranker is None and use_reranker:
            from matchmind.retrieval.reranker import build_reranker

            reranker = build_reranker(settings.reranker_model)
        self.reranker = reranker if use_reranker else None
        self.rerank_alpha = settings.rerank_blend_alpha
        if question_index is None:
            from matchmind.knowledge_base.example_questions import QuestionIndex

            question_index = QuestionIndex.load(question_index_path(), list(self.concepts.values()))
        self.question_index = question_index

    # ── public ────────────────────────────────────────────────
    def retrieve(
        self,
        query: RetrievalQuery,
        k: int = 3,
        *,
        relax: bool = False,
        features: SituationFeatures | None = None,
        match_state: MatchState | None = None,
    ) -> RetrievalResult:
        strategy = "hybrid" + ("+rerank" if self.reranker else "") + ("+relaxed" if relax else "")
        if not self.concepts:
            return RetrievalResult(items=[], confidence=0.0, strategy=strategy)

        # RETRIEVAL_PHASE_FILTER=false turns possession gating off (set-piece /
        # technique scoping always applies — see scope.py).
        allowed = allowed_ids(
            self.concepts.values(), query, k, relax=relax or not settings.retrieval_phase_filter
        )
        subqueries = query.subqueries()
        if relax and query.situation_text and query.situation_text not in subqueries:
            subqueries.append(query.situation_text)

        lists: list[tuple[list[str], float]] = []
        names: list[str] = []          # parallel to `lists`, for the trace
        dense_sim: dict[str, float] = {}
        try:
            for n, ranked in enumerate(self.store.dense_rank(subqueries, n=self.pool_size * 2)):
                ids = [cid for cid, sim in ranked if cid in allowed]
                for cid, sim in ranked:
                    if cid in allowed:
                        dense_sim[cid] = max(dense_sim.get(cid, 0.0), sim)
                lists.append((ids, DENSE_WEIGHT))
                names.append(f"dense{n}")
        except Exception as exc:  # noqa: BLE001 — BM25 + conditions still work
            logger.warning(f"[hybrid] dense ranking unavailable: {exc}")

        for n, sq in enumerate(subqueries):
            ids = [cid for cid, _ in self.bm25.rank(sq, allowed=allowed, n=self.pool_size)]
            lists.append((ids, BM25_WEIGHT))
            names.append(f"bm25_{n}")

        if self.question_index is not None and hasattr(self.store, "embed_queries"):
            try:
                ranked_lists = self.question_index.rank(self.store.embed_queries(subqueries), n=self.pool_size * 2)
                for n, ranked in enumerate(ranked_lists):
                    lists.append(([cid for cid, _ in ranked if cid in allowed][: self.pool_size], QUESTION_WEIGHT))
                    names.append("questions" if n == 0 else f"questions{n}")
            except Exception as exc:  # noqa: BLE001 — the other rankers still work
                logger.warning(f"[hybrid] example-question ranking unavailable: {exc}")

        if features is not None and match_state is not None:
            cond = condition_ranking(
                (self.concepts[cid] for cid in allowed), features, match_state
            )
            lists.append((cond[: self.pool_size], CONDITION_WEIGHT))
            names.append("conditions")

        fused = reciprocal_rank_fusion(lists)[: self.pool_size]
        if not fused:
            # Nothing matched anywhere (e.g. untranslated question, no phrases):
            # fall back to the allowed concepts in a stable order.
            fused = [(cid, 0.0) for cid in sorted(allowed)[: self.pool_size]]
        items = [
            ScoredConcept(
                concept_id=cid, score=score, dense_sim=dense_sim.get(cid),
                sources=self._sources(cid, lists, names),
            )
            for cid, score in fused
        ]
        items = self._rerank(query, items)
        items = self._diversify(items, k)
        top = items[:k]
        return RetrievalResult(items=top, confidence=self._confidence(top, dense_sim), strategy=strategy)

    def rank_ids(self, query: RetrievalQuery, k: int) -> list[str]:
        return self.retrieve(query, k).concept_ids

    def to_legacy_dicts(self, result: RetrievalResult) -> list[dict]:
        out = []
        for item in result.items:
            c = self.concepts[item.concept_id]
            out.append({
                "concept_id": c.concept_id,
                "title": c.title,
                "content": c.content,
                "category": c.category,
                "subcategory": c.subcategory,
                "phase": c.phase,
                "definition": c.definition,
                "conditions": list(c.conditions),
                "possible_actions": list(c.possible_actions),
                "contraindications": list(c.contraindications),
                "similarity": round(
                    item.rerank_score if item.rerank_score is not None else (item.dense_sim or 0.0), 4
                ),
                "source_name": c.source.name if c.source else None,
                "source_url": c.source.url if c.source else None,
            })
        return out

    # ── stages (overridden in later tasks) ────────────────────
    def _rerank(self, query: RetrievalQuery, items: list[ScoredConcept]) -> list[ScoredConcept]:
        if self.reranker is None or not items:
            return items
        try:
            scores = self.reranker.score(
                query.rerank_text(), [rerank_document(self.concepts[i.concept_id]) for i in items]
            )
        except Exception as exc:  # noqa: BLE001 — keep fusion order
            logger.warning(f"[hybrid] reranker failed, keeping fusion order: {exc}")
            return items
        # Blend instead of replacing: RRF already encodes agreement between dense,
        # BM25 and condition matching. Alpha = settings.rerank_blend_alpha (see
        # config.py for the bench history; 0.3 on the ~100-concept KB, 0.5 since 331).
        top_fused = max(i.score for i in items) or 1.0
        rescored = [
            i.model_copy(update={
                "rerank_score": round(s, 4),
                "score": self.rerank_alpha * s + (1 - self.rerank_alpha) * (i.score / top_fused),
            })
            for i, s in zip(items, scores)
        ]
        return sorted(rescored, key=lambda i: -i.score)

    def _diversify(self, items: list[ScoredConcept], k: int) -> list[ScoredConcept]:
        if len(items) <= k or self.diversity_max_sim >= 1.0:
            return items
        try:
            emb = self.store.get_embeddings([i.concept_id for i in items])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[hybrid] embeddings unavailable, skipping diversity: {exc}")
            return items
        keep = select_diverse([i.concept_id for i in items], emb, k, self.diversity_max_sim)
        by_id = {i.concept_id: i for i in items}
        return [by_id[cid] for cid in keep]

    @staticmethod
    def _confidence(top: list[ScoredConcept], dense_sim: dict[str, float]) -> float:
        if not top:
            return 0.0
        if top[0].rerank_score is not None:
            return float(top[0].rerank_score)
        return float(max((dense_sim.get(i.concept_id, 0.0) for i in top), default=0.0))

    @staticmethod
    def _sources(cid: str, lists: list[tuple[list[str], float]], names: list[str]) -> list[str]:
        """Which rankers had this concept and at what rank, e.g. ["dense0:3", "bm25_0:1"]."""
        return [f"{name}:{ids.index(cid) + 1}" for (ids, _), name in zip(lists, names) if cid in ids]


def question_index_path():
    """Where build_kb stores the example-question vectors (next to the Chroma index)."""
    from pathlib import Path

    return Path(settings.chroma_persist_dir) / "example_questions.npz"


_retriever: HybridRetriever | None = None


def get_hybrid_retriever(use_reranker: bool = True) -> HybridRetriever:
    """Process-wide singleton (BM25 build + model loads happen once).

    Shares retrieve_node's TacticalVectorStore instead of building a second one
    (a second MiniLM load, and the agent's main-thread pre-warm would otherwise
    warm the wrong object).
    """
    global _retriever
    if _retriever is None:
        from matchmind.agent.nodes.retrieve_node import _get_vector_store  # lazy: avoids an import cycle

        _retriever = HybridRetriever(
            concepts=load_all_concepts(), store=_get_vector_store(), use_reranker=use_reranker
        )
    return _retriever
