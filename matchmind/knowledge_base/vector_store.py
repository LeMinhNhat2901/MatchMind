"""
ChromaDB Vector Store — Tactical Knowledge Base Storage and Retrieval.

Wraps ChromaDB with our tactical domain logic:
- Upsert concepts with embeddings and metadata
- Retrieve top-k most similar concepts to a query
- Compute retrieval confidence (mean cosine similarity)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from matchmind.config import settings
from matchmind.schema.match_state import TacticalConcept

logger = logging.getLogger(__name__)


class TacticalVectorStore:
    """
    ChromaDB-backed vector store for football tactical concepts.

    Uses cosine similarity (via normalized embeddings + inner product).
    Persistence: concepts survive process restarts.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        self._persist_dir = str(persist_dir or settings.chroma_persist_dir)
        self._collection_name = collection_name or settings.chroma_collection_name

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{self._collection_name}' "
            f"at {self._persist_dir} — {self._collection.count()} docs"
        )

    # ──────────────────────────────────────────────────────────
    # Write operations
    # ──────────────────────────────────────────────────────────

    def upsert_concepts(
        self,
        concepts: list[TacticalConcept],
        embeddings: np.ndarray,
    ) -> None:
        """
        Upsert (insert or update) tactical concepts into the vector store.

        Args:
            concepts: List of TacticalConcept objects.
            embeddings: 2D numpy array, shape (len(concepts), embedding_dim).
        """
        if len(concepts) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(concepts)} concepts but {len(embeddings)} embeddings"
            )

        ids = [c.concept_id for c in concepts]
        docs = [c.content for c in concepts]
        metas = [
            {
                "title": c.title,
                "category": c.category,
                "phase": c.phase,
                "conditions": ",".join(c.conditions),
                "objective": ",".join(c.objective),
                "possible_actions": ",".join(c.possible_actions),
                "contraindications": ",".join(c.contraindications),
                "trigger_conditions": ",".join(c.trigger_conditions),
            }
            for c in concepts
        ]
        emb_list = embeddings.tolist()

        self._collection.upsert(
            ids=ids,
            embeddings=emb_list,
            documents=docs,
            metadatas=metas,
        )
        logger.info(f"Upserted {len(concepts)} concepts into ChromaDB")

    def delete_all(self) -> None:
        """Clear the entire collection (useful for rebuilding KB)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Deleted and recreated collection '{self._collection_name}'")

    # ──────────────────────────────────────────────────────────
    # Read operations
    # ──────────────────────────────────────────────────────────

    def retrieve_tactics(
        self,
        query_embedding: np.ndarray,
        k: int | None = None,
        filter_category: str | None = None,
        filter_phase: str | None = None,
    ) -> tuple[list[dict], float]:
        """
        Retrieve the top-k most relevant tactical concepts for a query embedding.

        Args:
            query_embedding: 1D normalized embedding vector.
            k: Number of results. Defaults to settings.retrieval_top_k.
            filter_category: Optional ChromaDB where filter on category metadata.
            filter_phase: Optional phase pre-filter. Matches concepts whose
                ``phase`` is this value OR ``"any"``. Falls back to an
                unfiltered query if the phase filter returns nothing.

        Returns:
            (results, confidence)
            - results: list of dicts with keys: concept_id, title, content, category, metadata
            - confidence: mean cosine similarity of top-k results (0–1)
        """
        k = k or settings.retrieval_top_k

        clauses: list[dict[str, Any]] = []
        if filter_category:
            clauses.append({"category": {"$eq": filter_category}})
        if filter_phase:
            clauses.append({"phase": {"$in": [filter_phase, "any"]}})

        where_filter: dict[str, Any] | None
        if len(clauses) > 1:
            where_filter = {"$and": clauses}
        elif clauses:
            where_filter = clauses[0]
        else:
            where_filter = None

        query_results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        # Phase filter can be too strict on a small KB — retry without it.
        if filter_phase and (not query_results["ids"] or not query_results["ids"][0]):
            logger.debug("Phase filter '%s' returned nothing — retrying unfiltered", filter_phase)
            return self.retrieve_tactics(query_embedding, k=k, filter_category=filter_category)

        if not query_results["ids"] or not query_results["ids"][0]:
            logger.warning("No results from ChromaDB query")
            return [], 0.0

        ids = query_results["ids"][0]
        docs = query_results["documents"][0]
        metas = query_results["metadatas"][0]
        distances = query_results["distances"][0]  # cosine distance (0=identical, 2=opposite)

        # Convert cosine distance → similarity score [0,1]
        similarities = [max(0.0, 1.0 - d / 2.0) for d in distances]
        confidence = float(np.mean(similarities)) if similarities else 0.0

        results = []
        for cid, doc, meta, sim in zip(ids, docs, metas, similarities):
            results.append(
                {
                    "concept_id": cid,
                    "title": meta.get("title", cid),
                    "content": doc,
                    "category": meta.get("category", ""),
                    "phase": meta.get("phase", ""),
                    "conditions": meta.get("conditions", "").split(","),
                    "possible_actions": meta.get("possible_actions", "").split(","),
                    "contraindications": meta.get("contraindications", "").split(","),
                    "similarity": round(sim, 4),
                }
            )

        logger.debug(
            f"Retrieved {len(results)} concepts, confidence={confidence:.3f}"
        )
        return results, confidence

    def get_all_titles(self) -> list[str]:
        """
        Return all concept titles in the collection.

        Used for anti-hallucination validation: cited_concepts MUST be subset of these.
        """
        results = self._collection.get(include=["metadatas"])
        return [m.get("title", "") for m in results["metadatas"]]

    def get_all_concept_ids(self) -> list[str]:
        """Return all concept IDs."""
        results = self._collection.get()
        return results["ids"]

    def count(self) -> int:
        """Number of concepts in the store."""
        return self._collection.count()

    def retrieve_by_query_text(
        self,
        query_text: str,
        embedder,
        k: int | None = None,
        filter_category: str | None = None,
    ) -> tuple[list[dict], float]:
        """
        Convenience method: embed query text on-the-fly then retrieve.

        Args:
            query_text: Natural language query string.
            embedder: TacticalEmbedder instance.
            k: Top-k results.
            filter_category: Optional category filter.
        """
        query_emb = embedder.embed_query(query_text)
        return self.retrieve_tactics(query_emb, k=k, filter_category=filter_category)
