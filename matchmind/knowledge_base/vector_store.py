"""
ChromaDB Vector Store — sử dụng ChromaDB's built-in embedding function.

Thay vì dùng custom TacticalEmbedder (gây crash ONNX Runtime trên Windows
khi chạy trong LangGraph thread context), dùng ChromaDB's own
SentenceTransformerEmbeddingFunction — ChromaDB handle threading an toàn hơn.

Đồng thời giữ TacticalEmbedder cho build_kb script (single-process, stable).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from matchmind.config import settings
from matchmind.schema.match_state import TacticalConcept

logger = logging.getLogger(__name__)

# chromadb 0.5.x + posthog>=6 spam ERROR logs on every call
# ("capture() takes 1 positional argument but 3 were given"). Opt out + mute.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


def _get_chroma_ef():
    """Get ChromaDB's built-in SentenceTransformer embedding function."""
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
    except Exception as exc:
        logger.warning(f"ChromaDB EF unavailable ({exc}), falling back to raw embeddings")
        return None


class TacticalVectorStore:
    """
    ChromaDB-backed vector store for football tactical concepts.

    Uses ChromaDB's built-in SentenceTransformerEmbeddingFunction for
    query-time embedding — avoids ONNX Runtime threading crashes on Windows.

    Build-time (upsert) still accepts pre-computed numpy embeddings.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        use_chroma_ef: bool = True,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        self._persist_dir = str(persist_dir or settings.chroma_persist_dir)
        self._collection_name = collection_name or settings.chroma_collection_name
        # anonymized_telemetry=False: chromadb 0.5.x ships a posthog client that
        # breaks with posthog>=6 ("capture() takes 1 positional argument").
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Use ChromaDB's EF for query — handles Windows threading safely
        self._ef = _get_chroma_ef() if use_chroma_ef else None

        # Collection with cosine similarity
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,  # None = raw embeddings mode
        )

        logger.info(
            f"ChromaDB '{self._collection_name}' at {self._persist_dir} "
            f"— {self._collection.count()} docs | EF: {'chroma_ef' if self._ef else 'manual'}"
        )

    # ──────────────────────────────────────────────────────────
    # Write operations
    # ──────────────────────────────────────────────────────────

    def upsert_concepts(
        self,
        concepts: list[TacticalConcept],
        embeddings: np.ndarray | None = None,
    ) -> None:
        """
        Upsert tactical concepts.

        If embeddings is None and use_chroma_ef=True, ChromaDB will auto-embed.
        If embeddings provided, uses those pre-computed vectors.
        """
        if len(concepts) == 0:
            return

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

        if embeddings is not None:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=docs,
                metadatas=metas,
            )
        else:
            # Let ChromaDB EF embed automatically
            self._collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
            )

        logger.info(f"Upserted {len(concepts)} concepts into ChromaDB")

    def delete_all(self) -> None:
        """Clear and recreate the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        logger.warning(f"Deleted and recreated collection '{self._collection_name}'")

    # ──────────────────────────────────────────────────────────
    # Read operations
    # ──────────────────────────────────────────────────────────

    def retrieve_tactics(
        self,
        query_embedding: np.ndarray | None = None,
        query_text: str | None = None,
        k: int | None = None,
        filter_category: str | None = None,
    ) -> tuple[list[dict], float]:
        """
        Retrieve top-k concepts.

        Accepts either:
        - query_embedding: pre-computed numpy vector (build_kb path)
        - query_text: raw text (agent path — ChromaDB EF embeds it safely)
        """
        k = k or settings.retrieval_top_k
        n = min(k, self._collection.count())
        if n == 0:
            return [], 0.0

        where_filter: dict[str, Any] | None = (
            {"category": {"$eq": filter_category}} if filter_category else None
        )

        if query_text is not None and self._ef is not None:
            # Let ChromaDB embed the query text — thread-safe on Windows
            query_results = self._collection.query(
                query_texts=[query_text],
                n_results=n,
                include=["documents", "metadatas", "distances"],
                where=where_filter,
            )
        elif query_embedding is not None:
            query_results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n,
                include=["documents", "metadatas", "distances"],
                where=where_filter,
            )
        else:
            logger.warning("No query_embedding or query_text provided")
            return [], 0.0

        if not query_results["ids"] or not query_results["ids"][0]:
            return [], 0.0

        ids = query_results["ids"][0]
        docs = query_results["documents"][0]
        metas = query_results["metadatas"][0]
        distances = query_results["distances"][0]

        similarities = [max(0.0, 1.0 - d / 2.0) for d in distances]
        confidence = float(np.mean(similarities)) if similarities else 0.0

        results = [
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
            for cid, doc, meta, sim in zip(ids, docs, metas, similarities)
        ]

        logger.debug(f"Retrieved {len(results)} concepts, confidence={confidence:.3f}")
        return results, confidence

    def retrieve_by_query_text(
        self,
        query_text: str,
        embedder=None,
        k: int | None = None,
        filter_category: str | None = None,
    ) -> tuple[list[dict], float]:
        """
        Retrieve by text — prefers ChromaDB EF, falls back to manual embedder.
        """
        if self._ef is not None:
            return self.retrieve_tactics(query_text=query_text, k=k, filter_category=filter_category)
        elif embedder is not None:
            emb = embedder.embed_query(query_text)
            return self.retrieve_tactics(query_embedding=emb, k=k, filter_category=filter_category)
        else:
            raise ValueError("Need either ChromaDB EF or an embedder to query by text")

    def get_all_titles(self) -> list[str]:
        results = self._collection.get(include=["metadatas"])
        return [m.get("title", "") for m in results["metadatas"]]

    def get_all_concept_ids(self) -> list[str]:
        return self._collection.get()["ids"]

    def count(self) -> int:
        return self._collection.count()
