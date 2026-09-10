"""
Tactical Knowledge Base — Embedder.

Loads all concept JSON files from the concepts/ directory,
embeds them using SentenceTransformer, and returns embeddings
ready for ChromaDB upsert.

Embedding model: all-MiniLM-L6-v2 (English, fast, 384-dim)
Alternative for multilingual: intfloat/multilingual-e5-base
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from matchmind.config import settings
from matchmind.schema.match_state import TacticalConcept

logger = logging.getLogger(__name__)


class TacticalEmbedder:
    """
    Embeds tactical concepts using SentenceTransformer.

    Lazy-loads the model on first call to avoid import-time GPU initialisation.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._model = None  # lazy load

    @property
    def model(self):
        """Lazy-load SentenceTransformer on first access."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_concept(self, concept: TacticalConcept) -> np.ndarray:
        """Embed a single tactical concept → 1D numpy array."""
        text = self._concept_to_text(concept)
        return self.model.encode([text], normalize_embeddings=True)[0]

    def embed_concepts(self, concepts: list[TacticalConcept]) -> np.ndarray:
        """
        Batch embed a list of concepts.

        Returns 2D numpy array of shape (len(concepts), embedding_dim).
        """
        texts = [self._concept_to_text(c) for c in concepts]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        logger.info(f"Embedded {len(concepts)} concepts, dim={embeddings.shape[1]}")
        return embeddings

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Embed a natural-language query for retrieval.

        Uses the same model as concept embedding for cosine similarity to work correctly.
        """
        return self.model.encode([query_text], normalize_embeddings=True)[0]

    @staticmethod
    def _concept_to_text(concept: TacticalConcept) -> str:
        """Convert a concept to a single text string for embedding."""
        parts = [f"{concept.title}.", concept.content]
        if concept.conditions:
            parts.append(f"Conditions: {', '.join(concept.conditions)}.")
        if concept.objective:
            parts.append(f"Objectives: {', '.join(concept.objective)}.")
        if concept.possible_actions:
            parts.append(f"Actions: {', '.join(concept.possible_actions)}.")
        return " ".join(parts)


def load_all_concepts(
    concepts_dir: Path | None = None,
) -> list[TacticalConcept]:
    """
    Load all JSON concept files from the concepts/ directory tree.

    Searches recursively for .json files in attacking/, defensive/, spatial/ subdirs.
    """
    if concepts_dir is None:
        concepts_dir = settings.knowledge_base_dir

    concepts_dir = Path(concepts_dir)
    if not concepts_dir.exists():
        raise FileNotFoundError(f"Concepts directory not found: {concepts_dir}")

    concepts = []
    json_files = sorted(concepts_dir.rglob("*.json"))

    logger.info(f"Found {len(json_files)} concept files in {concepts_dir}")

    for fpath in json_files:
        try:
            with open(fpath, encoding="utf-8-sig") as f:  # concept files carry a BOM
                data = json.load(f)
            concept = TacticalConcept(**data)
            concepts.append(concept)
        except Exception as exc:
            logger.warning(f"Failed to load concept from {fpath}: {exc}")

    logger.info(f"Successfully loaded {len(concepts)} concepts")
    return concepts
