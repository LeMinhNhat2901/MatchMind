"""
Near-duplicate gate for KB ingestion.

Many channels explain the same tactic, and concept synthesis writes a new
concept each time, so without a gate the KB fills with near-copies (one scan
found ~9 inverted-full-back concepts). Retrieval then returns five versions of
one idea and pushes everything else out of the top-k.

The gate embeds "title. content" and skips a new concept whose cosine
similarity to an existing (or same-run) concept is >= DEFAULT_THRESHOLD
(0.90). Reading pairs from the real KB with all-MiniLM-L6-v2: 0.88+ were
mostly near-copies ("Inverted Full-Back Attacking Shape" /
"Inverted Fullback System"), 0.84–0.88 were related but distinct ideas
("Low Block Counter-Attack" / "Narrow Block Counter-Strategy"). 0.88–0.90 also
held a few distinct pairs (3-4-3 vs 4-3-3 structures), so the user chose 0.90:
missing an occasional near-copy beats losing a distinct idea.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from matchmind.schema.match_state import TacticalConcept

DEFAULT_THRESHOLD = 0.90

Embed = Callable[[list[str]], np.ndarray]


def _text(c: TacticalConcept) -> str:
    return f"{c.title}. {c.content}"


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class DuplicateGate:
    def __init__(self, concepts: list[TacticalConcept], embed: Embed,
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        self.embed = embed
        self.threshold = threshold
        self._ids = [c.concept_id for c in concepts]
        self._vecs = _normalise(embed([_text(c) for c in concepts])) if concepts else np.zeros((0, 0))

    @classmethod
    def from_kb(cls, concepts_dir: Path | str, threshold: float = DEFAULT_THRESHOLD) -> "DuplicateGate":
        """Gate over every concept already on disk, using the KB's embedding model."""
        from sentence_transformers import SentenceTransformer

        from matchmind.config import settings
        from matchmind.knowledge_base.embedder import load_all_concepts

        model = SentenceTransformer(settings.embedding_model)
        embed = lambda texts: model.encode(texts, normalize_embeddings=True)  # noqa: E731
        return cls(load_all_concepts(concepts_dir), embed, threshold)

    def check(self, concept: TacticalConcept,
              pending: list[TacticalConcept] | None = None) -> tuple[str, float] | None:
        """(existing concept_id, similarity) when `concept` duplicates something
        in the KB or in `pending` (accepted earlier in this run, not saved yet)."""
        pending = pending or []
        v = _normalise(self.embed([_text(concept)] + [_text(p) for p in pending]))
        ids = self._ids + [p.concept_id for p in pending]
        vecs = v[1:] if not len(self._ids) else (np.vstack([self._vecs, v[1:]]) if pending else self._vecs)
        if not len(ids):
            return None
        sims = vecs @ v[0]
        best = int(np.argmax(sims))
        return (ids[best], float(sims[best])) if sims[best] >= self.threshold else None

    def add(self, concept: TacticalConcept) -> None:
        v = _normalise(self.embed([_text(concept)]))
        self._vecs = v if not len(self._ids) else np.vstack([self._vecs, v])
        self._ids.append(concept.concept_id)
