"""
Cross-encoder reranking of the fused candidate pool.

A bi-encoder (MiniLM) embeds query and concept separately; a cross-encoder
reads the (query, concept) pair together, which is much better at telling
"third-man run" from "one-two" for a specific question. It is only affordable
on a small pool (settings.retrieval_pool_size, default 20), which is exactly
what fusion produces.

The model is lazy-loaded on first use. If loading fails (offline first run,
HF outage) the error propagates to HybridRetriever._rerank, which logs it and
keeps the fusion order — reranking is an improvement, never a dependency.
"""
from __future__ import annotations

import logging
import math
from typing import Protocol

from matchmind.schema.match_state import TacticalConcept

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]: ...


def rerank_document(c: TacticalConcept) -> str:
    body = c.definition or c.content[:600]
    return f"{c.title}. {body}"


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
        scores = self._model.predict([(query, d) for d in docs], convert_to_numpy=True)
        # sentence-transformers 3.4 returns raw logits for ms-marco (measured:
        # -3.07 vs -11.38). Squash to [0, 1] so the top score can serve as
        # retrieval_confidence against a calibrated threshold; order is preserved.
        return [1.0 / (1.0 + math.exp(-float(s))) for s in scores]


def build_reranker(model_name: str) -> Reranker | None:
    return CrossEncoderReranker(model_name) if model_name.strip() else None
