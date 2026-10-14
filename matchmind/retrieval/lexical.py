"""
BM25 over the tactical KB (rank-bm25).

Why lexical at all: the KB is small (≈110 short docs) and full of exact
football terms ("third-man", "gegenpress", "half-space", "low block") that a
384-dim MiniLM embedding blurs together. BM25 catches exact-term matches the
dense model ranks 6th-10th; RRF (fusion.py) then merges both views.

The tokenizer is intentionally crude (lowercase, ASCII words, tiny stopword
list, strip -ing/-ed/-es/-s): it only has to be consistent between documents
and queries, and it is fully covered by test_lexical.py.
"""
from __future__ import annotations

import re

from matchmind.schema.match_state import TacticalConcept

_STOPWORDS = frozenset(
    "a an the and or of to in on at for with from by is are be was this that it its as "
    "into than then should what how when where which who do does can could would will "
    "player players team his her their he she they we our you your there here".split()
)


def _stem(token: str) -> str:
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower().replace("_", " "))
    return [_stem(w) for w in words if w not in _STOPWORDS and len(w) > 1]


def concept_document(c: TacticalConcept) -> str:
    """Title x3 (a title match is the strongest lexical signal), then the rest."""
    fields = [c.title] * 3 + [
        c.subcategory or "",
        c.definition or "",
        c.content,
        " ".join(c.conditions + c.trigger_conditions + c.possible_actions + c.player_behaviours),
        " ".join(c.example_questions),
    ]
    return " ".join(fields)


class BM25Index:
    def __init__(self, concepts: list[TacticalConcept]) -> None:
        from rank_bm25 import BM25Okapi

        self.ids = [c.concept_id for c in concepts]
        corpus = [tokenize(concept_document(c)) or ["_empty_"] for c in concepts]
        self._bm25 = BM25Okapi(corpus)

    def rank(
        self, query: str, allowed: set[str] | None = None, n: int = 20
    ) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        pairs = [
            (cid, float(score))
            for cid, score in zip(self.ids, scores)
            if score > 0 and (allowed is None or cid in allowed)
        ]
        pairs.sort(key=lambda p: -p[1])
        return pairs[:n]
