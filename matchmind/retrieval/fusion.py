"""Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = Σ_lists  weight / (k0 + rank_d)   (rank starts at 1)

Rank-based, so it needs no score normalisation between BM25 (unbounded),
cosine similarity and trigger fractions. k0=60 is the paper's default.
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[str], float]],
    k0: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ids, weight in ranked_lists:
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + weight / (k0 + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
