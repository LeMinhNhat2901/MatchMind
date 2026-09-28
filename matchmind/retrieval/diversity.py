"""Greedy near-duplicate suppression over an already-ranked list.

Walk the ranking; skip a candidate whose cosine similarity to an already
selected one is >= max_sim. If that leaves fewer than k, fill back with the
skipped ones in their original order (k results beat "diverse but short").
"""
from __future__ import annotations

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na and nb else 0.0


def select_diverse(
    ordered_ids: list[str],
    embeddings: dict[str, np.ndarray],
    k: int,
    max_sim: float,
) -> list[str]:
    chosen: list[str] = []
    skipped: list[str] = []
    for cid in ordered_ids:
        if len(chosen) >= k:
            break
        vec = embeddings.get(cid)
        if vec is not None and any(
            _cos(vec, embeddings[c]) >= max_sim for c in chosen if c in embeddings
        ):
            skipped.append(cid)
            continue
        chosen.append(cid)
    for cid in skipped:
        if len(chosen) >= k:
            break
        chosen.append(cid)
    return chosen
