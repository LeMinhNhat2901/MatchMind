"""
Retrieval benchmark over the hand-labelled golden set.

Primary retrieval metric from V1 on (the trigger_conditions proxy in
retrieval_quality.py becomes partly circular once condition matching is a
ranking signal). Pure functions + a runner that takes any
rank_fn(case, k) -> ranked concept ids, so dense / hybrid / rerank variants
are compared on exactly the same cases.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel, Field

GOLDEN_PATH = Path(__file__).parent / "golden" / "retrieval_golden.yaml"

try:
    from matchmind.retrieval.scope import SET_PIECE_TOKENS
except ImportError:  # Task 3 runs before the retrieval package exists
    SET_PIECE_TOKENS = ("corner", "free_kick", "penalty", "kick_off", "goal_kick", "set_piece", "throw_in")


class GoldenCase(BaseModel):
    id: str
    question: str
    in_possession: bool | None = None
    phase_of_play: str | None = None
    relevant: list[str] = Field(default_factory=list)
    acceptable: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    # True when the relevant concept was written after seeing this question (a KB
    # coverage gap being filled): counts for coverage, never for retrieval scores.
    coverage_only: bool = False


class BenchReport(BaseModel):
    n_cases: int
    hit_at_3: float | None
    ndcg_at_5: float | None
    mrr: float | None
    must_not_violation_rate: float
    set_piece_leak_rate: float
    per_case: list[dict]
    recall_at_5: float | None = None


class LayerReport(BaseModel):
    """The two layers below retrieval quality: is there an answer in the KB at
    all (coverage), and does the scope filter let the labelled answers through?"""
    coverage_rate: float
    uncovered: list[str]            # case ids with no relevant concept in the KB
    filter_recall: float | None     # labelled concepts (in the KB) the scope filter keeps
    filtered_out: list[str]         # "case:concept" pairs the filter wrongly blocks


def load_golden(path: Path | None = None) -> list[GoldenCase]:
    raw = yaml.safe_load((path or GOLDEN_PATH).read_text(encoding="utf-8"))
    return [GoldenCase(**row) for row in raw]


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def ndcg_at_k(ranked: list[str], gains: dict[str, float], k: int) -> float:
    dcg = sum(gains.get(cid, 0.0) / math.log2(i + 2) for i, cid in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def layer_report(
    cases: list[GoldenCase],
    kb_ids: set[str],
    allowed_for: Callable[[GoldenCase], set[str]],
) -> LayerReport:
    """KB coverage and filter correctness, independent of how anything is ranked.
    A case is covered when at least one of its `relevant` concepts exists in the KB."""
    uncovered, blocked, kept, total = [], [], 0, 0
    for case in cases:
        if not set(case.relevant) & kb_ids:
            uncovered.append(case.id)
        labelled = [cid for cid in dict.fromkeys(case.relevant + case.acceptable) if cid in kb_ids]
        if not labelled:
            continue
        allowed = allowed_for(case)
        for cid in labelled:
            total += 1
            if cid in allowed:
                kept += 1
            else:
                blocked.append(f"{case.id}:{cid}")
    n = len(cases)
    return LayerReport(
        coverage_rate=round((n - len(uncovered)) / n, 4) if n else 0.0,
        uncovered=uncovered,
        filter_recall=round(kept / total, 4) if total else None,
        filtered_out=blocked,
    )


def mrr(ranked: list[str], targets: set[str]) -> float:
    for i, cid in enumerate(ranked):
        if cid in targets:
            return 1.0 / (i + 1)
    return 0.0


def _is_set_piece(concept_id: str) -> bool:
    return any(tok in concept_id for tok in SET_PIECE_TOKENS)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def run_bench(
    rank_fn: Callable[[GoldenCase, int], list[str]],
    cases: list[GoldenCase],
    k: int = 5,
) -> BenchReport:
    per_case, hits, ndcgs, mrrs, recalls = [], [], [], [], []
    violations = leaks = 0
    for case in cases:
        ranked = list(rank_fn(case, k))[:k]
        row: dict = {"id": case.id, "ranked": ranked}
        if case.relevant and not case.coverage_only:
            gains = {cid: 1.0 for cid in case.acceptable} | {cid: 2.0 for cid in case.relevant}
            row["hit_at_3"] = hit_at_k(ranked, set(case.relevant), 3)
            row["ndcg_at_5"] = round(ndcg_at_k(ranked, gains, 5), 4)
            row["mrr"] = round(mrr(ranked, set(case.relevant) | set(case.acceptable)), 4)
            row["recall_at_5"] = round(recall_at_k(ranked, set(case.relevant), 5), 4)
            recalls.append(row["recall_at_5"])
            hits.append(row["hit_at_3"])
            ndcgs.append(row["ndcg_at_5"])
            mrrs.append(row["mrr"])
        bad = sorted(set(ranked) & set(case.must_not))
        if bad:
            violations += 1
            row["must_not_hit"] = bad
        if case.phase_of_play != "set_piece" and any(_is_set_piece(cid) for cid in ranked):
            leaks += 1
            row["set_piece_leak"] = True
        per_case.append(row)
    n = len(cases)
    return BenchReport(
        n_cases=n,
        hit_at_3=_mean(hits),
        ndcg_at_5=_mean(ndcgs),
        mrr=_mean(mrrs),
        must_not_violation_rate=round(violations / n, 4) if n else 0.0,
        set_piece_leak_rate=round(leaks / n, 4) if n else 0.0,
        per_case=per_case,
        recall_at_5=_mean(recalls),
    )
