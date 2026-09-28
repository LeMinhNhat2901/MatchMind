"""
Compare retrieval strategies on the golden set.

Usage:
    python -m matchmind.scripts.bench_retrieval                       # all strategies
    python -m matchmind.scripts.bench_retrieval --strategy dense
    python -m matchmind.scripts.bench_retrieval --fail-under hit3=0.8,ndcg5=0.7
Writes output/evaluation/retrieval_bench.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from matchmind.config import settings
from matchmind.evaluation.retrieval_bench import GoldenCase, load_golden, run_bench

app = typer.Typer()
console = Console()


def _dense_strategy() -> Callable[[GoldenCase, int], list[str]]:
    from matchmind.knowledge_base.vector_store import TacticalVectorStore

    store = TacticalVectorStore(use_chroma_ef=True)

    def rank(case: GoldenCase, k: int) -> list[str]:
        results, _ = store.retrieve_tactics(query_text=case.question, k=k)
        return [r["concept_id"] for r in results]

    return rank


def _hybrid_strategy(use_reranker: bool) -> Callable[[], Callable[[GoldenCase, int], list[str]]]:
    def factory():
        from matchmind.retrieval.hybrid import HybridRetriever
        from matchmind.retrieval.types import RetrievalQuery

        retriever = HybridRetriever(use_reranker=use_reranker)

        def rank(case: GoldenCase, k: int) -> list[str]:
            q = RetrievalQuery(question=case.question, in_possession=case.in_possession,
                               phase_of_play=case.phase_of_play)
            return retriever.rank_ids(q, k)

        return rank

    return factory


STRATEGIES: dict[str, Callable[[], Callable[[GoldenCase, int], list[str]]]] = {
    "dense": _dense_strategy,
    "hybrid_norerank": _hybrid_strategy(False),
    "hybrid": _hybrid_strategy(True),
}


def _parse_fail_under(spec: str) -> dict[str, float]:
    out = {}
    for part in filter(None, spec.split(",")):
        key, value = part.split("=")
        out[key.strip()] = float(value)
    return out


@app.command()
def main(
    strategy: str = typer.Option("", help="One strategy name; empty = all"),
    k: int = typer.Option(5),
    fail_under: str = typer.Option("", help="e.g. hit3=0.8,ndcg5=0.7 (applies to every strategy run)"),
) -> None:
    cases = load_golden()
    names = [strategy] if strategy else list(STRATEGIES)
    table = Table(title=f"Retrieval bench — {len(cases)} golden cases")
    for col in ("strategy", "hit@3", "nDCG@5", "MRR", "must_not", "set-piece leak"):
        table.add_column(col)
    results, failed = {}, []
    thresholds = _parse_fail_under(fail_under)
    for name in names:
        report = run_bench(STRATEGIES[name](), cases, k=k)
        results[name] = report.model_dump()
        table.add_row(name, str(report.hit_at_3), str(report.ndcg_at_5), str(report.mrr),
                      str(report.must_not_violation_rate), str(report.set_piece_leak_rate))
        if "hit3" in thresholds and (report.hit_at_3 or 0) < thresholds["hit3"]:
            failed.append(f"{name}: hit@3 {report.hit_at_3} < {thresholds['hit3']}")
        if "ndcg5" in thresholds and (report.ndcg_at_5 or 0) < thresholds["ndcg5"]:
            failed.append(f"{name}: nDCG@5 {report.ndcg_at_5} < {thresholds['ndcg5']}")
    console.print(table)
    out = settings.evaluation_output_dir / "retrieval_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"saved -> {out}")
    if failed:
        for f in failed:
            console.print(f"[red]{f}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
