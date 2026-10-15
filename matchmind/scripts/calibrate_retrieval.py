"""
Calibrate the retrieval-confidence threshold.

The re-retrieve branch fires when retrieval confidence < threshold. This
script runs labelled probe queries through the HybridRetriever and puts the
threshold in the middle of the gap between the weakest on-domain probe and the
strongest off-domain one (see choose_threshold), writing it to
output/retrieval_calibration.json, which
config.Settings.effective_retrieval_threshold picks up automatically.

Usage:
    python -m matchmind.scripts.build_kb --rebuild        # KB must exist first
    python -m matchmind.scripts.calibrate_retrieval
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import typer
from rich.console import Console

from matchmind.config import settings

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)

# Probe queries that SHOULD retrieve something relevant (good) vs. off-domain (bad).
GOOD_PROBES = [
    "wide player has the ball, full-back overlapping on the outside to create 2v1",
    "striker drops between the lines to receive in the pocket of space",
    "opponent presses high, quick vertical pass to break the first line",
    "numerical overload on the right flank, rotate the ball to exploit it",
    "third-man combination through a nearby midfielder into space behind",
    "switch the play to the free winger on the opposite flank",
    "counter-attack with space behind a high defensive line",
    "low block, need to create space by stretching the defense wide",
    "half-space is free, carry into it and threaten the back line",
    "recover the second ball after a long clearance in midfield",
    "man-mark the opponent's key playmaker out of the game",
    "gegenpress immediately after losing possession in the final third",
]
# Off-domain only. "what is the offside rule" used to be here, but with the
# reranker it (correctly) retrieves Offside Trap at 0.98 and pushed the
# threshold so high that almost every agent query re-retrieved.
BAD_PROBES = [
    "history of the world cup trophy",
    "who won the ballon d'or last year",
    "how to tie football boots correctly",
]


def choose_threshold(good: list[float], bad: list[float], percentile: float = 25.0) -> float:
    """Midpoint of the gap between the weakest good probe and the strongest bad
    one. With a reranker, good probes cluster near 1.0 and junk near 0, so a
    percentile of the good scores alone (the pre-V1 rule) lands near 1.0 and
    makes real agent queries (typically 0.6-0.99) re-retrieve every time. If the
    two sets overlap, fall back to the old percentile-with-guard rule."""
    good_a, bad_a = np.asarray(good, dtype=float), np.asarray(bad, dtype=float)
    if good_a.min() > bad_a.max():
        threshold = (float(good_a.min()) + float(bad_a.max())) / 2.0
    else:
        threshold = max(float(np.percentile(good_a, percentile)), float(bad_a.max()) + 0.01)
    return round(min(max(threshold, 0.1), 0.9), 3)


@app.command()
def main(
    percentile: float = typer.Option(25.0, "--percentile", help="Percentile of good top-1 sims"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    from matchmind.retrieval.hybrid import HybridRetriever
    from matchmind.retrieval.types import RetrievalQuery

    retriever = HybridRetriever()
    if not retriever.concepts:
        console.print("[red]KB is empty — run build_kb first.[/red]")
        raise typer.Exit(1)

    def top1(q: str) -> float:
        # Same unit the agent compares against: HybridRetriever confidence
        # (sigmoid rerank score of the top hit, or its dense similarity).
        return retriever.retrieve(RetrievalQuery(question=q), k=1).confidence

    good = np.array([top1(q) for q in GOOD_PROBES])
    bad = np.array([top1(q) for q in BAD_PROBES])

    threshold = choose_threshold(list(good), list(bad), percentile)

    out = settings.retrieval_calibration_file
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "percentile": percentile,
                "good_top1_mean": round(float(good.mean()), 3),
                "good_top1_min": round(float(good.min()), 3),
                "bad_top1_max": round(float(bad.max()), 3),
                "embedding_model": settings.embedding_model,
                "strategy": "hybrid",
                "reranker_model": settings.reranker_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(
        f"good top-1 sim: mean={good.mean():.3f} min={good.min():.3f} | "
        f"bad top-1 max={bad.max():.3f}"
    )
    console.print(f"[green]Calibrated RETRIEVAL_CONFIDENCE_THRESHOLD = {threshold} -> {out}[/green]")


if __name__ == "__main__":
    app()
