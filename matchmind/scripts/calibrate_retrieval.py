"""
Calibrate the retrieval-confidence threshold.

The re-retrieve branch fires when mean top-k cosine similarity < threshold.
A hard-coded 0.5 is wrong for MiniLM on a ~30-doc KB. This script runs a set
of labelled probe queries, takes the 25th percentile of the top-1 similarity
of the "good" ones, and writes it to output/retrieval_calibration.json.
config.Settings.effective_retrieval_threshold picks it up automatically.

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
from matchmind.knowledge_base.embedder import TacticalEmbedder
from matchmind.knowledge_base.vector_store import TacticalVectorStore

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
BAD_PROBES = [
    "what is the offside rule in football",
    "history of the world cup trophy",
    "how to tie football boots correctly",
]


@app.command()
def main(
    percentile: float = typer.Option(25.0, "--percentile", help="Percentile of good top-1 sims"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    embedder = TacticalEmbedder()
    store = TacticalVectorStore()
    if store.count() == 0:
        console.print("[red]KB is empty — run build_kb first.[/red]")
        raise typer.Exit(1)

    def top1(q: str) -> float:
        results, _ = store.retrieve_tactics(embedder.embed_query(q), k=1)
        return results[0]["similarity"] if results else 0.0

    good = np.array([top1(q) for q in GOOD_PROBES])
    bad = np.array([top1(q) for q in BAD_PROBES])

    threshold = float(np.percentile(good, percentile))
    # Guard: keep it above the best off-domain score so junk still triggers re-retrieve.
    threshold = max(threshold, float(bad.max()) + 0.01)
    threshold = round(min(max(threshold, 0.1), 0.9), 3)

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
