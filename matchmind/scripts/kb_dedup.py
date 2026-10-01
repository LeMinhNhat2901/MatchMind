"""
Review and archive near-duplicate concepts already in the knowledge base.

The ingest-time gate (knowledge_base/dedup.py) stops NEW near-copies; this
script finds the ones already in the KB. Nothing is ever deleted: archived
concepts move to knowledge_base/concepts/_archive/ (ignored by every loader)
and can be restored.

    python -m matchmind.scripts.kb_dedup                       # list clusters (read-only)
    python -m matchmind.scripts.kb_dedup --threshold 0.92      # stricter: only very close copies
    python -m matchmind.scripts.kb_dedup --archive yt_a yt_b   # archive these ids, rebuild the index
    python -m matchmind.scripts.kb_dedup --archive-suggested   # archive every non-keeper listed
    python -m matchmind.scripts.kb_dedup --restore yt_a        # bring one back, rebuild the index
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matchmind.config import settings
from matchmind.knowledge_base.dedup import (
    DEFAULT_THRESHOLD,
    archive_concepts,
    find_clusters,
    restore_concepts,
)

app = typer.Typer(add_completion=False)
console = Console()


def _rebuild() -> None:
    console.print("\n[bold]Rebuilding the vector index...[/bold]")
    subprocess.run([sys.executable, "-m", "matchmind.scripts.build_kb", "--rebuild"], check=False)


@app.command()
def main(
    threshold: float = typer.Option(DEFAULT_THRESHOLD, help="Cosine similarity that counts as a near-copy"),
    archive: list[str] = typer.Option(None, "--archive", help="Concept ids to archive (repeat or space-separate)"),
    archive_suggested: bool = typer.Option(False, help="Archive every non-keeper in the listed clusters"),
    restore: list[str] = typer.Option(None, "--restore", help="Archived concept ids to bring back"),
    rebuild: bool = typer.Option(True, "--rebuild/--no-rebuild", help="Rebuild the index after changes"),
) -> None:
    kb_dir = Path(settings.knowledge_base_dir)

    if restore:
        done = restore_concepts(kb_dir, list(restore))
        console.print(f"[green]restored {len(done)}[/green]: {', '.join(done) or '-'}")
        missing = sorted(set(restore) - set(done))
        if missing:
            console.print(f"[yellow]not in the archive: {', '.join(missing)}[/yellow]")
        if done and rebuild:
            _rebuild()
        return

    if archive:
        done = archive_concepts(kb_dir, list(archive))
        console.print(f"[green]archived {len(done)}[/green] -> {kb_dir / '_archive'}: {', '.join(done) or '-'}")
        missing = sorted(set(archive) - set(done))
        if missing:
            console.print(f"[yellow]not found: {', '.join(missing)}[/yellow]")
        if done and rebuild:
            _rebuild()
        return

    logging.disable(logging.WARNING)
    from sentence_transformers import SentenceTransformer

    from matchmind.knowledge_base.embedder import load_all_concepts

    concepts = load_all_concepts(kb_dir)
    model = SentenceTransformer(settings.embedding_model)
    clusters = find_clusters(concepts, lambda t: model.encode(t, normalize_embeddings=True), threshold)

    n_others = sum(len(o) for _, o in clusters)
    console.print(f"\n[bold]{len(concepts)} concepts · {len(clusters)} cluster(s) at >= {threshold:.2f} · "
                  f"{n_others} archive candidate(s)[/bold]\n")
    for k, (keep, others) in enumerate(clusters, 1):
        console.print(f"[bold]Cluster {k}[/bold]")
        console.print(f"  [green]KEEP   [/green] {keep.concept_id:<55} {keep.title}")
        for c, sim in others:
            console.print(f"  [yellow]archive[/yellow] {c.concept_id:<55} {c.title}  [dim](sim {sim:.2f})[/dim]")
        console.print()

    if archive_suggested and clusters:
        ids = [c.concept_id for _, others in clusters for c, _ in others]
        done = archive_concepts(kb_dir, ids)
        console.print(f"[green]archived {len(done)}[/green] -> {kb_dir / '_archive'}")
        if done and rebuild:
            _rebuild()
    elif clusters:
        console.print("[dim]Nothing changed. Archive with --archive <id> ... or --archive-suggested; "
                      "undo with --restore <id>.[/dim]")


if __name__ == "__main__":
    app()
