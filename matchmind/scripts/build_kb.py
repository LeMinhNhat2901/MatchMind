"""
Build Knowledge Base Script.

Loads all concept JSON files, embeds them with SentenceTransformer,
and upserts into ChromaDB.

Usage:
    python -m matchmind.scripts.build_kb --rebuild
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matchmind.config import settings
from matchmind.knowledge_base.embedder import TacticalEmbedder, load_all_concepts
from matchmind.knowledge_base.vector_store import TacticalVectorStore

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)


def _safe_stdout() -> None:
    """Best-effort UTF-8 stdout so Rich output never dies on a cp1252 console."""
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        rc = getattr(s, "reconfigure", None)
        if rc is not None and (getattr(s, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
            try:
                rc(encoding="utf-8", errors="replace")
            except Exception:
                pass


@app.command()
def main(
    rebuild: bool = typer.Option(False, "--rebuild", help="Delete existing collection and rebuild"),
    concepts_dir: str = typer.Option("", "--concepts-dir", help="Path to concepts directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build (or rebuild) the MatchMind tactical knowledge base in ChromaDB."""
    _safe_stdout()
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    console.print("\n[bold green]MatchMind Knowledge Base Builder[/bold green]")

    # ── 1. Load concepts ───────────────────────────────────────
    concepts_path = Path(concepts_dir) if concepts_dir else settings.knowledge_base_dir
    console.print(f"Loading concepts from: [cyan]{concepts_path}[/cyan]")
    concepts = load_all_concepts(concepts_path)
    if not concepts:
        console.print("[bold red]ERROR: no concepts found (run from the project root).[/bold red]")
        raise typer.Exit(1)

    cats: dict[str, int] = {}
    for c in concepts:
        cats[c.category] = cats.get(c.category, 0) + 1
    console.print(f"[green]Loaded {len(concepts)} concepts[/green]: " + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))

    # ── 2. Vector store ───────────────────────────────────────
    store = TacticalVectorStore(use_chroma_ef=True)
    existing = store.count()
    if existing > 0 and not rebuild:
        console.print(f"[yellow]KB already has {existing} concepts. Use --rebuild to force.[/yellow]")
        raise typer.Exit(0)
    if rebuild and existing > 0:
        console.print(f"[yellow]Deleting {existing} existing concepts...[/yellow]")
        store.delete_all()

    # ── 3. Embed ──────────────────────────────────────────────
    console.print(f"Embedding with model: [cyan]{settings.embedding_model}[/cyan]")
    embedder = TacticalEmbedder()
    embeddings = embedder.embed_concepts(concepts)
    console.print(f"[green]Embedded {len(concepts)} concepts -> {embeddings.shape[1]}-dim[/green]")

    # ── 4. Upsert ─────────────────────────────────────────────
    store.upsert_concepts(concepts, embeddings)
    console.print(
        f"[bold green]KB built:[/bold green] {store.count()} concepts at {settings.chroma_persist_dir}"
    )

    # ── 5. Sanity check ───────────────────────────────────────
    q = "player in the half-space with an open passing lane and a third-man option"
    results, confidence = store.retrieve_tactics(query_text=q, k=3)
    console.print(f"Sanity query confidence: [bold]{confidence:.3f}[/bold]")
    for r in results:
        console.print(f"  - [cyan]{r['title']}[/cyan] (sim={r['similarity']:.3f})")
    console.print("\n[bold green]Knowledge base is ready.[/bold green]")


if __name__ == "__main__":
    app()
