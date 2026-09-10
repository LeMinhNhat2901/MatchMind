"""
Build Knowledge Base Script.

Loads all concept JSON files, embeds them with SentenceTransformer,
and upserts into ChromaDB.

Usage:
    python -m matchmind.scripts.build_kb
    poetry run matchmind-build-kb
    poetry run matchmind-build-kb --rebuild
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matchmind.config import settings
from matchmind.knowledge_base.embedder import TacticalEmbedder, load_all_concepts
from matchmind.knowledge_base.vector_store import TacticalVectorStore

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def main(
    rebuild: bool = typer.Option(False, "--rebuild", help="Delete existing collection and rebuild"),
    concepts_dir: str = typer.Option("", "--concepts-dir", help="Path to concepts directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build (or rebuild) the MatchMind tactical knowledge base in ChromaDB."""

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    console.print("\n[bold green]🧠 MatchMind Knowledge Base Builder[/bold green]")

    # ── 1. Load concepts ───────────────────────────────────────
    concepts_path = Path(concepts_dir) if concepts_dir else settings.knowledge_base_dir
    console.print(f"\nLoading concepts from: [cyan]{concepts_path}[/cyan]")

    concepts = load_all_concepts(concepts_path)

    if not concepts:
        console.print("[bold red]ERROR: No concepts found! Run from project root.[/bold red]")
        raise typer.Exit(1)

    console.print(f"✅ Loaded [bold]{len(concepts)}[/bold] concepts:")
    categories = {}
    for c in concepts:
        categories[c.category] = categories.get(c.category, 0) + 1
    for cat, count in sorted(categories.items()):
        console.print(f"   • {cat}: {count} concepts")

    # ── 2. Initialise vector store ────────────────────────────
    store = TacticalVectorStore()
    existing = store.count()

    if existing > 0 and not rebuild:
        console.print(
            f"\n[yellow]Knowledge base already has {existing} concepts.[/yellow] "
            f"Use --rebuild to force rebuild."
        )
        raise typer.Exit(0)

    if rebuild and existing > 0:
        console.print(f"\n[yellow]Deleting {existing} existing concepts...[/yellow]")
        store.delete_all()

    # ── 3. Embed ───────────────────────────────────────────────
    console.print(f"\n[bold]Embedding with model:[/bold] {settings.embedding_model}")
    embedder = TacticalEmbedder()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Embedding concepts...", total=None)
        embeddings = embedder.embed_concepts(concepts)
        progress.remove_task(task)

    console.print(f"✅ Embedded {len(concepts)} concepts → {embeddings.shape[1]}-dim vectors")

    # ── 4. Upsert ──────────────────────────────────────────────
    store.upsert_concepts(concepts, embeddings)
    final_count = store.count()
    console.print(f"\n✅ [bold green]Knowledge base built![/bold green] {final_count} concepts in ChromaDB at: {settings.chroma_persist_dir}")

    # ── 5. Quick sanity test ──────────────────────────────────
    console.print("\n[bold]Running retrieval sanity test...[/bold]")
    test_query = "player in half-space with open passing lane and third-man opportunity"
    results, confidence = store.retrieve_by_query_text(test_query, embedder, k=3)
    console.print(f"Query: '{test_query[:60]}...'")
    console.print(f"Confidence: [bold]{confidence:.3f}[/bold] | Top results:")
    for r in results:
        console.print(f"  • [cyan]{r['title']}[/cyan] (sim={r['similarity']:.3f})")

    console.print("\n[bold green]✅ Knowledge base is ready![/bold green]")


if __name__ == "__main__":
    app()
