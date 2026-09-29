"""
Process every link in knowledge_base/sources/sources.txt into the knowledge base.

    python -m matchmind.scripts.ingest_sources              # process + lint + rebuild index
    python -m matchmind.scripts.ingest_sources --dry-run    # show what would be processed, no LLM calls
    python -m matchmind.scripts.ingest_sources --no-rebuild # skip the index rebuild

Safe to re-run: videos already processed are skipped without LLM calls, and
videos that failed (quota, no captions) are retried. Only YouTube videos and
channels are processed automatically; other links are listed as unsupported.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from matchmind.config import settings

app = typer.Typer()
console = Console()

DEFAULT_SOURCES = Path(settings.knowledge_base_dir).parent / "sources" / "sources.txt"


@app.command()
def main(
    file: str = typer.Option(str(DEFAULT_SOURCES), "--file", help="Sources file to read"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would be processed; no LLM calls"),
    rebuild: bool = typer.Option(True, "--rebuild/--no-rebuild", help="Rebuild the vector index if concepts were added"),
    sleep_s: float = typer.Option(2.0, help="Delay between LLM calls (raise it if you hit 429s)"),
) -> None:
    from matchmind.knowledge_base.concept_synthesis import existing_concept_ids
    from matchmind.knowledge_base.source_ingest import YouTubeIngestor, parse_sources_text

    for name in ("stdout", "stderr"):
        reconfigure = getattr(getattr(sys, name), "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")

    path = Path(file)
    if not path.exists():
        console.print(f"[red]Sources file not found: {path}[/red]")
        raise typer.Exit(1)
    entries = parse_sources_text(path.read_text(encoding="utf-8"))
    console.print(f"[bold green]{len(entries)} source line(s)[/bold green] in {path}")

    kb_dir = Path(settings.knowledge_base_dir)
    ingestor = YouTubeIngestor(
        concepts_root=kb_dir / "youtube",
        state_path=kb_dir / "_state" / "youtube_processed.json",
        known_ids=existing_concept_ids(kb_dir),
        sleep_s=sleep_s,
        dry_run=dry_run,
        log=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )
    report = ingestor.run(entries)

    table = Table(title="Ingest summary")
    table.add_column("")
    table.add_column("count", justify="right")
    if dry_run:
        table.add_row("videos to process", str(len(report.planned)))
    table.add_row("videos processed now", str(report.videos_new))
    table.add_row("videos skipped (already processed)", str(report.videos_skipped))
    table.add_row("videos failed (retried next run)", str(report.videos_failed))
    table.add_row("new concepts", str(report.concepts_written))
    table.add_row("chunks with no tactical concept", str(report.chunks_without_concept))
    table.add_row("unsupported lines", str(len(report.unsupported)))
    console.print(table)
    for line in report.unsupported:
        console.print(f"[yellow]unsupported (not processed yet): {line}[/yellow]")
    for err in report.errors:
        console.print(f"[red]{err}[/red]")
    if report.planned:
        console.print("would process: " + ", ".join(report.planned))

    if dry_run or report.concepts_written == 0:
        return
    console.print("\n[bold]Checking the knowledge base (kb_lint)...[/bold]")
    subprocess.run([sys.executable, "-m", "matchmind.scripts.kb_lint"], check=False)
    if rebuild:
        console.print("\n[bold]Rebuilding the vector index...[/bold]")
        result = subprocess.run([sys.executable, "-m", "matchmind.scripts.build_kb", "--rebuild"], check=False)
        if result.returncode != 0:
            console.print("[red]Index rebuild failed — run: python -m matchmind.scripts.build_kb --rebuild[/red]")
            raise typer.Exit(1)
        console.print("[green]Done — the new concepts are available to retrieval.[/green]")
    else:
        console.print("Run [bold]python -m matchmind.scripts.build_kb --rebuild[/bold] to make them searchable.")


if __name__ == "__main__":
    app()
