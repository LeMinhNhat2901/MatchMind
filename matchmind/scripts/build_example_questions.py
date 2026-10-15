"""
Write example questions (doc2query) for every concept that has none yet.

    python -m matchmind.scripts.build_example_questions              # all missing, then rebuild the index
    python -m matchmind.scripts.build_example_questions --limit 10   # try a few first
    python -m matchmind.scripts.build_example_questions --no-rebuild

Costs LLM tokens once per concept (batches of 5 concepts per call); safe to re-run,
since concepts that already have questions are skipped and failed batches are retried.
The prompt only sees the concept itself, never the golden or held-out evaluation sets.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matchmind.config import settings
from matchmind.knowledge_base.embedder import load_all_concepts
from matchmind.knowledge_base.example_questions import generate_example_questions, write_example_questions

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    limit: int = typer.Option(0, help="Only this many concepts (0 = all missing)"),
    batch_size: int = typer.Option(5, help="Concepts per LLM call"),
    rebuild: bool = typer.Option(True, "--rebuild/--no-rebuild", help="Rebuild the index afterwards"),
) -> None:
    from matchmind.llm_client import call_llm

    kb_dir = Path(settings.knowledge_base_dir)
    concepts = load_all_concepts(kb_dir)
    todo = [c for c in concepts if not c.example_questions]
    if limit:
        todo = todo[:limit]
    console.print(f"{len(concepts)} concepts, {len(todo)} without example questions")
    if not todo:
        return
    questions, report = generate_example_questions(
        todo, llm_call=call_llm, batch_size=batch_size, log=lambda m: console.print(f"[dim]{m}[/dim]"))
    written = write_example_questions(kb_dir, questions)
    console.print(f"[green]{written} concepts updated[/green] · {len(report.failed)} failed "
                  f"(re-run to retry) · {report.tokens} tokens")
    if report.failed:
        console.print(f"[yellow]failed: {', '.join(report.failed[:20])}{' …' if len(report.failed) > 20 else ''}[/yellow]")
    if written and rebuild:
        console.print("\n[bold]Rebuilding the vector index...[/bold]")
        subprocess.run([sys.executable, "-m", "matchmind.scripts.build_kb", "--rebuild"], check=False)


if __name__ == "__main__":
    app()
