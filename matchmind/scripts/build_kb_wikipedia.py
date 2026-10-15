"""
Build Knowledge Base from Wikipedia — discover -> fetch -> paraphrase -> save.

Implements the 5-step pipeline from knowledge_bonus.md, steps 1-4 (step 5,
embedding into ChromaDB, is the existing matchmind.scripts.build_kb):

    1. Discover article titles from 3 Wikipedia categories (tactics,
       terminology, skills — see knowledge_base/wikipedia_source.py)
    2. Fetch each article's plain-text extract via the MediaWiki API
    3. LLM paraphrases + restructures each into the TacticalConcept schema,
       filtering out non-tactical trivia (knowledge_base/concept_synthesis.py)
    4. Save each as JSON under knowledge_base/concepts/wikipedia/, with a
       `source` field citing the Wikipedia article (CC BY-SA 4.0)

Usage (matchmind conda env):
    conda activate matchmind
    python -m matchmind.scripts.build_kb_wikipedia --max-per-category 40 --max-skills 32
    python -m matchmind.scripts.build_kb_wikipedia --dry-run          # discovery only, no LLM calls
    python -m matchmind.scripts.build_kb_wikipedia --limit 5          # smoke test a handful first

Then embed the new concepts into the vector store:
    python -m matchmind.scripts.build_kb --rebuild
"""
from __future__ import annotations

import datetime as _dt
import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matchmind.config import settings
from matchmind.knowledge_base.concept_synthesis import existing_concept_ids, save_concept, synthesize_concept
from matchmind.knowledge_base.wikipedia_source import article_url, discover_articles, fetch_extract

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)

ID_PREFIX = "wiki_"
OUT_SUBDIR = "wikipedia"


def _safe_stdout() -> None:
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
    max_per_category: int = typer.Option(40, help="Max articles from each of tactics/terminology categories"),
    max_skills: int = typer.Option(32, help="Max individual-skill articles from the skills list page"),
    limit: int = typer.Option(0, help="If >0, only process the first N discovered articles (smoke test)"),
    out_dir: str = typer.Option("", help="Override concepts output dir (default: <knowledge_base_dir>/wikipedia)"),
    dry_run: bool = typer.Option(False, help="Discovery only — list titles, make no LLM calls, write nothing"),
    sleep_s: float = typer.Option(
        2.0,
        help="Delay between LLM calls. Groq's free tier is ~7000 input-tokens/minute, easy to "
        "exhaust with ~1500-2500 tokens/call — raise this (or run when not also using the "
        "dashboard interactively) if you see 429s.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _safe_stdout()
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    console.print("\n[bold green]MatchMind — Wikipedia Knowledge Base Ingestion[/bold green]")
    console.print(
        "[dim]Content is paraphrased (not copied) from Wikipedia, CC BY-SA 4.0. "
        "See knowledge_base/README.md for the attribution note.[/dim]\n"
    )

    # ── 1. Discover ─────────────────────────────────────────────
    console.print("[bold]Step 1/4[/bold] Discovering articles...")
    articles = discover_articles(max_per_category=max_per_category, max_skills=max_skills)
    if limit > 0:
        articles = articles[:limit]
    console.print(f"[green]Discovered {len(articles)} candidate articles[/green]")

    if dry_run:
        table = Table(title="Discovered articles (dry run — nothing written)")
        table.add_column("Title")
        table.add_column("Category hint")
        for a in articles:
            table.add_row(a.title, a.category_hint)
        console.print(table)
        return

    concepts_root = Path(out_dir) if out_dir else Path(settings.knowledge_base_dir) / OUT_SUBDIR
    known_ids = existing_concept_ids(Path(settings.knowledge_base_dir))
    console.print(f"[dim]{len(known_ids)} concepts already on disk (will skip re-synthesizing matching ids)[/dim]")

    written, skipped, failed = 0, 0, 0
    retrieved_at = _dt.date.today().isoformat()

    for i, a in enumerate(articles, 1):
        console.print(f"[{i}/{len(articles)}] {a.title}", end=" ")

        # ── 2. Fetch ────────────────────────────────────────────
        extract = fetch_extract(a.title)
        if not extract:
            console.print("[yellow]no extract, skipped[/yellow]")
            skipped += 1
            continue

        # ── 3. Synthesize (paraphrase + structure + relevance filter) ──
        concept = synthesize_concept(
            source_title=a.title,
            source_text=extract,
            source_name=f"Wikipedia: {a.title}",
            source_url=article_url(a.title),
            source_license="CC BY-SA 4.0",
            retrieved_at=retrieved_at,
            id_prefix=ID_PREFIX,
        )
        if concept is None:
            console.print("[yellow]skipped (not tactical / parse failed)[/yellow]")
            skipped += 1
            time.sleep(sleep_s)
            continue

        if concept.concept_id in known_ids:
            console.print(f"[dim]already have {concept.concept_id}[/dim]")
            skipped += 1
            time.sleep(sleep_s)
            continue

        # ── 4. Save ─────────────────────────────────────────────
        try:
            path = save_concept(concept, concepts_root)
            known_ids.add(concept.concept_id)
            written += 1
            console.print(f"[green]-> {path.name}[/green]")
        except Exception as exc:
            console.print(f"[red]write failed: {exc}[/red]")
            failed += 1

        time.sleep(sleep_s)

    console.print(
        f"\n[bold green]Done:[/bold green] {written} written, {skipped} skipped, {failed} failed "
        f"-> {concepts_root}"
    )
    if written:
        console.print(
            "\n[bold]Next:[/bold] embed the new concepts into the vector store:\n"
            "  python -m matchmind.scripts.build_kb --rebuild"
        )


if __name__ == "__main__":
    app()
