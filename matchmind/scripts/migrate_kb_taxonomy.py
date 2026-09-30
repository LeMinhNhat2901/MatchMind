"""
One-off migration: remap the 66 existing concepts' `category` field from the
v1 vocabulary to the v2 taxonomy (knowledge_base.md, 2026-09) — see
knowledge_base/concept_synthesis.py::ALLOWED_CATEGORIES for the new list.

Mechanical only (no LLM call): the v1 -> v2 category mapping is a clean 1:1
rename, no ambiguity. Does NOT backfill subcategory/definition/
spatial_requirements/player_behaviours/related_concepts for existing
concepts — those are optional new fields (schema v2 is backward compatible),
and populating them well for 66 concepts needs per-concept judgment, not a
mechanical script. New concepts synthesized from now on get them automatically
via concept_synthesis.py's updated prompt.

Usage (matchmind conda env):
    conda activate matchmind
    python -m matchmind.scripts.migrate_kb_taxonomy --dry-run
    python -m matchmind.scripts.migrate_kb_taxonomy
    python -m matchmind.scripts.build_kb --rebuild   # re-embed after migrating
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matchmind.config import settings
from matchmind.knowledge_base.embedder import concept_files
from matchmind.schema.match_state import TacticalConcept

app = typer.Typer()
console = Console()

CATEGORY_V1_TO_V2 = {
    "attacking_movement": "attacking",
    "defensive_organization": "defending",
    "spatial_principles": "fundamentals",
    "technical_skill": "individual",
    "general_terminology": "terminology",
}


@app.command()
def main(
    concepts_dir: str = typer.Option("", help="Override concepts dir (default: settings.knowledge_base_dir)"),
    dry_run: bool = typer.Option(False, help="Show what would change, write nothing"),
) -> None:
    root = Path(concepts_dir) if concepts_dir else Path(settings.knowledge_base_dir)
    files = concept_files(root)
    console.print(f"[bold green]MatchMind KB Taxonomy Migration[/bold green] — scanning {root}")

    table = Table(title="Category remap")
    table.add_column("File")
    table.add_column("Old category")
    table.add_column("New category")

    changed, skipped, failed = 0, 0, 0
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as exc:
            console.print(f"[red]{fpath.name}: read failed: {exc}[/red]")
            failed += 1
            continue

        old_category = data.get("category")
        new_category = CATEGORY_V1_TO_V2.get(old_category)
        if new_category is None:
            skipped += 1  # already v2, or unrecognized — leave untouched
            continue

        table.add_row(fpath.name, old_category, new_category)
        data["category"] = new_category

        if not dry_run:
            try:
                TacticalConcept(**data)  # validate before writing
            except Exception as exc:
                console.print(f"[red]{fpath.name}: post-migration validation failed: {exc}[/red]")
                failed += 1
                continue
            fpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        changed += 1

    console.print(table)
    console.print(
        f"\n[bold]{'Would change' if dry_run else 'Changed'}:[/bold] {changed}, "
        f"[dim]already v2 / skipped: {skipped}, failed: {failed}[/dim]"
    )
    if not dry_run and changed:
        console.print("\n[bold]Next:[/bold] re-embed so ChromaDB metadata matches:\n  python -m matchmind.scripts.build_kb --rebuild")


if __name__ == "__main__":
    app()
