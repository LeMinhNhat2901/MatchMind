"""
Build the Example Library from the local StatsBomb open-data clone.

Usage:
    python -m matchmind.scripts.build_example_library --matches 60 --per-match 150
Writes settings.example_library_dir/{examples.jsonl,index.npz}.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import typer
from rich.console import Console

from matchmind.config import settings

app = typer.Typer()
console = Console()


@app.command()
def main(
    matches: int = typer.Option(60, help="How many 360 matches to scan"),
    per_match: int = typer.Option(150, help="Max examples per match"),
    skip_eval_matches: bool = typer.Option(
        True, help="Leave out matches used by data_cache/statsbomb_testcases.json (no eval leakage)"
    ),
) -> None:
    from matchmind.data_sources.statsbomb_local import StatsBombLocalLoader
    from matchmind.examples.builder import build_examples
    from matchmind.examples.index import ExampleIndex

    loader = StatsBombLocalLoader(settings.statsbomb_open_data_dir)
    match_ids = loader.match_ids_with_360()
    if skip_eval_matches:
        try:
            from matchmind.scripts.download_data import load_cached_testcases

            held_out = {int(c["snapshot"].match_id) for c in load_cached_testcases()}
            match_ids = [m for m in match_ids if m not in held_out]
            console.print(f"held out {len(held_out)} evaluation matches")
        except FileNotFoundError:
            console.print("[yellow]no test-case cache yet — nothing held out[/yellow]")
    match_ids = match_ids[:matches]
    examples = build_examples(loader, match_ids, max_per_match=per_match)
    ExampleIndex.build(examples).save(settings.example_library_dir)
    console.print(f"[green]{len(examples)} examples from {len(match_ids)} matches -> {settings.example_library_dir}[/green]")
    console.print(f"actions: {dict(Counter(e.action_type for e in examples))}")
    console.print(f"outcomes: {dict(Counter(e.outcome for e in examples))}")


if __name__ == "__main__":
    app()
