"""
Download & cache external datasets so evaluation / CI never hits the network.

  * StatsBomb Open Data 360 test cases  -> data_cache/statsbomb_testcases.json
  * (optional) a Metrica sample frame set -> data_cache/metrica_frames.json

Usage:
    python -m matchmind.scripts.download_data --n-cases 30
    poetry run python -m matchmind.scripts.download_data
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import typer
from rich.console import Console

from matchmind.config import settings

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)

CACHE_DIR = Path("./data_cache")


@app.command()
def main(
    n_cases: int = typer.Option(30, "--n-cases", help="StatsBomb test cases to cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fetch StatsBomb 360 test cases once and store them as JSON."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]MatchMind — data cache builder[/bold green]")

    try:
        from matchmind.data_sources.statsbomb_adapter import StatsBombAdapter
    except ImportError as exc:
        console.print(f"[red]statsbombpy not available: {exc}[/red]")
        raise typer.Exit(1)

    adapter = StatsBombAdapter()
    console.print(f"Fetching {n_cases} StatsBomb 360 test cases (this hits the network once)...")
    cases = adapter.load_test_cases(n=n_cases)

    out = CACHE_DIR / "statsbomb_testcases.json"
    serialisable = [
        {
            "snapshot": c["snapshot"].model_dump(),
            "focus_player_id": c["focus_player_id"],
            "question": c["question"],
            "ground_truth_action": c.get("ground_truth_action"),
            "ground_truth_event_type": c.get("ground_truth_event_type"),
        }
        for c in cases
    ]
    out.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    console.print(f"[green]Cached {len(serialisable)} cases -> {out}[/green]")

    # Point evaluation at the cache
    console.print(
        "Set STATSBOMB_TESTCASE_CACHE or pass --from-cache to run_evaluation to use it."
    )


def load_cached_testcases(path: Path | None = None) -> list[dict]:
    """Re-hydrate cached StatsBomb test cases into MatchState objects."""
    from matchmind.schema.match_state import MatchState

    path = path or (CACHE_DIR / "statsbomb_testcases.json")
    if not path.exists():
        raise FileNotFoundError(
            f"No cached test cases at {path}. Run: python -m matchmind.scripts.download_data"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    for c in raw:
        c["snapshot"] = MatchState(**c["snapshot"])
    return raw


if __name__ == "__main__":
    app()
