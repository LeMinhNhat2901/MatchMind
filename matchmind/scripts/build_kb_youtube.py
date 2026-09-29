"""
Build Knowledge Base from YouTube transcripts — an optional, separate source
from the Wikipedia pipeline (knowledge_bonus.md: "build riêng" / build it as
its own thing, not folded into the main automated pipeline).

Pipeline:
    1. You supply either a curated list of video URLs (one per line,
       optionally "Channel Name | URL"), or a single --channel URL to pull
       its N most recent uploads from in one go (still every video from a
       channel YOU pointed at — not auto-discovery across unknown channels).
    2. Fetch each video's transcript via youtube-transcript-api (official
       captions API; NOT NotebookLM — see knowledge_base/youtube_source.py
       docstring for why that idea was dropped).
    3. Split the transcript into ~3000-char chunks and let the LLM extract
       0+ genuine tactical concepts per chunk (same synthesize_concept() used
       by the Wikipedia pipeline), skipping banter/sponsor reads/etc.
    4. Save each as JSON under knowledge_base/concepts/youtube/, with a
       `source` citing the channel name + a timestamped video link.

Idempotent across runs at the VIDEO level (knowledge_base/_state/
youtube_processed.json), not just at the concept level — a video already
transcribed/synthesized is skipped even if it produced zero concepts, or if
its LLM-generated concept titles (and therefore concept_ids) would differ on
a re-run. Otherwise --channel run twice, or a video that's in both
--urls-file and later pulled again via --channel, would burn LLM calls
re-processing the same transcript and could leave duplicate-content concepts
under different ids.

Requires (matchmind conda env):
    pip install youtube-transcript-api yt-dlp

Usage:
    conda activate matchmind
    python -m matchmind.scripts.build_kb_youtube --urls-file my_videos.txt
    python -m matchmind.scripts.build_kb_youtube --url "Tifo Football | https://youtube.com/watch?v=XXXXXXXXXXX"
    python -m matchmind.scripts.build_kb_youtube --channel "https://www.youtube.com/@SomeChannel" --max-videos 15

my_videos.txt format (one per line, blank lines / lines starting with # ignored):
    Tifo Football | https://www.youtube.com/watch?v=XXXXXXXXXXX
    https://youtu.be/YYYYYYYYYYY   # channel name optional; falls back to "YouTube"

Then embed the new concepts into the vector store:
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
from matchmind.knowledge_base.source_ingest import (
    SourceEntry,
    YouTubeIngestor,
    classify_line,
    parse_sources_text,
)

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)

OUT_SUBDIR = "youtube"


def _safe_stdout() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        rc = getattr(s, "reconfigure", None)
        if rc is not None and (getattr(s, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
            try:
                rc(encoding="utf-8", errors="replace")
            except Exception:
                pass


def entries_from_cli(urls_file: str, url: list[str], channel_url: str, max_videos: int) -> list[SourceEntry]:
    """CLI options -> the same SourceEntry list sources.txt produces, so this
    script and ingest_sources.py share one ingest loop (YouTubeIngestor)."""
    entries: list[SourceEntry] = []
    if urls_file:
        entries += parse_sources_text(Path(urls_file).read_text(encoding="utf-8"))
    for u in url:
        entry = classify_line(u)
        if entry is not None:
            entries.append(entry)
    if channel_url:
        entry = classify_line(f"{channel_url} | max={max_videos}")
        if entry is not None:
            if entry.kind != "youtube_channel":
                # Accept any channel URL form yt-dlp understands, even ones the
                # classifier doesn't recognise as a channel.
                entry = SourceEntry("youtube_channel", channel_url, max_videos=max_videos)
            entries.append(entry)
    return entries


@app.command()
def main(
    urls_file: str = typer.Option("", help="Text file of video URLs, one per line (see module docstring for format)"),
    url: list[str] = typer.Option([], help="A single 'Channel | URL' or bare URL — repeatable"),
    channel_url: str = typer.Option(
        "",
        "--channel",
        help=(
            "A channel URL (any form: @handle, /channel/UC..., with or without /videos) — "
            "pulls its most recent --max-videos uploads via yt-dlp (metadata only, no API "
            "key needed) instead of listing individual video URLs by hand."
        ),
    ),
    max_videos: int = typer.Option(20, help="Cap on videos pulled from --channel"),
    max_chunk_chars: int = typer.Option(3000, help="Transcript chunk size fed to the LLM per synthesis call"),
    out_dir: str = typer.Option("", help="Override concepts output dir (default: <knowledge_base_dir>/youtube)"),
    sleep_s: float = typer.Option(
        2.0,
        help="Delay between LLM calls. Groq's free tier is ~7000 input-tokens/minute — raise "
        "this if you see 429s, especially if also using the dashboard interactively.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    from matchmind.knowledge_base.concept_synthesis import existing_concept_ids

    _safe_stdout()
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    console.print("\n[bold green]MatchMind — YouTube Knowledge Base Ingestion[/bold green]")

    entries = entries_from_cli(urls_file, url, channel_url, max_videos)
    if not entries:
        console.print("[bold red]No videos given. Use --urls-file, --url, or --channel.[/bold red]")
        raise typer.Exit(1)

    kb_dir = Path(settings.knowledge_base_dir)
    concepts_root = Path(out_dir) if out_dir else kb_dir / OUT_SUBDIR
    report = YouTubeIngestor(
        concepts_root=concepts_root,
        state_path=kb_dir / "_state" / "youtube_processed.json",
        known_ids=existing_concept_ids(kb_dir),
        sleep_s=sleep_s,
        max_chunk_chars=max_chunk_chars,
        log=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    ).run(entries)

    for line in report.unsupported:
        console.print(f"[yellow]not a YouTube link, skipped: {line}[/yellow]")
    for err in report.errors:
        console.print(f"[red]{err}[/red]")
    console.print(
        f"\n[bold green]Done:[/bold green] {report.videos_new} video(s) processed, "
        f"{report.videos_skipped} already done, {report.videos_failed} failed; "
        f"{report.concepts_written} concept(s) written -> {concepts_root}"
    )
    if report.concepts_written:
        console.print(
            "\n[bold]Next:[/bold] embed the new concepts into the vector store:\n"
            "  python -m matchmind.scripts.build_kb --rebuild"
        )


if __name__ == "__main__":
    app()
