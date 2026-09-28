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

import datetime as _dt
import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matchmind.config import settings
from matchmind.knowledge_base.concept_synthesis import existing_concept_ids, save_concept, synthesize_concept
from matchmind.knowledge_base.youtube_source import (
    chunk_transcript,
    fetch_transcript,
    list_channel_videos,
    load_processed_videos,
    parse_video_id,
    save_processed_videos,
    timestamped_url,
)

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)

ID_PREFIX = "yt_"
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


def _parse_line(line: str) -> tuple[str | None, str] | None:
    """"Channel | url" or bare "url" -> (channel, url). channel is None when not given (the
    caller looks it up via YouTube's oEmbed API instead of defaulting to a generic label).
    None (the whole tuple) for blank/comment lines."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "|" in line:
        channel, url = line.split("|", 1)
        return channel.strip(), url.strip()
    return None, line


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
    _safe_stdout()
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    console.print("\n[bold green]MatchMind — YouTube Knowledge Base Ingestion[/bold green]")

    entries: list[tuple[str | None, str]] = []
    if urls_file:
        for line in Path(urls_file).read_text(encoding="utf-8").splitlines():
            parsed = _parse_line(line)
            if parsed:
                entries.append(parsed)
    for u in url:
        parsed = _parse_line(u)
        if parsed:
            entries.append(parsed)

    if channel_url:
        console.print(f"[dim]Listing up to {max_videos} videos from {channel_url}...[/dim]")
        try:
            channel_video_urls = list_channel_videos(channel_url, max_videos=max_videos)
        except ImportError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[bold red]Could not list channel videos: {exc}[/bold red]")
            raise typer.Exit(1)
        console.print(f"[green]Found {len(channel_video_urls)} video(s)[/green]")
        entries.extend((None, vurl) for vurl in channel_video_urls)

    if not entries:
        console.print("[bold red]No videos given. Use --urls-file, --url, or --channel.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[green]{len(entries)} video(s) queued[/green]")

    concepts_root = Path(out_dir) if out_dir else Path(settings.knowledge_base_dir) / OUT_SUBDIR
    known_ids = existing_concept_ids(Path(settings.knowledge_base_dir))
    processed_state_path = Path(settings.knowledge_base_dir) / "_state" / "youtube_processed.json"
    processed_videos = load_processed_videos(processed_state_path)
    retrieved_at = _dt.date.today().isoformat()
    written, skipped, failed = 0, 0, 0

    for channel, raw_url in entries:
        try:
            video_id = parse_video_id(raw_url)
        except ValueError as exc:
            console.print(f"[red]{raw_url}: {exc}[/red]")
            failed += 1
            continue

        if video_id in processed_videos:
            console.print(f"[dim]{raw_url}: already processed (video-level state), skipping[/dim]")
            skipped += 1
            continue

        if channel is None:
            try:
                from matchmind.knowledge_base.youtube_source import fetch_video_metadata

                channel = fetch_video_metadata(video_id)["channel"]
            except Exception as exc:
                logger.debug(f"oEmbed lookup failed for {video_id}: {exc}")
                channel = "YouTube"

        console.print(f"\n[bold]{channel}[/bold] — {raw_url}")
        try:
            segments = fetch_transcript(video_id)
        except ImportError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[red]transcript unavailable: {exc}[/red]")
            failed += 1
            continue

        chunks = chunk_transcript(segments, max_chars=max_chunk_chars)
        console.print(f"[dim]{len(segments)} transcript segments -> {len(chunks)} chunks[/dim]")

        for i, chunk in enumerate(chunks, 1):
            console.print(f"  chunk {i}/{len(chunks)}", end=" ")
            concept = synthesize_concept(
                source_title=f"{channel} transcript chunk {i}",
                source_text=chunk.text,
                source_name=channel,
                source_url=timestamped_url(video_id, chunk.start_seconds),
                source_license="fair-use paraphrase (transcript, not redistributed verbatim)",
                retrieved_at=retrieved_at,
                id_prefix=ID_PREFIX,
            )
            if concept is None:
                console.print("[yellow]skipped[/yellow]")
                skipped += 1
                time.sleep(sleep_s)
                continue
            if concept.concept_id in known_ids:
                console.print(f"[dim]already have {concept.concept_id}[/dim]")
                skipped += 1
                time.sleep(sleep_s)
                continue
            try:
                path = save_concept(concept, concepts_root)
                known_ids.add(concept.concept_id)
                written += 1
                console.print(f"[green]-> {path.name}[/green]")
            except Exception as exc:
                console.print(f"[red]write failed: {exc}[/red]")
                failed += 1
            time.sleep(sleep_s)

        # Mark processed once we've genuinely attempted every chunk (whatever the
        # per-chunk outcome) — NOT on a transcript-fetch failure above (that's
        # often transient/environmental and worth retrying next run).
        processed_videos.add(video_id)
        save_processed_videos(processed_state_path, processed_videos)

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
