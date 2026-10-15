"""
Build Narrator Style Guide from YouTube — a persona/voice description for
narrator_node, learned from real analyst commentary.

This is deliberately separate from build_kb_youtube.py: that script extracts
FOOTBALL CONTENT (tactical concepts) from transcripts, discarding voice/tone
in the process (it paraphrases into short, neutral schema fields). This
script extracts VOICE ONLY (sentence rhythm, metaphor use, technical depth,
how cause-and-effect gets narrated) and discards football content — see
knowledge_base/style_guide.py's docstring for why this has to be an explicit
persona instruction rather than relying on RAG to "teach" a style.

Pipeline: fetch transcript -> chunk -> clean (light ASR punctuation fix, no
summarizing) -> sample a few cleaned excerpts per video -> one LLM call
analyzes all excerpts for style patterns -> save as
knowledge_base/style_guide.md, which narrator_node.py loads automatically if
present.

Usage (matchmind conda env):
    conda activate matchmind
    python -m matchmind.scripts.build_style_guide --urls-file my_videos.txt
    python -m matchmind.scripts.build_style_guide --url "Tifo Football | https://www.youtube.com/watch?v=XXXXXXXXXXX"
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
from matchmind.knowledge_base.style_guide import clean_transcript_excerpt, extract_style_guide
from matchmind.knowledge_base.youtube_source import (
    chunk_transcript,
    fetch_transcript,
    parse_video_id,
)

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)

DEFAULT_OUT = Path(__file__).parent.parent / "knowledge_base" / "style_guide.md"


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
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "|" in line:
        channel, url = line.split("|", 1)
        return channel.strip(), url.strip()
    return None, line


@app.command()
def main(
    urls_file: str = typer.Option("", help="Text file of video URLs, one per line ('Channel | URL' or bare URL)"),
    url: list[str] = typer.Option([], help="A single 'Channel | URL' or bare URL — repeatable"),
    excerpts_per_video: int = typer.Option(1, help="How many transcript chunks to sample per video (kept small — this analyzes voice, not content, so it doesn't need full coverage)"),
    out_path: str = typer.Option("", help=f"Output path (default: {DEFAULT_OUT})"),
    sleep_s: float = typer.Option(1.5, help="Delay between LLM calls"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _safe_stdout()
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    console.print("\n[bold green]MatchMind — Narrator Style Guide Builder[/bold green]")
    console.print(
        "[dim]Analyzes VOICE only (sentence rhythm, metaphor, tone) — football content is "
        "discarded. See knowledge_base/style_guide.py for why this can't just be RAG.[/dim]\n"
    )

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

    if not entries:
        console.print("[bold red]No videos given. Use --urls-file or --url.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[green]{len(entries)} video(s) queued[/green]\n")

    cleaned_excerpts: list[tuple[str, str]] = []
    sources: list[str] = []

    for channel, raw_url in entries:
        try:
            video_id = parse_video_id(raw_url)
        except ValueError as exc:
            console.print(f"[red]{raw_url}: {exc}[/red]")
            continue

        if channel is None:
            try:
                from matchmind.knowledge_base.youtube_source import fetch_video_metadata

                channel = fetch_video_metadata(video_id)["channel"]
            except Exception:
                channel = "YouTube"

        console.print(f"[bold]{channel}[/bold] — {raw_url}")
        try:
            segments = fetch_transcript(video_id)
        except ImportError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[red]transcript unavailable: {exc}[/red]")
            continue

        chunks = chunk_transcript(segments)[:excerpts_per_video]
        for i, chunk in enumerate(chunks, 1):
            console.print(f"  cleaning excerpt {i}/{len(chunks)}", end=" ")
            cleaned = clean_transcript_excerpt(chunk.text)
            if cleaned:
                cleaned_excerpts.append((channel, cleaned))
                console.print("[green]ok[/green]")
            else:
                console.print("[yellow]skipped[/yellow]")
            time.sleep(sleep_s)

        sources.append(f"{channel} — {raw_url}")

    if not cleaned_excerpts:
        console.print("[bold red]No usable transcript excerpts — nothing to analyze.[/bold red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Analyzing style across {len(cleaned_excerpts)} excerpt(s)...[/bold]")
    style_guide = extract_style_guide(cleaned_excerpts)
    if style_guide is None:
        console.print("[bold red]Style extraction failed.[/bold red]")
        raise typer.Exit(1)

    out = Path(out_path) if out_path else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"<!-- Generated by build_style_guide.py on {_dt.date.today().isoformat()}. "
        f"Voice only — no football content is copied from these sources. Sources:\n"
        + "\n".join(f"  - {s}" for s in sources)
        + "\n-->\n\n"
    )
    out.write_text(header + style_guide + "\n", encoding="utf-8")

    console.print(f"\n[bold green]Style guide written to {out}[/bold green]")
    console.print("narrator_node.py loads this automatically on the next agent run.")


if __name__ == "__main__":
    app()
