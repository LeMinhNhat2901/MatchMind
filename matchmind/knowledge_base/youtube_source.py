"""
YouTube Source — transcript fetching for the KB ingestion pipeline.

Per knowledge_bonus.md: an automated NotebookLM pipeline was considered and
rejected — Google has no public API for the free NotebookLM tier, only an
unofficial reverse-engineered one (notebooklm-py) that runs on a personal
Google session cookie and can break at any time. The recommended, stable
alternative is used instead: youtube-transcript-api (official captions,
free, no auth) + the LLM already in the project (matchmind/llm_client.py) to
extract/paraphrase tactical concepts from the transcript text.

This module deliberately does NOT auto-discover "good" tactical-analysis
channels/videos — the caller supplies the video list explicitly (see
matchmind/scripts/build_kb_youtube.py --urls-file), same as the doc's
"ghi nguồn rõ ràng: tên kênh + link video" requirement implies a curated,
attributable list rather than a scraped one.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)


@dataclass
class TranscriptChunk:
    text: str
    start_seconds: float


def parse_video_id(url_or_id: str) -> str:
    """Accepts a full YouTube URL or a bare 11-char video id."""
    url_or_id = url_or_id.strip()
    match = _VIDEO_ID_RE.search(url_or_id)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    raise ValueError(f"Could not parse a YouTube video id from {url_or_id!r}")


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def fetch_video_metadata(video_id: str) -> dict:
    """{"channel": str, "title": str} via YouTube's public oEmbed endpoint (no API key).

    Used to auto-fill the channel name for attribution when the caller gives
    a bare URL instead of "Channel | URL".
    """
    import httpx

    resp = httpx.get(
        "https://www.youtube.com/oembed",
        params={"url": video_url(video_id), "format": "json"},
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return {"channel": data.get("author_name", "YouTube"), "title": data.get("title", video_id)}


POPULAR_SCAN_LIMIT = 300


def list_channel_videos(
    channel_url: str, *, max_videos: int = 20, sort: str = "newest", scan_limit: int = POPULAR_SCAN_LIMIT
) -> list[str]:
    """Video URLs from a channel's /videos tab (Shorts and streams excluded).

    sort="newest" (default): the most recent uploads.
    sort="popular": the most-viewed among the channel's last `scan_limit`
    uploads. The flat listing already carries view counts, so this needs no API
    key; channels with more uploads than `scan_limit` are only ranked over that
    recent window (keeps the listing to a few seconds).

    Uses yt-dlp in metadata-only ("flat playlist") mode — no download, just
    lists ids/titles, fast. Accepts a channel URL in any form (@handle,
    /channel/UC..., with or without a trailing /videos).

    Chosen over the official YouTube Data API (which the original CI/CD
    proposal used) to avoid needing a YOUTUBE_API_KEY / quota for a one-off,
    user-triggered batch pull. Also a deliberately different case from the
    NotebookLM pipeline this project already rejected: that was an
    unofficial, reverse-engineered wrapper around an internal API running on
    a personal session cookie; yt-dlp is a mature, openly-maintained,
    widely-used tool that reads YouTube's public page data, with a very
    different risk profile.

    Requires yt-dlp (pip install yt-dlp in the matchmind conda env — not
    installed by default, see pyproject.toml).
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError(
            "yt-dlp not installed. Run (in the matchmind conda env): pip install yt-dlp"
        ) from exc

    url = channel_url.rstrip("/")
    if not url.endswith("/videos"):
        url += "/videos"

    opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "playlistend": max(max_videos, scan_limit) if sort == "popular" else max_videos,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = [e for e in (info.get("entries") or []) if e.get("id")]
    if sort == "popular":
        entries = sorted(entries, key=lambda e: e.get("view_count") or 0, reverse=True)
    return [video_url(e["id"]) for e in entries[:max_videos]]


def fetch_transcript(video_id: str, *, languages: tuple[str, ...] = ("en", "vi")) -> list[dict]:
    """Raw transcript segments: [{"text", "start", "duration"}, ...].

    Tries `languages` in priority order first (manually-created captions
    preferred over auto-generated, per find_transcript's own ranking); if the
    video has captions in neither language, falls back to whatever transcript
    IS available rather than failing outright — tactical-analysis channels
    aren't only in English/Vietnamese.

    Requires youtube-transcript-api (pip install youtube-transcript-api in
    the matchmind conda env — not installed by default, see pyproject.toml).
    """
    try:
        from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi
    except ImportError as exc:
        raise ImportError(
            "youtube-transcript-api not installed. Run (in the matchmind conda env): "
            "pip install youtube-transcript-api"
        ) from exc

    api = YouTubeTranscriptApi()
    listing = api.list(video_id)
    try:
        transcript = listing.find_transcript(list(languages))
    except NoTranscriptFound:
        transcript = next(iter(listing))
        logger.info(
            f"No {'/'.join(languages)} transcript for {video_id}, using "
            f"{transcript.language_code!r} instead"
        )

    fetched = transcript.fetch()
    return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]


def chunk_transcript(
    segments: list[dict],
    *,
    max_chars: int = 3000,
) -> list[TranscriptChunk]:
    """Group consecutive transcript segments into ~max_chars windows.

    Chunking (rather than synthesizing from the whole transcript at once)
    keeps each LLM call's context small and lets a long video yield several
    concepts instead of just one diluted summary.
    """
    chunks: list[TranscriptChunk] = []
    buf: list[str] = []
    buf_start: float | None = None
    buf_len = 0

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if buf_start is None:
            buf_start = seg["start"]
        if buf_len + len(text) > max_chars and buf:
            chunks.append(TranscriptChunk(text=" ".join(buf), start_seconds=buf_start))
            buf, buf_len, buf_start = [], 0, seg["start"]
        buf.append(text)
        buf_len += len(text) + 1

    if buf and buf_start is not None:
        chunks.append(TranscriptChunk(text=" ".join(buf), start_seconds=buf_start))
    return chunks


def timestamped_url(video_id: str, start_seconds: float) -> str:
    return f"{video_url(video_id)}&t={int(start_seconds)}s"


def load_processed_videos(state_path) -> set[str]:
    """Video ids already run through the pipeline, for idempotent re-runs.

    Tracked separately from concept_synthesis.existing_concept_ids(): a
    processed video can legitimately yield zero concepts (nothing tactical in
    it), and even when it yields some, their concept_id is derived from an
    LLM-generated title that is NOT guaranteed identical across runs — so
    concept_id-based dedup alone would let the same video (e.g. via
    --channel twice, or a video that's both in --urls-file and pulled again
    by --channel) be re-transcribed and re-synthesized, burning LLM calls and
    producing duplicate-content concepts under different ids.
    """
    path = Path(state_path)
    if path.exists():
        try:
            return set(json.loads(path.read_text(encoding="utf-8")).get("video_ids", []))
        except Exception:  # noqa: BLE001 — a corrupt state file must not block a run
            return set()
    return set()


def save_processed_videos(state_path, video_ids: set[str]) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"video_ids": sorted(video_ids)}, indent=2),
        encoding="utf-8",
    )
