"""
sources.txt -> knowledge base.

One file (knowledge_base/sources/sources.txt) where the user pastes links, and
one command (scripts/ingest_sources.py) that processes all of them. Each line
is classified; only YouTube is automated today:

    youtube_video    "Label | https://youtu.be/…" or a bare video URL
    youtube_channel  "https://www.youtube.com/@Handle [| max=N] [| sort=popular]"
                     -> N newest uploads, or the N most-viewed with sort=popular
    unsupported      anything else (e.g. articles) — listed, never an error; other
                     source types need their own handler (websites have ToS and
                     per-site formats, see matchmind-kb-taxonomy-v2 decisions)

With a DuplicateGate (the CLI scripts always pass one), a concept that nearly
copies one already in the KB is skipped and reported instead of saved.

Re-running is safe and cheap: a video is marked processed only after every
transcript chunk went through the LLM without an error, and its concepts are
written only then (a video is atomic). So when the LLM quota runs out mid-run,
the unfinished videos are simply picked up again next time, without leaving
half a video's concepts behind to be duplicated.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from matchmind.knowledge_base.youtube_source import (
    chunk_transcript,
    load_processed_videos,
    parse_video_id,
    save_processed_videos,
    timestamped_url,
)

if TYPE_CHECKING:
    from matchmind.knowledge_base.dedup import DuplicateGate

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL_MAX = 10
ID_PREFIX = "yt_"
_CHANNEL_RE = re.compile(r"youtube\.com/(?:@[\w.\-]+|channel/[\w\-]+|c/[\w\-]+|user/[\w\-]+)", re.IGNORECASE)
_MAX_RE = re.compile(r"^max\s*=\s*(\d+)$", re.IGNORECASE)
_SORT_RE = re.compile(r"^sort\s*=\s*(newest|popular)$", re.IGNORECASE)


def _is_option(part: str) -> bool:
    return bool(_MAX_RE.match(part) or _SORT_RE.match(part))


@dataclass(frozen=True)
class SourceEntry:
    kind: Literal["youtube_video", "youtube_channel", "unsupported"]
    url: str
    label: str | None = None
    video_id: str | None = None
    max_videos: int = DEFAULT_CHANNEL_MAX
    line_no: int = 0
    sort: Literal["newest", "popular"] = "newest"


def classify_line(line: str, line_no: int = 0) -> SourceEntry | None:
    """One sources.txt line -> SourceEntry, or None for blank/comment lines.

    Format: [Label |] URL [| max=N] [| sort=newest|popular]   — any order after
    the URL; text after " #" is a comment.
    """
    line = line.split(" #", 1)[0].strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split("|")]
    url = next((p for p in parts if "://" in p or "youtu" in p.lower()), parts[-1])
    label = next((p for p in parts if p is not url and not _is_option(p) and p), None)
    max_match = next((_MAX_RE.match(p) for p in parts if _MAX_RE.match(p)), None)
    max_videos = int(max_match.group(1)) if max_match else DEFAULT_CHANNEL_MAX
    sort_match = next((_SORT_RE.match(p) for p in parts if _SORT_RE.match(p)), None)
    sort = sort_match.group(1).lower() if sort_match else "newest"

    try:
        return SourceEntry("youtube_video", url, label, video_id=parse_video_id(url), line_no=line_no)
    except ValueError:
        pass
    if _CHANNEL_RE.search(url):
        return SourceEntry("youtube_channel", url, label, max_videos=max_videos, line_no=line_no, sort=sort)
    return SourceEntry("unsupported", url, label, line_no=line_no)


def parse_sources_text(text: str) -> list[SourceEntry]:
    entries = []
    for n, line in enumerate(text.splitlines(), start=1):
        entry = classify_line(line, n)
        if entry is not None:
            entries.append(entry)
    return entries


@dataclass
class IngestReport:
    videos_new: int = 0
    videos_skipped: int = 0
    videos_failed: int = 0
    concepts_written: int = 0
    chunks_without_concept: int = 0
    unsupported: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    planned: list[str] = field(default_factory=list)
    duplicates_skipped: list[str] = field(default_factory=list)   # "new title ≈ existing_id (0.91)"
    aborted: bool = False


class YouTubeIngestor:
    """Runs the YouTube part of sources.txt. Every external call is injectable
    (the defaults are the real YouTube/LLM functions) so the loop is testable."""

    def __init__(
        self,
        *,
        concepts_root: Path,
        state_path: Path,
        known_ids: set[str],
        synthesize: Callable | None = None,
        llm_call: Callable | None = None,
        fetch_transcript: Callable[[str], list[dict]] | None = None,
        list_channel: Callable[[str, int, str], list[str]] | None = None,
        lookup_channel: Callable[[str], str] | None = None,
        save: Callable | None = None,
        sleep_s: float = 2.0,
        max_chunk_chars: int = 3000,
        max_consecutive_llm_errors: int = 3,
        dry_run: bool = False,
        dedup: "DuplicateGate | None" = None,
        log: Callable[..., None] = logger.info,
    ) -> None:
        from matchmind.knowledge_base import concept_synthesis, youtube_source
        from matchmind.llm_client import call_llm

        self.concepts_root = Path(concepts_root)
        self.state_path = Path(state_path)
        self.known_ids = known_ids
        self.synthesize = synthesize or concept_synthesis.synthesize_concept
        self.llm_call = llm_call or call_llm
        self.fetch_transcript = fetch_transcript or youtube_source.fetch_transcript
        self.list_channel = list_channel or (
            lambda url, n, sort="newest": youtube_source.list_channel_videos(url, max_videos=n, sort=sort))
        self.lookup_channel = lookup_channel or (lambda vid: youtube_source.fetch_video_metadata(vid)["channel"])
        self.save = save or concept_synthesis.save_concept
        self.sleep_s = sleep_s
        self.max_chunk_chars = max_chunk_chars
        self.max_consecutive_llm_errors = max_consecutive_llm_errors
        self.dry_run = dry_run
        self.dedup = dedup
        self.log = log
        self._llm_errors = 0
        self._consecutive_llm_errors = 0

    # The LLM wrapper handed to synthesize_concept: it swallows exceptions into
    # "None" (by design, for batch runs), so this is the only way to tell
    # "the LLM failed" apart from "this chunk had no tactical concept".
    def _tracked_llm(self, prompt: str, **kwargs):
        try:
            result = self.llm_call(prompt, **kwargs)
        except Exception:
            self._llm_errors += 1
            self._consecutive_llm_errors += 1
            raise
        self._consecutive_llm_errors = 0
        return result

    def run(self, entries: list[SourceEntry]) -> IngestReport:
        report = IngestReport()
        processed = load_processed_videos(self.state_path)
        queue: list[tuple[str, str | None, int]] = []      # (video_id, label, line_no)
        for e in entries:
            if e.kind == "unsupported":
                report.unsupported.append(f"line {e.line_no}: {e.url}")
            elif e.kind == "youtube_video":
                queue.append((e.video_id, e.label, e.line_no))
            else:
                try:
                    urls = self.list_channel(e.url, e.max_videos, e.sort)
                except Exception as exc:  # noqa: BLE001 — one bad channel must not stop the file
                    report.errors.append(f"line {e.line_no}: cannot list channel {e.url}: {exc}")
                    continue
                for u in urls:
                    try:
                        queue.append((parse_video_id(u), e.label, e.line_no))
                    except ValueError:
                        continue

        seen: set[str] = set()
        for video_id, label, line_no in queue:
            if video_id in seen:
                continue
            seen.add(video_id)
            if video_id in processed:
                report.videos_skipped += 1
                continue
            if self.dry_run:
                report.planned.append(video_id)
                continue
            if report.aborted:
                break
            self._ingest_video(video_id, label, line_no, processed, report)
            if self._consecutive_llm_errors >= self.max_consecutive_llm_errors:
                report.aborted = True
                report.errors.append(
                    f"stopped after {self._consecutive_llm_errors} consecutive LLM errors "
                    "(quota/outage?) — run the command again later to continue"
                )
                break
        return report

    def _ingest_video(self, video_id: str, label: str | None, line_no: int,
                      processed: set[str], report: IngestReport) -> None:
        channel = label
        if channel is None:
            try:
                channel = self.lookup_channel(video_id)
            except Exception:  # noqa: BLE001 — attribution fallback only
                channel = "YouTube"
        try:
            segments = self.fetch_transcript(video_id)
        except Exception as exc:  # noqa: BLE001 — often transient; retried next run
            report.videos_failed += 1
            report.errors.append(f"line {line_no}: {video_id}: transcript unavailable: {exc}")
            return

        chunks = chunk_transcript(segments, max_chars=self.max_chunk_chars)
        self.log(f"{channel} — {video_id}: {len(chunks)} chunk(s)")
        errors_before = self._llm_errors
        concepts = []
        retrieved_at = _dt.date.today().isoformat()
        for i, chunk in enumerate(chunks, 1):
            concept = self.synthesize(
                source_title=f"{channel} transcript chunk {i}",
                source_text=chunk.text,
                source_name=channel,
                source_url=timestamped_url(video_id, chunk.start_seconds),
                source_license="fair-use paraphrase (transcript, not redistributed verbatim)",
                retrieved_at=retrieved_at,
                id_prefix=ID_PREFIX,
                llm_call=self._tracked_llm,
            )
            if self.sleep_s:
                time.sleep(self.sleep_s)
            if self._llm_errors > errors_before:
                break                                   # the video will be retried as a whole
            if concept is None:
                report.chunks_without_concept += 1
            elif concept.concept_id not in self.known_ids:
                dup = self.dedup.check(concept, pending=concepts) if self.dedup else None
                if dup is None:
                    concepts.append(concept)
                else:
                    report.duplicates_skipped.append(f"{concept.title} ≈ {dup[0]} ({dup[1]:.2f})")

        if self._llm_errors > errors_before:
            report.videos_failed += 1
            report.errors.append(f"line {line_no}: {video_id}: LLM error — will retry next run")
            return

        for concept in concepts:
            self.save(concept, self.concepts_root)
            self.known_ids.add(concept.concept_id)
            if self.dedup:
                self.dedup.add(concept)
            report.concepts_written += 1
        processed.add(video_id)
        save_processed_videos(self.state_path, processed)
        report.videos_new += 1
