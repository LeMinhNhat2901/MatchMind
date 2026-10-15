"""Tests for youtube_source's pure helpers (no network, no youtube-transcript-api needed)."""
from __future__ import annotations

import pytest

from matchmind.knowledge_base.youtube_source import (
    chunk_transcript,
    load_processed_videos,
    parse_video_id,
    save_processed_videos,
    timestamped_url,
)


def test_parse_video_id_from_watch_url():
    assert parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_parse_video_id_from_short_url():
    assert parse_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_parse_video_id_bare_id():
    assert parse_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_parse_video_id_invalid_raises():
    with pytest.raises(ValueError):
        parse_video_id("not a url or id")


def test_chunk_transcript_groups_by_max_chars():
    segments = [
        {"text": "hello world", "start": 0.0, "duration": 2.0},
        {"text": "this is a tactics video", "start": 2.0, "duration": 2.0},
        {"text": "about pressing", "start": 4.0, "duration": 2.0},
    ]
    chunks = chunk_transcript(segments, max_chars=20)
    assert len(chunks) >= 2
    assert chunks[0].start_seconds == 0.0
    assert "hello world" in chunks[0].text


def test_chunk_transcript_skips_empty_segments():
    segments = [{"text": "  ", "start": 0.0, "duration": 1.0}, {"text": "real content", "start": 1.0, "duration": 1.0}]
    chunks = chunk_transcript(segments, max_chars=100)
    assert len(chunks) == 1
    assert chunks[0].text == "real content"


def test_timestamped_url():
    url = timestamped_url("abc12345678", 90.4)
    assert url == "https://www.youtube.com/watch?v=abc12345678&t=90s"


def test_load_processed_videos_missing_file_returns_empty_set(tmp_path):
    assert load_processed_videos(tmp_path / "does_not_exist.json") == set()


def test_save_and_load_processed_videos_round_trip(tmp_path):
    state_path = tmp_path / "_state" / "youtube_processed.json"
    save_processed_videos(state_path, {"abc12345678", "xyz98765432"})
    assert load_processed_videos(state_path) == {"abc12345678", "xyz98765432"}


def test_load_processed_videos_corrupt_file_returns_empty_set(tmp_path):
    state_path = tmp_path / "youtube_processed.json"
    state_path.write_text("not valid json", encoding="utf-8")
    assert load_processed_videos(state_path) == set()
