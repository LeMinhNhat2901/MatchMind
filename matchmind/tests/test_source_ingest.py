"""sources.txt -> knowledge base: line classification + the YouTube ingest loop.

Every network/LLM dependency is injected, so these tests never touch YouTube
or an LLM.
"""
from __future__ import annotations

from matchmind.knowledge_base.source_ingest import (
    DEFAULT_CHANNEL_MAX,
    YouTubeIngestor,
    classify_line,
    parse_sources_text,
)
from matchmind.knowledge_base.youtube_source import load_processed_videos
from matchmind.tests.fixtures.concepts import make_concept

VID_A = "https://www.youtube.com/watch?v=AAAAAAAAAAA"
VID_B = "https://youtu.be/BBBBBBBBBBB?si=share"


# ── classification ────────────────────────────────────────────────────────────

def test_blank_and_comment_lines_are_ignored():
    assert classify_line("") is None
    assert classify_line("   ") is None
    assert classify_line("# a note") is None


def test_video_with_label_and_inline_comment():
    e = classify_line(f"Tifo Football | {VID_A}   # great one")
    assert (e.kind, e.label, e.video_id) == ("youtube_video", "Tifo Football", "AAAAAAAAAAA")


def test_bare_short_link_video():
    e = classify_line(VID_B)
    assert (e.kind, e.label, e.video_id) == ("youtube_video", None, "BBBBBBBBBBB")


def test_channel_lines_default_and_override_max():
    e = classify_line("https://www.youtube.com/@TifoFootball")
    assert (e.kind, e.max_videos) == ("youtube_channel", DEFAULT_CHANNEL_MAX)
    e = classify_line("Tifo | https://www.youtube.com/@TifoFootball/videos | max=20")
    assert (e.kind, e.label, e.max_videos) == ("youtube_channel", "Tifo", 20)
    assert classify_line("https://www.youtube.com/channel/UCabcdefghijk").kind == "youtube_channel"


def test_other_websites_are_unsupported_not_errors():
    e = classify_line("https://www.fifatrainingcentre.com/en/game/some-article.php")
    assert e.kind == "unsupported"


def test_parse_keeps_line_numbers():
    entries = parse_sources_text(f"# header\n\n{VID_A}\nhttps://example.com/x\n")
    assert [(e.line_no, e.kind) for e in entries] == [(3, "youtube_video"), (4, "unsupported")]


# ── ingest loop ───────────────────────────────────────────────────────────────

SEGMENTS = [{"text": "part one of the analysis", "start": 0.0, "duration": 5.0},
            {"text": "part two of the analysis", "start": 5.0, "duration": 5.0}]


class Fakes:
    def __init__(self, *, llm_fails: bool = False, transcript_fails: frozenset[str] | set[str] = frozenset()):
        self.llm_fails = llm_fails
        self.transcript_fails = transcript_fails
        self.saved: list[str] = []
        self.channel_calls: list[tuple[str, int]] = []

    def llm(self, prompt, **kw):
        if self.llm_fails:
            raise RuntimeError("429 tokens per day")
        return "{}", 1

    def synthesize(self, *, source_url, llm_call, **kw):
        # Same contract as concept_synthesis.synthesize_concept: an LLM error is
        # swallowed and reported as "no concept" (None).
        try:
            llm_call("prompt")
        except Exception:
            return None
        vid = source_url.split("v=")[1][:11]
        return make_concept(f"yt_{vid.lower()}_{len(self.saved)}")

    def fetch_transcript(self, video_id):
        if video_id in self.transcript_fails:
            raise RuntimeError("no captions")
        return SEGMENTS

    def list_channel(self, url, max_videos):
        self.channel_calls.append((url, max_videos))
        return [VID_A, VID_B][:max_videos]

    def save(self, concept, out_dir):
        self.saved.append(concept.concept_id)
        return out_dir / f"{concept.concept_id}.json"


def _ingestor(tmp_path, fakes, **kw):
    return YouTubeIngestor(
        concepts_root=tmp_path / "youtube", state_path=tmp_path / "_state" / "yt.json", known_ids=set(),
        synthesize=fakes.synthesize, llm_call=fakes.llm, fetch_transcript=fakes.fetch_transcript,
        list_channel=fakes.list_channel, lookup_channel=lambda vid: "Some Channel", save=fakes.save,
        sleep_s=0.0, max_chunk_chars=30, log=lambda *a, **k: None, **kw,
    )


def test_new_video_writes_concepts_and_is_marked_processed(tmp_path):
    fakes = Fakes()
    report = _ingestor(tmp_path, fakes).run(parse_sources_text(VID_A))
    assert report.videos_new == 1 and report.concepts_written == 2   # 2 chunks at 30 chars
    assert load_processed_videos(tmp_path / "_state" / "yt.json") == {"AAAAAAAAAAA"}


def test_already_processed_video_is_skipped_without_llm_calls(tmp_path):
    fakes = Fakes()
    _ingestor(tmp_path, fakes).run(parse_sources_text(VID_A))
    fakes.llm_fails = True                                     # any LLM call would now error
    report = _ingestor(tmp_path, fakes).run(parse_sources_text(VID_A))
    assert report.videos_skipped == 1 and report.videos_new == 0 and not report.errors


def test_llm_failure_leaves_video_unprocessed_for_the_next_run(tmp_path):
    report = _ingestor(tmp_path, Fakes(llm_fails=True)).run(parse_sources_text(VID_A))
    assert report.videos_failed == 1 and report.concepts_written == 0
    assert load_processed_videos(tmp_path / "_state" / "yt.json") == set()


def test_repeated_llm_failures_abort_the_run(tmp_path):
    # A video stops at its first LLM error, so each failed video adds one to the
    # consecutive-error count: with a limit of 2, the third video is never tried.
    vid_c = "https://www.youtube.com/watch?v=CCCCCCCCCCC"
    report = _ingestor(tmp_path, Fakes(llm_fails=True), max_consecutive_llm_errors=2).run(
        parse_sources_text(f"{VID_A}\n{VID_B}\n{vid_c}\n"))
    assert report.aborted
    assert report.videos_failed == 2


def test_transcript_failure_is_reported_and_not_marked(tmp_path):
    report = _ingestor(tmp_path, Fakes(transcript_fails={"AAAAAAAAAAA"})).run(parse_sources_text(VID_A))
    assert report.videos_failed == 1 and report.errors
    assert load_processed_videos(tmp_path / "_state" / "yt.json") == set()


def test_channel_line_expands_and_dedups_against_listed_video(tmp_path):
    fakes = Fakes()
    text = f"{VID_A}\nhttps://www.youtube.com/@Tifo | max=2\n"
    report = _ingestor(tmp_path, fakes).run(parse_sources_text(text))
    assert fakes.channel_calls == [("https://www.youtube.com/@Tifo", 2)]
    assert report.videos_new == 2                              # A once (not twice) + B


def test_unsupported_lines_are_listed(tmp_path):
    report = _ingestor(tmp_path, Fakes()).run(parse_sources_text("https://example.com/article\n"))
    assert report.unsupported == ["line 1: https://example.com/article"]


def test_legacy_build_kb_youtube_cli_maps_options_to_entries(tmp_path):
    """build_kb_youtube now delegates to YouTubeIngestor (it used to mark a video
    processed even when every chunk failed with an LLM error)."""
    from matchmind.scripts.build_kb_youtube import entries_from_cli

    urls = tmp_path / "v.txt"
    urls.write_text(f"Tifo | {VID_A}\n", encoding="utf-8")
    entries = entries_from_cli(str(urls), [VID_B], "https://www.youtube.com/@Tifo", 7)
    assert [(e.kind, e.label) for e in entries] == [
        ("youtube_video", "Tifo"), ("youtube_video", None), ("youtube_channel", None)]
    assert entries[-1].max_videos == 7


def test_dry_run_calls_nothing(tmp_path):
    fakes = Fakes(llm_fails=True)
    report = _ingestor(tmp_path, fakes, dry_run=True).run(parse_sources_text(f"{VID_A}\n"))
    assert report.videos_new == 0 and report.planned == ["AAAAAAAAAAA"] and not fakes.saved
