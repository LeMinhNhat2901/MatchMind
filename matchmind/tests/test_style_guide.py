"""Tests for style_guide.py's extraction helpers (fake llm_call, no network)."""
from __future__ import annotations

from matchmind.knowledge_base.style_guide import clean_transcript_excerpt, extract_style_guide


def _fake_llm(response: str):
    def _call(prompt: str, **kwargs) -> tuple[str, int]:
        return response, 10

    return _call


def test_clean_transcript_excerpt_too_short_skips_without_llm_call():
    calls = []

    def _tracking_llm(prompt: str, **kwargs):
        calls.append(prompt)
        return "cleaned", 0

    result = clean_transcript_excerpt("too short", llm_call=_tracking_llm)
    assert result is None
    assert calls == []


def test_clean_transcript_excerpt_happy_path():
    result = clean_transcript_excerpt(
        "um so basically when the team presses high" * 5,
        llm_call=_fake_llm("So, basically, when the team presses high..."),
    )
    assert result == "So, basically, when the team presses high..."


def test_clean_transcript_excerpt_llm_failure_returns_none():
    def _failing_llm(prompt: str, **kwargs):
        raise RuntimeError("429")

    result = clean_transcript_excerpt("x" * 100, llm_call=_failing_llm)
    assert result is None


def test_extract_style_guide_happy_path():
    guide = extract_style_guide(
        [("Tifo Football", "Some cleaned analyst commentary." * 10)],
        llm_call=_fake_llm("- Uses short, punchy sentences\n- Frequent metaphor use"),
    )
    assert guide == "- Uses short, punchy sentences\n- Frequent metaphor use"


def test_extract_style_guide_empty_input_returns_none_without_llm_call():
    calls = []

    def _tracking_llm(prompt: str, **kwargs):
        calls.append(prompt)
        return "x", 0

    guide = extract_style_guide([], llm_call=_tracking_llm)
    assert guide is None
    assert calls == []


def test_extract_style_guide_llm_failure_returns_none():
    def _failing_llm(prompt: str, **kwargs):
        raise RuntimeError("rate limited")

    guide = extract_style_guide([("X", "y" * 100)], llm_call=_failing_llm)
    assert guide is None
