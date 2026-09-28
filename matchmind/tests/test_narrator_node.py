"""Tests for narrator_node's style-guide loading and voice injection."""
from __future__ import annotations

from matchmind.agent.nodes import narrator_node


def test_voice_section_empty_when_no_style_guide(monkeypatch, tmp_path):
    monkeypatch.setattr(narrator_node, "STYLE_GUIDE_PATH", tmp_path / "does_not_exist.md")
    narrator_node._load_style_guide.cache_clear()
    assert narrator_node._voice_section() == ""


def test_voice_section_includes_style_guide_content(monkeypatch, tmp_path):
    guide_path = tmp_path / "style_guide.md"
    guide_path.write_text("- Uses short, punchy sentences", encoding="utf-8")
    monkeypatch.setattr(narrator_node, "STYLE_GUIDE_PATH", guide_path)
    narrator_node._load_style_guide.cache_clear()
    section = narrator_node._voice_section()
    assert "Uses short, punchy sentences" in section
    narrator_node._load_style_guide.cache_clear()


def test_narrator_node_noop_when_no_advice():
    state = {"tactical_advice": None, "trace": []}
    result = narrator_node.narrator_node(state)
    assert result is state
