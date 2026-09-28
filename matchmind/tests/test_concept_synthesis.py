"""
Tests for the LLM-driven synthesis step shared by the Wikipedia and YouTube
ingestion pipelines. The real LLM is never called here — a fake `llm_call`
is injected (synthesize_concept accepts it via the llm_call= parameter).
"""
from __future__ import annotations

import json

from matchmind.knowledge_base.concept_synthesis import (
    ALLOWED_CATEGORIES,
    ALLOWED_PHASES,
    slugify,
    synthesize_concept,
)


def _fake_llm(payload: dict):
    def _call(prompt: str, **kwargs) -> tuple[str, int]:
        return json.dumps(payload), 42

    return _call


def test_slugify():
    assert slugify("Gegenpressing") == "gegenpressing"
    assert slugify("False 9") == "false_9"
    assert slugify("  Half-Space / Exploitation!!  ") == "half_space_exploitation"


def test_synthesize_concept_happy_path():
    payload = {
        "skip": False,
        "title": "Gegenpressing",
        "category": "defending",
        "subcategory": "counter_press",
        "phase": "defensive_transition",
        "definition": "Pressing the opponent immediately after losing the ball to win it back quickly.",
        "content": "A" * 150,
        "conditions": ["ball_just_lost"],
        "objective": ["win_ball_back_quickly"],
        "possible_actions": ["press_ball_carrier"],
        "contraindications": ["team_out_of_shape"],
        "trigger_conditions": ["phase_of_play_defensive_transition"],
        "spatial_requirements": ["compact_team_shape"],
        "player_behaviours": ["angle_press"],
        "related_concepts": ["high_press"],
    }
    concept = synthesize_concept(
        source_title="Gegenpressing",
        source_text="Gegenpressing is a tactic where the team presses immediately after losing the ball." * 3,
        source_name="Wikipedia: Gegenpressing",
        source_url="https://en.wikipedia.org/wiki/Gegenpressing",
        source_license="CC BY-SA 4.0",
        retrieved_at="2026-09-11",
        id_prefix="wiki_",
        llm_call=_fake_llm(payload),
    )
    assert concept is not None
    assert concept.concept_id == "wiki_gegenpressing"
    assert concept.category == "defending"
    assert concept.subcategory == "counter_press"
    assert concept.definition is not None and "losing the ball" in concept.definition
    assert concept.spatial_requirements == ["compact_team_shape"]
    assert concept.player_behaviours == ["angle_press"]
    assert concept.related_concepts == ["high_press"]
    assert concept.source is not None
    assert concept.source.name == "Wikipedia: Gegenpressing"
    assert concept.source.license == "CC BY-SA 4.0"


def test_synthesize_concept_skip_path():
    concept = synthesize_concept(
        source_title="4th place trophy",
        source_text="Trivia about a specific award given to fourth-place finishers." * 3,
        source_name="Wikipedia: 4th place trophy",
        source_url="https://en.wikipedia.org/wiki/4th_place_trophy",
        source_license="CC BY-SA 4.0",
        retrieved_at="2026-09-11",
        id_prefix="wiki_",
        llm_call=_fake_llm({"skip": True, "reason": "not a tactical concept"}),
    )
    assert concept is None


def test_synthesize_concept_short_source_skipped_without_llm_call():
    calls = []

    def _tracking_llm(prompt: str, **kwargs):
        calls.append(prompt)
        return json.dumps({"skip": True}), 0

    concept = synthesize_concept(
        source_title="Too short",
        source_text="short",
        source_name="x",
        source_url="x",
        source_license="x",
        retrieved_at="2026-09-11",
        id_prefix="wiki_",
        llm_call=_tracking_llm,
    )
    assert concept is None
    assert calls == []  # must not spend an LLM call on obviously-too-short text


def test_synthesize_concept_invalid_category_falls_back():
    payload = {
        "skip": False,
        "title": "Some Concept",
        "category": "not_a_real_category",
        "phase": "not_a_real_phase",
        "content": "B" * 150,
    }
    concept = synthesize_concept(
        source_title="Some Concept",
        source_text="x" * 150,
        source_name="x",
        source_url="x",
        source_license="x",
        retrieved_at="2026-09-11",
        id_prefix="wiki_",
        llm_call=_fake_llm(payload),
    )
    assert concept is not None
    assert concept.category in ALLOWED_CATEGORIES
    assert concept.phase in ALLOWED_PHASES


def test_synthesize_concept_unparseable_output_returns_none():
    def _garbage_llm(prompt: str, **kwargs):
        return "not json at all", 0

    concept = synthesize_concept(
        source_title="Garbage",
        source_text="x" * 150,
        source_name="x",
        source_url="x",
        source_license="x",
        retrieved_at="2026-09-11",
        id_prefix="wiki_",
        llm_call=_garbage_llm,
    )
    assert concept is None
