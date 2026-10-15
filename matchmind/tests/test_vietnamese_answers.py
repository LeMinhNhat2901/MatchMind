"""Tests for native-language narration: number grounding, glossary, the
narrator's answer/rationale split, and the dashboard's localize_advice fallback.
Fake LLM calls only — no network."""
from __future__ import annotations

import json

from matchmind.agent import translation
from matchmind.agent.nodes import narrator_node
from matchmind.agent.translation import (
    _placeholder_legend,
    load_glossary,
    localize_advice,
    translate_from_english,
    ungrounded_numbers,
)
from matchmind.schema.match_state import AlternativeAction, TacticalAdvice


def _advice(**overrides) -> TacticalAdvice:
    base = dict(
        recommended_action="Pass to player 9 in the half-space",
        reasoning="Player 11 has 8.2m of space ahead and a 2v1 overload; the pass raises control by 3.1%.",
        confidence=0.82,
        evidence=[],
        cited_concepts=["Half-Space"],
        alternatives=[AlternativeAction(action="Shoot", why_not="Angle too tight", delta_pitch_control=-1.4)],
        focus_player_id=11,
    )
    base.update(overrides)
    return TacticalAdvice(**base)


# ── ungrounded_numbers ────────────────────────────────────────────────────────

DATA = "player 11, 8.23 m, confidence 0.82, delta 3.1, pass_to_9, alt -1.4"


def test_grounding_accepts_rounded_and_scaled_numbers():
    text = "Số 11 nên chuyền cho số 9: còn 8.2 mét khoảng trống, độ tin cậy 82%, tăng 3.1%, phương án sút giảm 1.4%."
    assert ungrounded_numbers(text, DATA, {9, 11}) == []


def test_grounding_rejects_wrong_player_reference():
    """The production bug: 'Player 11' rendered as 'Cầu thủ 1'."""
    assert ungrounded_numbers("Cầu thủ 1 nên chuyền bóng.", DATA, {9, 11}) == ["1"]


def test_grounding_rejects_invented_statistic():
    assert ungrounded_numbers("Kiểm soát tăng 27%.", DATA, {9, 11}) == ["27"]


def test_grounding_allows_small_counts_outside_player_refs():
    assert ungrounded_numbers("Có 2 phương án ở 1/3 cuối sân.", DATA, {9, 11}) == []


def test_distance_after_so_is_not_a_player_reference():
    # "số 8 mét" style phrasing must be checked as a distance, not a shirt number.
    assert ungrounded_numbers("khoảng trống 8 m phía trước", DATA, {11}) == []


# ── glossary + translation prompt ─────────────────────────────────────────────

def test_glossary_only_for_vietnamese():
    load_glossary.cache_clear()
    glossary = load_glossary("Vietnamese")
    assert glossary and "nách trung lộ" in glossary
    assert "<!--" not in glossary
    assert load_glossary("Spanish") is None


def test_translation_prompt_has_glossary_and_typed_legend():
    seen = {}

    def _llm(prompt: str, **kwargs):
        seen["prompt"] = prompt
        return "Số @@0@@ nên chuyền, còn @@1@@m khoảng trống, tăng @@2@@.", 5

    out = translate_from_english("Player 11 should pass, 8.2m of space, +3.1%.", "Vietnamese", llm_call=_llm)
    assert out == "Số 11 nên chuyền, còn 8.2m khoảng trống, tăng +3.1%."
    assert "nách trung lộ" in seen["prompt"]
    assert "@@0@@ = a player's shirt number" in seen["prompt"]
    assert "@@2@@ = a percentage" in seen["prompt"]
    assert "8.2" not in seen["prompt"].split("TEXT:")[1]  # values stay hidden


def test_placeholder_legend_types():
    legend = _placeholder_legend("Player 7 is 12m away, 40% control, 3 options")
    assert "@@0@@ = a player's shirt number" in legend
    assert "@@1@@ = a distance" in legend
    assert "@@2@@ = a percentage" in legend
    assert "@@3@@ = a number" in legend


# ── localize_advice ───────────────────────────────────────────────────────────

def _no_llm(prompt: str, **kwargs):
    raise AssertionError("should not translate")


def test_localize_is_noop_when_narrated_natively():
    advice = _advice(narrative="Số 11 nên chuyền.", rationale=["Còn 8.2 m khoảng trống."],
                     narrative_language="Vietnamese")
    out = localize_advice(advice, "Vietnamese", llm_call=_no_llm)
    assert out.answer == "Số 11 nên chuyền."
    assert out.rationale == ["Còn 8.2 m khoảng trống."]


def test_localize_translates_english_fallback():
    advice = _advice(narrative="Player 11 should pass.", rationale=["Space is 8.2m.", "Shot angle tight."],
                     narrative_language="English")

    def _llm(prompt: str, **kwargs):
        if "- Space" in prompt:
            return "- Khoảng trống @@0@@m.\n- Góc sút hẹp.", 5
        return "Số @@0@@ nên chuyền.", 5

    out = localize_advice(advice, "Vietnamese", llm_call=_llm)
    assert out.answer == "Số 11 nên chuyền."
    assert out.rationale == ["Khoảng trống 8.2m.", "Góc sút hẹp."]


def test_localize_keeps_english_bullets_when_line_count_changes():
    advice = _advice(narrative="Pass.", rationale=["A is 8.2m.", "B."], narrative_language="English")

    def _llm(prompt: str, **kwargs):
        if "- A" in prompt:
            return "- A và B gộp lại @@0@@m.", 5
        return "Chuyền.", 5

    assert localize_advice(advice, "Vietnamese", llm_call=_llm).rationale == ["A is 8.2m.", "B."]


# ── narrator_node ─────────────────────────────────────────────────────────────

def _state(language: str = "Vietnamese") -> dict:
    return {"tactical_advice": _advice(), "trace": [], "question": "What should player 11 do?",
            "response_language": language, "total_tokens_used": 0}


def _patch_llm(monkeypatch, responses: list[str]):
    prompts: list[str] = []

    def _llm(prompt: str, **kwargs):
        prompts.append(prompt)
        return responses[len(prompts) - 1], 50

    monkeypatch.setattr(narrator_node, "call_narrator_llm", _llm)
    return prompts


def test_narrator_writes_natively_with_rationale(monkeypatch):
    prompts = _patch_llm(monkeypatch, [json.dumps({
        "answer": "Số 11 nên chuyền vào nách trung lộ cho số 9.",
        "rationale": ["Số 11 còn 8.2 m khoảng trống phía trước.", "Đường chuyền tăng kiểm soát 3.1%."],
    }, ensure_ascii=False)])
    out = narrator_node.narrator_node(_state())
    advice = out["tactical_advice"]
    assert advice.narrative.startswith("Số 11")
    assert len(advice.rationale) == 2
    assert advice.narrative_language == "Vietnamese"
    assert len(prompts) == 1 and "Write in Vietnamese" in prompts[0]
    assert "nách trung lộ" in prompts[0]  # glossary injected


def test_narrator_falls_back_to_english_when_numbers_ungrounded(monkeypatch):
    prompts = _patch_llm(monkeypatch, [
        json.dumps({"answer": "Cầu thủ 1 nên chuyền.", "rationale": []}, ensure_ascii=False),
        json.dumps({"answer": "Player 11 should pass to 9.", "rationale": ["8.2m of space ahead."]}),
    ])
    out = narrator_node.narrator_node(_state())
    advice = out["tactical_advice"]
    assert len(prompts) == 2
    assert advice.narrative_language == "English"
    assert advice.narrative == "Player 11 should pass to 9."
    assert out["trace"][-1]["rejected"]


def test_narrator_leaves_narrative_empty_when_all_attempts_fail(monkeypatch):
    def _boom(prompt: str, **kwargs):
        raise RuntimeError("429")

    monkeypatch.setattr(narrator_node, "call_narrator_llm", _boom)
    out = narrator_node.narrator_node(_state())
    assert out["tactical_advice"].narrative is None
    assert out["tactical_advice"].rationale == []


def test_narrator_english_makes_single_attempt(monkeypatch):
    prompts = _patch_llm(monkeypatch, ["not json at all"])
    out = narrator_node.narrator_node(_state("English"))
    assert len(prompts) == 1
    assert out["tactical_advice"].narrative is None


def test_translation_module_uses_narrator_llm_by_default():
    assert translate_from_english.__kwdefaults__["llm_call"] is translation.call_narrator_llm
