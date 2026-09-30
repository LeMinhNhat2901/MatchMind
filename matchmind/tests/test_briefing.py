"""The Coaching briefing card shows bullets only — never the chat answer again.

User feedback (2026-09-30): the narrative paragraph already appears in the chat;
repeating it in the briefing card is noise. Show the rationale bullets.
"""
from matchmind.dashboard.components.tactical_advice import briefing_bullets
from matchmind.schema.match_state import TacticalAdvice


def _advice(**kw):
    base = dict(recommended_action="x", reasoning="First reason here. Second reason here. Third reason.",
                confidence=0.8, evidence=[], cited_concepts=[], alternatives=[])
    base.update(kw)
    return TacticalAdvice(**base)


def test_briefing_uses_rationale_bullets_not_the_narrative():
    a = _advice(narrative="A long persuasive paragraph.", rationale=["Point one.", "Point two."])
    bullets = briefing_bullets(a)
    assert bullets == ["Point one.", "Point two."]
    assert all("persuasive paragraph" not in b for b in bullets)


def test_briefing_falls_back_to_reasoning_sentences():
    bullets = briefing_bullets(_advice())
    assert bullets[:2] == ["First reason here.", "Second reason here."]
