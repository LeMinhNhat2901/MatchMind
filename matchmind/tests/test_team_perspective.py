"""Score and game state must be told from the FOCUS player's team, not always home.

User report (2026-09-29): asked about away centre-back #23 (away leads 1-0), the
answer said "our team is trailing 0-1, so press #10 now" — every score line read
"home trailing", which the LLM took as the focus player's own team.
"""
from __future__ import annotations

from matchmind.agent.nodes.counterfactual_node import assemble_evidence_node
from matchmind.agent.nodes.narrator_node import _build_data
from matchmind.schema.match_state import TacticalAdvice
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import make_sample_match_state

# Sample: home 0 - 1 away, minute 62. #10 is home, #23 is away.


def test_match_state_score_from_either_side():
    s, _ = make_sample_match_state()
    assert s.score_for("away") == (1, 0) and s.result_for("away") == "leading"
    assert s.score_for("home") == (0, 1) and s.result_for("home") == "trailing"


def test_description_uses_focus_team_perspective():
    s, _ = make_sample_match_state()
    away = encode_situation(s, 23).natural_language_description
    home = encode_situation(s, 10).natural_language_description
    assert "Player 23's team (away) is leading 1–0" in away
    assert "trailing" not in away
    assert "Player 10's team (home) is trailing 0–1" in home


def test_context_evidence_uses_focus_team_perspective():
    s, _ = make_sample_match_state()
    ctx = assemble_evidence_node({"match_state": s, "situation_features": encode_situation(s, 23),
                                  "retrieved_tactics": [], "counterfactual_analysis": []})
    text = ctx["evidence_set"]["context"][0]["content"]
    assert "Player 23's team (away) is leading 1–0" in text
    assert "Possession: home (the opponent)" in text


def test_urgency_is_judged_for_the_focus_team():
    s, _ = make_sample_match_state(minute=85)          # away leads 1-0 late on
    def urgency(focus):
        ev = assemble_evidence_node({"match_state": s, "situation_features": encode_situation(s, focus),
                                     "retrieved_tactics": [], "counterfactual_analysis": []})
        return ev["evidence_set"]["context"][0]["content"]
    assert "protect the lead" in urgency(23)
    assert "chasing the game" in urgency(10)


def test_narrator_data_states_score_for_focus_team():
    s, _ = make_sample_match_state()
    advice = TacticalAdvice(recommended_action="x", reasoning="r", confidence=0.9, evidence=[],
                            cited_concepts=[], alternatives=[], focus_player_id=23)
    ctx = _build_data({"match_state": s, "situation_features": encode_situation(s, 23)}, advice)["match_context"]
    assert ctx["focus_team"] == "away"
    assert (ctx["focus_team_goals"], ctx["opponent_goals"]) == (1, 0)
    assert ctx["focus_team_is"] == "leading"
