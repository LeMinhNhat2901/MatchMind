"""The reasoner prompt shows similar real situations only when there are some."""
from matchmind.agent.nodes.reasoner_node import _build_prompt
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import make_sample_match_state


def _state(examples):
    ms, focus = make_sample_match_state()
    return {
        "match_state": ms, "focus_player_id": focus, "question": "What should player 10 do?",
        "situation_features": encode_situation(ms, focus), "retrieved_tactics": [],
        "candidate_actions": [], "counterfactual_text": "",
        "evidence_set": {"stats": [], "context": [], "rag": [], "example": examples},
        "similar_examples_summary": "Across 1 similar real situations (StatsBomb Open Data): 1 pass." if examples else "",
    }


def test_prompt_includes_examples_section_and_ids():
    prompt = _build_prompt(_state([{"id": "sb_1_2", "source": "example", "content": "Carrier outnumbered"}]))
    assert "SIMILAR REAL SITUATIONS" in prompt
    assert "[sb_1_2]" in prompt
    assert '"example"' in prompt


def test_prompt_omits_examples_section_when_empty():
    assert "SIMILAR REAL SITUATIONS" not in _build_prompt(_state([]))
