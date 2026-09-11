"""
Tests for validator_node's handling of recommended_action_id / action_id —
the ids that link a TacticalAdvice back to a geometric CandidateAction so
the renderer can draw an arrow.

No LLM call: raw_llm_output is a hand-built JSON string, exactly like what
validator_node parses in the real pipeline.
"""
from __future__ import annotations

import json

from matchmind.agent.nodes.validator_node import validator_node
from matchmind.situation_engine.candidate_actions import generate_candidate_actions
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import make_sample_match_state


def _base_state(state, focus_id, feats, cands, raw: dict) -> dict:
    return {
        "match_state": state,
        "focus_player_id": focus_id,
        "situation_features": feats,
        "candidate_actions": [c.model_dump() for c in cands],
        "retrieved_tactics": [],
        "raw_llm_output": json.dumps(raw),
        "trace": [],
    }


def test_valid_action_id_is_kept():
    state, focus_id = make_sample_match_state()
    feats = encode_situation(state, focus_id)
    cands = generate_candidate_actions(state, feats)
    real_id = cands[0].action

    raw = {
        "recommended_action": "Pass forward",
        "recommended_action_id": real_id,
        "alternatives": [],
        "reasoning": "test",
        "confidence": 0.8,
        "evidence": [{"id": "x", "source": "stats", "content": "y"}],
        "cited_concepts": [],
    }
    out = validator_node(_base_state(state, focus_id, feats, cands, raw))
    assert out["validation_passed"] is True
    assert out["tactical_advice"].recommended_action_id == real_id


def test_hallucinated_action_id_is_nulled_not_failed():
    """A bad action_id is cosmetic (no arrow drawn) — it must NOT trigger a
    regenerate loop the way a bad cited_concept does."""
    state, focus_id = make_sample_match_state()
    feats = encode_situation(state, focus_id)
    cands = generate_candidate_actions(state, feats)

    raw = {
        "recommended_action": "Pass to player 6",
        "recommended_action_id": "not_a_real_candidate",
        "alternatives": [
            {"action": "dribble", "action_id": "also_fake", "why_not": "x", "delta_pitch_control": None}
        ],
        "reasoning": "test",
        "confidence": 0.8,
        "evidence": [{"id": "x", "source": "stats", "content": "y"}],
        "cited_concepts": [],
    }
    out = validator_node(_base_state(state, focus_id, feats, cands, raw))
    advice = out["tactical_advice"]

    assert out["validation_errors"] == []
    assert out["validation_passed"] is True
    assert advice.recommended_action_id is None
    assert advice.alternatives[0].action_id is None


def test_candidate_actions_are_attached_to_advice():
    state, focus_id = make_sample_match_state()
    feats = encode_situation(state, focus_id)
    cands = generate_candidate_actions(state, feats)

    raw = {
        "recommended_action": "Pass forward",
        "recommended_action_id": cands[0].action,
        "alternatives": [],
        "reasoning": "test",
        "confidence": 0.8,
        "evidence": [{"id": "x", "source": "stats", "content": "y"}],
        "cited_concepts": [],
    }
    out = validator_node(_base_state(state, focus_id, feats, cands, raw))
    advice = out["tactical_advice"]
    assert len(advice.candidate_actions) == len(cands)
    assert advice.resolve_action(advice.recommended_action_id) is not None
