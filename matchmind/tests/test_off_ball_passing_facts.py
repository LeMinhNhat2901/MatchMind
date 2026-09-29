"""Passing-lane facts describe the BALL CARRIER's options. For an off-ball focus
player they must not reach the prompts.

Real run (2026-09-29): asked about the keeper (#1) while #10 had the ball, the
answer said the keeper's move "opens 5 passing lanes for #10 (to #2-#6)" — those
were the keeper's own lanes, computed as if he had the ball, and the LLM pinned
them on #10.
"""
from __future__ import annotations

from matchmind.agent.nodes.counterfactual_node import assemble_evidence_node
from matchmind.agent.nodes.narrator_node import _build_data
from matchmind.schema.match_state import TacticalAdvice
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import make_sample_match_state

LANE_TEXT = "Open passing lanes"


def _state(focus: int):
    ms, _ = make_sample_match_state()
    feats = encode_situation(ms, focus)
    return {"match_state": ms, "focus_player_id": focus, "situation_features": feats,
            "retrieved_tactics": [], "counterfactual_analysis": [], "trace": []}


def test_off_ball_description_has_no_own_passing_lanes():
    off = _state(1)["situation_features"]
    assert not off.is_ball_carrier
    assert LANE_TEXT not in off.natural_language_description
    assert "Third-man run opportunity" not in off.natural_language_description


def test_ball_carrier_description_keeps_passing_lanes():
    carrier = _state(10)["situation_features"]
    assert carrier.is_ball_carrier
    assert LANE_TEXT in carrier.natural_language_description


def test_off_ball_stats_evidence_has_no_passing_lanes():
    stats = assemble_evidence_node(_state(1))["evidence_set"]["stats"][0]["content"]
    assert LANE_TEXT not in stats and "Progressive passes available" not in stats
    carrier_stats = assemble_evidence_node(_state(10))["evidence_set"]["stats"][0]["content"]
    assert LANE_TEXT in carrier_stats


def test_off_ball_narrator_data_has_no_passing_lanes():
    state = _state(1)
    advice = TacticalAdvice(recommended_action="x", reasoning="r", confidence=0.9, evidence=[],
                            cited_concepts=[], alternatives=[], focus_player_id=1)
    measurements = _build_data(state, advice)["key_measurements"]
    assert "open_passing_lane_to_shirt_numbers" not in measurements
    assert "progressive_passes_available" not in measurements
