"""
Tests for the Spearman pitch control model + counterfactual analyzer.

Pure NumPy — no ChromaDB / LLM / network needed.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from matchmind.pitch_control.counterfactual_analyzer import CounterfactualAnalyzer
from matchmind.pitch_control.spearman_model import (
    compute_control_at_point,
    compute_controlled_area,
    generate_pitch_control_for_match_state,
)
from matchmind.schema.match_state import CandidateAction
from matchmind.situation_engine.candidate_actions import generate_candidate_actions
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import (
    make_counter_attack_state,
    make_sample_match_state,
)


def test_grid_is_probability():
    state, _ = make_sample_match_state()
    _, _, ppcf = generate_pitch_control_for_match_state(state, "home")
    assert ppcf.shape[0] > 1 and ppcf.shape[1] > 1
    assert ppcf.min() >= 0.0
    assert ppcf.max() <= 1.0
    # A pitch is not fully controlled by one team
    assert 0.05 < compute_controlled_area(ppcf) < 0.95


def test_control_at_point_between_zero_and_one():
    state, _ = make_sample_match_state()
    p = compute_control_at_point(state, 60.0, 34.0, "home")
    assert 0.0 <= p <= 1.0


def test_grid_is_fast():
    """Regression guard for the vectorised integration (was ~65s unvectorised)."""
    state, _ = make_sample_match_state()
    t0 = time.time()
    generate_pitch_control_for_match_state(state, "home")
    assert time.time() - t0 < 8.0


def test_counterfactual_delta_direction():
    """On a counter-attack, playing the runners in should not lose control
    versus a backward recycle."""
    state, pid = make_counter_attack_state()
    feats = encode_situation(state, pid)
    cands = generate_candidate_actions(state, feats)
    results = CounterfactualAnalyzer(state, pid, "home").analyze(cands)

    by_name = {r.action_name: r for r in results}
    assert results, "analyzer returned nothing"
    for r in results:
        if r.pitch_control_delta is not None:
            assert -1.0 <= r.pitch_control_delta <= 1.0

    if "hold_and_recycle" in by_name and any(n.startswith("pass_to_") for n in by_name):
        recycle = by_name["hold_and_recycle"].pitch_control_delta
        best_pass = max(
            (by_name[n].pitch_control_delta or -9 for n in by_name if n.startswith("pass_to_")),
            default=None,
        )
        if recycle is not None and best_pass is not None:
            assert best_pass >= recycle - 0.15  # forward option is not dramatically worse


def test_analyze_accepts_plain_strings_and_candidate_objects():
    state, pid = make_sample_match_state()
    an = CounterfactualAnalyzer(state, pid, "home")
    r_str = an.analyze(["hold_and_recycle"])
    r_obj = an.analyze([CandidateAction(action="hold_and_recycle", label="hold")])
    assert r_str[0].action_name == r_obj[0].action_name == "hold_and_recycle"


def test_degraded_warning_when_no_velocity(caplog):
    state, _ = make_sample_match_state()
    assert state.has_velocity is False  # synthetic fixture
    with caplog.at_level("WARNING"):
        generate_pitch_control_for_match_state(state, "home")
    assert any("degraded" in m.lower() for m in caplog.messages)
