"""
Tests for the advice renderers (arrows for recommended/alternative actions).

No LLM involved — TacticalAdvice objects are built by hand from the same
rule-based candidate_actions the real reasoner would have been offered.
"""
from __future__ import annotations

from pathlib import Path

from matchmind.schema.match_state import EvidenceItem, TacticalAdvice
from matchmind.situation_engine.candidate_actions import generate_candidate_actions
from matchmind.situation_engine.encoder import encode_situation
from matchmind.situation_engine.team_focus import pick_dual_focus_players
from matchmind.tests.fixtures.sample_data import make_sample_match_state
from matchmind.visualization.snapshot_renderer import (
    render_advice_snapshot,
    render_dual_advice_snapshot,
    render_snapshot,
)


def _advice_for(state, focus_id: int, recommended_id: str | None = None) -> TacticalAdvice:
    feats = encode_situation(state, focus_id)
    candidates = generate_candidate_actions(state, feats)
    rec_id = recommended_id or candidates[0].action
    return TacticalAdvice(
        recommended_action=f"Do the {rec_id} thing",
        recommended_action_id=rec_id,
        alternatives=[],
        reasoning="test",
        confidence=0.8,
        evidence=[EvidenceItem(id="x", source="stats", content="test")],
        cited_concepts=[],
        situation_features=feats,
        candidate_actions=candidates,
        focus_player_id=focus_id,
        match_id=state.match_id,
        timestamp=state.timestamp,
    )


def test_render_snapshot_smoke(tmp_path):
    state, focus_id = make_sample_match_state()
    out = render_snapshot(state, focus_id, str(tmp_path / "base.png"))
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_resolve_action_matches_candidate():
    state, focus_id = make_sample_match_state()
    advice = _advice_for(state, focus_id)
    resolved = advice.resolve_action(advice.recommended_action_id)
    assert resolved is not None
    assert resolved.action == advice.recommended_action_id


def test_resolve_action_none_for_unknown_id():
    state, focus_id = make_sample_match_state()
    advice = _advice_for(state, focus_id)
    assert advice.resolve_action("totally_made_up_action") is None
    assert advice.resolve_action(None) is None


def test_render_advice_snapshot_onball(tmp_path):
    state, focus_id = make_sample_match_state()  # focus_id is the ball carrier
    advice = _advice_for(state, focus_id)
    out = render_advice_snapshot(state, advice, str(tmp_path / "advice_on.png"))
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_render_advice_snapshot_offball(tmp_path):
    state, _ = make_sample_match_state()
    advice = _advice_for(state, 9, recommended_id="run_in_behind")  # off-ball winger
    out = render_advice_snapshot(state, advice, str(tmp_path / "advice_off.png"))
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_render_advice_snapshot_no_match_still_renders(tmp_path):
    """An id that doesn't resolve must not crash — just no arrow."""
    state, focus_id = make_sample_match_state()
    advice = _advice_for(state, focus_id, recommended_id=None)
    advice = advice.model_copy(update={"recommended_action_id": "not_a_real_action"})
    out = render_advice_snapshot(state, advice, str(tmp_path / "advice_none.png"))
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_render_dual_advice_snapshot(tmp_path):
    state, _ = make_sample_match_state()
    focus = pick_dual_focus_players(state)
    att_advice = _advice_for(state, focus.attacking_player_id)
    def_advice = _advice_for(state, focus.defending_player_id)
    out = render_dual_advice_snapshot(state, att_advice, def_advice, str(tmp_path / "dual.png"))
    assert Path(out).exists() and Path(out).stat().st_size > 0
