"""Goalkeepers must get goalkeeper options, not outfield runs.

User report (2026-09-28): asked what the home goalkeeper (#1, at x=5) should do
while #10 has the ball, the agent answered "drift into the half-space to open a
progressive passing lane" — because candidate generation and the retrieval query
were role-agnostic and offered the keeper run_in_behind / drop_into_pocket /
provide_width like any attacker.
"""
from __future__ import annotations

from matchmind.retrieval.query_builder import build_retrieval_query
from matchmind.schema.match_state import BallState
from matchmind.situation_engine.candidate_actions import generate_candidate_actions
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import make_sample_match_state

OUTFIELD_OFF_BALL = {"run_in_behind", "check_to_receive", "provide_width", "drop_into_pocket", "overlapping_run"}
OWN_BOX_X = 16.5


def _gk_case(**updates):
    state, _ = make_sample_match_state()
    if updates:
        state = state.model_copy(update=updates)
    feats = encode_situation(state, 1)
    return state, feats, generate_candidate_actions(state, feats)


def test_keeper_off_ball_while_team_attacks_gets_keeper_options():
    state, feats, actions = _gk_case()
    assert not feats.is_ball_carrier and state.possession == "home"
    names = {a.action for a in actions}
    assert not names & OUTFIELD_OFF_BALL, names
    assert "gk_sweeper_position" in names
    assert all(not a.moves_ball for a in actions)
    # Stays behind its own back line — never sent to the half-space at x≈35+.
    own_back_line = min(p.x for p in state.players if p.team == "home" and p.id != 1)
    assert all(a.target_xy[0] < own_back_line for a in actions if a.target_xy)


def test_sweeper_label_names_our_own_back_line():
    """Real LLM run: an ambiguous "behind the back line" was read as the
    OPPONENT's line ("their high line leaves space behind them")."""
    _, _, actions = _gk_case()
    label = next(a.label for a in actions if a.action == "gk_sweeper_position")
    assert "our own back line" in label
    assert "opponent" not in label.lower()


def test_keeper_when_opponent_has_ball_positions_between_ball_and_goal():
    state, _ = make_sample_match_state()
    carrier = next(p for p in state.players if p.team == "away" and p.role != "goalkeeper")
    state = state.model_copy(update={
        "possession": "away",
        "ball": BallState(x=carrier.x, y=carrier.y),
    })
    feats = encode_situation(state, 1)
    actions = generate_candidate_actions(state, feats)
    names = {a.action for a in actions}
    assert "gk_set_position" in names
    assert "press_ball_carrier" not in names
    assert all(a.target_xy[0] <= OWN_BOX_X for a in actions if a.target_xy)


def test_keeper_on_the_ball_distributes_instead_of_dribbling():
    state, _ = make_sample_match_state()
    gk = state.get_player(1, team="home")
    state = state.model_copy(update={"ball": BallState(x=gk.x + 0.5, y=gk.y)})
    feats = encode_situation(state, 1)
    assert feats.is_ball_carrier
    names = {a.action for a in generate_candidate_actions(state, feats)}
    assert not {n for n in names if n.startswith("dribble")}, names
    assert "shoot" not in names
    assert any(n.startswith("pass_to_") for n in names) or "gk_long_distribution" in names


def test_retrieval_query_for_keeper_talks_about_keeping_not_running_in_behind():
    state, feats, _ = _gk_case()
    q = build_retrieval_query(feats, state, "What should the goalkeeper do?")
    text = " ".join(q.tactical_phrases)
    assert "goalkeeper" in text
    assert "run in behind" not in text and "off-ball movement" not in text
