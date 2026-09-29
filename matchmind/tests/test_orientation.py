"""Attacking direction: every team must be analysed attacking its OWN way.

User report (2026-09-29): for away centre-back #23 (x=82, right in front of his
own goal at x=105) the answer spoke of "23 m of free space ahead" — the space
toward his own goal — because every calculation assumed each team attacks +x.
"""
from __future__ import annotations

import pytest

from matchmind.schema.match_state import BallState, CandidateAction, MatchState, PlayerState, TacticalAdvice
from matchmind.situation_engine.candidate_actions import generate_candidate_actions
from matchmind.situation_engine.encoder import encode_situation
from matchmind.situation_engine.orientation import orient_for_team, rotate_state, unorient_advice
from matchmind.tests.fixtures.sample_data import make_sample_match_state


# ── direction inference ───────────────────────────────────────────────────────

def test_direction_inferred_from_goalkeepers():
    s, _ = make_sample_match_state()             # home keeper x=5, away keeper x=100
    assert s.attacks_right("home") is True
    assert s.attacks_right("away") is False


def test_explicit_direction_wins_over_inference():
    s, _ = make_sample_match_state()
    s = s.model_copy(update={"home_attacks_right": False})
    assert s.attacks_right("home") is False and s.attacks_right("away") is True


def test_direction_without_keepers_uses_team_depth():
    s, _ = make_sample_match_state()
    s = s.model_copy(update={"players": [p for p in s.players if p.role != "goalkeeper"]})
    assert s.attacks_right("home") is True       # home outfield sits deeper (lower x)


# ── rotation ──────────────────────────────────────────────────────────────────

def test_rotate_state_is_a_180_degree_turn_and_an_involution():
    s, _ = make_sample_match_state()
    r = rotate_state(s)
    p, q = s.get_player(11, "home"), r.get_player(11, "home")
    assert (q.x, q.y, q.vx, q.vy) == pytest.approx((105 - p.x, 68 - p.y, -p.vx, -p.vy))
    assert (r.ball.x, r.ball.y) == pytest.approx((105 - s.ball.x, 68 - s.ball.y))
    assert r.attacks_right("home") is False       # direction flips with the pitch
    back = rotate_state(r)
    assert back.get_player(11, "home").x == pytest.approx(p.x)
    assert back.attacks_right("home") is True


def test_orient_is_a_no_op_for_a_team_already_attacking_right():
    s, _ = make_sample_match_state()
    oriented, flipped = orient_for_team(s, "home")
    assert not flipped and oriented.get_player(10, "home").x == s.get_player(10, "home").x
    oriented, flipped = orient_for_team(s, "away")
    assert flipped and oriented.attacks_right("away") is True
    _, flipped_again = orient_for_team(oriented, "away")
    assert not flipped_again                       # idempotent: safe to call twice


# ── features ──────────────────────────────────────────────────────────────────

def test_away_centre_back_is_in_his_own_defensive_third():
    s, _ = make_sample_match_state()
    f = encode_situation(s, 23)                    # away CB at x=82, own goal at x=105
    assert f.is_in_defensive_third and not f.is_in_attacking_third


def test_home_features_are_unchanged():
    s, _ = make_sample_match_state()
    f = encode_situation(s, 3)                     # home CB at x=28
    assert f.is_in_defensive_third and not f.is_in_attacking_third


def test_away_keeper_candidate_targets_stay_near_his_own_goal():
    s, _ = make_sample_match_state()
    oriented, flipped = orient_for_team(s, "away")
    feats = encode_situation(oriented, 21)
    actions = generate_candidate_actions(oriented, feats)
    advice = TacticalAdvice(recommended_action="x", reasoning="r", confidence=0.9, evidence=[],
                            cited_concepts=[], alternatives=[], candidate_actions=actions)
    real = unorient_advice(advice, flipped)
    targets = [a.target_xy for a in real.candidate_actions if a.target_xy]
    assert targets and all(x > 80 for x, _ in targets)   # keeper stays by x=105, not x=0


def test_unorient_advice_maps_targets_back_to_the_real_pitch():
    advice = TacticalAdvice(recommended_action="x", reasoning="r", confidence=0.9, evidence=[], cited_concepts=[],
                            alternatives=[], candidate_actions=[CandidateAction(action="a", label="a", target_xy=(20.0, 10.0))])
    assert unorient_advice(advice, True).candidate_actions[0].target_xy == pytest.approx((85.0, 58.0))
    assert unorient_advice(advice, False).candidate_actions[0].target_xy == (20.0, 10.0)


def test_agent_runs_the_graph_on_the_oriented_pitch_and_returns_real_targets():
    from matchmind.agent.graph import TacticalAgent

    seen = {}

    class FakeGraph:
        def invoke(self, state):
            seen["state"] = state["match_state"]
            advice = TacticalAdvice(recommended_action="x", reasoning="r", confidence=0.9, evidence=[],
                                    cited_concepts=[], alternatives=[],
                                    candidate_actions=[CandidateAction(action="a", label="a", target_xy=(20.0, 10.0))])
            return {**state, "tactical_advice": advice, "trace": []}

    agent = TacticalAgent.__new__(TacticalAgent)       # skip model pre-warm
    agent._graph = FakeGraph()
    s, _ = make_sample_match_state()
    advice, _ = agent.analyze(s, 23, focus_team="away", question="q")
    assert seen["state"].attacks_right("away") is True
    assert seen["state"].get_player(23, "away").x == pytest.approx(105 - 82)
    assert advice.candidate_actions[0].target_xy == pytest.approx((85.0, 58.0))


def test_single_team_frame_without_keepers_still_orients():
    s = MatchState(match_id="m", timestamp=0.0, ball=BallState(x=60, y=34), possession="home", players=[
        PlayerState(id=1, team="home", x=60, y=34), PlayerState(id=2, team="away", x=70, y=30)])
    assert s.attacks_right("home") in (True, False)   # no crash on sparse frames
