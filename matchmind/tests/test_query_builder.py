import re

from matchmind.retrieval.query_builder import build_retrieval_query
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import make_sample_match_state


def _query(question="How does player 10 escape the press?", **state_overrides):
    state, focus = make_sample_match_state()
    if state_overrides:
        state = state.model_copy(update=state_overrides)
    feats = encode_situation(state, focus)
    return build_retrieval_query(feats, state, question), feats


def test_question_is_kept_without_digits():
    q, _ = _query()
    assert "escape the press" in q.question
    assert not re.search(r"\d", q.question)


def test_phrases_have_no_numbers_and_reflect_features():
    q, feats = _query()
    assert q.tactical_phrases, "expected at least one phrase"
    assert not any(re.search(r"\d", p) for p in q.tactical_phrases)
    if feats.third_man_opportunity:
        assert "third-man combination" in q.tactical_phrases
    if feats.local_numerical_advantage < 0:
        assert any("outnumbered" in p for p in q.tactical_phrases)


def test_possession_flag_from_match_state():
    q, feats = _query()
    assert q.in_possession is (feats.focus_player_team == "home")   # sample: home has the ball


def test_out_of_possession_gets_defending_phrases():
    q, _ = _query(possession="away")
    assert q.in_possession is False
    assert "defending out of possession" in q.tactical_phrases
    assert "third-man combination" not in q.tactical_phrases


def test_unknown_possession():
    q, _ = _query(possession=None)
    assert q.in_possession is None


def test_set_piece_phrase():
    q, _ = _query(phase_of_play="set_piece")
    assert "set piece routine" in q.tactical_phrases


def test_situation_text_kept_for_fallback():
    q, feats = _query()
    assert q.situation_text == feats.natural_language_description
