from matchmind.evaluation import retrieval_quality
from matchmind.retrieval.triggers import (
    TRIGGER_PREDICATES, condition_ranking, trigger_match_fraction,
)
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.concepts import make_concept
from matchmind.tests.fixtures.sample_data import make_sample_match_state


def _situation():
    state, focus = make_sample_match_state()
    return encode_situation(state, focus), state


def test_backwards_compatible_aliases():
    assert retrieval_quality._TRIGGERS is TRIGGER_PREDICATES
    assert retrieval_quality.concept_relevance_score is trigger_match_fraction


def test_fraction_none_when_nothing_checkable():
    feats, state = _situation()
    assert trigger_match_fraction(make_concept("x", triggers=("made_up_token",)), feats, state) is None


def test_condition_ranking_orders_by_fraction_and_filters():
    feats, state = _situation()
    true_tok = "half_space_occupied_or_available"             # predicate is constant True
    false_tok = ("progressive_pass_not_available" if feats.progressive_passes_available > 0
                 else "progressive_pass_available")           # false for this situation
    assert not TRIGGER_PREDICATES[false_tok](feats, state)

    full = make_concept("full", triggers=(true_tok, true_tok))                 # 2/2
    half = make_concept("half", triggers=(true_tok, false_tok))                # 1/2
    none = make_concept("none", triggers=(false_tok, false_tok))               # 0/2
    single = make_concept("single", triggers=(true_tok,))                      # < min_checkable
    assert condition_ranking([none, half, single, full], feats, state) == ["full", "half"]
