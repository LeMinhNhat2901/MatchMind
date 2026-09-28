from matchmind.retrieval.scope import allowed_ids, concept_scope, is_allowed
from matchmind.retrieval.types import RetrievalQuery
from matchmind.tests.fixtures.concepts import make_concept

CORNER = make_concept("fifa_front_post_corner_strategy", title="Front Post Corner Strategy")
RABONA = make_concept("wiki_rabona_kicking_technique", title="Rabona Kicking Technique", category="individual")
OVERLAP = make_concept("overlapping_run", title="Overlapping Run", phase="possession")
LOWBLOCK = make_concept("low_block", title="Low Block", category="defending", phase="out_of_possession")
GENERAL = make_concept("wiki_width_and_depth_principles", title="Width and Depth", phase="general")


def test_scope_classification():
    assert concept_scope(CORNER) == "set_piece"
    assert concept_scope(RABONA) == "technique"
    assert concept_scope(OVERLAP) == "open_play"


def test_set_piece_subcategory_variants_count_as_set_piece():
    # Real KB: this corner concept's id has no set-piece token, only its subcategory.
    front_post = make_concept("fifa_front_post_overload_vs_zonal_defense", subcategory="set_piece")
    assert concept_scope(front_post) == "set_piece"
    assert concept_scope(make_concept("x", subcategory="set_piece_defense")) == "set_piece"


def test_set_piece_only_in_set_piece_phase():
    assert not is_allowed(CORNER, RetrievalQuery(phase_of_play="open_play"))
    assert is_allowed(CORNER, RetrievalQuery(phase_of_play="set_piece"))


def test_set_piece_concepts_still_respect_possession():
    """Golden g27: an attacking corner question returned defensive-corner concepts."""
    defending_corner = make_concept("fifa_corner_kick_defensive_systems", category="defending",
                                    phase="out_of_possession")
    attacking = RetrievalQuery(phase_of_play="set_piece", in_possession=True)
    assert not is_allowed(defending_corner, attacking)
    assert is_allowed(CORNER, attacking)
    assert is_allowed(defending_corner, RetrievalQuery(phase_of_play="set_piece", in_possession=False))


def test_set_piece_never_allowed_even_when_relaxed_in_open_play():
    assert not is_allowed(CORNER, RetrievalQuery(phase_of_play="open_play"), relax=True)


def test_technique_needs_explicit_mention():
    assert not is_allowed(RABONA, RetrievalQuery(question="What should the winger do?"))
    assert is_allowed(RABONA, RetrievalQuery(question="Should he try a rabona cross here?"))


def test_possession_gating():
    attacking = RetrievalQuery(in_possession=True)
    defending = RetrievalQuery(in_possession=False)
    assert is_allowed(OVERLAP, attacking) and not is_allowed(OVERLAP, defending)
    assert is_allowed(LOWBLOCK, defending) and not is_allowed(LOWBLOCK, attacking)
    assert is_allowed(GENERAL, attacking) and is_allowed(GENERAL, defending)
    assert is_allowed(LOWBLOCK, attacking, relax=True)


def test_unknown_possession_does_not_gate():
    assert is_allowed(LOWBLOCK, RetrievalQuery()) and is_allowed(OVERLAP, RetrievalQuery())


def test_allowed_ids_relaxes_when_too_few():
    concepts = [OVERLAP, LOWBLOCK, GENERAL]
    strict = allowed_ids(concepts, RetrievalQuery(in_possession=False), k=1)
    assert strict == {"low_block", "wiki_width_and_depth_principles"}
    relaxed = allowed_ids(concepts, RetrievalQuery(in_possession=False), k=3)
    assert relaxed == {"overlapping_run", "low_block", "wiki_width_and_depth_principles"}


def test_subqueries_and_rerank_text():
    q = RetrievalQuery(question="How to escape the press?", tactical_phrases=["third-man combination", "tight space"],
                       situation_text="Player 10 at (68m, 42m)", extra=["expanded text"])
    assert q.subqueries() == ["How to escape the press?", "third-man combination. tight space", "expanded text"]
    assert q.rerank_text().startswith("How to escape the press?")
    assert RetrievalQuery(situation_text="only this").subqueries() == ["only this"]
