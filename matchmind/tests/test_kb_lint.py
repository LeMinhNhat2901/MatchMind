from matchmind.knowledge_base.embedder import load_all_concepts
from matchmind.scripts.kb_lint import lint_concepts
from matchmind.tests.fixtures.concepts import make_concept


def _msgs(issues, level=None):
    return [i.message for i in issues if level is None or i.level == level]


def test_defending_concept_with_possession_phase_warns():
    issues = lint_concepts([make_concept("x", category="defending", phase="possession")])
    assert any("phase" in m for m in _msgs(issues, "warning"))


def test_trigger_phase_contradicting_concept_phase_warns():
    c = make_concept("g", category="defending", phase="defensive_transition",
                     triggers=("phase_of_play_attacking_transition",))
    assert any("trigger" in m for m in _msgs(lint_concepts([c]), "warning"))


def test_unknown_related_concept_warns():
    # related_concepts are LLM-suggested hints (concept_synthesis.py) — a dangling
    # one is worth fixing but must not block CI.
    issues = lint_concepts([make_concept("a", related=("does_not_exist",))])
    assert any("does_not_exist" in m for m in _msgs(issues, "warning"))


def test_duplicate_title_is_error():
    issues = lint_concepts([make_concept("a", title="Same"), make_concept("b", title="Same")])
    assert _msgs(issues, "error")


def test_real_kb_has_no_errors():
    errors = [i for i in lint_concepts(load_all_concepts()) if i.level == "error"]
    assert errors == [], errors


def test_gegenpressing_is_a_defensive_transition_concept():
    g = next(c for c in load_all_concepts() if c.concept_id == "gegenpressing")
    assert g.phase == "defensive_transition"
    assert "phase_of_play_defensive_transition" in g.trigger_conditions
