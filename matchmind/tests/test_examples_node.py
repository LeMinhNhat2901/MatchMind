from matchmind.agent.nodes import examples_node as en
from matchmind.examples.builder import feature_vector
from matchmind.examples.index import ExampleIndex
from matchmind.examples.schema import SituationExample
from matchmind.schema.match_state import EvidenceItem
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import make_sample_match_state


def _state():
    ms, focus = make_sample_match_state()
    return {"match_state": ms, "focus_player_id": focus, "situation_features": encode_situation(ms, focus), "trace": []}


def _index(state):
    vec = feature_vector(state["situation_features"], state["match_state"])

    def ex(i, match):
        return SituationExample(example_id=f"e{i}", match_id=match, event_index=i, minute=1, action_type="pass",
                                action_detail="progressive pass", outcome="shot", summary=f"s{i}",
                                vector=[v + i for v in vec], source_url="https://github.com/statsbomb/open-data")

    return ExampleIndex.build([ex(0, "other"), ex(1, "other"), ex(2, state["match_state"].match_id)])


def test_node_without_library_is_a_noop(monkeypatch):
    monkeypatch.setattr(en, "get_example_index", lambda: None)
    out = en.retrieve_examples_node(_state())
    assert out["similar_examples"] == [] and out["similar_examples_summary"] == ""


def test_node_returns_examples_excluding_same_match(monkeypatch):
    state = _state()
    monkeypatch.setattr(en, "get_example_index", lambda: _index(state))
    out = en.retrieve_examples_node(state)
    ids = [e["example_id"] for e in out["similar_examples"]]
    assert "e2" not in ids and ids
    assert "similar real situations" in out["similar_examples_summary"]
    assert out["trace"][-1]["node"] == "retrieve_examples"


def test_evidence_item_accepts_example_source():
    assert EvidenceItem(id="e1", source="example", content="x").source == "example"


def test_graph_runs_examples_before_evidence_assembly():
    from matchmind.agent.graph import build_tactical_graph

    edges = {(e.source, e.target) for e in build_tactical_graph().get_graph().edges}
    assert ("counterfactual_analysis", "retrieve_examples") in edges
    assert ("retrieve_examples", "assemble_evidence") in edges


def test_assemble_evidence_includes_example_bucket():
    from matchmind.agent.nodes.counterfactual_node import assemble_evidence_node

    state = _state()
    state["similar_examples"] = [{"example_id": "sb_1_2", "summary": "Carrier outnumbered", "similarity": 0.91}]
    out = assemble_evidence_node(state)
    assert out["evidence_set"]["example"][0]["id"] == "sb_1_2"
    assert out["evidence_set"]["example"][0]["source"] == "example"
