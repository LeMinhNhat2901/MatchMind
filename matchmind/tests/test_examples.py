from matchmind.data_sources.statsbomb_local import StatsBombLocalLoader
from matchmind.examples.builder import FEATURE_NAMES, build_examples, describe_action, feature_vector
from matchmind.examples.index import ExampleIndex, summarize_neighbours
from matchmind.examples.labeling import label_outcome
from matchmind.examples.schema import SituationExample
from matchmind.situation_engine.encoder import encode_situation
from matchmind.tests.fixtures.sample_data import make_sample_match_state
from matchmind.tests.fixtures.statsbomb_mini import MATCH_ID, write_mini_open_data


def _e(t, minute, second, x, team=10, possession=1):
    return {"type": {"name": t}, "minute": minute, "second": second, "location": [x, 40.0],
            "team": {"id": team}, "possession": possession, "possession_team": {"id": team}}


def test_outcome_shot_within_window():
    events = [_e("Pass", 1, 0, 60), _e("Carry", 1, 5, 80), _e("Shot", 1, 9, 105)]
    assert label_outcome(events, 0) == "shot"


def test_outcome_progressed():
    events = [_e("Pass", 1, 0, 50), _e("Carry", 1, 6, 70), _e("Pass", 1, 20, 72)]
    assert label_outcome(events, 0) == "progressed"      # +20 SB units ~ +17.5 m


def test_outcome_lost_when_possession_changes_to_other_team():
    events = [_e("Pass", 1, 0, 50), _e("Ball Recovery", 1, 3, 55, team=20, possession=2)]
    assert label_outcome(events, 0) == "lost"


def test_outcome_retained_when_same_team_new_possession():
    events = [_e("Pass", 1, 0, 50), _e("Pass", 1, 3, 52, team=10, possession=2)]
    assert label_outcome(events, 0) == "retained"


def test_outcome_window_expires():
    events = [_e("Pass", 1, 0, 50), _e("Shot", 1, 30, 105)]
    assert label_outcome(events, 0) == "retained"


def test_describe_action():
    p = {"type": {"name": "Pass"}, "location": [50.0, 40.0],
         "pass": {"end_location": [70.0, 40.0], "height": {"name": "Ground Pass"}}}
    assert describe_action(p) == ("pass", "progressive pass (ground pass)")
    p["pass"]["switch"] = True
    assert describe_action(p)[1].startswith("switch of play")
    assert describe_action({"type": {"name": "Shot"}, "location": [100, 40]}) == ("shot", "shot")


def test_feature_vector_shape():
    state, focus = make_sample_match_state()
    vec = feature_vector(encode_situation(state, focus), state)
    assert len(vec) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vec)


def test_build_examples_from_mini_data(tmp_path):
    loader = StatsBombLocalLoader(write_mini_open_data(tmp_path / "data"))
    examples = build_examples(loader, [MATCH_ID])
    assert examples, "expected at least one example"
    assert all(e.action_type in {"pass", "carry", "dribble", "shot"} for e in examples)
    assert all(e.event_index != 2 for e in examples)   # the corner (set piece) is excluded
    assert all(e.source_url.startswith("https://github.com/statsbomb/open-data") for e in examples)


def _ex(i, vec, match="m1", action="pass", outcome="progressed"):
    return SituationExample(example_id=f"e{i}", match_id=match, event_index=i, minute=1,
                            action_type=action, action_detail=f"{action} detail", outcome=outcome,
                            summary=f"example {i}", vector=vec, source_url="https://github.com/statsbomb/open-data")


def test_index_query_nearest_and_exclude(tmp_path):
    idx = ExampleIndex.build([_ex(0, [0.0, 0.0]), _ex(1, [10.0, 10.0], match="m2"), _ex(2, [9.0, 10.0])])
    got = [e.example_id for e, _ in idx.query([9.5, 10.0], k=2)]
    assert set(got) == {"e1", "e2"}
    got_ex = [e.example_id for e, _ in idx.query([9.5, 10.0], k=2, exclude_match_id="m2")]
    assert "e1" not in got_ex
    idx.save(tmp_path / "lib")
    assert len(ExampleIndex.load(tmp_path / "lib")) == 3


def test_summary_counts_actions_and_outcomes():
    text = summarize_neighbours([(_ex(0, [0, 0], action="pass", outcome="shot"), 0.9),
                                 (_ex(1, [0, 0], action="pass", outcome="retained"), 0.8),
                                 (_ex(2, [0, 0], action="carry", outcome="lost"), 0.7)])
    assert "3 similar real situations" in text
    assert "2 pass" in text and "1 carry" in text
    assert "1 led to a shot" in text
