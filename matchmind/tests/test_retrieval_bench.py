import math

from matchmind.evaluation.retrieval_bench import (
    GoldenCase, hit_at_k, load_golden, mrr, ndcg_at_k, run_bench,
)
from matchmind.knowledge_base.embedder import load_all_concepts


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


def test_ndcg_perfect_and_partial():
    gains = {"a": 2.0, "b": 1.0}
    assert ndcg_at_k(["a", "b", "x"], gains, 3) == 1.0
    partial = ndcg_at_k(["x", "a"], gains, 3)
    assert 0.0 < partial < 1.0
    assert math.isclose(ndcg_at_k(["x"], gains, 3), 0.0)


def test_mrr():
    assert mrr(["x", "y", "a"], {"a"}) == 1 / 3
    assert mrr(["x"], {"a"}) == 0.0


def test_run_bench_counts_must_not_and_set_piece_leaks():
    cases = [
        GoldenCase(id="c1", question="q", relevant=["a"], must_not=["bad"]),
        GoldenCase(id="c2", question="q", phase_of_play="open_play", must_not=["z"]),
    ]
    ranked = {"c1": ["bad", "a"], "c2": ["fifa_front_post_corner_strategy"]}
    report = run_bench(lambda case, k: ranked[case.id], cases, k=5)
    assert report.hit_at_3 == 1.0                  # only c1 has relevant labels
    assert report.must_not_violation_rate == 0.5   # c1 violated, c2 not
    assert report.set_piece_leak_rate == 0.5       # c2 is open play and got a corner concept


def test_golden_ids_exist_in_kb():
    ids = {c.concept_id for c in load_all_concepts()}
    for case in load_golden():
        for cid in case.relevant + case.acceptable + case.must_not:
            assert cid in ids, f"{case.id}: unknown concept id {cid!r}"


def test_golden_has_enough_cases():
    assert len(load_golden()) >= 25
