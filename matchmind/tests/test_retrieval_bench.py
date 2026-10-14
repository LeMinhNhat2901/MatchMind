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


# ── benchmark layers: KB coverage · filter correctness · retrieval quality ────

def test_recall_at_5_counts_relevant_found():
    cases = [GoldenCase(id="q", question="x", relevant=["a", "b"], acceptable=["c"])]
    report = run_bench(lambda case, k: ["a", "c", "z"], cases, k=5)
    assert report.recall_at_5 == 0.5
    assert report.per_case[0]["recall_at_5"] == 0.5


def test_layers_report_coverage_and_filter_blocks():
    from matchmind.evaluation.retrieval_bench import layer_report

    cases = [
        GoldenCase(id="covered", question="x", relevant=["a"], acceptable=["b"]),
        GoldenCase(id="gap", question="y", relevant=[], acceptable=["b"]),          # KB has no answer
        GoldenCase(id="ghost", question="z", relevant=["not_in_kb"]),                # label points nowhere
    ]
    allowed = {"covered": {"a"}, "gap": {"b"}, "ghost": set()}                       # "b" blocked for "covered"
    rep = layer_report(cases, kb_ids={"a", "b"}, allowed_for=lambda c: allowed[c.id])
    assert rep.coverage_rate == round(1 / 3, 4)
    assert sorted(rep.uncovered) == ["gap", "ghost"]
    assert rep.filter_recall == round(2 / 3, 4)                                      # a@covered, b@gap pass; b@covered blocked
    assert rep.filtered_out == ["covered:b"]


def test_coverage_only_cases_are_not_scored():
    """Concepts written after seeing a question can't fairly score that question:
    it counts for KB coverage only."""
    from matchmind.evaluation.retrieval_bench import layer_report

    cases = [GoldenCase(id="scored", question="x", relevant=["a"]),
             GoldenCase(id="gap_filled", question="y", relevant=["b"], coverage_only=True)]
    report = run_bench(lambda case, k: ["a"], cases, k=5)
    assert report.ndcg_at_5 == 1.0 and report.hit_at_3 == 1.0          # "gap_filled" (a miss) not averaged in
    assert "ndcg_at_5" not in report.per_case[1]
    assert layer_report(cases, {"a", "b"}, lambda c: {"a", "b"}).coverage_rate == 1.0
