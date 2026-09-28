from matchmind.retrieval.lexical import BM25Index, tokenize
from matchmind.tests.fixtures.concepts import make_concept

CONCEPTS = [
    make_concept("third_man_run", title="Third-Man Run (Combination Play)",
                 content="A third player runs onto a lay-off; the first pass bypasses the marker."),
    make_concept("gegenpressing", title="Gegenpressing", category="defending", phase="defensive_transition",
                 content="Counter-press immediately after losing the ball to win it back high."),
    make_concept("low_block", title="Low Block", category="defending", phase="out_of_possession",
                 content="Sit deep, stay compact, protect the penalty box."),
]


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The pressing players pressed the passes") == ["press", "press", "pass"]
    assert tokenize("third_man run") == ["third", "man", "run"]


def test_rank_prefers_lexical_match():
    idx = BM25Index(CONCEPTS)
    assert idx.rank("third-man combination")[0][0] == "third_man_run"
    assert idx.rank("counter press after losing the ball")[0][0] == "gegenpressing"


def test_rank_respects_allowed_and_drops_zero_scores():
    idx = BM25Index(CONCEPTS)
    ranked = idx.rank("counter press after losing the ball", allowed={"low_block", "third_man_run"})
    assert all(cid in {"low_block", "third_man_run"} for cid, _ in ranked)
    assert all(score > 0 for _, score in ranked)


def test_empty_or_stopword_query_returns_nothing():
    idx = BM25Index(CONCEPTS)
    assert idx.rank("") == []
    assert idx.rank("the and of") == []


def test_vietnamese_query_does_not_crash():
    ranked = BM25Index(CONCEPTS).rank("cầu thủ nên làm gì")
    assert isinstance(ranked, list)
