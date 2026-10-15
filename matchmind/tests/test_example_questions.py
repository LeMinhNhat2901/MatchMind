"""doc2query: example coach questions per concept, written once by an LLM.

Users rarely ask in textbook terms ("use a nearby midfielder to find the runner"
means "third-man run"), so each concept stores a few plain-language questions it
answers. They feed BM25 and a separate dense ranker (max similarity over a
concept's questions). Every LLM / embedding call is injected here.
"""
from __future__ import annotations

import json

import numpy as np

from matchmind.knowledge_base.example_questions import (
    QuestionIndex,
    generate_example_questions,
    write_example_questions,
)
from matchmind.retrieval.lexical import concept_document, tokenize
from matchmind.tests.fixtures.concepts import make_concept


def _llm_answering(mapping):
    calls = []

    def llm(prompt, **kw):
        calls.append(prompt)
        ids = [cid for cid in mapping if cid in prompt]
        return json.dumps({cid: mapping[cid] for cid in ids}), 10
    return llm, calls


def test_generation_batches_skips_done_and_cleans_output():
    done = make_concept("done").model_copy(update={"example_questions": ["already here?"]})
    todo = [make_concept(f"c{i}") for i in range(3)]
    llm, calls = _llm_answering({
        "c0": ["How do we beat the press?", "  How do we beat the press?  ", "Who drops in?"],
        "c1": ["Where should the winger stand?"],
        "c2": "not a list",
    })
    got, report = generate_example_questions([done, *todo], llm_call=llm, batch_size=2)
    assert len(calls) == 2                                    # 3 concepts to do, batches of 2
    assert "done" not in "".join(calls)                       # never re-asks for finished concepts
    assert got == {"c0": ["How do we beat the press?", "Who drops in?"],
                   "c1": ["Where should the winger stand?"]}
    assert report.failed == ["c2"]


def test_llm_error_skips_the_batch_without_crashing():
    def broken(prompt, **kw):
        raise RuntimeError("429")
    got, report = generate_example_questions([make_concept("a")], llm_call=broken)
    assert got == {} and report.failed == ["a"]


def test_write_keeps_every_other_field(tmp_path):
    c = make_concept("yt_x")
    path = tmp_path / "youtube" / "yt_x.json"
    path.parent.mkdir()
    path.write_text(c.model_dump_json(indent=2), encoding="utf-8")
    assert write_example_questions(tmp_path, {"yt_x": ["Q1?", "Q2?"]}) == 1
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["example_questions"] == ["Q1?", "Q2?"] and data["title"] == c.title


def test_bm25_document_includes_example_questions(monkeypatch):
    from matchmind.config import settings
    monkeypatch.setattr(settings, "retrieval_bm25_questions", True)
    c = make_concept("third_man_run").model_copy(update={"example_questions": ["Use a nearby midfielder to find the runner"]})
    assert "midfielder" in tokenize(concept_document(c))


def _embed(texts):
    """Topic = first word: same first word -> same direction."""
    axes = {"press": [1, 0, 0], "width": [0, 1, 0]}
    return np.array([axes.get(t.split()[0].lower(), [0, 0, 1]) for t in texts], dtype=float)


def test_question_index_ranks_by_best_matching_question(tmp_path):
    concepts = [
        make_concept("a").model_copy(update={"example_questions": ["width on the left?", "press higher?"]}),
        make_concept("b").model_copy(update={"example_questions": ["width please?"]}),
        make_concept("c"),                                     # no questions: never ranked here
    ]
    idx = QuestionIndex.build(concepts, _embed)
    [ranked] = idx.rank(_embed(["press now"]), n=5)
    assert [cid for cid, _ in ranked] == ["a", "b"]          # a's "press" question wins; b is next
    path = tmp_path / "q.npz"
    idx.save(path)
    again = QuestionIndex.load(path, concepts)
    assert again is not None and again.ids == idx.ids
    changed = [concepts[0].model_copy(update={"example_questions": ["new?"]}), *concepts[1:]]
    assert QuestionIndex.load(path, changed) is None          # stale after the questions change


def test_switches_turn_both_uses_off(monkeypatch):
    """Both uses of example questions are config switches, so retrieval can run
    with or without them while the evaluation decides."""
    from matchmind.config import settings
    from matchmind.retrieval.lexical import concept_document as doc

    c = make_concept("x").model_copy(update={"example_questions": ["Zebra crossing question?"]})
    monkeypatch.setattr(settings, "retrieval_bm25_questions", False)
    assert "zebra" not in tokenize(doc(c))
    monkeypatch.setattr(settings, "retrieval_bm25_questions", True)
    assert "zebra" in tokenize(doc(c))
