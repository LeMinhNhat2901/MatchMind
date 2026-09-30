"""Near-duplicate gate for KB ingestion.

Many channels explain the same idea, and the LLM wrote a fresh concept every
time (~9 near-identical inverted-full-back concepts): the top-k then repeats one
idea and retrieval quality dropped (nDCG@5 0.70 -> 0.53). The gate skips a new
concept whose text is >= threshold cosine-similar to one already in the KB.
"""
from __future__ import annotations

import numpy as np

from matchmind.knowledge_base.dedup import DuplicateGate
from matchmind.tests.fixtures.concepts import make_concept


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Topic = first word of the title: same topic -> identical vector."""
    topics = {"inverted": [1, 0, 0], "pressing": [0, 1, 0], "tiki": [0.6, 0.8, 0]}
    return np.array([topics.get(t.split()[0].lower(), [0, 0, 1]) for t in texts], dtype=float)


def _c(cid, title):
    c = make_concept(cid)
    return c.model_copy(update={"title": title})


def test_near_duplicate_of_kb_concept_is_flagged():
    gate = DuplicateGate([_c("a", "Inverted fullback system")], embed=_fake_embed, threshold=0.88)
    hit = gate.check(_c("b", "Inverted full-back build-up"))
    assert hit is not None and hit[0] == "a" and hit[1] >= 0.88


def test_distinct_concept_passes():
    gate = DuplicateGate([_c("a", "Inverted fullback system")], embed=_fake_embed, threshold=0.88)
    assert gate.check(_c("b", "Pressing traps")) is None
    assert gate.check(_c("c", "Tiki taka")) is None          # cos 0.6 — related, not a duplicate


def test_pending_concepts_of_the_same_run_count():
    gate = DuplicateGate([], embed=_fake_embed)
    first = _c("a", "Pressing high")
    assert gate.check(first) is None
    assert gate.check(_c("b", "Pressing trap"), pending=[first])[0] == "a"
    gate.add(first)
    assert gate.check(_c("c", "Pressing again"))[0] == "a"


def test_ingestor_skips_duplicates_and_reports_them(tmp_path):
    from matchmind.tests.test_source_ingest import VID_A, Fakes, _ingestor
    from matchmind.knowledge_base.source_ingest import parse_sources_text

    fakes = Fakes()
    gate = DuplicateGate([], embed=lambda texts: np.ones((len(texts), 3)))   # everything is one topic
    report = _ingestor(tmp_path, fakes, dedup=gate).run(parse_sources_text(VID_A))
    # 2 chunks -> 2 concepts; the second duplicates the first.
    assert report.concepts_written == 1 and len(report.duplicates_skipped) == 1
    assert report.videos_new == 1
