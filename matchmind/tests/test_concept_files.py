"""Bookkeeping files under the concepts tree (e.g. _state/youtube_processed.json,
written by the ingest scripts) are not concepts and must never be loaded as one."""
from __future__ import annotations

import json

from matchmind.knowledge_base.concept_synthesis import existing_concept_ids
from matchmind.knowledge_base.embedder import concept_files, load_all_concepts
from matchmind.tests.fixtures.concepts import make_concept


def test_underscore_dirs_are_skipped(tmp_path, caplog):
    (tmp_path / "youtube").mkdir()
    (tmp_path / "youtube" / "c.json").write_text(make_concept("yt_c").model_dump_json(), encoding="utf-8")
    (tmp_path / "_state").mkdir()
    (tmp_path / "_state" / "youtube_processed.json").write_text(json.dumps({"video_ids": ["x"]}), encoding="utf-8")

    assert [p.name for p in concept_files(tmp_path)] == ["c.json"]
    assert [c.concept_id for c in load_all_concepts(tmp_path)] == ["yt_c"]
    assert "Failed to load" not in caplog.text
    assert existing_concept_ids(tmp_path) == {"yt_c"}
