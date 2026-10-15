"""
Dump cached StatsBomb situations + candidate concepts for hand labelling.

Usage:
    python -m matchmind.scripts.label_situations --n 15
Fill `relevant:` in output/evaluation/situation_labeling.yaml, then move the
labelled rows into matchmind/evaluation/golden/ (future: situation-level bench).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import typer
import yaml

from matchmind.config import settings

app = typer.Typer()


@app.command()
def main(n: int = typer.Option(15)) -> None:
    from matchmind.retrieval import build_retrieval_query, get_hybrid_retriever
    from matchmind.scripts.download_data import load_cached_testcases
    from matchmind.situation_engine.encoder import encode_situation

    retriever = get_hybrid_retriever()
    rows = []
    for i, case in enumerate(load_cached_testcases()[:n]):
        state = case["snapshot"]
        feats = encode_situation(state, case["focus_player_id"])
        q = build_retrieval_query(feats, state, case["question"])
        candidates = retriever.retrieve(q, k=8, features=feats, match_state=state).concept_ids
        rows.append({"case_index": i, "situation": feats.natural_language_description,
                     "phrases": q.tactical_phrases, "candidates": candidates, "relevant": []})
    out = settings.evaluation_output_dir / "situation_labeling.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(rows, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    app()
