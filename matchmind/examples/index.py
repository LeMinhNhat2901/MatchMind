"""kNN over standardised feature vectors (NumPy — a few thousand 13-d rows need no vector DB)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from matchmind.examples.schema import SituationExample


class ExampleIndex:
    def __init__(self, examples: list[SituationExample], mean: np.ndarray, std: np.ndarray) -> None:
        self.examples = examples
        self.mean, self.std = mean, std
        if examples:
            z = (np.asarray([e.vector for e in examples], dtype=float) - mean) / std
        else:
            z = np.zeros((0, len(mean)))
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        self._unit = np.divide(z, norms, out=np.zeros_like(z), where=norms > 0)

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def build(cls, examples: list[SituationExample]) -> "ExampleIndex":
        mat = np.asarray([e.vector for e in examples], dtype=float)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        std[std == 0] = 1.0
        return cls(examples, mean, std)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "index.npz", mean=self.mean, std=self.std)
        with (directory / "examples.jsonl").open("w", encoding="utf-8") as fh:
            for e in self.examples:
                fh.write(e.model_dump_json() + "\n")

    @classmethod
    def load(cls, directory: Path) -> "ExampleIndex":
        directory = Path(directory)
        stats = np.load(directory / "index.npz")
        lines = (directory / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        examples = [SituationExample(**json.loads(line)) for line in lines if line.strip()]
        return cls(examples, stats["mean"], stats["std"])

    def query(
        self, vector: list[float], k: int = 5, exclude_match_id: str | None = None
    ) -> list[tuple[SituationExample, float]]:
        if not self.examples:
            return []
        z = (np.asarray(vector, dtype=float) - self.mean) / self.std
        norm = np.linalg.norm(z)
        sims = self._unit @ (z / norm) if norm > 0 else np.zeros(len(self.examples))
        out = []
        for i in np.argsort(-sims):
            e = self.examples[int(i)]
            if exclude_match_id is not None and e.match_id == exclude_match_id:
                continue   # never let an evaluation case "retrieve itself"
            out.append((e, round(float(sims[i]), 4)))
            if len(out) >= k:
                break
        return out


def summarize_neighbours(neighbours: list[tuple[SituationExample, float]]) -> str:
    if not neighbours:
        return ""
    n = len(neighbours)
    actions = Counter(e.action_type for e, _ in neighbours)
    shots = Counter(e.action_type for e, _ in neighbours if e.outcome == "shot")
    parts = []
    for action, count in actions.most_common():
        text = f"{count} {action}"
        if shots.get(action):
            text += f" ({shots[action]} led to a shot)"
        parts.append(text)
    outcomes = Counter(e.outcome for e, _ in neighbours)
    outcome_text = ", ".join(f"{c} {o}" for o, c in outcomes.most_common())
    return (f"Across {n} similar real situations (StatsBomb Open Data): "
            f"{', '.join(parts)}. Outcomes: {outcome_text}.")
