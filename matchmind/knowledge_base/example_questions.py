"""
Example questions per concept (doc2query).

Users rarely ask in textbook terms: "the direct pass to the striker is blocked,
use a nearby midfielder to find the runner" means "third-man run", but neither
the embedding model nor the MS MARCO cross-encoder makes that link (the golden
bench scored that concept 0.001). So each concept stores 3-5 plain-language
questions a coach or fan would ask that it answers, written once by an LLM from
the concept itself (never from the evaluation sets). They are used twice:

  - BM25 indexes them as part of the concept document (lexical.concept_document);
  - QuestionIndex embeds each question separately and ranks a concept by its
    best-matching question, as an extra list in the hybrid retriever's RRF.

Generation costs tokens once per concept; retrieval then costs nothing extra.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from matchmind.schema.match_state import TacticalConcept

logger = logging.getLogger(__name__)

MIN_QUESTIONS, MAX_QUESTIONS = 1, 5

_PROMPT = """You write example questions for a football tactics knowledge base.
For EACH concept below, write 4 different questions that a coach, player or fan might
type into a chat when this concept is the right answer.

Rules:
- Describe the match situation or problem in everyday football language; do NOT reuse
  the concept's title words or its technical name.
- Vary the angle: one short question, one describing a concrete situation on the pitch,
  one asking what a specific player should do, one about a recurring problem.
- English, one sentence each, end with a question mark.

Return ONLY a JSON object mapping each concept id to a list of 4 strings.

Concepts:
{concepts}
"""


@dataclass
class GenerationReport:
    done: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    tokens: int = 0


def _concept_brief(c: TacticalConcept) -> str:
    body = (c.definition or c.content)[:500]
    return f"- id: {c.concept_id}\n  title: {c.title}\n  phase: {c.phase}\n  summary: {body}"


def _clean(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for q in raw:
        if isinstance(q, str) and q.strip() and q.strip() not in out:
            out.append(q.strip())
    return out[:MAX_QUESTIONS]


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def generate_example_questions(
    concepts: list[TacticalConcept],
    llm_call: Callable[..., tuple[str, int]],
    batch_size: int = 5,
    log: Callable[[str], None] = logger.info,
) -> tuple[dict[str, list[str]], GenerationReport]:
    """{concept_id: questions} for every concept that has none yet. A failed batch
    or a malformed answer is skipped and reported; the next run retries it."""
    todo = [c for c in concepts if not c.example_questions]
    got: dict[str, list[str]] = {}
    report = GenerationReport()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        prompt = _PROMPT.format(concepts="\n".join(_concept_brief(c) for c in batch))
        try:
            text, tokens = llm_call(prompt, max_tokens=300 * len(batch), temperature=0.4,
                                    json_mode=True, thinking_budget=0)
            report.tokens += tokens or 0
            data = _parse(text)
        except Exception as exc:  # noqa: BLE001 — quota/outage: retried next run
            log(f"batch {i // batch_size + 1}: LLM error {exc}")
            data = {}
        for c in batch:
            qs = _clean(data.get(c.concept_id))
            if len(qs) >= MIN_QUESTIONS:
                got[c.concept_id] = qs
                report.done.append(c.concept_id)
            else:
                report.failed.append(c.concept_id)
        log(f"batch {i // batch_size + 1}/{(len(todo) + batch_size - 1) // batch_size}: "
            f"{len(report.done)} done, {len(report.failed)} failed")
    return got, report


def write_example_questions(concepts_dir: Path | str, questions: dict[str, list[str]]) -> int:
    """Store questions in each concept's JSON file (all other fields untouched)."""
    from matchmind.knowledge_base.embedder import concept_files

    written = 0
    for path in concept_files(concepts_dir):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        qs = questions.get(data.get("concept_id"))
        if qs:
            data["example_questions"] = qs
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written += 1
    return written


def _fingerprint(concepts: list[TacticalConcept]) -> str:
    payload = json.dumps(sorted((c.concept_id, c.example_questions) for c in concepts if c.example_questions))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QuestionIndex:
    """One unit vector per example question; a concept scores its best question."""

    def __init__(self, ids: list[str], matrix: np.ndarray, fingerprint: str) -> None:
        self.ids = ids                 # concept id per row (repeats once per question)
        self.matrix = matrix
        self.fingerprint = fingerprint

    @classmethod
    def build(cls, concepts: list[TacticalConcept], embed: Callable[[list[str]], np.ndarray]) -> "QuestionIndex":
        ids, texts = [], []
        for c in concepts:
            for q in c.example_questions:
                ids.append(c.concept_id)
                texts.append(q)
        matrix = np.asarray(embed(texts), dtype=float) if texts else np.zeros((0, 0))
        if len(texts):
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        return cls(ids, matrix, _fingerprint(concepts))

    def rank(self, query_vectors: np.ndarray, n: int) -> list[list[tuple[str, float]]]:
        """Per query: concepts by their best question's cosine, ties by id."""
        if not self.ids:
            return [[] for _ in range(len(query_vectors))]
        q = np.asarray(query_vectors, dtype=float)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        out = []
        for cos in q @ self.matrix.T:
            best: dict[str, float] = {}
            for cid, s in zip(self.ids, cos):
                if s > best.get(cid, -2.0):
                    best[cid] = float(s)
            out.append(sorted(best.items(), key=lambda p: (-p[1], p[0]))[:n])
        return out

    def save(self, path: Path | str) -> None:
        np.savez(path, ids=np.array(self.ids, dtype=object), matrix=self.matrix,
                 fingerprint=np.array(self.fingerprint))

    @classmethod
    def load(cls, path: Path | str, concepts: list[TacticalConcept]) -> "QuestionIndex | None":
        """None when missing or built from different questions (rebuild the KB)."""
        path = Path(path)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        if str(data["fingerprint"]) != _fingerprint(concepts):
            return None
        return cls([str(i) for i in data["ids"]], data["matrix"], str(data["fingerprint"]))
