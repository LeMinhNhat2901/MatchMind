"""Value types shared by the retrieval pipeline (matchmind/retrieval/)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """Everything retrieval needs, already stripped of raw coordinates.

    question          — the user's question (English, after chat translation).
    tactical_phrases  — short football phrases derived from SituationFeatures
                        (query_builder.py); the situation's meaning without numbers.
    situation_text    — the legacy numeric description; only used as a last
                        resort / on the relaxed retry.
    in_possession     — does the focus player's team have the ball (None = unknown).
    phase_of_play     — MatchState.phase_of_play.
    extra             — additional sub-queries (the re-retrieve branch's expansion).
    """

    question: str = ""
    tactical_phrases: list[str] = Field(default_factory=list)
    situation_text: str = ""
    in_possession: bool | None = None
    phase_of_play: str | None = None
    extra: list[str] = Field(default_factory=list)

    def subqueries(self) -> list[str]:
        out: list[str] = []
        if self.question.strip():
            out.append(self.question.strip())
        if self.tactical_phrases:
            out.append(". ".join(self.tactical_phrases))
        out.extend(e for e in self.extra if e.strip())
        if not out and self.situation_text.strip():
            out.append(self.situation_text.strip())
        return out

    def rerank_text(self) -> str:
        parts = [self.question.strip()] + self.tactical_phrases[:4]
        text = ". ".join(p for p in parts if p)
        return text or self.situation_text


class ScoredConcept(BaseModel):
    concept_id: str
    score: float
    dense_sim: float | None = None
    rerank_score: float | None = None
    sources: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    items: list[ScoredConcept]
    confidence: float
    strategy: str

    @property
    def concept_ids(self) -> list[str]:
        return [i.concept_id for i in self.items]
