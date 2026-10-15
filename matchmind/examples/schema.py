"""One real, recorded football situation and what the team did next."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SituationExample(BaseModel):
    example_id: str                       # f"sb_{match_id}_{event_index}"
    match_id: str
    event_index: int
    minute: int
    action_type: Literal["pass", "carry", "dribble", "shot"]
    action_detail: str                    # "progressive pass (ground pass)"
    outcome: Literal["shot", "progressed", "retained", "lost"]
    summary: str                          # one templated sentence, no LLM
    vector: list[float]                   # raw (un-standardised) FEATURE_NAMES values
    source_url: str                       # StatsBomb open-data events file
