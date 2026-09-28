"""Tiny TacticalConcept factory for retrieval tests (no KB files, no Chroma)."""
from __future__ import annotations

from matchmind.schema.match_state import TacticalConcept


def make_concept(
    concept_id: str,
    *,
    title: str | None = None,
    content: str | None = None,
    category: str = "attacking",
    phase: str = "possession",
    triggers: tuple[str, ...] = (),
    related: tuple[str, ...] = (),
    subcategory: str | None = None,
    definition: str | None = None,
) -> TacticalConcept:
    title = title or concept_id.replace("_", " ").title()
    return TacticalConcept(
        concept_id=concept_id,
        category=category,
        subcategory=subcategory,
        phase=phase,
        title=title,
        definition=definition,
        content=content or f"{title}. " + "Filler tactical explanation text. " * 5,
        trigger_conditions=list(triggers),
        related_concepts=list(related),
    )
