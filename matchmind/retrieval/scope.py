"""
Which concepts may be retrieved for a given situation.

The KB mixes three kinds of knowledge that should not compete with each other:
open-play tactics, set-piece routines (corners, free kicks...) and individual
showboat/finishing techniques (rabona, scorpion kick...). Before V1 a corner
routine could outrank "third-man run" for an open-play question simply because
it shared words like "overload" or "front post".

Rules:
- set_piece concepts: only when phase_of_play == "set_piece" (never relaxed —
  a corner routine is wrong advice in open play, not merely less relevant).
- technique concepts: only when the question names the technique.
- in/out of possession: a team with the ball gets possession/attacking-
  transition/general concepts, a team without it gets out_of_possession/
  defensive-transition/general ones. Relaxed on the re-retrieve branch and
  whenever gating leaves fewer than k concepts.
"""
from __future__ import annotations

import re
from typing import Iterable, Literal

from matchmind.retrieval.types import RetrievalQuery
from matchmind.schema.match_state import TacticalConcept

SET_PIECE_TOKENS: tuple[str, ...] = (
    "corner", "free_kick", "penalty", "kick_off", "goal_kick", "set_piece", "throw_in",
)
TECHNIQUE_IDS: frozenset[str] = frozenset({
    "wiki_bicycle_kick_technique", "wiki_scorpion_kick_technique",
    "wiki_rabona_kicking_technique", "wiki_rainbow_kick_technique",
    "wiki_flip_flap_dribbling_feint", "wiki_marseille_turn_dribble",
    "wiki_step_over_dribbling_feint", "wiki_cruyff_turn_evasive_dribble",
    "wiki_runaround_dribble_technique", "wiki_toe_poke_finishing_technique",
    "wiki_chip_shot_technique", "wiki_volley_and_half_volley_technique",
    "wiki_curling_and_bending_technique",
})
_IN_POSSESSION = {"possession", "attacking_transition"}
_OUT_OF_POSSESSION = {"out_of_possession", "defensive_transition"}
_TITLE_STOPWORDS = {"technique", "kick", "kicking", "shot", "turn", "dribble", "dribbling",
                    "feint", "finishing", "evasive", "and", "the"}


def concept_scope(concept: TacticalConcept) -> Literal["open_play", "set_piece", "technique"]:
    cid = concept.concept_id.lower()
    if any(tok in cid for tok in SET_PIECE_TOKENS) or (concept.subcategory or "").startswith("set_piece"):
        return "set_piece"
    if concept.concept_id in TECHNIQUE_IDS:
        return "technique"
    return "open_play"


def _mentions_technique(concept: TacticalConcept, question: str) -> bool:
    words = {w for w in re.findall(r"[a-z]+", concept.title.lower())
             if len(w) > 3 and w not in _TITLE_STOPWORDS}
    q = question.lower()
    return any(w in q for w in words)


def is_allowed(concept: TacticalConcept, query: RetrievalQuery, *, relax: bool = False) -> bool:
    scope = concept_scope(concept)
    if scope == "set_piece" and query.phase_of_play != "set_piece":
        return False
    if scope == "technique" and not _mentions_technique(concept, query.question):
        return False
    if relax or query.in_possession is None:
        return True
    if query.in_possession:
        return concept.phase not in _OUT_OF_POSSESSION
    return concept.phase not in _IN_POSSESSION


def allowed_ids(
    concepts: Iterable[TacticalConcept],
    query: RetrievalQuery,
    k: int,
    *,
    relax: bool = False,
) -> set[str]:
    concepts = list(concepts)
    ids = {c.concept_id for c in concepts if is_allowed(c, query, relax=relax)}
    if len(ids) < k and not relax:
        ids = {c.concept_id for c in concepts if is_allowed(c, query, relax=True)}
    return ids
