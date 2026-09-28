"""
Concept Synthesis — turns raw external text (a Wikipedia article, a YouTube
transcript chunk, ...) into a validated ``TacticalConcept`` via the LLM.

Shared by the Wikipedia and YouTube ingestion scripts (see
matchmind/scripts/build_kb_wikipedia.py and build_kb_youtube.py). Kept
provider-agnostic — it calls through matchmind.llm_client.call_llm, so it
follows whichever LLM_PROVIDER is configured (see matchmind-llm-provider
memory: Gemini is the default, but the .env may point at Groq/Anthropic/OpenAI).

Per knowledge_bonus.md: content must be PARAPHRASED (restructured summary),
never a verbatim copy, and every concept built this way carries a `source`
attribution (name + url + license) so the repo can show provenance.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

from matchmind.evaluation.retrieval_quality import _TRIGGERS
from matchmind.llm_client import call_llm
from matchmind.schema.match_state import ConceptSource, TacticalConcept

logger = logging.getLogger(__name__)

# Taxonomy v2 (knowledge_base.md, 2026-09) — 6 tactical-family buckets plus a
# 7th ("terminology") for glossary-style entries that genuinely don't belong
# to a tactical family. Replaces the old 5-value v1 vocabulary
# (attacking_movement/defensive_organization/spatial_principles/
# technical_skill/general_terminology) — see scripts/migrate_kb_taxonomy.py
# for the one-off remap of the concepts that shipped under the old names.
ALLOWED_CATEGORIES = [
    "fundamentals",
    "attacking",
    "defending",
    "transitions",
    "formations",
    "individual",
    "terminology",
]

# Example subcategories per category, shown to the LLM as a guide — it may
# pick a different one if it genuinely fits better (subcategory is informal,
# unlike category which is validated against ALLOWED_CATEGORIES).
SUBCATEGORY_EXAMPLES = {
    "fundamentals": ["principles_of_play", "width", "depth", "support", "mobility", "penetration"],
    "attacking": [
        "build_up", "positional_play", "overload", "isolation", "overlap",
        "underlap", "third_man", "half_space", "final_third",
    ],
    "defending": [
        "high_press", "mid_block", "low_block", "pressing_triggers", "pressing_traps",
        "compactness", "defensive_line", "zonal_defending",
    ],
    "transitions": ["counter_attack", "counter_press", "rest_defence", "defensive_transition"],
    "formations": ["433", "4231", "343", "352", "formation_transitions"],
    "individual": ["scanning", "body_orientation", "receiving", "movement", "decision_making"],
    "terminology": [],
}

# Loose phase-of-play vocabulary already used by the hand-written concepts
# (this is TacticalConcept.phase, a free string field — NOT the same
# controlled vocabulary as MatchState.phase_of_play).
ALLOWED_PHASES = [
    "possession",
    "attacking_transition",
    "defensive_transition",
    "out_of_possession",
    "general",
]

# The subset of trigger_conditions tokens the evaluator can actually check
# (matchmind/evaluation/retrieval_quality.py::_TRIGGERS). Concepts are free to
# use other tokens too — they just won't count toward the retrieval-quality
# proxy metric — but reusing these where genuinely applicable keeps new
# concepts measurable.
KNOWN_TRIGGER_TOKENS = sorted(_TRIGGERS.keys())


def slugify(text: str) -> str:
    """Lowercase, non-alnum -> underscore, collapse repeats, strip edges."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from LLM output (handles ```json fences).

    Mirrors matchmind.agent.nodes.validator_node._extract_json — duplicated
    rather than imported since that module is an internal agent-graph node,
    not a shared utility.
    """
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.debug(f"JSON decode error: {exc}")
        return None


_SYNTHESIS_PROMPT = """You are building a tactical knowledge base entry for a football (soccer) \
tactical-advice AI, from a source text about "{source_title}". The source text may be in any \
language (English, Vietnamese, ...) — regardless of its language, write ALL output fields \
("title", "content", condition/objective/action tokens, "reason") in ENGLISH, so the knowledge \
base stays in one consistent language.

STEP 1 — RELEVANCE CHECK
Decide whether this text describes a genuine TACTICAL or POSITIONAL concept usable for giving a \
player/coach real-time situational advice: team shape, pressing, movement patterns, spatial \
principles, a defined on-ball/off-ball technique, or a well-defined tactical term.
It is NOT usable if it's trivia, a specific match/competition/club/person, a law of the game \
unrelated to tactics, or a list/disambiguation page with no single coherent concept.
If NOT usable, respond with EXACTLY this JSON and nothing else:
{{"skip": true, "reason": "<one short sentence>"}}

STEP 2 — IF USABLE, PARAPHRASE AND STRUCTURE
Do NOT copy sentences verbatim from the source — write your own restructured summary (120-220 \
words) in the "content" field. Respond with ONLY this JSON structure:

{{
  "skip": false,
  "title": "Concise concept title (3-6 words)",
  "category": "one of: {categories}",
  "subcategory": "a specific concept family within that category, e.g. {subcategory_hint} — \
invent a short_snake_case one if none of these fit",
  "phase": "one of: {phases}",
  "definition": "ONE concise sentence defining the concept — not the full explanation.",
  "content": "Your paraphrased 120-220 word explanation of the concept, including when/why it \
applies on the pitch.",
  "conditions": ["short_snake_case_condition", "..."],
  "objective": ["short_snake_case_objective", "..."],
  "possible_actions": ["short_snake_case_action", "..."],
  "contraindications": ["short_snake_case_contraindication (a.k.a. risk)", "..."],
  "trigger_conditions": ["short_snake_case_token (a.k.a. trigger)", "..."],
  "spatial_requirements": ["short_snake_case_condition about team shape/spacing needed for this \
to work, e.g. 'compact_team_shape', 'short_distance_between_lines' — empty list if not applicable"],
  "player_behaviours": ["short_snake_case_individual_action a player performs as part of this \
concept, e.g. 'angle_press', 'cover_passing_lane' — empty list if not applicable"],
  "related_concepts": ["short_snake_case_concept_name this connects to, e.g. 'pressing_trap' — \
these are informal names, not required to be exact concept_ids — empty list if none obvious"]
}}

For "trigger_conditions" specifically, PREFER reusing these existing tokens where genuinely \
applicable (pick 0-4): {triggers}
Only invent a new short_snake_case token if none of the above genuinely fit.

SOURCE TEXT:
\"\"\"
{source_text}
\"\"\"
"""


def synthesize_concept(
    *,
    source_title: str,
    source_text: str,
    source_name: str,
    source_url: str,
    source_license: str,
    retrieved_at: str,
    id_prefix: str,
    llm_call: Callable[..., tuple[str, int]] = call_llm,
) -> TacticalConcept | None:
    """Turn one piece of raw source text into a validated TacticalConcept.

    Returns None when the LLM judged the source not tactically usable, or
    when its output didn't parse/validate (logged, never raised — a single
    bad article must not kill a batch run).
    """
    if not source_text or len(source_text.strip()) < 80:
        logger.info(f"Skipping {source_title!r}: source text too short")
        return None

    all_subcats = sorted({s for examples in SUBCATEGORY_EXAMPLES.values() for s in examples})
    prompt = _SYNTHESIS_PROMPT.format(
        source_title=source_title,
        categories=", ".join(ALLOWED_CATEGORIES),
        subcategory_hint=", ".join(all_subcats),
        phases=", ".join(ALLOWED_PHASES),
        triggers=", ".join(KNOWN_TRIGGER_TOKENS),
        source_text=source_text[:6000],
    )

    try:
        # thinking_budget=0: this is a simple structured-extraction call, not a task
        # that benefits from Gemini 3.x "thinking" — and thinking tokens are carved out
        # of max_tokens, so leaving it on with a small budget truncates the JSON before
        # it can be parsed (same class of issue as the reasoner node, see
        # matchmind-env-gotchas memory). Ignored by non-Gemini providers.
        text, _tokens = llm_call(
            prompt, max_tokens=1800, temperature=0.2, json_mode=True, thinking_budget=0
        )
    except Exception as exc:  # noqa: BLE001 — a rate-limited/failed call must not kill a batch run
        logger.warning(f"LLM call failed for {source_title!r}: {exc}")
        return None

    data = _extract_json(text)
    if data is None:
        logger.warning(f"Could not parse LLM output for {source_title!r}")
        return None

    if data.get("skip"):
        logger.info(f"Skipping {source_title!r}: {data.get('reason', 'not tactical')}")
        return None

    title = str(data.get("title") or source_title).strip()
    category = data.get("category") if data.get("category") in ALLOWED_CATEGORIES else ALLOWED_CATEGORIES[-1]
    phase = data.get("phase") if data.get("phase") in ALLOWED_PHASES else "general"

    try:
        return TacticalConcept(
            concept_id=f"{id_prefix}{slugify(title)}",
            category=category,
            subcategory=(str(data["subcategory"]).strip() if data.get("subcategory") else None),
            phase=phase,
            definition=(str(data["definition"]).strip() if data.get("definition") else None),
            title=title,
            content=str(data.get("content", "")).strip(),
            conditions=[str(x) for x in data.get("conditions", [])],
            objective=[str(x) for x in data.get("objective", [])],
            possible_actions=[str(x) for x in data.get("possible_actions", [])],
            contraindications=[str(x) for x in data.get("contraindications", [])],
            trigger_conditions=[str(x) for x in data.get("trigger_conditions", [])],
            spatial_requirements=[str(x) for x in data.get("spatial_requirements", [])],
            player_behaviours=[str(x) for x in data.get("player_behaviours", [])],
            related_concepts=[str(x) for x in data.get("related_concepts", [])],
            source=ConceptSource(
                name=source_name,
                url=source_url,
                license=source_license,
                retrieved_at=retrieved_at,
            ),
        )
    except Exception as exc:  # noqa: BLE001 — a bad LLM payload must not kill a batch run
        logger.warning(f"Validation failed for {source_title!r}: {exc}")
        return None


def existing_concept_ids(concepts_dir: Path) -> set[str]:
    """Concept ids already on disk (hand-written or from a previous run), for idempotent re-runs."""
    ids: set[str] = set()
    for fpath in Path(concepts_dir).rglob("*.json"):
        try:
            with open(fpath, encoding="utf-8-sig") as f:
                data = json.load(f)
            if "concept_id" in data:
                ids.add(data["concept_id"])
        except Exception:  # noqa: BLE001 — a malformed file must not block discovery
            continue
    return ids


def save_concept(concept: TacticalConcept, out_dir: Path) -> Path:
    """Write one concept as pretty JSON, named after its concept_id."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{concept.concept_id}.json"
    path.write_text(
        json.dumps(concept.model_dump(exclude_none=True), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
