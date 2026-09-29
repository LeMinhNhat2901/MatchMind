"""
Narrator Node — Tier 2: natural-language retelling of an already-validated
TacticalAdvice, written directly in the user's language.

Why this exists: forcing the reasoner LLM to emit strict JSON in one shot
(Tier 1, reasoner_node.py) makes its `reasoning` field read like a technical
report ("numerical_advantage=-2, space_ahead=8.2m...") rather than analysis a
person would actually say. Rather than relaxing the JSON constraint (which
would reopen the door to unvalidatable, possibly-hallucinated output), this
node runs AFTER validation as a second, narrow LLM call over the validated
advice plus the measurements behind it. It is constrained to restate only
what's already in that data — no new numbers, no new tactical terms.

It produces two things (user feedback: an answer full of percentages and
distances doesn't persuade anyone, but the numbers must still be available):
- advice.narrative — persuasive prose answering the question, no statistics;
- advice.rationale — short bullets carrying the concrete numbers behind the
  decision and why each alternative lost.

Language: it writes natively in state["response_language"]. The earlier
design (narrate in English, then translate) read as machine-translated
Vietnamese — a translation prompt has to follow the English sentence
structure, and the placeholder tokens protecting the numbers hid what each
number meant. Writing natively drops both problems; number safety moves to a
post-check instead (translation.ungrounded_numbers: every number in the
output must exist in the data, and player references must be real player
ids). If the native attempt fails that check, it retries in English and the
dashboard falls back to translating (translation.localize_advice).

If every attempt fails, `advice.narrative` stays None and callers fall back to
`advice.reasoning` — a narration failure must never fail the whole run.

Voice: RAG does not teach an LLM a writing style — it only pastes retrieved
text into the prompt for that one call. So a consistent analyst voice is a
persona/prompt-instruction problem: this node includes an optional style
guide (knowledge_base/style_guide.md, built offline from real YouTube analyst
transcripts by scripts/build_style_guide.py) and, for Vietnamese, the voice
rules + terminology in knowledge_base/glossary_vi.md.
"""
from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path

from matchmind.agent.state import AgentState
from matchmind.agent.translation import (
    _extract_json,
    glossary_section,
    is_english,
    ungrounded_numbers,
)
from matchmind.llm_client import call_narrator_llm, last_provider
from matchmind.schema.match_state import TacticalAdvice

logger = logging.getLogger(__name__)

STYLE_GUIDE_PATH = Path(__file__).parent.parent.parent / "knowledge_base" / "style_guide.md"

_NARRATION_PROMPT = """You are a football (soccer) tactical analyst answering a coach's question. \
Write in {language}, natively — as a {language}-speaking analyst would say it, never as a \
sentence-by-sentence translation of English.
{voice_section}{glossary_section}
QUESTION:
{question}

DATA (a fully computed, already-validated analysis — the ONLY facts you may use):
{data_json}

Produce two parts, both in {language}:

1. "answer" — 3 to 5 sentences of persuasive, flowing prose that answers the question directly: \
what the player should do, and why it is the right idea, told as cause and effect (what it \
exploits, what it opens up, what it avoids). Refer to players only by shirt number. Put NO \
statistics in the answer: no percentages, metres, coordinates, confidence scores or \
pitch-control figures — the reader should be convinced by the football logic, not by numbers.

2. "rationale" — 3 to 5 short bullet points that justify the decision with the concrete numbers \
from DATA: the key measurements, the pitch-control change of the chosen action, and why each \
alternative was rejected. Each bullet is one sentence. Describe positions in football words \
("between the lines", "in the half-space", "behind the defence"), never as raw (x, y) \
coordinates. Say numerical disadvantage naturally ("outnumbered by 2") rather than as a signed \
number, and write confidence as a percentage.

STRICT: only restate facts, numbers and concepts that appear in DATA. Every number you write must \
come from DATA. Do not invent statistics, tactical claims, or real teams, coaches or players.

Respond with ONLY this JSON:
{{"answer": "...", "rationale": ["...", "..."]}}"""


@lru_cache(maxsize=1)
def _load_style_guide() -> str | None:
    """Read knowledge_base/style_guide.md once per process. None if absent/unreadable."""
    try:
        if STYLE_GUIDE_PATH.exists():
            text = STYLE_GUIDE_PATH.read_text(encoding="utf-8").strip()
            # The generated file starts with a provenance comment — not prompt material.
            text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
            return text or None
    except Exception as exc:  # noqa: BLE001 — a missing/unreadable style guide is not fatal
        logger.warning(f"Could not read style guide at {STYLE_GUIDE_PATH}: {exc}")
    return None


def _voice_section() -> str:
    guide = _load_style_guide()
    if not guide:
        return ""
    return f"\nWrite in this voice:\n{guide}\n"


def _r(v: float | None, nd: int = 1) -> float | None:
    return None if v is None else round(float(v), nd)


def _pct(v: float | None) -> float | None:
    """Pitch-control fractions (0.031) → percentage points (3.1), the unit users read."""
    return None if v is None else round(float(v) * 100, 1)


def _build_data(state: AgentState, advice: TacticalAdvice) -> dict:
    """Everything the narrator may say, and nothing else. The number check runs
    against exactly this, so a figure missing here can't be used in the answer."""
    match_state = state.get("match_state")
    features = state.get("situation_features")
    rec = advice.resolve_action(advice.recommended_action_id)

    data: dict = {
        "focus_player_shirt_number": advice.focus_player_id,
        "recommended_action": advice.recommended_action,
        "recommended_action_detail": rec.label if rec else None,
        "reasoning": advice.reasoning,
        "confidence": advice.confidence,
        "alternatives_considered": [
            {"action": a.action, "why_not": a.why_not, "pitch_control_change_pct": a.delta_pitch_control}
            for a in advice.alternatives
        ],
        "cited_tactical_concepts": advice.cited_concepts,
    }
    if match_state is not None:
        data["match_context"] = {
            "minute": match_state.minute,
            "score_home": match_state.score_home,
            "score_away": match_state.score_away,
        }
    if features is not None:
        data["key_measurements"] = {
            "focus_player_role": features.focus_player_role,
            "has_the_ball": features.is_ball_carrier,
            "nearest_opponent_distance_m": _r(features.nearest_opponent_distance),
            "nearest_teammate_distance_m": _r(features.nearest_teammate_distance),
            "free_space_ahead_m": _r(features.space_ahead),
            "local_numerical_advantage": features.local_numerical_advantage,
            "in_attacking_third": features.is_in_attacking_third,
        }
        if features.is_ball_carrier:
            # Only the carrier's own passing options — an off-ball player's
            # "lanes" were misread as the carrier's (see encoder.py).
            data["key_measurements"]["progressive_passes_available"] = features.progressive_passes_available
            data["key_measurements"]["open_passing_lane_to_shirt_numbers"] = features.open_passing_lane_player_ids

    wanted = {advice.recommended_action_id} | {a.action_id for a in advice.alternatives}
    cf = state.get("counterfactual_analysis") or []
    rows = [r for r in cf if r.get("action_name") in wanted] or cf[:4]
    if rows:
        data["pitch_control_simulation"] = [
            {
                "action": r.get("description") or r.get("action_name"),
                "is_recommended": r.get("action_name") == advice.recommended_action_id,
                "team_control_change_pct": _pct(r.get("pitch_control_delta")),
                "control_at_target_pct": _pct(r.get("pitch_control_at_target")),
            }
            for r in rows
        ]
    return data


def _player_ids(state: AgentState, advice: TacticalAdvice, data_json: str) -> set[int]:
    """Shirt numbers the answer may legitimately name."""
    ids: set[int] = set()
    if advice.focus_player_id is not None:
        ids.add(advice.focus_player_id)
    for c in advice.candidate_actions:
        if c.target_player_id is not None:
            ids.add(c.target_player_id)
    features = state.get("situation_features")
    if features is not None:
        ids.update(features.open_passing_lane_player_ids)
        if features.ball_carrier_id is not None:
            ids.add(features.ball_carrier_id)
    ids.update(int(n) for n in re.findall(r"(?:player\s*#?|_to_)(\d+)", data_json, re.IGNORECASE))
    return ids


def _narrate(
    language: str, question: str, data_json: str, player_ids: set[int]
) -> tuple[str, list[str], int] | str:
    """One narration attempt. Returns (answer, rationale, tokens) or a failure reason."""
    prompt = _NARRATION_PROMPT.format(
        language=language,
        voice_section=_voice_section(),
        glossary_section=glossary_section(language),
        question=question,
        data_json=data_json,
    )
    # thinking_budget=0: a paraphrase, not a reasoning task — Gemini 3.x thinking
    # tokens would just eat the output budget (see matchmind-env-gotchas).
    # max_tokens is generous because Vietnamese costs ~2-3x the tokens of English.
    raw, tokens = call_narrator_llm(
        prompt, max_tokens=1600, temperature=0.4, json_mode=True, thinking_budget=0
    )
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return "unparseable JSON"
    answer = str(parsed.get("answer") or "").strip()
    rationale_raw = parsed.get("rationale") or []
    if isinstance(rationale_raw, str):
        rationale_raw = [rationale_raw]
    rationale = [str(b).strip().lstrip("-• ").strip() for b in rationale_raw if str(b).strip()][:6]
    if not answer:
        return "empty answer"

    bad = ungrounded_numbers("\n".join([answer, *rationale]), data_json, player_ids)
    if bad:
        return f"numbers not in data: {bad}"
    return answer, rationale, tokens


def narrator_node(state: AgentState) -> AgentState:
    """Node: write advice.narrative + advice.rationale in the user's language (best-effort)."""
    t0 = time.time()
    advice = state.get("tactical_advice")
    trace = state.get("trace") or []

    if advice is None:
        # Validation never produced advice (exhausted retries with no parseable
        # output) — nothing to narrate.
        return state

    language = state.get("response_language") or "English"
    data_json = json.dumps(_build_data(state, advice), indent=2, ensure_ascii=False)
    player_ids = _player_ids(state, advice, data_json)
    question = state.get("question") or ""

    attempts = [language] if is_english(language) else [language, "English"]
    tokens_total = 0
    failures: list[str] = []
    for lang in attempts:
        try:
            result = _narrate(lang, question, data_json, player_ids)
        except Exception as exc:  # noqa: BLE001 — narration is a nice-to-have, never fail the run over it
            result = f"LLM error: {str(exc)[:160]}"
        if isinstance(result, str):
            failures.append(f"{lang}: {result}")
            logger.warning(f"[narrator] {lang} attempt rejected — {result}")
            continue
        answer, rationale, tokens = result
        tokens_total += tokens
        advice = advice.model_copy(
            update={"narrative": answer, "rationale": rationale, "narrative_language": lang}
        )
        break

    elapsed = (time.time() - t0) * 1000
    step = {"node": "narrator", "latency_ms": round(elapsed, 1), "tokens": tokens_total,
            "language": advice.narrative_language, "provider": last_provider()}
    if failures:
        step["rejected"] = failures
    trace.append(step)
    logger.debug(f"[narrator] {elapsed:.1f}ms | language={advice.narrative_language}")

    return {
        **state,
        "tactical_advice": advice,
        "total_tokens_used": (state.get("total_tokens_used") or 0) + tokens_total,
        "trace": trace,
    }
