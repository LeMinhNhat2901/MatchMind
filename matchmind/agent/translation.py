"""
Chat Translation Layer — lets a user type/read in any language while the
tactical-reasoning pipeline (prompts, knowledge base, schema) stays
English-only internally.

This sits OUTSIDE the LangGraph agent graph on purpose: translation is
orthogonal to tactical reasoning (the graph shouldn't need to know or care
what language the user typed in), so it's a thin adapter around
TacticalAgent.analyze()'s existing `question` parameter rather than new graph
nodes. Used by the dashboard chat interface (dashboard/app.py).

Both functions are resilient by design: a translation failure must never
block the chat, so each falls back to passing the original English/source
text through unchanged rather than raising.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from matchmind.llm_client import call_llm, call_narrator_llm

if TYPE_CHECKING:
    from matchmind.schema.match_state import TacticalAdvice

logger = logging.getLogger(__name__)


@dataclass
class DetectedTranslation:
    original_text: str
    english_text: str
    language: str  # e.g. "Vietnamese", "English", "Spanish"
    is_english: bool


_DETECT_PROMPT = """Detect the language of this message and translate it to English.

MESSAGE:
\"\"\"{text}\"\"\"

Respond with ONLY this JSON:
{{"language": "<language name in English, e.g. Vietnamese, English, Spanish>", "is_english": true or false, "english_text": "<the message translated to English; if it's already English, repeat it unchanged>"}}"""

_TRANSLATE_BACK_PROMPT = """Rewrite the following football tactical analysis in natural {language}, \
the way a {language}-speaking football analyst would say it to a coach. Keep every fact and its \
meaning, but do NOT mirror the English sentence structure — reorder, split or merge sentences \
freely so it reads as if it was written in {language} from the start.
{glossary_section}
The text contains placeholder tokens like @@0@@, @@1@@. Each stands in for a number — what kind \
of number is listed below, so you can phrase the words around it naturally (e.g. a shirt number \
reads like "số @@0@@" in Vietnamese). Copy every placeholder EXACTLY, character-for-character. Do \
NOT translate, change, merge, drop or write out the number instead of the placeholder.
{legend}
TEXT:
\"\"\"{text}\"\"\"

Respond with ONLY the {language} text, nothing else."""

GLOSSARY_DIR = Path(__file__).parent.parent / "knowledge_base"


def is_same_language(a: str | None, b: str | None) -> bool:
    return bool(a and b) and a.strip().lower() == b.strip().lower()


def is_english(language: str | None) -> bool:
    return not language or language.strip().lower() == "english"


@lru_cache(maxsize=8)
def load_glossary(language: str | None) -> str | None:
    """Voice rules + football terminology for `language`, if we have a file for it.

    Only Vietnamese exists today (knowledge_base/glossary_vi.md). Without it the
    model translates terms like "half-space" or "third-man run" literally and
    differently every time — a main reason Vietnamese answers read as
    machine-translated.
    """
    if not language or "vietnam" not in language.lower():
        return None
    path = GLOSSARY_DIR / "glossary_vi.md"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning(f"Could not read glossary {path}: {exc}")
        return None
    # Drop the maintainer comment at the top — it's not prompt material.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
    return text or None


def glossary_section(language: str | None) -> str:
    glossary = load_glossary(language)
    return f"\nFollow these {language} writing rules and terms:\n{glossary}\n" if glossary else ""


# Any standalone number (incl. sign, decimals, trailing %) — covers player references
# ("Player 11"), coordinates, distances, percentages, confidence scores, etc. in one
# pattern. Deliberately leaves surrounding words (e.g. "Player") untouched so the
# translator can still render them naturally ("Cầu thủ 11") — only the digit itself
# is protected, since that's the part a translation can silently get wrong.
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?")


def _protect_numeric_refs(text: str) -> tuple[str, dict[str, str]]:
    """Swap every number for an opaque @@N@@ token before it goes anywhere near a
    translation LLM call.

    Why: relying on a prompt instruction ("preserve every number exactly") is not
    enough — seen in production, a translation turned "Player 11" into "Cầu thủ 1",
    silently misattributing the whole recommendation to a different player. An
    opaque token the model has no reason to "translate" is far less likely to be
    altered than a digit embedded in natural prose, and importantly: if the model
    DOES mishandle a token, the failure mode becomes "a placeholder is missing/out of
    place" (visibly broken, caught below) rather than "a wrong-but-plausible number"
    (silently wrong) — restoration is a literal dict lookup, so IF the token
    round-trips intact, the value is guaranteed correct, never re-generated by the LLM.
    """
    mapping: dict[str, str] = {}

    def _sub(m: re.Match) -> str:
        token = f"@@{len(mapping)}@@"
        mapping[token] = m.group(0)
        return token

    protected = _NUMBER_RE.sub(_sub, text)
    return protected, mapping


_PLAYER_BEFORE_RE = re.compile(r"(?:player|#|no\.?|number)\s*$", re.IGNORECASE)
_METRES_AFTER_RE = re.compile(r"^\s*(?:m\b|metres?\b|meters?\b)", re.IGNORECASE)


def _placeholder_legend(text: str) -> str:
    """Say what KIND of number each @@N@@ token hides, never its value.

    With bare tokens the model can't tell a shirt number from a distance, so it
    phrases every one the same stiff way. The value itself stays hidden so it
    still can't be re-generated (and mangled) by the model. Token indices match
    _protect_numeric_refs, which numbers matches of the same regex in order.
    """
    lines = []
    for idx, m in enumerate(_NUMBER_RE.finditer(text)):
        before, after = text[: m.start()], text[m.end():]
        if m.group(0).endswith("%"):
            kind = "a percentage"
        elif _PLAYER_BEFORE_RE.search(before):
            kind = "a player's shirt number"
        elif _METRES_AFTER_RE.match(after):
            kind = "a distance (the unit follows it)"
        else:
            kind = "a number"
        lines.append(f"@@{idx}@@ = {kind}")
    return ("\nPlaceholders:\n" + "\n".join(lines) + "\n") if lines else ""


def _restore_numeric_refs(text: str, mapping: dict[str, str]) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


def _extract_json(text: str) -> dict | None:
    """Same tolerant extraction used elsewhere (validator_node, concept_synthesis) —
    handles a provider wrapping JSON in ```json fences despite json_mode=True."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    json_str = match.group(1).strip() if match else text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def detect_and_translate_to_english(
    text: str,
    *,
    llm_call: Callable[..., tuple[str, int]] = call_llm,
) -> DetectedTranslation:
    """Best-effort: on any failure, assumes English and passes the text through unchanged."""
    text = text.strip()
    fallback = DetectedTranslation(original_text=text, english_text=text, language="English", is_english=True)
    if not text:
        return fallback

    try:
        prompt = _DETECT_PROMPT.format(text=text[:2000])
        raw, _tokens = llm_call(prompt, max_tokens=600, temperature=0.0, json_mode=True, thinking_budget=0)
        data = _extract_json(raw)
        if data is None:
            return fallback
        return DetectedTranslation(
            original_text=text,
            english_text=str(data.get("english_text") or text),
            language=str(data.get("language") or "English"),
            is_english=bool(data.get("is_english", True)),
        )
    except Exception as exc:  # noqa: BLE001 — a broken detector must not block the chat
        logger.warning(f"Language detection/translation failed, assuming English: {exc}")
        return fallback


def translate_from_english(
    text: str,
    language: str,
    *,
    llm_call: Callable[..., tuple[str, int]] = call_narrator_llm,
) -> str:
    """Best-effort: returns `text` unchanged on failure or when `language` is English."""
    if not text or not language or language.strip().lower() == "english":
        return text
    protected, mapping = _protect_numeric_refs(text)
    try:
        prompt = _TRANSLATE_BACK_PROMPT.format(
            language=language,
            glossary_section=glossary_section(language),
            legend=_placeholder_legend(text),
            text=protected[:4000],
        )
        translated, _tokens = llm_call(prompt, max_tokens=1500, temperature=0.3, thinking_budget=0)
        translated = translated.strip() or protected
        if not all(token in translated for token in mapping):
            # At least one placeholder was dropped or mangled by the model before we
            # ever get to restore it — safer to show the untranslated-but-numerically
            # -correct English than either a missing figure or (if we restored what
            # DID match) a sentence silently missing a figure it should have had.
            logger.warning(f"Translation to {language!r} dropped a placeholder token, returning English")
            return text
        return _restore_numeric_refs(translated, mapping)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Translation to {language!r} failed, returning English: {exc}")
        return text


_PLAYER_MENTION_RE = re.compile(r"\bplayer\s*#?\s*(\d+)\b", re.IGNORECASE)


def mentioned_player_mismatch(question_en: str, focus_id: int) -> int | None:
    """If the (English-translated) question names a player number different from
    the currently-selected focus player, return that number — else None.

    The chat dashboard intentionally does NOT infer the focus player from free
    text — rejected due to shirt-number-collision risk (the same id can belong
    to a different player on each team; only an explicit Team+Player selection
    can disambiguate that safely, see matchmind-dashboard-redesign memory). So
    typing "what should player 11 do" while the sidebar is still on player 1
    silently analyses player 1 regardless of what the question says. This
    surfaces that mismatch so the caller (dashboard/app.py) can warn instead of
    leaving it invisible.
    """
    match = _PLAYER_MENTION_RE.search(question_en)
    if match and int(match.group(1)) != focus_id:
        return int(match.group(1))
    return None


# ── Number grounding ──────────────────────────────────────────────────────────
#
# narrator_node writes the answer directly in the user's language (no English
# draft + translation hop, which is what made answers read machine-translated).
# That gives up the @@N@@ placeholder protection, so correctness is checked
# AFTER writing instead: every number in the output must already exist in the
# data the narrator was given. This still catches the production bug that
# motivated placeholders ("Player 11" -> "Cầu thủ 1").

_ANY_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_PLAYER_REF_RE = re.compile(
    r"(?:#|\bsố|\bcầu thủ|\bplayer|\bno\.?|\bnumber)\s*(?:số\s*)?#?\s*(\d+)(?![\d.,]*\s*(?:%|m\b|mét))",
    re.IGNORECASE,
)
# Small counts ("2 phương án", "1/3 cuối sân") are too common in natural prose to
# demand they appear in the data — but never accepted as a player reference.
_FREE_SMALL_INTEGERS = {0, 1, 2, 3}


def _to_float(token: str) -> float:
    return float(token.replace(",", "."))


def ungrounded_numbers(text: str, source: str, player_ids: set[int]) -> list[str]:
    """Numbers in `text` that can't be traced back to `source`.

    - A number used as a player reference ("số 11", "Player 11", "#11") must be
      exactly one of `player_ids`.
    - Any other number must match a number in `source` within display rounding
      (8.23 may be written 8.2 or 8), or be that number x100 / /100 (a 0.82
      confidence written as 82%), or be a small count 0-3. Signs are ignored:
      "-2.1%" in the data is naturally written "giảm 2.1%".
    """
    allowed: set[float] = set()
    for tok in _ANY_NUMBER_RE.findall(source):
        v = abs(_to_float(tok))
        allowed.update({v, v * 100, v / 100})

    def _grounded(v: float) -> bool:
        if v.is_integer() and int(v) in _FREE_SMALL_INTEGERS:
            return True
        return any(abs(v - a) <= 0.051 or (v.is_integer() and round(a) == v) for a in allowed)

    bad: list[str] = []
    player_spans: set[tuple[int, int]] = set()
    for m in _PLAYER_REF_RE.finditer(text):
        player_spans.add(m.span(1))
        if int(m.group(1)) not in player_ids:
            bad.append(m.group(1))
    for m in _ANY_NUMBER_RE.finditer(text):
        if m.span() in player_spans:
            continue
        if not _grounded(abs(_to_float(m.group(0)))):
            bad.append(m.group(0))
    return bad


# ── Dashboard-facing view ─────────────────────────────────────────────────────

@dataclass
class LocalizedAnswer:
    answer: str
    rationale: list[str] = field(default_factory=list)


def _translate_rationale(
    bullets: list[str], language: str, llm_call: Callable[..., tuple[str, int]]
) -> list[str]:
    """Translate all bullets in ONE call; fall back to the English bullets if the
    model merges/splits lines (a count mismatch means the pairing can't be trusted)."""
    if not bullets:
        return []
    joined = "\n".join(f"- {b}" for b in bullets)
    translated = translate_from_english(joined, language, llm_call=llm_call)
    if translated == joined:
        return bullets
    lines = [re.sub(r"^\s*[-•*]\s*", "", ln).strip() for ln in translated.splitlines() if ln.strip()]
    return lines if len(lines) == len(bullets) else bullets


def localize_advice(
    advice: "TacticalAdvice",
    language: str,
    *,
    llm_call: Callable[..., tuple[str, int]] = call_narrator_llm,
) -> LocalizedAnswer:
    """What the chat shows: the persuasive answer + the numbers-backed rationale,
    in the user's language.

    narrator_node normally already wrote both natively in `language` — then this
    is a no-op. Translation (with placeholder protection) is only the fallback
    for when native narration failed its number check or didn't run.
    """
    answer = advice.narrative or advice.reasoning
    rationale = list(advice.rationale or [])
    if is_english(language):
        return LocalizedAnswer(answer, rationale)
    if advice.narrative and is_same_language(advice.narrative_language, language):
        return LocalizedAnswer(answer, rationale)
    return LocalizedAnswer(
        translate_from_english(answer, language, llm_call=llm_call),
        _translate_rationale(rationale, language, llm_call),
    )
