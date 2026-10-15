"""
Style Guide Extraction — learns a description of an analyst's VOICE from
YouTube transcripts, for the narrator_node persona.

Important distinction (per user feedback, 2026-09-15): RAG does not make an
LLM "learn" a writing style — it just pastes retrieved text into the prompt
for that one call, with no accumulation across calls and no weight updates.
Whatever tone shows up in a RAG-retrieved concept is an uncontrolled side
effect, not something to rely on. A consistent voice has to come from an
explicit persona/style instruction in the prompt instead — this module
builds that instruction, once, offline, and narrator_node applies it on
every call, not by stuffing transcripts into narrator_node's context.

Two-step pipeline per the user's design:
    1. `clean_transcript_excerpt` — light ASR cleanup (punctuation, drop
       filler words) WITHOUT summarizing or rephrasing, so the speaker's own
       sentence rhythm and word choice survive. Raw YouTube auto-captions are
       unusable for style analysis as-is: no punctuation, filler words,
       misrecognitions.
    2. `extract_style_guide` — reads the cleaned excerpts and describes voice
       characteristics ONLY (sentence length, metaphor use, technical depth,
       how cause-and-effect is narrated) — explicitly NOT a summary of
       football content and NOT a set of quotable phrases, since narrator_node
       must never present copied wording as its own commentary.
"""
from __future__ import annotations

import logging
from typing import Callable

from matchmind.llm_client import call_llm

logger = logging.getLogger(__name__)


_CLEAN_PROMPT = """The text below is a raw, auto-generated YouTube caption excerpt from a \
football tactical-analysis video — no punctuation, filler words, occasional misheard words.

Add punctuation and paragraph breaks, and remove filler words/false starts ("um", "like", \
"you know", repeated words). Do NOT summarize, shorten, or reword anything — keep the speaker's \
own sentence structure and word choice as close to verbatim as possible. Output ONLY the cleaned \
text, nothing else.

RAW EXCERPT:
\"\"\"
{raw_text}
\"\"\""""

_STYLE_EXTRACTION_PROMPT = """Below are cleaned transcript excerpts from football \
tactical-analysis YouTube videos, labelled by channel.

Describe ONLY the speaking/writing STYLE common across these excerpts — NOT the football content \
itself (ignore what they're saying about football; focus on how they say it). Cover:
- Typical sentence length and rhythm (short and punchy vs long and flowing)
- Use of metaphor/imagery vs literal, technical language
- Technical/jargon depth (do they explain terms or assume knowledge?)
- How cause-and-effect is narrated ("because X, so Y" vs implicit/assumed)
- Typical framing (do they open with a question, a claim, a scene-setting description?)

Output 5-8 short bullet points describing HOW to write in this voice — instructions for a writer \
to follow, not a description of these specific videos. Do NOT quote or closely paraphrase exact \
phrases from the excerpts — describe the PATTERN only; this is a style guide, not a template to \
copy from.

EXCERPTS:
{excerpts}"""


def clean_transcript_excerpt(
    raw_text: str,
    *,
    llm_call: Callable[..., tuple[str, int]] = call_llm,
) -> str | None:
    """Punctuate/de-filler a raw ASR excerpt without summarizing it. None on failure."""
    if not raw_text or len(raw_text.strip()) < 80:
        return None
    prompt = _CLEAN_PROMPT.format(raw_text=raw_text[:4000])
    try:
        text, _tokens = llm_call(prompt, max_tokens=1200, temperature=0.1, thinking_budget=0)
        text = text.strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 — one bad excerpt must not kill a batch run
        logger.warning(f"Transcript cleaning failed: {exc}")
        return None


def extract_style_guide(
    excerpts: list[tuple[str, str]],
    *,
    llm_call: Callable[..., tuple[str, int]] = call_llm,
) -> str | None:
    """excerpts: [(channel_name, cleaned_text), ...] -> a style-guide bullet list, or None."""
    if not excerpts:
        return None
    labelled = "\n\n".join(f"[{channel}]\n{text}" for channel, text in excerpts)
    prompt = _STYLE_EXTRACTION_PROMPT.format(excerpts=labelled[:12000])
    try:
        text, _tokens = llm_call(prompt, max_tokens=800, temperature=0.3, thinking_budget=0)
        text = text.strip()
        return text or None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Style guide extraction failed: {exc}")
        return None
