"""
Reasoner Node — LLM Generation.

Builds a comprehensive prompt from all assembled evidence and calls the LLM
to generate structured tactical advice with counterfactual reasoning.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

from matchmind.agent.state import AgentState
from matchmind.config import settings

logger = logging.getLogger(__name__)


def reasoner_node(state: AgentState) -> AgentState:
    """
    Node: Generate tactical advice via LLM.

    Builds a rich prompt combining:
    - Situation features (computed stats)
    - Retrieved tactical concepts (RAG)
    - Counterfactual analysis (pitch control)
    - Match context
    - Focus question

    Sends pitch image + text to LLM (multimodal).
    """
    t0 = time.time()

    prompt = _build_prompt(state)

    if settings.llm_provider == "anthropic":
        raw_output, tokens = _call_anthropic(prompt, state.get("pitch_image_path"))
    else:
        raw_output, tokens = _call_openai(prompt, state.get("pitch_image_path"))

    elapsed = (time.time() - t0) * 1000
    logger.debug(f"[reasoner] {elapsed:.1f}ms | tokens={tokens}")

    trace = state.get("trace") or []
    trace.append({"node": "reasoner", "latency_ms": round(elapsed, 1), "tokens": tokens})

    return {
        **state,
        "raw_llm_output": raw_output,
        "total_tokens_used": (state.get("total_tokens_used") or 0) + tokens,
        "trace": trace,
    }


def _build_prompt(state: AgentState) -> str:
    """Build the full text prompt for the LLM."""
    match_state = state["match_state"]
    features = state["situation_features"]
    evidence = state.get("evidence_set") or {}
    retrieved = state.get("retrieved_tactics") or []
    cf_text = state.get("counterfactual_text") or ""
    candidates = state.get("candidate_actions") or []
    question = state["question"]
    focus_id = state["focus_player_id"]

    candidates_text = (
        "\n".join(f"  • {c['action']} — {c['label']}" for c in candidates)
        or "  (none generated)"
    )

    # Retrieved tactical concepts (RAG evidence)
    rag_text = "\n".join(
        f"  • {r['title']}: {r['content'][:300]}"
        for r in retrieved
    )

    # Stats evidence
    stats_text = " | ".join(
        e["content"] for e in evidence.get("stats", [])
    )

    # Context evidence
    context_text = " | ".join(
        e["content"] for e in evidence.get("context", [])
    )

    # Valid cited concepts (anti-hallucination constraint)
    valid_titles = [r["title"] for r in retrieved]
    valid_titles_str = ", ".join(f'"{t}"' for t in valid_titles)

    prompt = f"""You are an expert football tactical analyst. Analyse the situation and provide specific, actionable tactical advice.

═══════════════════════════════════════════════════════════════
MATCH CONTEXT
═══════════════════════════════════════════════════════════════
{context_text}

═══════════════════════════════════════════════════════════════
SITUATION — PLAYER {focus_id}
═══════════════════════════════════════════════════════════════
{features.natural_language_description}

KEY COMPUTED STATISTICS:
{stats_text}

═══════════════════════════════════════════════════════════════
TACTICAL KNOWLEDGE (RETRIEVED FROM KNOWLEDGE BASE)
═══════════════════════════════════════════════════════════════
{rag_text}

═══════════════════════════════════════════════════════════════
CANDIDATE ACTIONS (rule-based shortlist — evaluate these first)
═══════════════════════════════════════════════════════════════
{candidates_text}

═══════════════════════════════════════════════════════════════
QUANTITATIVE ACTION ANALYSIS (PITCH CONTROL MODEL)
═══════════════════════════════════════════════════════════════
{cf_text}

═══════════════════════════════════════════════════════════════
QUESTION
═══════════════════════════════════════════════════════════════
{question}

═══════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════
Respond with ONLY valid JSON matching this exact structure:

{{
  "recommended_action": "Specific, concrete action (not vague). Include player IDs if relevant.",
  "alternatives": [
    {{
      "action": "Alternative action description",
      "why_not": "Specific reason why this is inferior in this exact situation",
      "delta_pitch_control": null
    }}
  ],
  "reasoning": "Step-by-step reasoning connecting the evidence (spatial features + retrieved concepts + pitch control) to your recommendation. Be specific.",
  "confidence": 0.82,
  "evidence": [
    {{"id": "concept_id_or_feature_name", "source": "rag", "content": "What this evidence tells us"}},
    {{"id": "computed_features", "source": "stats", "content": "Key statistic used"}},
    {{"id": "match_context", "source": "context", "content": "How context influences decision"}}
  ],
  "cited_concepts": ["Title1", "Title2"],
  "situation_features": null
}}

CRITICAL CONSTRAINTS:
- The computed statistics and pitch-control numbers above are the PRIMARY evidence.
  The pitch image is supporting context only — never override the numbers with it.
- cited_concepts MUST ONLY contain titles copied verbatim from this list: [{valid_titles_str}]
- Do NOT invent concepts not in the list above
- recommended_action must be SPECIFIC (e.g., "Pass to player 9 in the channel" not "Pass forward")
- Prefer an action from the CANDIDATE ACTIONS shortlist; if you deviate, justify why
- Include at least 2 alternatives with why_not explanations
- Use pitch control delta values from the quantitative analysis; note it assumes
  static defenders (upper bound) when you cite it
"""

    return prompt


def _call_anthropic(prompt: str, image_path: str | None) -> tuple[str, int]:
    """Call Claude API with optional pitch image."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    content = []

    # Add image if available
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64,
            },
        })
        content.append({
            "type": "text",
            "text": "Above is the current match situation visualised on the pitch (red = home, blue = away, white = ball). The highlighted player is the focus player.\n\n" + prompt,
        })
    else:
        content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=1000,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return raw, tokens


def _call_openai(prompt: str, image_path: str | None) -> tuple[str, int]:
    """Call OpenAI GPT-4V API with optional pitch image."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

    client = OpenAI(api_key=settings.openai_api_key)

    content = []
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })
    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.choices[0].message.content
    tokens = response.usage.total_tokens
    return raw, tokens
