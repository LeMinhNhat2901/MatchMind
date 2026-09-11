"""
Reasoner Node — LLM Generation.

Builds a comprehensive prompt from all assembled evidence and calls the LLM
to generate structured tactical advice with counterfactual reasoning.
"""
from __future__ import annotations

import logging
import time

from matchmind.agent.state import AgentState
from matchmind.llm_client import call_llm

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

    # Provider is chosen by settings.llm_provider (default: gemini). json_mode
    # makes Gemini/OpenAI emit valid JSON directly; the validator still tolerates
    # markdown-fenced JSON for providers that ignore it.
    # Budget must cover reasoning + alternatives + evidence AND Gemini 3.x
    # "thinking" tokens, or the JSON comes back truncated → parse failure.
    raw_output, tokens = call_llm(
        prompt,
        image_path=state.get("pitch_image_path"),
        max_tokens=8000,
        temperature=0.3,
        json_mode=True,
    )

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

    if features.is_ball_carrier:
        possession_banner = f"Player {focus_id} HAS THE BALL."
        possession_rule = (
            "- Player HAS the ball: recommend an on-ball action (pass / dribble / "
            "shoot / hold)."
        )
    else:
        possession_banner = (
            f"Player {focus_id} DOES NOT have the ball "
            f"(Player {features.ball_carrier_id} has it)."
        )
        possession_rule = (
            "- Player is OFF the ball: you MUST NOT recommend a pass, dribble, carry, "
            "or shot FOR this player — they cannot pass a ball they don't have. "
            "Recommend an OFF-BALL action instead: a run, a movement to create/exploit "
            "space, or positioning to support the actual ball carrier "
            f"(Player {features.ball_carrier_id})."
        )

    prompt = f"""You are an expert football tactical analyst. Analyse the situation and provide specific, actionable tactical advice.

{possession_banner}

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
  "recommended_action_id": "canonical_id_from_candidate_actions_or_null",
  "alternatives": [
    {{
      "action": "Alternative action description",
      "action_id": "canonical_id_from_candidate_actions_or_null",
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
{possession_rule}
- The computed statistics and pitch-control numbers above are the PRIMARY evidence.
  The pitch image is supporting context only — never override the numbers with it.
- cited_concepts MUST ONLY contain titles copied verbatim from this list: [{valid_titles_str}]
- Do NOT invent concepts not in the list above
- recommended_action must be SPECIFIC (e.g., "Pass to player 9 in the channel" not "Pass forward")
- Prefer an action from the CANDIDATE ACTIONS shortlist; if you deviate, justify why
- recommended_action_id (and each alternative's action_id) MUST be copied EXACTLY (character
  for character) from the "action" field shown in the CANDIDATE ACTIONS shortlist above —
  this is how the pitch diagram draws an arrow for your recommendation. Use null ONLY if your
  recommendation truly does not correspond to any shortlist entry.
- Include at least 2 alternatives with why_not explanations
- Use pitch control delta values from the quantitative analysis; note it assumes
  static defenders (upper bound) when you cite it
"""

    return prompt

