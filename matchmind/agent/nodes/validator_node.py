"""
Validator Node — Anti-Hallucination and Schema Validation.

Parses the raw LLM output into TacticalAdvice and checks:
1. cited_concepts ⊆ retrieved concept titles
2. confidence in [0, 1]
3. recommended_action is not empty or too vague
4. JSON is parseable

On failure, sets validation_errors for the conditional edge to trigger regeneration.
"""
from __future__ import annotations

import json
import logging
import re
import time

from matchmind.agent.state import AgentState
from matchmind.config import settings
from matchmind.schema.match_state import (
    AlternativeAction,
    EvidenceItem,
    TacticalAdvice,
)

logger = logging.getLogger(__name__)

# Vague action phrases that trigger actionability failure
VAGUE_PHRASES = [
    "attack better", "play well", "do something", "be more aggressive",
    "improve positioning", "play smarter", "move forward",
]


def validator_node(state: AgentState) -> AgentState:
    """
    Node: Parse and validate LLM output.

    Input:  state.raw_llm_output, state.retrieved_tactics, state.situation_features
    Output: state.tactical_advice (if valid), state.validation_errors
    """
    t0 = time.time()

    raw = state.get("raw_llm_output") or ""
    retrieved = state.get("retrieved_tactics") or []
    retrieved_titles = [r["title"] for r in retrieved]
    features = state.get("situation_features")

    errors: list[str] = []
    advice: TacticalAdvice | None = None

    # ── Step 1: Parse JSON ───────────────────────────────────
    parsed = _extract_json(raw)
    if parsed is None:
        errors.append(f"JSON parse failed. Raw output: {raw[:200]}")
    else:
        # ── Step 2: Schema validation ────────────────────────
        try:
            alternatives = [
                AlternativeAction(**alt) for alt in parsed.get("alternatives", [])
            ]
            evidence = [
                EvidenceItem(**e) for e in parsed.get("evidence", [])
            ]

            advice = TacticalAdvice(
                recommended_action=parsed.get("recommended_action", ""),
                alternatives=alternatives,
                reasoning=parsed.get("reasoning", ""),
                confidence=float(parsed.get("confidence", 0.5)),
                evidence=evidence,
                cited_concepts=parsed.get("cited_concepts", []),
                situation_features=features,
                focus_player_id=state.get("focus_player_id"),
                match_id=state["match_state"].match_id,
                timestamp=state["match_state"].timestamp,
            )
        except Exception as exc:
            errors.append(f"Schema validation failed: {exc}")

    if advice is not None:
        # ── Step 3: Anti-hallucination check ─────────────────
        invalid_cites = [
            c for c in advice.cited_concepts
            if c not in retrieved_titles
        ]
        if invalid_cites:
            errors.append(
                f"Cited concepts not in retrieved list: {invalid_cites}. "
                f"Valid titles: {retrieved_titles}"
            )

        # ── Step 4: Confidence range ──────────────────────────
        if not 0.0 <= advice.confidence <= 1.0:
            errors.append(f"confidence={advice.confidence} out of [0,1] range")
            advice = advice.model_copy(update={"confidence": max(0.0, min(1.0, advice.confidence))})

        # ── Step 5: Actionability check ──────────────────────
        action_lower = advice.recommended_action.lower().strip()
        if not action_lower:
            errors.append("recommended_action is empty")
        elif any(vague in action_lower for vague in VAGUE_PHRASES):
            errors.append(f"recommended_action is too vague: '{advice.recommended_action}'")

    elapsed = (time.time() - t0) * 1000
    passed = len(errors) == 0

    logger.debug(
        f"[validator] {elapsed:.1f}ms | passed={passed} | "
        f"errors={errors[:2] if errors else '[]'}"
    )

    trace = state.get("trace") or []
    trace.append({
        "node": "validator",
        "latency_ms": round(elapsed, 1),
        "passed": passed,
        "errors": errors[:3],
    })

    return {
        **state,
        "tactical_advice": advice,
        "validation_errors": errors,
        "validation_passed": passed,
        "trace": trace,
    }


def check_validation(state: AgentState) -> str:
    """
    Conditional edge function after validator_node.

    Returns:
        "regenerate" if there are errors AND retries not exhausted
        "end" if validation passed or max retries reached
    """
    errors = state.get("validation_errors") or []
    retry_count = state.get("generation_retry_count") or 0
    max_retries = settings.agent_max_retries

    if errors and retry_count < max_retries:
        logger.info(
            f"[check_validation] Errors found, retry {retry_count}/{max_retries}: {errors[:1]}"
        )
        return "regenerate"

    if errors:
        logger.warning(
            f"[check_validation] Validation still failing after {retry_count} retries — accepting anyway"
        )

    return "end"


def increment_generation_retry(state: AgentState) -> AgentState:
    """Increment the generation retry counter before re-generating."""
    return {
        **state,
        "generation_retry_count": (state.get("generation_retry_count") or 0) + 1,
        "validation_errors": [],
    }


def _extract_json(text: str) -> dict | None:
    """
    Extract the first valid JSON object from LLM output.

    Handles cases where the LLM wraps JSON in markdown code blocks.
    """
    # Try to extract from ```json ... ``` blocks first
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Try to find the JSON object directly
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.debug(f"JSON decode error: {exc}")
        return None
