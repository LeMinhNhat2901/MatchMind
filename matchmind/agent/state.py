"""
LangGraph Agent State.

Defines the shared state TypedDict that flows through all nodes in the graph.
Every node reads from and writes to this state.
"""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from matchmind.schema.match_state import (
    MatchState,
    SituationFeatures,
    TacticalAdvice,
)


class AgentState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph tactical agent.

    Fields marked with defaults are set by the graph's entry point.
    Optional fields are populated by specific nodes.
    """

    # ── Input (set at START) ────────────────────────────────
    match_state: MatchState
    focus_player_id: int
    question: str

    # ── Situation encoding (encode_situation node) ──────────
    situation_features: SituationFeatures | None

    # ── Pitch rendering (render_pitch node) ─────────────────
    pitch_image_path: str | None

    # ── Pitch control (counterfactual node) ──────────────────
    candidate_actions: list[dict] | None     # CandidateAction dicts (shared w/ reasoner)
    pitch_control_grid: list | None          # serialised from np.ndarray
    pitch_control_baseline: float | None
    counterfactual_analysis: list[dict] | None  # ActionAnalysis as dicts
    counterfactual_text: str | None          # formatted text for prompt

    # ── RAG retrieval (retrieve node) ────────────────────────
    retrieved_tactics: list[dict] | None
    retrieval_confidence: float | None
    retrieval_retry_count: int               # counts re-retrieve attempts
    expanded_query: str | None              # used in re-retrieve

    # ── Evidence assembly (assemble_evidence node) ────────────
    evidence_set: dict[str, Any] | None     # {"rag": [...], "stats": {...}, "context": {...}}

    # ── Generation (generate_reasoning node) ─────────────────
    raw_llm_output: str | None
    tactical_advice: TacticalAdvice | None

    # ── Validation (validate_output node) ────────────────────
    validation_errors: list[str]
    generation_retry_count: int             # counts regeneration attempts
    validation_passed: bool

    # ── Observability ─────────────────────────────────────────
    trace: list[dict] | None               # list of step traces for logging
    total_tokens_used: int
