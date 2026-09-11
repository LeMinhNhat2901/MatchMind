"""
Unified Match Schema — The Single Contract Between All Components.

Every data source (StatsBomb, Metrica, Live API, CV) MUST produce a MatchState.
Every downstream module (Situation Engine, Agent, Evaluator) ONLY consumes MatchState.

Design principle: adapters adapt IN, nothing leaks OUT.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────────────────────
# Controlled vocabulary — phase of play
# Used for MatchState.phase_of_play AND as the retrieval metadata filter key.
# "open_play" is the catch-all so adapters never fail validation.
# ──────────────────────────────────────────────────────────────────────────────

PhaseOfPlay = Literal[
    "build_up",
    "progression",
    "final_third",
    "attacking_transition",
    "defensive_transition",
    "settled_defense",
    "set_piece",
    "open_play",
]

PHASE_VOCAB: frozenset[str] = frozenset(
    [
        "build_up",
        "progression",
        "final_third",
        "attacking_transition",
        "defensive_transition",
        "settled_defense",
        "set_piece",
        "open_play",
    ]
)


def normalize_phase(raw: str | None) -> PhaseOfPlay:
    """Map any incoming phase string to the controlled vocabulary.

    Unknown values collapse to ``"open_play"`` so nothing downstream breaks.
    """
    if not raw:
        return "open_play"
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "possession": "progression",
        "in_possession": "progression",
        "out_of_possession": "settled_defense",
        "transition": "attacking_transition",
        "counter": "attacking_transition",
        "counter_attack": "attacking_transition",
        "counterattack": "attacking_transition",
        "attack": "final_third",
        "attacking": "final_third",
        "defense": "settled_defense",
        "defence": "settled_defense",
        "defending": "settled_defense",
        "buildup": "build_up",
        "setpiece": "set_piece",
        "dead_ball": "set_piece",
        "open": "open_play",
    }
    if key in PHASE_VOCAB:
        return key  # type: ignore[return-value]
    return aliases.get(key, "open_play")  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# Core match state models
# ──────────────────────────────────────────────────────────────────────────────


class BallState(BaseModel):
    """Ball position on the pitch (metres, standard coordinate system)."""

    x: float = Field(..., ge=0.0, le=105.0, description="Longitudinal position (0=left goal, 105=right goal)")
    y: float = Field(..., ge=0.0, le=68.0, description="Lateral position (0=bottom, 68=top)")
    z: float = Field(default=0.0, ge=0.0, description="Height above pitch (metres)")


class PlayerState(BaseModel):
    """
    Single player's state at one instant.

    Coordinate system: 0–105m (x, attack → right) × 0–68m (y, bottom → top).
    This matches Metrica / StatsBomb / standard tracking conventions.
    """

    id: int = Field(..., description="Jersey / tracking ID of the player")
    team: Literal["home", "away"] = Field(..., description="Which side this player is on")
    role: str | None = Field(
        default=None,
        description="Positional role e.g. 'center_back', 'attacking_midfielder'",
    )
    x: float = Field(..., ge=0.0, le=105.0)
    y: float = Field(..., ge=0.0, le=68.0)
    # Velocity components — required for pitch control model; default 0 when not available
    vx: float = Field(default=0.0, description="Velocity in x direction (m/s)")
    vy: float = Field(default=0.0, description="Velocity in y direction (m/s)")

    @property
    def speed(self) -> float:
        """Scalar speed in m/s."""
        return (self.vx**2 + self.vy**2) ** 0.5

    @property
    def position(self) -> tuple[float, float]:
        """(x, y) tuple for geometry calculations."""
        return (self.x, self.y)


class MatchState(BaseModel):
    """
    The single unified representation of a football match snapshot.

    All source adapters (StatsBomb, Metrica, Live, CV) return this model.
    All downstream modules accept this model as input.
    """

    match_id: str = Field(..., description="Unique match identifier")
    timestamp: float = Field(
        ..., ge=0.0, description="Seconds from kick-off"
    )

    # Match score
    score_home: int = Field(default=0, ge=0)
    score_away: int = Field(default=0, ge=0)

    # Match context
    possession: Literal["home", "away"] | None = Field(
        default=None, description="Which team currently has possession (possession_team)"
    )
    possession_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Cumulative possession share for home team (0-100). Optional; live/season stat.",
    )
    phase_of_play: PhaseOfPlay | None = Field(
        default=None,
        description="Tactical phase from the PhaseOfPlay controlled vocabulary",
    )

    # On-pitch objects
    ball: BallState
    players: list[PlayerState] = Field(..., min_length=1)

    # StatsBomb 360 camera-visible polygon (pitch coords, closed ring — first
    # point repeated at the end). Players outside it were never tracked, which
    # is why a 360 snapshot legitimately has fewer than 22 players (typically
    # ~16, StatsBomb's own data ranges 4-21 — never all 22). None for sources
    # that track everyone (Metrica, CV) or don't provide the polygon.
    visible_area: list[tuple[float, float]] | None = Field(
        default=None,
        description="360 camera field-of-view polygon in pitch metres, if known.",
    )

    # True only when player velocities (vx, vy) are real (Metrica frames, CV tracking).
    # StatsBomb 360 is a single freeze-frame → False → pitch control runs degraded.
    has_velocity: bool = Field(
        default=False,
        description="Whether player vx/vy are real. False → pitch control uses static approximation.",
    )

    # Provenance — helps with debugging but agent does NOT use this for logic
    source: Literal["statsbomb", "metrica", "live", "cv", "synthetic"] = Field(
        default="statsbomb"
    )

    @field_validator("phase_of_play", mode="before")
    @classmethod
    def _coerce_phase(cls, v: Any) -> Any:
        if v is None or v in PHASE_VOCAB:
            return v
        return normalize_phase(str(v))

    # Optional raw event metadata (for evaluation ground truth)
    event_type: str | None = Field(
        default=None,
        description="StatsBomb event type that follows this snapshot (ground truth for evaluation)",
    )
    event_player_id: int | None = Field(
        default=None,
        description="Player who performed the following action (ground truth)",
    )

    @field_validator("players")
    @classmethod
    def must_have_both_teams(cls, players: list[PlayerState]) -> list[PlayerState]:
        teams = {p.team for p in players}
        if len(teams) < 2:
            raise ValueError("MatchState must contain players from both 'home' and 'away' teams")
        return players

    def get_player(
        self, player_id: int, team: Literal["home", "away"] | None = None
    ) -> PlayerState | None:
        """Retrieve a player by ID, or None if not found.

        `id` is NOT guaranteed globally unique — some sources (Metrica shirt
        numbers) reuse 1-11 on both teams. Without `team`, this returns
        whichever match comes first (matches historical behaviour). Pass
        `team` whenever it's known to disambiguate correctly; see
        `has_ambiguous_id()` / `agent.graph.TacticalAgent.analyze(focus_team=...)`.
        """
        candidates = (p for p in self.players if p.id == player_id)
        if team is not None:
            return next((p for p in candidates if p.team == team), None)
        return next(candidates, None)

    def has_ambiguous_id(self, player_id: int) -> bool:
        """True if `player_id` matches players on BOTH teams (get_player(id) would be ambiguous)."""
        teams = {p.team for p in self.players if p.id == player_id}
        return len(teams) > 1

    def get_team_players(self, team: Literal["home", "away"]) -> list[PlayerState]:
        """All players belonging to a team."""
        return [p for p in self.players if p.team == team]

    def get_opponents(self, team: Literal["home", "away"]) -> list[PlayerState]:
        """Players on the opposing team."""
        opp = "away" if team == "home" else "home"
        return [p for p in self.players if p.team == opp]

    @property
    def minute(self) -> int:
        """Match minute (integer)."""
        return int(self.timestamp // 60)

    @property
    def home_winning(self) -> bool:
        return self.score_home > self.score_away

    @property
    def away_winning(self) -> bool:
        return self.score_away > self.score_home

    @property
    def score_diff_from_home(self) -> int:
        """Positive = home winning, negative = home losing."""
        return self.score_home - self.score_away


# ──────────────────────────────────────────────────────────────────────────────
# Situation representation (computed by Situation Engine, not adapters)
# ──────────────────────────────────────────────────────────────────────────────


class SituationFeatures(BaseModel):
    """
    Structured tactical feature vector computed from MatchState by the Situation Engine.

    This is pure computation — NO LLM involved.
    The Agent reasons ON TOP OF these features, not on raw (x, y) coordinates.
    """

    focus_player_id: int
    focus_player_team: Literal["home", "away"]
    focus_player_role: str | None = None

    # ── Possession — WHO has the ball ─────────────────────────
    # Without this, candidate-action generation and the reasoner prompt have
    # no way to tell an on-ball player from an off-ball one, and end up
    # recommending passes/dribbles/shots for players who don't have the ball.
    is_ball_carrier: bool = Field(
        default=True, description="Whether the focus player currently has the ball"
    )
    ball_carrier_id: int | None = Field(
        default=None, description="Player id presumed to currently have the ball"
    )

    # ── Spatial (individual) ─────────────────────────────────
    distance_to_ball: float = Field(..., description="metres")
    nearest_opponent_distance: float = Field(..., description="metres")
    nearest_teammate_distance: float = Field(..., description="metres")
    space_ahead: float = Field(..., description="Free space (m) toward attacking goal")

    # ── Passing ─────────────────────────────────────────────
    passing_lane_open: bool = Field(..., description="At least one unobstructed passing lane exists")
    progressive_passes_available: int = Field(
        default=0, description="Number of teammates reachable with a progressive pass"
    )
    third_man_opportunity: bool = Field(
        default=False, description="A third-man combination is geometrically feasible"
    )
    switch_play_viable: bool = Field(
        default=False, description="Long switch to opposite flank is available"
    )
    open_passing_lane_player_ids: list[int] = Field(
        default_factory=list, description="IDs of teammates with open lane"
    )

    # ── Local numerical balance ──────────────────────────────
    local_numerical_advantage: int = Field(
        ..., description="Teammates minus opponents within 15m radius (+ve = overload)"
    )
    overload_left: bool = Field(default=False)
    overload_right: bool = Field(default=False)

    # ── Half-space & positional ──────────────────────────────
    half_space_occupied: bool = Field(
        default=False, description="Focus player is in a half-space zone"
    )
    is_in_attacking_third: bool = Field(default=False)
    is_in_defensive_third: bool = Field(default=False)

    # ── Team structure ───────────────────────────────────────
    defensive_line_height: float = Field(
        ..., description="y-coordinate of opposing defensive line"
    )
    team_compactness: float = Field(
        ..., description="Average inter-player distance for focus player's team (m)"
    )
    attacking_width: float = Field(
        ..., description="Lateral spread of attacking team (m)"
    )
    numerical_superiority_zone: bool = Field(
        default=False, description="Numerical superiority in the key tactical zone"
    )

    # ── Pitch control (filled by pitch_control module, optional at encode stage) ─
    pitch_control_at_ball: float | None = Field(
        default=None, description="Pitch control probability for focus team at ball position (0–1)"
    )

    # ── Natural language summary (rule-based, not LLM) ──────
    natural_language_description: str = Field(
        ..., description="Human-readable summary for RAG query and prompt context"
    )

    # ── Raw computed stats dict (passed as evidence to agent) ─
    computed_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional computed statistics as key-value pairs",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Candidate actions — rule-based options fed to BOTH the counterfactual engine
# and the reasoner. Generated by situation_engine.candidate_actions.
# ──────────────────────────────────────────────────────────────────────────────


class CandidateAction(BaseModel):
    """A single tactical option to be evaluated by the counterfactual engine."""

    action: str = Field(..., description="Canonical id, e.g. 'pass_to_9', 'run_in_behind'")
    label: str = Field(..., description="Human-readable description")
    target_xy: tuple[float, float] | None = Field(
        default=None,
        description="Simulated target (metres) — where the BALL goes if moves_ball, else where the PLAYER runs",
    )
    target_player_id: int | None = Field(
        default=None, description="Receiver id for pass_to_* actions"
    )
    moves_ball: bool = Field(
        default=True,
        description=(
            "True: this action moves the ball to target_xy (pass/dribble/shot/switch — "
            "requires the focus player to actually have the ball). "
            "False: an OFF-BALL movement (run/positioning) — the ball stays with its "
            "real carrier; only the focus player relocates to target_xy."
        ),
    )
    origin: Literal["default", "situation", "passing_lane"] = "default"


# ──────────────────────────────────────────────────────────────────────────────
# Match event — V2 live incremental state. MVP adapters return MatchState directly;
# only the live adapter + state engine use MatchEvent.
# ──────────────────────────────────────────────────────────────────────────────


class MatchEvent(BaseModel):
    """A single event in a live stream, applied to the current MatchState."""

    match_id: str
    timestamp: float = Field(..., ge=0.0, description="Seconds from kick-off")
    type: Literal[
        "pass", "carry", "shot", "recovery", "duel", "tracking_frame", "other"
    ]
    player_id: int | None = None
    outcome: str | None = None
    ball: BallState | None = None
    players: list[PlayerState] | None = Field(
        default=None, description="Full frame when type == 'tracking_frame'"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tactical KB concept
# ──────────────────────────────────────────────────────────────────────────────


class TacticalConcept(BaseModel):
    """A single entry in the Tactical Knowledge Base."""

    concept_id: str
    category: str
    phase: str
    title: str
    content: str
    conditions: list[str] = Field(default_factory=list)
    objective: list[str] = Field(default_factory=list)
    possible_actions: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    trigger_conditions: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Agent output — Structured Tactical Advice
# ──────────────────────────────────────────────────────────────────────────────


class EvidenceItem(BaseModel):
    """A single piece of evidence used by the agent."""

    id: str = Field(..., description="Concept ID (RAG), feature name (stats), or field name (context)")
    source: Literal["rag", "stats", "context"] = Field(
        ..., description="Which evidence bucket this came from"
    )
    content: str = Field(..., description="Human-readable evidence text")


class AlternativeAction(BaseModel):
    """A considered-but-rejected alternative tactical option."""

    action: str = Field(..., description="Description of the alternative action")
    action_id: str | None = Field(
        default=None,
        description=(
            "Canonical id copied verbatim from the CANDIDATE ACTIONS shortlist "
            "(e.g. 'pass_to_9', 'run_in_behind') — lets the renderer draw this "
            "alternative as an arrow. Null if it doesn't match a shortlist entry."
        ),
    )
    why_not: str = Field(..., description="Reason this alternative is inferior in this situation")
    delta_pitch_control: float | None = Field(
        default=None,
        description=(
            "Change in controlled pitch area vs the recommended action, in "
            "PERCENTAGE POINTS (e.g. -2.7 means 2.7pp worse). Display as-is with a '%'."
        ),
    )

    @field_validator("delta_pitch_control", mode="before")
    @classmethod
    def _coerce_delta(cls, v: Any) -> Any:
        """LLMs return '+2.1%' / '-3%' / '0.021' / 'null' — normalise to percentage points."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            # A fraction like 0.021 almost certainly means 2.1 percentage points.
            return round(float(v) * 100, 2) if abs(v) < 1 else float(v)
        if isinstance(v, str):
            s = v.strip().replace("+", "").strip()
            had_pct = s.endswith("%")
            s = s.rstrip("%").strip()
            if s.lower() in ("", "null", "none", "n/a", "na"):
                return None
            try:
                num = float(s)
            except ValueError:
                return None
            if not had_pct and abs(num) < 1:
                num *= 100  # bare fraction → percentage points
            return round(num, 2)
        return v


class TacticalAdvice(BaseModel):
    """
    The final structured output of the MatchMind agent.

    Anti-hallucination guarantee: cited_concepts MUST be a subset of retrieved titles.
    This is enforced by the validator node in LangGraph.
    """

    recommended_action: str = Field(
        ..., description="Specific, actionable recommendation (not vague)"
    )
    recommended_action_id: str | None = Field(
        default=None,
        description=(
            "Canonical id copied verbatim from the CANDIDATE ACTIONS shortlist "
            "(e.g. 'pass_to_9', 'run_in_behind', 'overlapping_run') that this "
            "recommendation corresponds to. Lets the renderer draw an arrow for "
            "it instead of leaving the reader to infer movement from coordinates. "
            "Null if the recommendation doesn't match a shortlist entry."
        ),
    )
    alternatives: list[AlternativeAction] = Field(
        default_factory=list,
        description="Considered alternatives with why-not reasoning",
    )
    reasoning: str = Field(
        ..., description="Step-by-step reasoning linking evidence to recommendation"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(
        ..., description="All evidence used, with source attribution"
    )
    cited_concepts: list[str] = Field(
        ...,
        description="Tactical concepts from RAG that support this recommendation. MUST be from retrieved list.",
    )
    situation_features: SituationFeatures | None = Field(
        default=None, description="The computed features used as input"
    )
    candidate_actions: list[CandidateAction] = Field(
        default_factory=list,
        description=(
            "The rule-based shortlist offered to the LLM (situation_engine.candidate_actions), "
            "carried along so recommended_action_id / alternatives[].action_id can be resolved "
            "to a target_xy for arrow rendering without re-running the situation engine."
        ),
    )
    focus_player_id: int | None = None
    match_id: str | None = None
    timestamp: float | None = None

    def resolve_action(self, action_id: str | None) -> "CandidateAction | None":
        """Look up a CandidateAction by id from this advice's shortlist, or None."""
        if not action_id:
            return None
        return next((c for c in self.candidate_actions if c.action == action_id), None)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation models
# ──────────────────────────────────────────────────────────────────────────────


class EvaluationCase(BaseModel):
    """One test case for evaluation."""

    snapshot: MatchState
    focus_player_id: int
    question: str
    ground_truth_action: str | None = Field(
        default=None, description="Actual action taken by player (from StatsBomb events)"
    )
    ground_truth_event_type: str | None = None


class EvaluationReport(BaseModel):
    """Full evaluation report across all 5 dimensions.

    Dim 0 Retrieval Quality · Dim 1 Agreement · Dim 2 Groundedness ·
    Dim 3 Actionability · Dim 4 Consistency · + system performance.
    """

    n_cases: int

    # Dim 0 — Retrieval Quality (bonus.md §6 + revise.md §9)
    precision_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_k: int | None = Field(default=None, description="k used for precision/recall")
    retrieval_confidence_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    reretrieve_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of cases that hit the re-retrieve branch"
    )

    # Dim 1 — Agreement
    agreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    # Dim 2 — Groundedness
    citation_validity_rate: float = Field(..., ge=0.0, le=1.0)
    logical_consistency_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    groundedness_score: float = Field(..., ge=0.0, le=1.0)

    # Dim 3 — Actionability
    actionability_score: float | None = Field(default=None, ge=1.0, le=5.0)

    # Dim 4 — Consistency
    consistency_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    consistency_temperature: float | None = Field(
        default=None, description="Sampling temperature used for the repeat runs"
    )
    quantitative_alignment: float | None = Field(default=None, ge=0.0, le=1.0)

    # System performance
    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    avg_tokens_used: int | None = None
    total_cost_usd: float | None = None

    # Metadata
    model_used: str | None = None
    embedding_model: str | None = None
    notes: str | None = None
