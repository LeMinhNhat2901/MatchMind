"""
Evaluation Framework — Main Evaluator.

Runs all 5 evaluation dimensions:
  0. Retrieval Quality — precision@k / recall@k of the RAG retriever
  1. Agreement         — does recommendation match actual player action?
  2. Groundedness      — are citations valid? Is reasoning internally consistent?
  3. Actionability     — is the recommendation specific enough to act on?
  4. Consistency       — is output stable? Aligned with pitch control?

Designed to be CI-ready (pytest) and independent of the dashboard.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from matchmind.config import settings
from matchmind.evaluation.retrieval_quality import (
    aggregate_retrieval_quality,
    evaluate_retrieval,
)
from matchmind.knowledge_base.embedder import load_all_concepts
from matchmind.schema.match_state import EvaluationReport, TacticalAdvice

logger = logging.getLogger(__name__)


def _last_retrieve_step(trace: list[dict]) -> dict:
    """Return the final 'retrieve' trace entry (has concept_ids + titles)."""
    steps = [s for s in (trace or []) if s.get("node") == "retrieve"]
    return steps[-1] if steps else {}

# ──────────────────────────────────────────────────────────────────────────────
# Dimension 1: Agreement
# ──────────────────────────────────────────────────────────────────────────────

# Map StatsBomb event types to broad action categories
AGREEMENT_MAP = {
    # Pass actions
    "pass": ["pass", "through_ball", "combination", "cross", "switch", "feed"],
    "cross": ["cross", "switch", "deliver", "overlap"],
    "carry": ["carry", "dribble", "drive", "progress", "advance"],
    "dribble": ["dribble", "carry", "drive", "1v1", "beat"],
    "shot": ["shoot", "shot", "fire", "attempt"],
    "clearance": ["clear", "clearance", "head"],
    "pressure": ["press", "pressure", "close_down", "defend"],
    "tackle": ["tackle", "challenge", "challenge"],
    "interception": ["intercept", "block"],
}


def compute_agreement(
    advice: TacticalAdvice,
    ground_truth_event_type: str,
) -> bool:
    """
    Check if the recommended action matches the ground truth event type.

    Uses a fuzzy mapping — checks if any keyword from the action category
    appears in the recommended action text.
    """
    if not ground_truth_event_type:
        return False

    gt_lower = ground_truth_event_type.lower()
    recommended = advice.recommended_action.lower()

    # Find matching action keywords
    for event_type, keywords in AGREEMENT_MAP.items():
        if event_type in gt_lower or any(kw in gt_lower for kw in keywords):
            # Check if recommended action contains any matching keyword
            if any(kw in recommended for kw in keywords):
                return True

    # Fallback: direct word overlap
    gt_words = set(gt_lower.split())
    rec_words = set(recommended.split())
    overlap = gt_words & rec_words
    return len(overlap) > 0


def compute_agreement_rate(
    results: list[dict],
) -> float:
    """Compute agreement rate across multiple evaluation cases."""
    if not results:
        return 0.0
    agreed = sum(1 for r in results if r.get("agreement", False))
    return agreed / len(results)


# ──────────────────────────────────────────────────────────────────────────────
# Dimension 2: Groundedness
# ──────────────────────────────────────────────────────────────────────────────

def compute_groundedness(
    advice: TacticalAdvice,
    retrieved_titles: list[str],
    logical_consistency: float | None = None,
) -> dict:
    """
    Compute groundedness metrics.

    1. citation_validity_rate: fraction of cited_concepts in retrieved_titles
    2. evidence_coverage: fraction of evidence items with valid source attribution
    3. logical_consistency: 0/1 from the LLM-as-judge check (optional)
    4. no_invented_concepts: True if 0 invalid citations
    """
    cited = advice.cited_concepts
    if not cited:
        citation_validity = 1.0  # no citations = no invalid citations
        no_invented = True
    else:
        valid = [c for c in cited if c in retrieved_titles]
        citation_validity = len(valid) / len(cited)
        no_invented = citation_validity == 1.0

    valid_sources = {"rag", "stats", "context"}
    evidence_coverage = (
        sum(1 for e in advice.evidence if e.source in valid_sources) / max(len(advice.evidence), 1)
    )

    if logical_consistency is None:
        groundedness_score = (citation_validity * 0.7) + (evidence_coverage * 0.3)
    else:
        groundedness_score = (
            citation_validity * 0.5 + logical_consistency * 0.3 + evidence_coverage * 0.2
        )

    return {
        "citation_validity_rate": round(citation_validity, 4),
        "no_invented_concepts": no_invented,
        "evidence_coverage": round(evidence_coverage, 4),
        "logical_consistency_rate": logical_consistency,
        "groundedness_score": round(groundedness_score, 4),
    }


def compute_logical_consistency_llm(advice: TacticalAdvice) -> float | None:
    """
    LLM-as-judge: does the reasoning contradict the computed SituationFeatures?

    Returns 1.0 (consistent), 0.0 (contradiction found), or None (no judge available).
    Falls back to None when features are missing or the API call fails.
    """
    feats = advice.situation_features
    if feats is None:
        return None
    try:
        from matchmind.llm_client import call_llm

        facts = feats.model_dump(
            include={
                "distance_to_ball",
                "nearest_opponent_distance",
                "space_ahead",
                "local_numerical_advantage",
                "passing_lane_open",
                "half_space_occupied",
                "third_man_opportunity",
                "switch_play_viable",
                "defensive_line_height",
            }
        )
        prompt = (
            "You are checking a football tactical explanation for internal consistency.\n\n"
            f"COMPUTED FACTS (ground truth):\n{json.dumps(facts, indent=2)}\n\n"
            f"REASONING TO CHECK:\n\"{advice.reasoning[:800]}\"\n\n"
            "Does the reasoning state anything that directly contradicts the computed "
            "facts (e.g. claims the player is free when nearest_opponent_distance is 2m)?\n"
            "Answer with ONLY 'CONSISTENT' or 'CONTRADICTION'."
        )
        text, _ = call_llm(prompt, max_tokens=24, temperature=0.0, thinking_budget=0)
        return 0.0 if "CONTRADICT" in text.strip().upper() else 1.0
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Logical-consistency judge unavailable: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Dimension 3: Actionability (LLM rubric)
# ──────────────────────────────────────────────────────────────────────────────

def compute_actionability(
    advice: TacticalAdvice,
    use_llm: bool = True,
) -> float:
    """
    Score actionability of the recommendation on a 1–5 scale.

    Uses LLM rubric if use_llm=True, otherwise uses rule-based heuristics.

    Rule-based heuristics (used when use_llm=False for cost control):
    - Mentions player ID: +1
    - Mentions a direction (left, right, channel, half-space): +1
    - Mentions a specific tactical concept: +1
    - Length > 20 words: +1
    - Contains a conditional "if": +1
    Max = 5
    """
    action = advice.recommended_action.lower()
    words = action.split()

    if use_llm:
        return _llm_actionability_score(advice)

    score = 1.0  # base

    # Specificity indicators
    import re
    if re.search(r"player \d+", action):
        score += 1
    if any(d in action for d in ["left", "right", "channel", "half-space", "half space", "forward"]):
        score += 1
    if any(c in action for c in ["overlap", "dribble", "pass", "shoot", "switch", "press"]):
        score += 1
    if len(words) >= 15:
        score += 0.5
    if "if" in action or "when" in action:
        score += 0.5

    return min(5.0, score)


def _llm_actionability_score(advice: TacticalAdvice) -> float:
    """Use the configured LLM (Gemini by default) to score actionability on a 1–5 rubric."""
    try:
        from matchmind.llm_client import call_llm

        prompt = f"""Score this tactical football recommendation on Actionability (1-5):

Recommendation: "{advice.recommended_action}"
Reasoning: "{advice.reasoning[:300]}"

Rubric:
1 = Completely vague ("play better", "attack")
2 = Somewhat specific but missing key details
3 = Specific about what but not where/when
4 = Specific about what, where, and when
5 = Fully specific: player ID, action, location, timing, and condition

Respond with ONLY a single number (1.0 to 5.0).
"""
        text, _ = call_llm(prompt, max_tokens=24, temperature=0.0, thinking_budget=0)
        # keep the first float-looking token
        import re

        m = re.search(r"[0-5](?:\.\d+)?", text)
        return float(m.group(0)) if m else compute_actionability(advice, use_llm=False)
    except Exception as exc:
        logger.warning(f"LLM actionability scoring failed: {exc} — using rule-based")
        return compute_actionability(advice, use_llm=False)


# ──────────────────────────────────────────────────────────────────────────────
# Dimension 4: Consistency
# ──────────────────────────────────────────────────────────────────────────────

def compute_consistency(
    advice: TacticalAdvice,
    counterfactual_best_action: str | None,
) -> dict:
    """
    Compute consistency:
    1. quantitative_alignment: does recommended action match pitch-control-best action?
    2. confidence_calibration: is confidence plausible given evidence?
    """
    # Quantitative alignment
    if counterfactual_best_action and advice.recommended_action:
        rec_lower = advice.recommended_action.lower()
        cf_lower = counterfactual_best_action.lower()
        # Check word overlap between recommendation and counterfactual best
        rec_words = set(rec_lower.split())
        cf_words = set(cf_lower.split())
        overlap = len(rec_words & cf_words) / max(len(cf_words), 1)
        quantitative_alignment = min(1.0, overlap * 2.0)  # scale up
    else:
        quantitative_alignment = None

    # Confidence calibration: confidence should correlate with number of alternatives
    n_alts = len(advice.alternatives)
    citation_count = len(advice.cited_concepts)
    # Heuristic: high confidence should be backed by multiple evidence sources
    evidence_backed = len(set(e.source for e in advice.evidence)) >= 2
    confidence_calibrated = (
        (advice.confidence <= 0.9 or evidence_backed)  # high confidence only when well-evidenced
        and (advice.confidence >= 0.3)  # not too low
    )

    return {
        "quantitative_alignment": round(quantitative_alignment, 4) if quantitative_alignment else None,
        "confidence_calibrated": confidence_calibrated,
        "evidence_diversity": len(set(e.source for e in advice.evidence)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class MatchMindEvaluator:
    """
    Main evaluation runner for all 4 dimensions.

    Usage:
        evaluator = MatchMindEvaluator()
        report = evaluator.run_evaluation(test_cases, agent)
        evaluator.save_report(report, "output/evaluation/report.json")
    """

    def __init__(
        self,
        use_llm_for_actionability: bool = True,
        retrieval_k: int = 3,
    ) -> None:
        self.use_llm = use_llm_for_actionability
        self.retrieval_k = retrieval_k
        self._concepts = None  # lazy — only load KB when retrieval quality is scored

    @property
    def concepts(self):
        if self._concepts is None:
            self._concepts = load_all_concepts()
        return self._concepts

    def evaluate_single(
        self,
        advice: TacticalAdvice,
        retrieved_titles: list[str],
        ground_truth_event_type: str | None = None,
        counterfactual_best_action: str | None = None,
    ) -> dict:
        """Evaluate a single TacticalAdvice across Dims 1-4 (Dim 0 handled in run_evaluation)."""
        result: dict[str, Any] = {}

        # Dim 1: Agreement
        if ground_truth_event_type:
            result["agreement"] = compute_agreement(advice, ground_truth_event_type)
        else:
            result["agreement"] = None

        # Dim 2: Groundedness (+ LLM-as-judge logical consistency when enabled)
        lc = compute_logical_consistency_llm(advice) if self.use_llm else None
        g = compute_groundedness(advice, retrieved_titles, logical_consistency=lc)
        result.update(g)

        # Dim 3: Actionability
        result["actionability_score"] = compute_actionability(advice, use_llm=self.use_llm)

        # Dim 4: Consistency
        c = compute_consistency(advice, counterfactual_best_action)
        result.update(c)

        return result

    def run_evaluation(
        self,
        test_cases: list[dict],
        agent,
        n_cases: int | None = None,
    ) -> EvaluationReport:
        """
        Run evaluation on N test cases.

        Args:
            test_cases: List of dicts from adapter.load_test_cases().
            agent: TacticalAgent instance.
            n_cases: Max cases to evaluate.

        Returns:
            EvaluationReport with all dimensions computed.
        """
        if n_cases:
            test_cases = test_cases[:n_cases]

        results = []
        latencies = []
        tokens = []

        logger.info(f"Running evaluation on {len(test_cases)} cases...")

        for i, case in enumerate(test_cases):
            try:
                t0 = time.time()
                advice, trace = agent.analyze(
                    match_state=case["snapshot"],
                    focus_player_id=case["focus_player_id"],
                    question=case.get("question"),
                )
                elapsed = (time.time() - t0) * 1000

                # Retrieved concepts come from the last 'retrieve' trace entry
                rstep = _last_retrieve_step(trace)
                retrieved_titles = rstep.get("titles", [])
                retrieved_ids = rstep.get("concept_ids", [])
                did_reretrieve = any(s.get("node") == "expand_query" for s in trace)

                # Evaluate Dims 1-4
                r = self.evaluate_single(
                    advice=advice,
                    retrieved_titles=retrieved_titles,
                    ground_truth_event_type=case.get("ground_truth_event_type"),
                    counterfactual_best_action=case.get("counterfactual_best_action"),
                )

                # Dim 0: Retrieval Quality (proxy ground truth from trigger_conditions)
                if advice.situation_features is not None and retrieved_ids:
                    try:
                        rq = evaluate_retrieval(
                            retrieved_ids=retrieved_ids,
                            concepts=self.concepts,
                            feats=advice.situation_features,
                            state=case["snapshot"],
                            k=self.retrieval_k,
                        )
                        r["precision_at_k"] = rq["precision_at_k"]
                        r["recall_at_k"] = rq["recall_at_k"]
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"Retrieval-quality scoring failed on case {i}: {exc}")

                r["retrieval_confidence"] = rstep.get("confidence")
                r["did_reretrieve"] = did_reretrieve
                r["case_id"] = i
                results.append(r)
                latencies.append(elapsed)

                # Token usage from trace
                total_tok = sum(s.get("tokens", 0) for s in trace if "tokens" in s)
                tokens.append(total_tok)

                logger.info(f"Case {i+1}/{len(test_cases)}: {r.get('agreement')=}, score={r.get('actionability_score'):.1f}")

            except Exception as exc:
                logger.warning(f"Case {i} failed: {exc}")
                results.append({"case_id": i, "error": str(exc)})

        # Aggregate
        valid = [r for r in results if "error" not in r]
        n_valid = len(valid)

        if n_valid == 0:
            logger.error("No valid evaluation cases!")
            return EvaluationReport(
                n_cases=len(test_cases),
                citation_validity_rate=0.0,
                groundedness_score=0.0,
            )

        agreement_vals = [r["agreement"] for r in valid if r.get("agreement") is not None]
        agreement_rate = sum(agreement_vals) / len(agreement_vals) if agreement_vals else None

        avg_citation = sum(r.get("citation_validity_rate", 0) for r in valid) / n_valid
        avg_groundedness = sum(r.get("groundedness_score", 0) for r in valid) / n_valid
        avg_actionability = sum(r.get("actionability_score", 3) for r in valid) / n_valid

        qa_vals = [r.get("quantitative_alignment") for r in valid if r.get("quantitative_alignment") is not None]
        avg_qa = sum(qa_vals) / len(qa_vals) if qa_vals else None

        lc_vals = [r.get("logical_consistency_rate") for r in valid if r.get("logical_consistency_rate") is not None]
        avg_lc = sum(lc_vals) / len(lc_vals) if lc_vals else None

        # Dim 0 aggregate
        rq_agg = aggregate_retrieval_quality(
            [r for r in valid if r.get("precision_at_k") is not None], k=self.retrieval_k
        )
        conf_vals = [r["retrieval_confidence"] for r in valid if r.get("retrieval_confidence") is not None]
        reretrieve_rate = (
            sum(1 for r in valid if r.get("did_reretrieve")) / n_valid if n_valid else None
        )

        lat_sorted = sorted(latencies)
        p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else None

        return EvaluationReport(
            n_cases=n_valid,
            # Dim 0
            precision_at_k=rq_agg["precision_at_k"],
            recall_at_k=rq_agg["recall_at_k"],
            retrieval_k=self.retrieval_k,
            retrieval_confidence_mean=round(sum(conf_vals) / len(conf_vals), 4) if conf_vals else None,
            reretrieve_rate=round(reretrieve_rate, 4) if reretrieve_rate is not None else None,
            # Dim 1
            agreement_rate=agreement_rate,
            # Dim 2
            citation_validity_rate=round(avg_citation, 4),
            logical_consistency_rate=round(avg_lc, 4) if avg_lc is not None else None,
            groundedness_score=round(avg_groundedness, 4),
            # Dim 3
            actionability_score=round(avg_actionability, 2),
            # Dim 4
            quantitative_alignment=round(avg_qa, 4) if avg_qa else None,
            # System performance
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else None,
            p95_latency_ms=round(p95, 1) if p95 is not None else None,
            avg_tokens_used=int(sum(tokens) / len(tokens)) if tokens else None,
            model_used=settings.llm_model,
            embedding_model=settings.embedding_model,
        )

    def run_consistency(
        self,
        test_cases: list[dict],
        agent,
        n_runs: int = 3,
        max_cases: int = 5,
    ) -> dict:
        """
        Dim 4 — Consistency: re-run the SAME snapshot ``n_runs`` times and measure
        how often the modal recommended_action recurs.

        Relies on LLM sampling non-determinism. For a stricter test, plumb
        temperature=0.7 through reasoner_node (follow-up); at temperature 0 this
        mostly measures parser stability.
        """
        import statistics
        from collections import Counter

        rates: list[float] = []
        for case in test_cases[:max_cases]:
            actions: list[str] = []
            for _ in range(n_runs):
                try:
                    advice, _ = agent.analyze(
                        match_state=case["snapshot"],
                        focus_player_id=case["focus_player_id"],
                        question=case.get("question"),
                    )
                    actions.append(advice.recommended_action.strip().lower()[:60])
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"consistency run failed: {exc}")
            if actions:
                modal_count = Counter(actions).most_common(1)[0][1]
                rates.append(modal_count / len(actions))

        return {
            "consistency_rate": round(statistics.mean(rates), 4) if rates else None,
            "n_cases": len(rates),
            "n_runs": n_runs,
        }

    def save_report(self, report: EvaluationReport, output_path: str | Path) -> None:
        """Save evaluation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)
        logger.info(f"Evaluation report saved to {output_path}")
