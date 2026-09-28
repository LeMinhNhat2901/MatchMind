"""
Tactical Advice Panel Component — Chief Tactical Analyst Report & Decision Matrix.

Presents MatchMind tactical intelligence as a professional coaching decision matrix
(inspired by StatsBomb IQ, Opta Pro, and Wyscout), replacing AI-style prose with
structured scouting cards, concise tactical takeaways, and pitch control deltas.
"""
from __future__ import annotations

import html

import streamlit as st

from matchmind.dashboard.components.ui import icon, render_html
from matchmind.schema.match_state import TacticalAdvice


def _delta_chip(delta: float | None) -> str:
    """Pitch-control delta chip. Unknown stays unknown ("—"): never show a
    placeholder number, since users read these as computed model output."""
    if delta is None:
        return '<span class="delta-chip delta-neutral" title="No pitch-control delta computed">—</span>'
    cls = "delta-pos" if delta >= 0 else "delta-neg"
    return f'<span class="delta-chip {cls}">{delta:+.1f}% PC</span>'


def render_advice_panel(advice: TacticalAdvice, trace: list[dict] | None = None) -> None:
    """Render the structured tactical analyst report and decision matrix."""
    conf = advice.confidence
    if conf >= 0.75:
        tone, conf_label, directive_border = "badge-green", "High confidence", "directive-border-green"
    elif conf >= 0.50:
        tone, conf_label, directive_border = "badge-amber", "Moderate confidence", "directive-border-amber"
    else:
        tone, conf_label, directive_border = "badge-red", "Low confidence", "directive-border-red"
    conf_pill = f'<span class="tactical-badge {tone}"><span class="status-dot"></span>{conf:.0%} · {conf_label}</span>'

    target_player = f"Player #{advice.focus_player_id}" if advice.focus_player_id is not None else "Tactical focus"
    action_code = html.escape(advice.recommended_action_id or "unmatched")
    rec_action_escaped = html.escape(advice.recommended_action)

    render_html(
        f"""
        <div class="directive-card {directive_border}">
            <div class="directive-card-top">
                <span class="directive-primary-badge">Primary directive · {target_player}</span>
                {conf_pill}
            </div>
            <div class="directive-main-text">{rec_action_escaped}</div>
            <div class="directive-card-bottom">
                <span class="action-id-pill">action: {action_code}</span>
            </div>
        </div>
        """
    )

    render_html('<div class="mm-kicker">Decision matrix</div>')

    # The recommended action's own delta, when the LLM also listed it among the
    # alternatives (it usually doesn't — then it's genuinely unknown here).
    rec_delta = next(
        (a.delta_pitch_control for a in advice.alternatives
         if a.action_id and a.action_id == advice.recommended_action_id),
        None,
    )
    rec_cand = advice.resolve_action(advice.recommended_action_id)
    if rec_cand is not None:
        kind = "Ball action" if rec_cand.moves_ball else "Off-ball run"
        rec_detail = (
            f'<div class="decision-risk-text"><span class="decision-risk-label">{kind}:</span> '
            f"{html.escape(rec_cand.label)}</div>"
        )
    else:
        rec_detail = ""

    cards = [
        f"""
        <div class="decision-card decision-card-preferred">
            <div class="decision-card-top">
                <span class="decision-status-pill status-pref-pill">{icon("check", 14)}Preferred</span>
                {_delta_chip(rec_delta)}
            </div>
            <div class="decision-action-title">{rec_action_escaped}</div>
            {rec_detail}
        </div>
        """
    ]
    for alt in advice.alternatives:
        if alt.action_id and alt.action_id == advice.recommended_action_id:
            continue
        cards.append(
            f"""
            <div class="decision-card decision-card-alt">
                <div class="decision-card-top">
                    <span class="decision-status-pill status-alt-pill">{icon("x", 14)}Rejected</span>
                    {_delta_chip(alt.delta_pitch_control)}
                </div>
                <div class="decision-action-title">{html.escape(alt.action)}</div>
                <div class="decision-risk-text">
                    <span class="decision-risk-label">Why not:</span> {html.escape(alt.why_not)}
                </div>
            </div>
            """
        )
    render_html(f'<div class="decision-deck">{"".join(cards)}</div>')

    render_html('<div class="mm-kicker">Coaching briefing</div>')

    # advice.narrative (when present) is a second LLM pass that paraphrases the
    # structured reasoning below into one analyst-style paragraph, constrained to
    # restate only facts already in reasoning/alternatives/cited_concepts — never
    # new ones. Prefer it: naive sentence-splitting of the raw structured
    # `reasoning` field (the old fallback) reads like a spec sheet, not analysis.
    if advice.narrative:
        briefing_body = f'<div class="briefing-text">{html.escape(advice.narrative)}</div>'
    else:
        raw_reasoning = advice.reasoning
        sentences = [s.strip() for s in raw_reasoning.replace("\n", ". ").split(". ") if len(s.strip()) > 8]
        if len(sentences) >= 2:
            bullet_items = "".join(f'<li class="briefing-li">{html.escape(s)}.</li>' for s in sentences[:4])
            briefing_body = f'<ul class="briefing-list">{bullet_items}</ul>'
        else:
            briefing_body = f'<div class="briefing-text">{html.escape(raw_reasoning)}</div>'

    if advice.narrative and advice.rationale:
        rationale_items = "".join(f'<li class="briefing-li">{html.escape(b)}</li>' for b in advice.rationale)
        briefing_body += f'<ul class="briefing-list">{rationale_items}</ul>'

    render_html(f'<div class="briefing-box">{briefing_body}</div>')

    if advice.narrative:
        with st.expander("Structured reasoning (pre-narration)", icon=":material/psychology:"):
            st.markdown(advice.reasoning)

    if advice.cited_concepts:
        render_html('<div class="mm-kicker">Playbook references</div>')
        chips = "".join(
            f'<span class="concept-chip">{icon("book", 13)}{html.escape(c)}</span>' for c in advice.cited_concepts
        )
        render_html(f'<div class="concept-chips-wrap">{chips}</div>')

    with st.expander("Evidence sources", icon=":material/fact_check:"):
        ev_items = "".join(
            f'<div class="ev-row"><span class="ev-tag">{html.escape(e.source.upper())}</span>'
            f"{html.escape(e.id)} — {html.escape(e.content[:140])}</div>"
            for e in advice.evidence[:6]
        )
        render_html(f'<div class="ev-container">{ev_items}</div>')
