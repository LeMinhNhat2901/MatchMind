"""
Tactical Advice Panel — displays agent output.
"""
from __future__ import annotations

import streamlit as st

from matchmind.schema.match_state import TacticalAdvice


def render_advice_panel(advice: TacticalAdvice, trace: list[dict]) -> None:
    """Render the tactical advice output panel."""
    st.markdown("## 🧠 Tactical Agent Output")

    # Confidence colour
    conf = advice.confidence
    if conf >= 0.75:
        pill_class = "pill-green"
    elif conf >= 0.5:
        pill_class = "pill-amber"
    else:
        pill_class = "pill-red"

    # Header with confidence
    header_col, conf_col = st.columns([4, 1])
    with header_col:
        st.markdown("### ✅ Recommended Action")
    with conf_col:
        st.markdown(
            f'<span class="pill {pill_class}">⭐ {conf:.0%} confidence</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="recommend-box"><b>{advice.recommended_action}</b></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Layout: reasoning + alternatives + evidence
    reason_col, cf_col = st.columns([3, 2], gap="medium")

    with reason_col:
        st.markdown("### 💭 Reasoning")
        st.markdown(advice.reasoning)

        if advice.cited_concepts:
            st.markdown("**📚 Cited Concepts:**")
            for concept in advice.cited_concepts:
                st.markdown(f"  - 📌 *{concept}*")

    with cf_col:
        st.markdown("### 🔄 Alternatives Considered")
        if advice.alternatives:
            for alt in advice.alternatives[:3]:
                delta_str = ""
                if alt.delta_pitch_control is not None:
                    sign = "+" if alt.delta_pitch_control >= 0 else ""
                    delta_str = f" | Δ PC: {sign}{alt.delta_pitch_control:.1f}%"

                with st.expander(f"❌ {alt.action[:40]}"):
                    st.markdown(f"**Why not:** {alt.why_not}{delta_str}")
        else:
            st.caption("No alternatives recorded.")

    # Evidence table
    st.divider()
    st.markdown("### 📋 Evidence Used")

    rag_ev = [e for e in advice.evidence if e.source == "rag"]
    stats_ev = [e for e in advice.evidence if e.source == "stats"]
    ctx_ev = [e for e in advice.evidence if e.source == "context"]

    ev_col1, ev_col2, ev_col3 = st.columns(3)

    with ev_col1:
        st.markdown("**📚 RAG (Tactical KB)**")
        for e in rag_ev:
            st.markdown(f"- {e.id}")

    with ev_col2:
        st.markdown("**📊 Stats (Computed)**")
        for e in stats_ev:
            st.caption(e.content[:120])

    with ev_col3:
        st.markdown("**🌍 Context (Match)**")
        for e in ctx_ev:
            st.caption(e.content[:120])

    # Pipeline trace
    if trace:
        with st.expander("🔍 Agent Pipeline Trace"):
            total_ms = sum(s.get("latency_ms", 0) for s in trace)
            total_tok = sum(s.get("tokens", 0) for s in trace)
            st.caption(f"Total: {total_ms:.0f}ms | {total_tok} tokens")

            import pandas as pd
            trace_df = pd.DataFrame([
                {
                    "Node": s.get("node", ""),
                    "Latency (ms)": s.get("latency_ms", 0),
                    "Notes": str({k: v for k, v in s.items() if k not in ["node", "latency_ms"]})[:80],
                }
                for s in trace
            ])
            st.dataframe(trace_df, use_container_width=True, hide_index=True)
