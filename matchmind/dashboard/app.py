"""
MatchMind Streamlit Dashboard — Main Application.

Run with: streamlit run matchmind/dashboard/app.py

Features:
- Load StatsBomb or synthetic match snapshots
- Visualise pitch with player positions + pitch control heatmap
- Display situation features computed by the Situation Engine
- Run the tactical agent and display structured advice
- Show counterfactual analysis with pitch control deltas
- Display evaluation metrics (run from evaluation output)
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

# Page config MUST be first Streamlit call
st.set_page_config(
    page_title="MatchMind — Football Tactical Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

from matchmind.dashboard.components.pitch_view import render_pitch_panel
from matchmind.dashboard.components.situation_panel import render_situation_panel
from matchmind.dashboard.components.tactical_advice import render_advice_panel

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark premium theme */
    .stApp { background: linear-gradient(135deg, #0a0f1e 0%, #1a2035 100%); }
    .main .block-container { padding: 1.5rem 2rem; }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        backdrop-filter: blur(10px);
    }

    /* Headers */
    h1, h2, h3 { color: #E8F0FE !important; font-family: 'Inter', sans-serif; }
    h1 { background: linear-gradient(135deg, #4F90E8, #A855F7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    /* Sidebar */
    .css-1d391kg { background: rgba(15,20,40,0.95); }

    /* Success/warning pills */
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 100px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .pill-green { background: rgba(74,222,128,0.15); color: #4ADE80; border: 1px solid rgba(74,222,128,0.3); }
    .pill-blue  { background: rgba(96,165,250,0.15); color: #60A5FA; border: 1px solid rgba(96,165,250,0.3); }
    .pill-amber { background: rgba(251,191,36,0.15);  color: #FBD740; border: 1px solid rgba(251,191,36,0.3); }
    .pill-red   { background: rgba(248,113,113,0.15); color: #F87171; border: 1px solid rgba(248,113,113,0.3); }

    /* Recommend action box */
    .recommend-box {
        background: linear-gradient(135deg, rgba(79,144,232,0.15), rgba(168,85,247,0.15));
        border: 1px solid rgba(79,144,232,0.4);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
    }

    /* Confidence bar */
    .confidence-label { color: #94A3B8; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# ⚽ MatchMind")
st.markdown("**Real-Time Multi-Agent Football Tactical Intelligence**")
st.divider()

# ── Load Match State ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_match_state(source: str, mid: int | None, eidx: int | None):
    if source == "StatsBomb Open Data":
        try:
            from matchmind.data_sources.statsbomb_adapter import StatsBombAdapter
            adapter = StatsBombAdapter()
            return adapter.load_snapshot(match_id=mid, event_index=eidx)
        except Exception as e:
            st.warning(f"StatsBomb load failed: {e}. Using synthetic data.")

    from matchmind.tests.fixtures.sample_data import make_sample_match_state
    state, _ = make_sample_match_state()
    return state


def _player_label(p) -> str:
    role = f" · {p.role.replace('_', ' ').title()}" if p.role else ""
    return f"#{p.id}{role}"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎮 Match Setup")

    data_source = st.radio(
        "Data Source",
        options=["Synthetic (Demo)", "StatsBomb Open Data"],
        index=0,
        help="'Synthetic' works without API keys. 'StatsBomb' requires statsbombpy.",
    )

    if data_source == "StatsBomb Open Data":
        match_id = st.number_input("Match ID", value=3788741, step=1)
        event_idx = st.slider("Event Index", 0, 500, 150, step=10)
    else:
        match_id = None
        event_idx = None

    # Loaded here (not after the sidebar) so the player selectors below can
    # list this snapshot's actual players instead of a blind numeric spinner.
    state = load_match_state(data_source, match_id, event_idx)

    st.divider()
    st.markdown("### 🎯 Analysis Target")

    dual_mode = st.checkbox(
        "Analyse both teams (attack vs defense)",
        value=False,
        help=(
            "Auto-picks the ball carrier for the attacking side and their "
            "nearest marker for the defending side, and asks a tactical "
            "question for each — e.g. 'home has the ball, what should home "
            "do?' alongside 'what should away do to stop it?'. Runs the "
            "agent twice (2x LLM calls)."
        ),
    )

    focus_id: int | None = None
    focus_team: str | None = None

    if not dual_mode:
        team_choice = st.radio(
            "Team",
            options=["Home", "Away"],
            horizontal=True,
            help=(
                "Pick the team first — two different teams can share the same "
                "shirt number (e.g. both have a #10), so the player list below "
                "is scoped to this team to avoid picking the wrong one."
            ),
        )
        focus_team = "home" if team_choice == "Home" else "away"
        team_players = sorted(
            [p for p in state.players if p.team == focus_team], key=lambda p: p.id
        )

        if team_players:
            selected_player = st.selectbox(
                "Focus Player",
                options=team_players,
                format_func=_player_label,
                help="The player you want tactical advice for.",
            )
            focus_id = selected_player.id
        else:
            st.warning(f"No {team_choice} players in this snapshot.")

        with st.expander("Type a player ID instead"):
            manual_id = st.text_input(
                "Player ID",
                value="",
                placeholder=f"e.g. {team_players[0].id if team_players else 10}",
                help="Overrides the dropdown above if you know the exact id.",
            )
            if manual_id.strip():
                try:
                    focus_id = int(manual_id.strip())
                except ValueError:
                    st.error("Player ID must be a whole number.")

    show_pitch_control = st.checkbox(
        "Show Pitch Control Heatmap",
        value=False,
        help="Overlay pitch control surface (computationally expensive)",
    )

    st.divider()
    run_agent = st.button("🧠 Run Tactical Agent", use_container_width=True, type="primary")
    st.caption("Requires your LLM provider's API key in .env (see LLM_PROVIDER)")


# ── Main Layout — Two Columns ──────────────────────────────────────────────────
col_pitch, col_info = st.columns([3, 2], gap="medium")

# The pitch panel shows the LAST completed analysis (arrow on the minimap),
# independent of whatever the sidebar is currently set to, until re-run.
shown_advice = None if dual_mode else st.session_state.get("last_advice")
shown_dual = st.session_state.get("last_dual") if dual_mode else None
preview_focus_id = focus_id if focus_id is not None else (
    shown_advice.focus_player_id if shown_advice is not None else None
)

with col_pitch:
    render_pitch_panel(
        state,
        preview_focus_id,
        show_pitch_control=show_pitch_control,
        advice=shown_advice,
        dual_advice=shown_dual,
    )

with col_info:
    if not dual_mode and focus_id is not None:
        render_situation_panel(state, focus_id)
    else:
        st.info("Situation panel is shown in single-player mode — pick a team + player on the left.")

st.divider()

# ── Agent Results ──────────────────────────────────────────────────────────────
if run_agent:
    if not dual_mode and focus_id is None:
        st.error("Pick a team and a focus player first.")
        st.stop()

    with st.spinner("🧠 Running MatchMind Agent Pipeline..."):
        try:
            from matchmind.agent.graph import TacticalAgent
            agent = TacticalAgent()
            if dual_mode:
                attacking, defending = agent.analyze_dual(state)
                st.session_state["last_dual"] = (attacking[0], defending[0])
                st.session_state["last_dual_trace"] = (attacking[1], defending[1])
                st.session_state.pop("last_advice", None)
            else:
                advice, trace = agent.analyze(state, focus_id, focus_team=focus_team)
                st.session_state["last_advice"] = advice
                st.session_state["last_trace"] = trace
                st.session_state.pop("last_dual", None)
            st.success("✅ Agent completed analysis!")
        except Exception as exc:
            st.error(f"Agent failed: {exc}")
            st.info("Make sure your LLM provider's API key is set in .env and the knowledge base is built.")
            st.stop()
    st.rerun()  # redraw the pitch panel with the fresh advice/arrow

if dual_mode and "last_dual" in st.session_state:
    att_advice, def_advice = st.session_state["last_dual"]
    att_trace, def_trace = st.session_state.get("last_dual_trace", ([], []))
    st.markdown("## ⚔️ Attack vs Defense")
    tab_att, tab_def = st.tabs(["🟡 Attacking team", "🟣 Defending team"])
    with tab_att:
        render_advice_panel(att_advice, att_trace)
    with tab_def:
        render_advice_panel(def_advice, def_trace)
elif not dual_mode and "last_advice" in st.session_state:
    render_advice_panel(
        st.session_state["last_advice"],
        st.session_state.get("last_trace", []),
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "MatchMind | Built with LangGraph + ChromaDB + Spearman Pitch Control | "
    "Data: StatsBomb Open Data"
)
