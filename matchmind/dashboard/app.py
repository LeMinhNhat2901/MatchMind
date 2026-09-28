"""
MatchMind Streamlit Dashboard — Pro Football Tactical Intelligence Suite.

Run with:
    streamlit run matchmind/dashboard/app.py

A high-performance tactical intelligence console for football analysts,
coaches, and scouts. Inspired by StatsBomb 360, Opta Pro, SofaScore, and Wyscout.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="MatchMind — Tactical Intelligence",
    page_icon=":material/sports_soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from matchmind.dashboard.components.pitch_view import render_pitch_panel
from matchmind.dashboard.components.situation_panel import render_situation_panel
from matchmind.dashboard.components.tactical_advice import render_advice_panel
from matchmind.dashboard.components.ui import icon, load_theme, render_html
from matchmind.agent.translation import (
    detect_and_translate_to_english,
    localize_advice,
    mentioned_player_mismatch,
)


@st.cache_resource(show_spinner="Loading tactical agent...")
def get_agent():
    """One TacticalAgent per server process — chat sends far more requests than the
    old single-button flow, so re-instantiating per message (which re-warms the
    retrieval store) would make every turn noticeably slower for no benefit."""
    from matchmind.agent.graph import TacticalAgent

    return TacticalAgent()

load_theme()


# ── Load Match State ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_match_state(source: str, mid: int | None, eidx: int | None):
    if source == "StatsBomb Open Data":
        try:
            from matchmind.data_sources.statsbomb_adapter import StatsBombAdapter

            adapter = StatsBombAdapter()
            return adapter.load_snapshot(match_id=mid, event_index=eidx)
        except Exception as e:
            st.warning(f"StatsBomb load failed: {e}. Using synthetic demo data.")

    from matchmind.tests.fixtures.sample_data import make_sample_match_state

    state, _ = make_sample_match_state()
    return state


def _queue_prompt(widget_key: str, prompts: dict[str, str]) -> None:
    """Pill callback: queue the chosen prompt for the chat, then clear the pill
    so the same suggestion can be picked again later."""
    choice = st.session_state.get(widget_key)
    if choice:
        st.session_state["pending_prompt"] = prompts[choice]
    st.session_state[widget_key] = None


def _prompt_pills(widget_key: str, prompts: dict[str, str], icons: dict[str, str]) -> None:
    st.pills(
        "Gợi ý câu hỏi",
        options=list(prompts),
        format_func=lambda o: f"{icons[o]} {o}",
        key=widget_key,
        on_change=_queue_prompt,
        args=(widget_key, prompts),
        label_visibility="collapsed",
    )


def _player_label(p) -> str:
    role = f" · {p.role.replace('_', ' ').title()}" if p.role else ""
    return f"#{p.id}{role}"


# ── Sidebar: Tactical Workstation Dock ─────────────────────────────────────────
with st.sidebar:
    render_html(
        """
        <div class="sidebar-brand">
            <span class="brand-logo-badge">MM</span>
            <div class="brand-title-wrap">
                <span class="brand-name">MatchMind</span>
                <span class="brand-sub">Tactical Intelligence</span>
            </div>
        </div>
        """
    )
    render_html(f'<div class="sidebar-section-title">{icon("radio", 16)}Match feed</div>')

    data_source = st.radio(
        "Data Feed",
        options=["Synthetic (Demo)", "StatsBomb Open Data"],
        index=0,
        help="'Synthetic' runs immediately without external API downloads.",
    )

    if data_source == "StatsBomb Open Data":
        match_id = st.number_input("Match ID", value=3788741, step=1)
        event_idx = st.slider("Event Snapshot #", 0, 500, 150, step=10)
    else:
        match_id = None
        event_idx = None

    state = load_match_state(data_source, match_id, event_idx)

    st.markdown("---")
    render_html(f'<div class="sidebar-section-title">{icon("target", 16)}Analysis target</div>')

    dual_mode = st.checkbox(
        "Dual Mode (Attack vs Defense)",
        value=False,
        help="Evaluates attacking ball carrier options alongside defensive counter-measures in parallel.",
    )

    focus_id: int | None = None
    focus_team: str | None = None

    if not dual_mode:
        team_choice = st.radio(
            "Squad Scope",
            options=["Home Team", "Away Team"],
            horizontal=True,
        )
        focus_team = "home" if team_choice == "Home Team" else "away"
        team_players = sorted(
            [p for p in state.players if p.team == focus_team], key=lambda p: p.id
        )

        if team_players:
            selected_player = st.selectbox(
                "Focus Player Dossier",
                options=team_players,
                format_func=_player_label,
                help="Select the player to evaluate tactical options for.",
            )
            focus_id = selected_player.id
        else:
            st.warning(f"No {team_choice} players detected in this snapshot.")

        with st.expander("Manual Jersey ID"):
            manual_id = st.text_input(
                "Jersey ID",
                value="",
                placeholder=f"e.g. {team_players[0].id if team_players else 10}",
            )
            if manual_id.strip():
                try:
                    focus_id = int(manual_id.strip())
                except ValueError:
                    st.error("Player ID must be a number.")

    st.markdown("---")
    render_html(f'<div class="sidebar-section-title">{icon("layers", 16)}Map layers</div>')
    show_pitch_control = st.checkbox(
        "Pitch control surface",
        value=False,
        help="Overlay the Spearman (2018) probabilistic pitch-control surface for the focus team.",
    )

    st.markdown("---")
    if dual_mode:
        run_agent = st.button(
            "Run dual analysis", use_container_width=True, type="primary", icon=":material/bolt:"
        )
        st.caption("Analyses the ball carrier's options and the defending response in one pass.")
    else:
        run_agent = False
        st.caption("Ask about the selected player in the **Directive** tab — any language works.")


# ── Chat Context Reset ──────────────────────────────────────────────────────────
# A conversation is scoped to one (team, player) target — reset it when that
# changes, BEFORE the pitch panel renders below, so a stale arrow/advice from
# the previous player never flashes on screen after switching the roster pick.
if not dual_mode:
    chat_context_key = (focus_team, focus_id)
    if st.session_state.get("chat_context_key") != chat_context_key:
        st.session_state["chat_context_key"] = chat_context_key
        st.session_state["chat_messages"] = []
        st.session_state.pop("last_advice", None)
        st.session_state.pop("last_trace", None)


# ── Top Broadcast Scoreboard Banner ────────────────────────────────────────────
score_h = state.score_home
score_a = state.score_away
minute_str = f"{state.minute}'" if state.minute else "00'"
phase_str = (state.phase_of_play or "open_play").replace("_", " ").upper()
possession_team = (state.possession or "home").upper()

render_html(
    f"""
    <div class="broadcast-banner" role="banner">
        <div class="broadcast-brand">
            <span class="brand-logo-badge">MM</span>
            <div class="brand-title-wrap">
                <span class="brand-name">MatchMind</span>
                <span class="brand-sub">Tactical Intelligence Suite</span>
            </div>
        </div>
        <div class="scorebug" aria-label="Score: Home {score_h}, Away {score_a}, minute {minute_str}">
            <span class="scorebug-team home">Home</span>
            <span class="scorebug-score">{score_h}<span class="sep">–</span>{score_a}</span>
            <span class="scorebug-team away">Away</span>
            <span class="scorebug-clock"><span class="live-dot"></span>{minute_str}</span>
        </div>
        <div class="match-status-wrap">
            <span class="tactical-phase-pill">{icon("activity", 13)}{phase_str}</span>
            <span class="tactical-badge badge-blue">{icon("ball", 13)}Possession · {possession_team}</span>
        </div>
    </div>
    """
)


# ── Agent Execution Logic (Dual Mode only — single-player mode runs from the
#    chat input inside the TACTICAL DIRECTIVE tab instead) ─────────────────────
if run_agent and dual_mode:
    with st.spinner("Running tactical engine: situation → playbook retrieval → pitch control → validation..."):
        try:
            agent = get_agent()
            attacking, defending = agent.analyze_dual(state)
            st.session_state["last_dual"] = (attacking[0], defending[0])
            st.session_state["last_dual_trace"] = (attacking[1], defending[1])
            st.session_state.pop("last_advice", None)
            st.success("Tactical directive generated successfully.")
        except Exception as exc:
            st.error(f"Tactical Agent failed: {exc}")
            st.info("Ensure GROQ_API_KEY or GOOGLE_API_KEY is configured in .env and the vector KB is built.")
            st.stop()
    st.rerun()


# ── Main Coordinated Workstation Layout (Split-Screen Studio) ──────────────────
col_pitch, col_console = st.columns([57, 43], gap="large")

shown_advice = None if dual_mode else st.session_state.get("last_advice")
shown_dual = st.session_state.get("last_dual") if dual_mode else None
preview_focus_id = (
    focus_id if focus_id is not None
    else (shown_advice.focus_player_id if shown_advice is not None else None)
)

# ── LEFT COLUMN: Pitch Canvas & Match Timeline ────────────────────────────────
with col_pitch:
    render_pitch_panel(
        state,
        preview_focus_id,
        show_pitch_control=show_pitch_control,
        advice=shown_advice,
        dual_advice=shown_dual,
    )

    carrier_text = f"#{state.event_player_id}" if state.event_player_id else "Contested"
    render_html(
        f"""
        <div class="scrubber-strip">
            <span><span class="k">Snapshot</span><span class="v">{state.match_id}</span></span>
            <span><span class="k">Phase</span><span class="v">{phase_str.title()}</span></span>
            <span><span class="k">Ball carrier</span><span class="v">{carrier_text}</span></span>
            <span><span class="k">Players</span><span class="v">{len(state.players)}</span></span>
        </div>
        """
    )


# ── RIGHT COLUMN: Tactical Decision & Performance Console ──────────────────────
with col_console:
    if dual_mode:
        tab_att, tab_def, tab_scout = st.tabs([
            ":material/sports_soccer: Attack",
            ":material/shield: Defense",
            ":material/person: Dossier",
        ])

        dual = st.session_state.get("last_dual")
        if dual is not None:
            att_advice, def_advice = dual
            att_trace, def_trace = st.session_state.get("last_dual_trace", ([], []))
            with tab_att:
                render_advice_panel(att_advice, att_trace)
            with tab_def:
                render_advice_panel(def_advice, def_trace)
        else:
            with tab_att:
                render_html(
                    f"""
                    <div class="candidate-pre-box">
                        {icon("swords", 28)}
                        <div class="t">Dual analysis ready</div>
                        <div>Press <b>Run dual analysis</b> in the sidebar to evaluate the ball carrier's
                        options and the defending team's best response together.</div>
                    </div>
                    """
                )
            with tab_def:
                st.caption("The defending side's directive appears here after the dual analysis runs.")

        with tab_scout:
            # Dual mode has no manual player pick — show the auto-picked ball carrier.
            scout_id = dual[0].focus_player_id if dual is not None else None
            if scout_id is not None:
                render_situation_panel(state, scout_id)
            else:
                st.info("Run the dual analysis to see the ball carrier's dossier.")

    else:
        # Single Player Mode: Integrated 3-Tab Console
        tab_directive, tab_dossier, tab_evidence = st.tabs([
            ":material/forum: Directive",
            ":material/person: Dossier",
            ":material/menu_book: Method",
        ])

        with tab_directive:
            if focus_id is None:
                st.info("Select a player from the squad list in the sidebar.")
            else:
                focus = state.get_player(focus_id)
                role_label = (focus.role or "Player").replace("_", " ").title() if focus else "Player"
                chat_messages = st.session_state.get("chat_messages", [])

                # 1. Copilot header — clear button only once there's history to clear
                col_head, col_clear = st.columns([6, 1], vertical_alignment="center")
                with col_head:
                    render_html(
                        f"""
                        <div class="copilot-header-bar">
                            <div class="copilot-title-wrap">
                                {icon("message", 16)}<span>Tactical Copilot</span>
                            </div>
                            <span class="copilot-target-badge">#{focus_id} · {role_label}</span>
                        </div>
                        """
                    )
                with col_clear:
                    # Icon-only (the column is too narrow for icon + text); the
                    # material ligature ("delete") doubles as the accessible name.
                    if chat_messages and st.button(
                        ":material/delete:", key="clear_chat_btn", help="Xóa lịch sử hội thoại",
                        use_container_width=True,
                    ):
                        st.session_state["chat_messages"] = []
                        st.session_state.pop("last_advice", None)
                        st.session_state.pop("last_trace", None)
                        st.rerun()

                # 2. Welcome card + quick prompts while the chat is empty
                if not chat_messages:
                    render_html(
                        f"""
                        <div class="copilot-welcome-card">
                            <div class="welcome-title">Trợ lý chiến thuật trực tiếp</div>
                            <div class="welcome-desc">
                                Đặt câu hỏi về <b>cầu thủ #{focus_id} ({role_label})</b> bằng tiếng Việt hoặc bất kỳ ngôn ngữ nào.
                                Hệ thống tự dịch, tra cẩm nang chiến thuật, mô phỏng kiểm soát sân và đưa ra chỉ đạo tối ưu.
                            </div>
                            <div class="suggested-label">Gợi ý câu hỏi nhanh</div>
                        </div>
                        """
                    )

                    _prompt_pills(
                        "starter_pills",
                        {
                            "Tối ưu nhất lúc này?": f"Tình huống này cầu thủ #{focus_id} nên làm gì là tối ưu nhất?",
                            "Nên chuyền cho ai?": f"Cầu thủ #{focus_id} nên chuyền bóng cho ai để đạt kiểm soát sân cao nhất?",
                            "Thoát pressing đối thủ?": f"Làm thế nào để cầu thủ #{focus_id} thoát khỏi sự áp sát của đối thủ gần nhất?",
                            "Khai thác nách trung lộ?": f"Cầu thủ #{focus_id} có phương án khai thác khoảng trống nách trung lộ (half-space) hoặc sau lưng hàng thủ không?",
                        },
                        {
                            "Tối ưu nhất lúc này?": ":material/bolt:",
                            "Nên chuyền cho ai?": ":material/call_split:",
                            "Thoát pressing đối thủ?": ":material/shield:",
                            "Khai thác nách trung lộ?": ":material/north_east:",
                        },
                    )
                else:
                    with st.expander("Gợi ý câu hỏi tiếp theo", icon=":material/lightbulb:"):
                        _prompt_pills(
                            "followup_pills",
                            {
                                "So sánh các hướng chuyền?": "So sánh các phương án chuyền bóng khả thi và rủi ro bị cắt bóng?",
                                "Di chuyển không bóng thế nào?": "Nếu không cầm bóng, cầu thủ này nên chạy chỗ vào vị trí nào để kéo giãn đội hình?",
                            },
                            {
                                "So sánh các hướng chuyền?": ":material/compare_arrows:",
                                "Di chuyển không bóng thế nào?": ":material/directions_run:",
                            },
                        )

                # 3. Message stream
                for msg in chat_messages:
                    avatar = ":material/person:" if msg["role"] == "user" else ":material/sports_soccer:"
                    with st.chat_message(msg["role"], avatar=avatar):
                        if msg.get("mismatch_warning"):
                            st.warning(
                                f"Bạn hỏi về cầu thủ #{msg['mismatch_warning']}, nhưng câu trả "
                                f"lời dưới đây là cho **cầu thủ #{focus_id}** — người đang được chọn "
                                f"ở sidebar. Muốn hỏi về cầu thủ #{msg['mismatch_warning']}, hãy chọn "
                                f"đúng người đó ở sidebar trước rồi hỏi lại.",
                                icon=":material/warning:",
                            )
                        st.markdown(msg["content"])
                        if msg.get("rationale"):
                            with st.expander("Vì sao lại là quyết định này?", icon=":material/query_stats:"):
                                st.markdown("\n".join(f"- {b}" for b in msg["rationale"]))
                        if msg.get("detected_language") and msg["detected_language"] != "English":
                            render_html(
                                f'<div class="translation-pill">{icon("globe", 12)}Tự động dịch từ {msg["detected_language"]}</div>'
                            )

                # 4. Handle Input (from chat input or quick prompt buttons)
                user_input = st.chat_input(f"Hỏi về cầu thủ #{focus_id} (Tiếng Việt hoặc bất kỳ ngôn ngữ nào)...")
                if not user_input and "pending_prompt" in st.session_state:
                    user_input = st.session_state.pop("pending_prompt")

                if user_input:
                    chat_messages.append({"role": "user", "content": user_input})
                    with st.spinner("Đang dịch & phân tích chiến thuật..."):
                        try:
                            detected = detect_and_translate_to_english(user_input)
                            mismatch = mentioned_player_mismatch(detected.english_text, focus_id)
                            agent = get_agent()
                            advice, trace = agent.analyze(
                                state, focus_id, focus_team=focus_team,
                                question=detected.english_text, language=detected.language,
                            )
                            localized = localize_advice(advice, detected.language)
                            chat_messages.append({
                                "role": "assistant",
                                "content": localized.answer,
                                "rationale": localized.rationale,
                                "detected_language": detected.language,
                                "mismatch_warning": mismatch,
                            })
                            st.session_state["last_advice"] = advice
                            st.session_state["last_trace"] = trace
                            st.session_state["last_chat_language"] = detected.language
                        except Exception as exc:
                            chat_messages.append({
                                "role": "assistant",
                                "content": f"**Quá trình phân tích gặp lỗi:** {exc}",
                            })
                    st.session_state["chat_messages"] = chat_messages
                    st.rerun()

                # 5. Synchronized Structured Directive & Decision Matrix
                if "last_advice" in st.session_state:
                    render_html('<div class="mm-kicker">Chi tiết chỉ đạo mới nhất</div>')
                    render_advice_panel(
                        st.session_state["last_advice"],
                        st.session_state.get("last_trace", []),
                    )

                    # 6. Optional "what happens next?" — projects the recommendation
                    # forward one step (receiver gets the ball / the run completes
                    # and pulls a marker) and asks the agent again. Opt-in only —
                    # this is a second full LLM pass, so it never runs on its own.
                    current_advice = st.session_state["last_advice"]
                    followup_action = current_advice.resolve_action(current_advice.recommended_action_id)
                    if followup_action is not None and followup_action.target_xy is not None and followup_action.action != "shoot":
                        if st.button(
                            "Giả lập: chuyện gì xảy ra tiếp theo?",
                            key=f"btn_followup_{len(chat_messages)}",
                            use_container_width=True,
                            icon=":material/fast_forward:",
                        ):
                            with st.spinner("Đang giả lập tình huống tiếp theo..."):
                                try:
                                    agent = get_agent()
                                    lang = st.session_state.get("last_chat_language", "English")
                                    result = agent.analyze_followup(state, current_advice, focus_team, language=lang)
                                    if result is None:
                                        st.info("Hành động này không có bước tiếp theo phù hợp để giả lập.")
                                    else:
                                        followup_advice, followup_trace = result
                                        localized = localize_advice(followup_advice, lang)
                                        chat_messages.append({
                                            "role": "assistant",
                                            "content": f"**Kịch bản giả định — bước tiếp theo:**\n\n{localized.answer}",
                                            "rationale": localized.rationale,
                                            "detected_language": lang,
                                        })
                                        st.session_state["chat_messages"] = chat_messages
                                        st.session_state["last_advice"] = followup_advice
                                        st.session_state["last_trace"] = followup_trace
                                except Exception as exc:
                                    st.error(f"Giả lập thất bại: {exc}")
                                    st.stop()
                            st.rerun()

        with tab_dossier:
            if focus_id is not None:
                render_situation_panel(state, focus_id)
            else:
                st.info("Select a player from the squad list in the sidebar.")

        with tab_evidence:
            from matchmind.config import settings

            kb_count = sum(1 for _ in Path(settings.knowledge_base_dir).rglob("*.json"))
            render_html('<div class="mm-kicker">How MatchMind decides</div>')
            render_html(
                f"""
                <div class="briefing-box">
                    <ul class="briefing-list">
                        <li class="briefing-li"><b>Situation engine:</b> distances, pressure, local numerical balance, passing lanes and team shape computed from player coordinates on a 105 × 68 m pitch.</li>
                        <li class="briefing-li"><b>Pitch control:</b> Spearman (2018) probabilistic model scores each candidate action by the change in controlled space.</li>
                        <li class="briefing-li"><b>Tactical playbook:</b> {kb_count} concepts in a ChromaDB vector store (hand-written, Wikipedia, FIFA Training Centre and YouTube analysis, each with its source attached).</li>
                        <li class="briefing-li"><b>Validation:</b> recommendations must map to a real candidate action and may only cite concepts that were actually retrieved.</li>
                    </ul>
                </div>
                """
            )


render_html(
    """
    <div class="mm-footer">
        <span>MatchMind · Multi-agent tactical graph · Spearman pitch control</span>
        <span>StatsBomb 360 · Metrica tracking · Synthetic demo</span>
    </div>
    """
)
