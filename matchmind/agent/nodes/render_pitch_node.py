"""
Render Pitch Node.

Renders the match snapshot as a PNG image and stores the path in state.
The image is passed to the VLM alongside the text prompt.
"""
from __future__ import annotations

import logging
import time

# NOTE: matplotlib is intentionally NOT imported here. On Windows + Python 3.13,
# importing matplotlib before sentence-transformers / torch finish initialising
# hard-crashes the process (0xC0000005, no traceback). snapshot_renderer imports
# it lazily (Agg backend) only when a pitch is actually drawn — by which point
# TacticalAgent.__init__ has already pre-warmed the retrieval model.
from matchmind.agent.state import AgentState
from matchmind.visualization.snapshot_renderer import render_snapshot

logger = logging.getLogger(__name__)


def render_pitch_node(state: AgentState) -> AgentState:
    """
    Node: Render pitch snapshot → PNG image.

    Input:  state.match_state, state.focus_player_id
    Output: state.pitch_image_path
    """
    t0 = time.time()

    from matchmind.config import settings
    import uuid

    output_dir = settings.pitch_images_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{state['match_state'].match_id}_{state['focus_player_id']}_{uuid.uuid4().hex[:8]}.png"
    save_path = str(output_dir / fname)

    render_snapshot(
        state=state["match_state"],
        focus_player_id=state["focus_player_id"],
        save_path=save_path,
    )

    elapsed = (time.time() - t0) * 1000
    logger.debug(f"[render_pitch] {elapsed:.1f}ms | saved to {save_path}")

    trace = state.get("trace") or []
    trace.append({"node": "render_pitch", "latency_ms": round(elapsed, 1), "image": save_path})

    return {**state, "pitch_image_path": save_path, "trace": trace}
