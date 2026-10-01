"""The pitch is drawn broadcast-style: y = 0 (the attacking team's LEFT touchline
in StatsBomb/Metrica data — left backs average y≈11 of 80, right backs y≈68) at
the TOP. It used to be at the bottom, so a team attacking right had its left
flank drawn where its right flank should be, contradicting "overload on the
left flank" in the text (user report 2026-09-30)."""
from __future__ import annotations

import math

from matchmind.schema.match_state import PlayerState
from matchmind.visualization.pitch_renderer import plot_pitch
from matchmind.visualization.snapshot_renderer import ball_glyph_position


def test_y_zero_is_drawn_at_the_top():
    fig, ax = plot_pitch()
    bottom, top = ax.get_ylim()
    assert bottom > top          # inverted axis: y grows downward on screen


def test_ball_glyph_still_sits_up_right_of_the_carrier_on_screen():
    carrier = PlayerState(id=5, team="home", x=50.0, y=30.0)
    gx, gy = ball_glyph_position((50.0, 30.0), [carrier], player_radius_m=2.0)
    assert gx > 50.0 and gy < 30.0   # smaller y = higher on the inverted axis
    assert math.hypot(gx - 50.0, gy - 30.0) < 2.0


def test_projection_trails_are_drawn(tmp_path):
    """Estimated moves show a faded dot at the old spot + a dashed trail."""
    from matchmind.schema.match_state import BallState, MatchState
    from matchmind.situation_engine.team_shift import Move
    from matchmind.visualization.snapshot_renderer import render_snapshot

    state = MatchState(match_id="t", timestamp=0.0, source="synthetic", ball=BallState(x=60.0, y=30.0),
                       players=[PlayerState(id=5, team="home", x=60.0, y=30.0),
                                PlayerState(id=4, team="away", x=65.0, y=32.0)])
    plain = render_snapshot(state, 5, str(tmp_path / "a.png"), annotate=False)
    trails = render_snapshot(state, 5, str(tmp_path / "b.png"), annotate=False,
                             trails=[Move("away", 4, (75.0, 40.0), (65.0, 32.0))])
    assert open(plain, "rb").read() != open(trails, "rb").read()
