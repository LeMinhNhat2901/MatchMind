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
