"""Where the ball '+' is drawn relative to its carrier.

User report (2026-09-29): the ball looked far from the player on the ball. The
data was right (distance 0.0 m) — the renderer pushed the glyph out to just
touch the dot plus a 2 pt gap, which after the bigger player dots of the UI
redesign read as "ball away from the player". The glyph may overlap the dot;
what matters is that contact is visible.
"""
from __future__ import annotations

import math

from matchmind.schema.match_state import PlayerState
from matchmind.visualization.snapshot_renderer import ball_glyph_position

R = 1.2   # player-dot radius in metres at the rendered scale


def test_ball_on_its_carrier_overlaps_the_dot_rim():
    carrier = PlayerState(id=10, team="home", x=50.0, y=30.0)
    x, y = ball_glyph_position((50.0, 30.0), [carrier], R)
    d = math.hypot(x - 50.0, y - 30.0)
    assert 0 < d < R            # on the dot's edge, overlapping it (visible contact)


def test_loose_ball_is_drawn_exactly_where_it_is():
    player = PlayerState(id=10, team="home", x=50.0, y=30.0)
    assert ball_glyph_position((60.0, 30.0), [player], R) == (60.0, 30.0)


def test_glyph_stays_on_the_pitch():
    carrier = PlayerState(id=10, team="home", x=104.9, y=67.9)
    x, y = ball_glyph_position((104.9, 67.9), [carrier], R)
    assert 0 <= x <= 105 and 0 <= y <= 68


def test_no_players_no_crash():
    assert ball_glyph_position((40.0, 20.0), [], R) == (40.0, 20.0)
