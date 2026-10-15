"""
Jersey inference for anonymous StatsBomb 360 freeze-frame players.

360 frames only identify the event's actor. Every other dot is anonymous — but
the same match's event stream says exactly who touched the ball where and when
(Pass, Ball Receipt, Carry, Pressure, Duel, ...). A dot is matched to a player by
predicting where each on-pitch player should be at the frame's timestamp
(interpolating between their touches, blended with their average position in
that half) and solving a per-team Hungarian assignment.

Shown only when it is reliable. Spike on 12 matches / ~3000 frames
(leave-one-out on the actor, chance ~14%):
    player touched the ball within 3 s          88% correct, 93% with the margin gate
    no touch for 10 s+ (typical off-ball runner) ~55% correct — NOT shown
So a dot gets an identity only when that player has a touch within
TOUCH_WINDOW_S of the frame AND the assignment beats the runner-up by
MIN_MARGIN; everyone else keeps a placeholder id. V2/V3 (tracking) will know
every identity per frame, so this is a V1-only bridge.

Coordinates are raw StatsBomb units (120 x 80). Each event is stored in its
own team's attacking frame, so another team's touches are rotated 180°.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

SB_LENGTH, SB_WIDTH = 120.0, 80.0
TOUCH_WINDOW_S = 3.0
MIN_MARGIN = 2.0
_UNKNOWN_COST = 8.0            # a dot may stay unassigned (e.g. player not in the candidate list)
_KEEPER_MISMATCH = 1e6
_MIN_PRIOR_VAR = 25.0          # m² floor for the "average position in this half" prior
_END = (9, 1e9)


@dataclass(frozen=True)
class EventRecord:
    """The few fields of a StatsBomb event this module needs."""
    id: str
    type: str
    team: str
    player_id: int | None
    location: tuple[float, float] | None
    period: int
    t: float                                   # seconds within the period (from `timestamp`)
    replacement_id: int | None = None          # Substitution: who comes on
    lineup: list[tuple[int, str]] | None = None  # Starting XI: (player_id, position name)

    @classmethod
    def from_open_data(cls, e: dict) -> "EventRecord":
        """A raw open-data JSON event."""
        lineup = None
        if (e.get("tactics") or {}).get("lineup"):
            lineup = [(int(p["player"]["id"]), (p.get("position") or {}).get("name", ""))
                      for p in e["tactics"]["lineup"]]
        loc = e.get("location")
        return cls(
            id=e["id"], type=e["type"]["name"], team=(e.get("team") or {}).get("name", ""),
            player_id=int(e["player"]["id"]) if e.get("player") else None,
            location=(float(loc[0]), float(loc[1])) if isinstance(loc, (list, tuple)) and len(loc) >= 2 else None,
            period=int(e.get("period", 1)), t=parse_timestamp(e.get("timestamp")),
            replacement_id=(((e.get("substitution") or {}).get("replacement") or {}).get("id")),
            lineup=lineup,
        )

    @classmethod
    def from_statsbombpy(cls, row: Any) -> "EventRecord":
        """A row of statsbombpy's flattened events DataFrame."""
        def val(key):
            v = row.get(key)
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

        tactics = val("tactics")
        lineup = None
        if isinstance(tactics, dict) and tactics.get("lineup"):
            lineup = [(int(p["player"]["id"]), (p.get("position") or {}).get("name", ""))
                      for p in tactics["lineup"]]
        loc = val("location")
        pid, rep = val("player_id"), val("substitution_replacement_id")
        return cls(
            id=str(row["id"]), type=str(val("type") or ""), team=str(val("team") or ""),
            player_id=int(pid) if pid is not None else None,
            location=(float(loc[0]), float(loc[1])) if isinstance(loc, (list, tuple)) and len(loc) >= 2 else None,
            period=int(val("period") or 1), t=parse_timestamp(val("timestamp")),
            replacement_id=int(rep) if rep is not None else None, lineup=lineup,
        )


@dataclass(frozen=True)
class FrameDot:
    teammate: bool
    actor: bool
    keeper: bool
    x: float
    y: float


def parse_timestamp(ts: Any) -> float:
    if not isinstance(ts, str) or ts.count(":") != 2:
        return 0.0
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


@dataclass
class _Touch:
    period: int
    t: float
    x: float
    y: float
    event_id: str


@dataclass
class _Stint:
    player_id: int
    start: tuple[int, float]
    end: tuple[int, float] = _END


class JerseyInference:
    """Built once per match from its events; `identify` is then cheap per frame."""

    def __init__(self, events: Iterable[EventRecord]) -> None:
        events = list(events)
        self._by_id = {e.id: e for e in events}
        self._touches: dict[int, list[_Touch]] = {}
        self._keepers: set[int] = set()
        self._stints: dict[str, list[_Stint]] = {}
        open_stint: dict[int, _Stint] = {}

        for e in sorted(events, key=lambda e: (e.period, e.t)):
            if e.type == "Starting XI" and e.lineup:
                for pid, pos in e.lineup:
                    stint = _Stint(pid, (1, 0.0))
                    self._stints.setdefault(e.team, []).append(stint)
                    open_stint[pid] = stint
                    if pos == "Goalkeeper":
                        self._keepers.add(pid)
            elif e.type == "Substitution" and e.player_id is not None and e.replacement_id is not None:
                if e.player_id in open_stint:
                    open_stint.pop(e.player_id).end = (e.period, e.t)
                stint = _Stint(e.replacement_id, (e.period, e.t))
                self._stints.setdefault(e.team, []).append(stint)
                open_stint[e.replacement_id] = stint
                if e.player_id in self._keepers:
                    self._keepers.add(e.replacement_id)
            if e.player_id is not None and e.location is not None:
                self._touches.setdefault(e.player_id, []).append(
                    _Touch(e.period, e.t, e.location[0], e.location[1], e.id))
        self._team_of = {s.player_id: team for team, stints in self._stints.items() for s in stints}

    # ── public ────────────────────────────────────────────────────────────────

    def identify(self, event_id: str, dots: list[FrameDot], actor_id: int | None = None) -> dict[int, int]:
        """{index into `dots`: StatsBomb player_id} for the dots identified with
        confidence. The actor's dot is never included (the caller knows it)."""
        event = self._by_id.get(event_id)
        if event is None:
            return {}
        out: dict[int, int] = {}
        for teammate in (True, False):
            idx = [i for i, d in enumerate(dots) if d.teammate == teammate and not d.actor]
            team = event.team if teammate else next((t for t in self._stints if t != event.team), None)
            if idx and team:
                flip = not teammate
                out.update(self._assign(event, [dots[i] for i in idx], idx, team, flip, actor_id))
        return out

    # ── internals ─────────────────────────────────────────────────────────────

    def _on_pitch(self, team: str, period: int, t: float) -> list[int]:
        return [s.player_id for s in self._stints.get(team, []) if s.start <= (period, t) <= s.end]

    def _predict(self, pid: int, period: int, t: float, skip_event: str, flip: bool):
        pts = [p for p in self._touches.get(pid, []) if p.period == period and p.event_id != skip_event]
        if not pts:
            return None
        xy = np.array([[p.x, p.y] for p in pts])
        if flip:
            xy = np.array([SB_LENGTH, SB_WIDTH]) - xy
        prior_mu, prior_var = xy.mean(0), max(float(xy.var(0).mean()), _MIN_PRIOR_VAR)
        times = [p.t for p in pts]
        i = bisect.bisect_left(times, t)
        if 0 < i < len(pts):
            span = times[i] - times[i - 1]
            w = 0.5 if span <= 0 else (t - times[i - 1]) / span
            mu = xy[i - 1] * (1 - w) + xy[i] * w
            dt = min(t - times[i - 1], times[i] - t)
        else:
            j = i - 1 if i == len(pts) else i
            mu, dt = xy[j], abs(t - times[j])
        var_touch = (2.0 + dt) ** 2
        prec = 1 / var_touch + 1 / prior_var
        return (mu / var_touch + prior_mu / prior_var) / prec, 1 / prec, dt

    def _assign(self, event: EventRecord, dots: list[FrameDot], idx: list[int],
                team: str, flip: bool, actor_id: int | None) -> dict[int, int]:
        preds = {}
        for pid in self._on_pitch(team, event.period, event.t):
            if pid == actor_id:
                continue
            p = self._predict(pid, event.period, event.t, event.id, flip)
            if p is not None:
                preds[pid] = p
        if not preds:
            return {}
        pids = list(preds)
        # Extra "unknown" columns so every dot can stay unassigned.
        cost = np.full((len(dots), len(pids) + len(dots)), _UNKNOWN_COST)
        for r, d in enumerate(dots):
            p = np.array([d.x, d.y])
            for c, pid in enumerate(pids):
                mu, var, _ = preds[pid]
                cost[r, c] = ((p - mu) ** 2).sum() / (2 * var) + np.log(var)
                if d.keeper != (pid in self._keepers):
                    cost[r, c] += _KEEPER_MISMATCH
        rows, cols = linear_sum_assignment(cost)
        out = {}
        for r, c in zip(rows, cols):
            if c >= len(pids):
                continue
            pid = pids[c]
            real = np.sort(cost[r, : len(pids)])
            margin = real[1] - real[0] if len(real) > 1 else float("inf")
            if preds[pid][2] <= TOUCH_WINDOW_S and margin >= MIN_MARGIN and cost[r, c] == real[0]:
                out[idx[r]] = pid
        return out
