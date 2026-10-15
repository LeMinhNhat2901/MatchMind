"""What happened after an on-ball event — deterministic labels from the event stream.

Within `window_s` seconds and the same possession:
  shot        — the acting team takes a shot
  progressed  — the acting team's ball location advances >= progress_m metres
  lost        — possession passes to the other team
  retained    — anything else (kept the ball, no clear progress)
"""
from __future__ import annotations

SB_X_TO_M = 105.0 / 120.0


def _t(event: dict) -> float:
    return float(event.get("minute", 0)) * 60.0 + float(event.get("second", 0))


def label_outcome(events: list[dict], idx: int, window_s: float = 15.0, progress_m: float = 15.0) -> str:
    start = events[idx]
    team_id = (start.get("team") or {}).get("id")
    possession = start.get("possession")
    t0, x0 = _t(start), float(start["location"][0])
    progressed = False
    for event in events[idx + 1:]:
        if _t(event) - t0 > window_s:
            break
        if event.get("possession") != possession:
            owner = (event.get("possession_team") or {}).get("id")
            return "lost" if owner != team_id else ("progressed" if progressed else "retained")
        same_team = (event.get("team") or {}).get("id") == team_id
        if same_team and event["type"]["name"] == "Shot":
            return "shot"
        loc = event.get("location")
        if same_team and loc and (float(loc[0]) - x0) * SB_X_TO_M >= progress_m:
            progressed = True
    return "progressed" if progressed else "retained"
