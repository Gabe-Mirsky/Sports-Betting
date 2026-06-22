"""Generic, sport-agnostic event planner (research-only).

Pure time-window logic shared by the NBA and World Cup watchers. Given a list of
events (each with a start time) and a ``now``, it labels every event with a
status and a recommended action. It performs NO I/O and NO API calls, so it is
trivially testable and cannot, by itself, spend quota or change any model gate.

Recommended actions:
    EARLY_SNAPSHOT    - match is 24-48h out: optionally grab an opening line
    CLOSING_SNAPSHOT  - match tips within the closing window (default 60 min)
    POSTGAME_RESULTS  - match ended recently: fetch results/status
    SKIP              - nothing is due for this event right now
    ERROR             - the event has no usable start time

This module enables no betting, parlays, predictions, or recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

ACTION_EARLY = "EARLY_SNAPSHOT"
ACTION_CLOSING = "CLOSING_SNAPSHOT"
ACTION_POSTGAME = "POSTGAME_RESULTS"
ACTION_SKIP = "SKIP"
ACTION_ERROR = "ERROR"

STATUS_SCHEDULED = "scheduled"        # future, nothing due yet
STATUS_EARLY_WINDOW = "early_window"  # 24-48h out
STATUS_STARTING_SOON = "starting_soon"  # within closing window
STATUS_IN_PROGRESS = "in_progress"    # kicked off, not yet results-checkable
STATUS_ENDED = "ended"                # finished / results-checkable
STATUS_UNKNOWN = "unknown"            # no usable start time


@dataclass(frozen=True)
class EventWindows:
    """Time-window thresholds (minutes/hours). Defaults suit soccer + NBA."""

    closing_window_minutes: float = 60.0
    pregame_grace_minutes: float = 10.0      # still fire CLOSING just after kickoff
    early_min_hours: float = 24.0
    early_max_hours: float = 48.0
    postgame_after_minutes: float = 150.0    # treat as ended once this long past start
    postgame_lookback_minutes: float = 420.0  # stop checking results after this

    @classmethod
    def from_config(cls, defaults: dict[str, Any] | None) -> "EventWindows":
        d = defaults or {}
        return cls(
            closing_window_minutes=float(d.get("closing_window_minutes", 60.0)),
            pregame_grace_minutes=float(d.get("pregame_grace_minutes", 10.0)),
            early_min_hours=float(d.get("early_snapshot_min_hours", 24.0)),
            early_max_hours=float(d.get("early_snapshot_max_hours", 48.0)),
            postgame_after_minutes=float(d.get("postgame_after_minutes", 150.0)),
            postgame_lookback_minutes=float(d.get("postgame_lookback_minutes", 420.0)),
        )


def _parse_start(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts.to_pydatetime()


def plan_event(
    event: dict[str, Any],
    now: datetime,
    windows: EventWindows,
    *,
    allow_early: bool = True,
) -> dict[str, Any]:
    """Label one event with status + recommended action (pure)."""

    start = _parse_start(event.get("event_start_time") or event.get("commence_time"))
    base = {
        "event_id": str(event.get("event_id") or event.get("id") or ""),
        "league": event.get("league"),
        "sport_key": event.get("sport_key"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "event_start_time": start.isoformat() if start else event.get("event_start_time"),
        "minutes_until_event": None,
        "event_status": STATUS_UNKNOWN,
        "recommended_action": ACTION_ERROR,
        "reason": "",
    }

    if start is None:
        base["reason"] = "missing or unparseable event_start_time"
        return base

    mins = round((start - now).total_seconds() / 60.0, 2)
    base["minutes_until_event"] = mins

    # An explicit "completed" flag from a results source wins immediately.
    if bool(event.get("completed")):
        base["event_status"] = STATUS_ENDED
        base["recommended_action"] = ACTION_POSTGAME
        base["reason"] = "source marked event completed"
        return base

    closing_hi = windows.closing_window_minutes
    closing_lo = -windows.pregame_grace_minutes
    early_lo = windows.early_min_hours * 60.0
    early_hi = windows.early_max_hours * 60.0
    post_after = -windows.postgame_after_minutes
    post_lookback = -windows.postgame_lookback_minutes

    if closing_lo <= mins <= closing_hi:
        base["event_status"] = STATUS_STARTING_SOON
        base["recommended_action"] = ACTION_CLOSING
        base["reason"] = f"within closing window ({closing_hi:.0f} min, {windows.pregame_grace_minutes:.0f} grace)"
    elif post_lookback <= mins <= post_after:
        base["event_status"] = STATUS_ENDED
        base["recommended_action"] = ACTION_POSTGAME
        base["reason"] = "ended recently; results checkable"
    elif early_lo <= mins <= early_hi:
        base["event_status"] = STATUS_EARLY_WINDOW
        base["recommended_action"] = ACTION_EARLY if allow_early else ACTION_SKIP
        base["reason"] = "in early-snapshot window (24-48h)" + ("" if allow_early else "; early disabled")
    elif closing_hi < mins < early_lo:
        base["event_status"] = STATUS_SCHEDULED
        base["recommended_action"] = ACTION_SKIP
        base["reason"] = "between early and closing windows; nothing due"
    elif post_after < mins < closing_lo:
        base["event_status"] = STATUS_IN_PROGRESS
        base["recommended_action"] = ACTION_SKIP
        base["reason"] = "match in progress; wait for results window"
    elif mins > early_hi:
        base["event_status"] = STATUS_SCHEDULED
        base["recommended_action"] = ACTION_SKIP
        base["reason"] = "too far in the future"
    else:  # mins < post_lookback
        base["event_status"] = STATUS_ENDED
        base["recommended_action"] = ACTION_SKIP
        base["reason"] = "ended too long ago; results window closed"

    return base


def plan_events(
    events: Iterable[dict[str, Any]],
    now: datetime | None = None,
    windows: EventWindows | None = None,
    *,
    allow_early: bool = True,
) -> list[dict[str, Any]]:
    """Label a list of events. ``now`` defaults to UTC now."""

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    windows = windows or EventWindows()
    return [plan_event(e, now, windows, allow_early=allow_early) for e in events]


def actions_due(planned: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group planned events by the action they request (excludes SKIP)."""

    out: dict[str, list[dict[str, Any]]] = {
        ACTION_EARLY: [], ACTION_CLOSING: [], ACTION_POSTGAME: [], ACTION_ERROR: [],
    }
    for p in planned:
        action = p.get("recommended_action")
        if action in out:
            out[action].append(p)
    return out
