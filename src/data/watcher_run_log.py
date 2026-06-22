"""Shared, path-parameterized dedup run-log for event watchers (research-only).

Mirrors the proven NBA-watcher dedup rules but takes the log path as an argument
so multiple watchers (e.g. World Cup) can reuse it WITHOUT importing or modifying
the NBA watcher. An action for an item is "done" after one SUCCESS; failures are
retried up to ``max_attempts`` so a transient API error does not permanently skip
an event. ``dry_run`` rows never count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_run_log(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return rows


def append_run_log(path: str | Path, entry: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def action_state(history: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    """Count successes/attempts per (item_id, action). dry_run rows ignored."""
    state: dict[tuple[str, str], dict[str, int]] = {}
    for row in history:
        key = (str(row.get("event_id", row.get("game_id"))), str(row.get("action")))
        bucket = state.setdefault(key, {"success": 0, "attempts": 0})
        status = str(row.get("status"))
        if status in ("success", "failed"):
            bucket["attempts"] += 1
        if status == "success":
            bucket["success"] += 1
    return state


def needs_action(state: dict, item_id: str, action: str, max_attempts: int) -> bool:
    bucket = state.get((item_id, action))
    if not bucket:
        return True
    if bucket["success"] > 0:
        return False  # done once -> never repeat
    return bucket["attempts"] < max_attempts  # retry failures up to the cap
