"""Persistent Kalshi sync state for manual catch-up runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = PROJECT_ROOT / "data" / "kalshi_sync_state.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_sync_state() -> dict[str, Any]:
    return {
        "last_market_backfill_ts": None,
        "last_candle_download_ts": None,
        "last_successful_run": None,
        "failed_market_tickers": [],
        "failed_candle_tickers": [],
    }


def load_sync_state(path: str | Path | None = None) -> dict[str, Any]:
    state_path = Path(path) if path else DEFAULT_STATE_PATH
    if not state_path.exists():
        return default_sync_state()
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_sync_state()
    state = default_sync_state()
    if isinstance(loaded, dict):
        state.update(loaded)
    return state


def save_sync_state(state: dict[str, Any], path: str | Path | None = None) -> Path:
    state_path = Path(path) if path else DEFAULT_STATE_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state_path


def mark_successful_run(
    state: dict[str, Any],
    market_backfill_ts: str | None = None,
    candle_download_ts: str | None = None,
) -> dict[str, Any]:
    now = utc_now_iso()
    updated = dict(default_sync_state())
    updated.update(state)
    updated["last_market_backfill_ts"] = market_backfill_ts or now
    updated["last_candle_download_ts"] = candle_download_ts or now
    updated["last_successful_run"] = now
    return updated
