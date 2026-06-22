"""World Cup (FIFA) game-market odds collection + normalization (research-only).

Separate from the player-prop pipeline on purpose: the World Cup value is in
GAME/TEAM markets (1X2 / match winner, totals), which do not fit the player-prop
schema. This module:

  * lists events via the FREE ``/sports/{key}/events`` endpoint (0 credits),
  * pulls game-market odds via the cheap bulk ``/sports/{key}/odds`` endpoint
    (cost = n_markets x n_regions, independent of match count),
  * normalizes outcomes into ``world_cup_odds_snapshots_normalized.csv``.

A strict quota guard runs BEFORE any credit-spending request, and a hard
per-run event cap and credit cap make accidental quota burn impossible. It
enables no models, recommendations, bets, parlays, or predictions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from data.prop_collection import make_default_fetch_json, _odds_api_url  # reuse quota-tracking fetch

# Normalized game-market schema (Phase 6). Player fields are appended ONLY when
# player props are actually observed (never for 1X2 / totals).
WORLD_CUP_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "source", "league", "sport_key", "event_id", "event_start_time", "snapshot_time",
    "home_team", "away_team", "market_type", "bookmaker", "outcome_name", "price", "line",
    "is_pregame", "is_closing_like",
)
PLAYER_EXTRA_COLUMNS: tuple[str, ...] = ("player_name", "player_team", "prop_type")


def load_world_cup_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.setdefault("defaults", {})
    data.setdefault("quota", {})
    data.setdefault("markets", {})
    data.setdefault("output", {})
    return data


def _iso(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    return "" if ts is pd.NaT or pd.isna(ts) else ts.isoformat()


def events_from_payload(payload: Any, sport_key: str) -> list[dict[str, Any]]:
    """Map an Odds API /events (or /odds) payload to planner-shaped event dicts."""
    out: list[dict[str, Any]] = []
    if not isinstance(payload, list):
        return out
    for ev in payload:
        if not isinstance(ev, dict):
            continue
        out.append({
            "event_id": str(ev.get("id") or ""),
            "league": "WORLD_CUP",
            "sport_key": ev.get("sport_key") or sport_key,
            "home_team": ev.get("home_team"),
            "away_team": ev.get("away_team"),
            "event_start_time": _iso(ev.get("commence_time")),
        })
    return out


def fetch_world_cup_events(
    api_key: str,
    config: dict[str, Any],
    fetch_json: Callable[[str], Any],
) -> list[dict[str, Any]]:
    """List World Cup events. Uses the FREE /events endpoint (0 credits)."""
    src = config.get("source", {})
    base_url = src.get("base_url", "https://api.the-odds-api.com/v4")
    sport_key = _sport_key(config)
    url = _odds_api_url(base_url, f"/sports/{sport_key}/events", {"apiKey": api_key})
    payload = fetch_json(url)
    return events_from_payload(payload, sport_key)


def estimate_credit_cost(config: dict[str, Any]) -> int:
    markets = list(config.get("markets", {}).keys())
    regions = [r for r in str(config.get("source", {}).get("regions", "us")).split(",") if r]
    return max(1, len(markets)) * max(1, len(regions))


def _markets_within_budget(config: dict[str, Any]) -> list[str]:
    """Configured Odds API market keys, trimmed so cost <= max_credits_per_run."""
    markets = list(config.get("markets", {}).keys())
    regions = [r for r in str(config.get("source", {}).get("regions", "us")).split(",") if r] or ["us"]
    cap = int(config.get("quota", {}).get("max_credits_per_run", 2) or 2)
    max_markets = max(1, cap // max(1, len(regions)))
    return markets[:max_markets]


def fetch_world_cup_odds(
    api_key: str,
    config: dict[str, Any],
    fetch_json: Callable[[str], Any],
) -> Any:
    """Bulk game-market odds for the sport (credit-spending; call only after guard)."""
    src = config.get("source", {})
    base_url = src.get("base_url", "https://api.the-odds-api.com/v4")
    sport_key = _sport_key(config)
    params = {
        "apiKey": api_key,
        "regions": src.get("regions", "us"),
        "oddsFormat": src.get("odds_format", "decimal"),
        "markets": ",".join(_markets_within_budget(config)),
    }
    url = _odds_api_url(base_url, f"/sports/{sport_key}/odds", params)
    return fetch_json(url)


def _sport_key(config: dict[str, Any]) -> str:
    leagues = config.get("leagues", {})
    wc = leagues.get("WORLD_CUP", {}) if isinstance(leagues, dict) else {}
    return wc.get("sport_key") or "soccer_fifa_world_cup"


def normalize_world_cup_odds(
    payload: Any,
    config: dict[str, Any],
    *,
    run_time: datetime,
    raw_source_file: str = "",
    max_events: int | None = None,
) -> pd.DataFrame:
    """Normalize a bulk /odds payload into the World Cup game-market schema."""
    market_map = config.get("markets", {})
    closing_minutes = float(config.get("defaults", {}).get("closing_window_minutes", 60.0))
    rows: list[dict[str, Any]] = []
    if not isinstance(payload, list):
        return pd.DataFrame(columns=list(WORLD_CUP_SNAPSHOT_COLUMNS))

    events = payload if max_events is None else payload[:max_events]
    for ev in events:
        if not isinstance(ev, dict):
            continue
        event_id = str(ev.get("id") or "")
        start_iso = _iso(ev.get("commence_time"))
        start_ts = pd.to_datetime(start_iso, errors="coerce", utc=True)
        is_closing = False
        is_pregame = True  # unknown start -> assume pregame (collection runs pre-kick)
        if start_ts is not pd.NaT and not pd.isna(start_ts):
            mins_to_kick = (start_ts - pd.Timestamp(run_time)).total_seconds() / 60.0
            is_closing = -10.0 <= mins_to_kick <= closing_minutes
            is_pregame = mins_to_kick > 0.0
        for bk in ev.get("bookmakers", []) or []:
            book = bk.get("key")
            for mk in bk.get("markets", []) or []:
                src_key = mk.get("key")
                market_type = market_map.get(src_key, src_key)
                for oc in mk.get("outcomes", []) or []:
                    rows.append({
                        "source": "odds_api",
                        "league": "WORLD_CUP",
                        "sport_key": ev.get("sport_key") or _sport_key(config),
                        "event_id": event_id,
                        "event_start_time": start_iso,
                        "snapshot_time": pd.Timestamp(run_time).isoformat(),
                        "home_team": ev.get("home_team"),
                        "away_team": ev.get("away_team"),
                        "market_type": market_type,
                        "bookmaker": book,
                        "outcome_name": oc.get("name"),
                        "price": oc.get("price"),
                        "line": oc.get("point"),
                        "is_pregame": bool(is_pregame),
                        "is_closing_like": bool(is_closing),
                    })
    frame = pd.DataFrame(rows, columns=list(WORLD_CUP_SNAPSHOT_COLUMNS))
    return frame


def append_snapshots(existing: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Append new rows, dropping exact duplicates on the natural snapshot key."""
    if new.empty:
        return existing, 0
    combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new.copy()
    key_cols = ["snapshot_time", "event_id", "market_type", "bookmaker", "outcome_name", "line"]
    before = len(combined)
    combined = combined.drop_duplicates(subset=[c for c in key_cols if c in combined.columns], keep="first")
    return combined.reset_index(drop=True), before - len(combined)


def collect_world_cup_kalshi_markets(
    project_root: str | Path,
    *,
    max_pages: int = 1,
    limit: int = 200,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Bounded, SEPARATE Kalshi prediction-market check for World Cup contracts.

    Kalshi data is a different category (yes/no contract prices) and is written to
    its own file — it is NEVER merged into the sportsbook odds CSV. Best-effort and
    bounded (single page by default); public reads, no key required.
    """
    root = Path(project_root)
    now = now or datetime.now(timezone.utc)
    out = {
        "report": "world_cup_kalshi_markets", "generated_at_utc": now.isoformat(),
        "research_only": True, "approved": False, "category": "prediction_market",
        "markets_scanned": 0, "world_cup_markets_found": 0, "samples": [], "error": None,
        "note": "Kalshi yes/no contract prices; separate from sportsbook odds (never merged).",
    }
    try:
        from data.kalshi_client import KalshiAPIClient
        client = KalshiAPIClient.from_env() if hasattr(KalshiAPIClient, "from_env") else KalshiAPIClient()
        markets = client.get_markets({"max_pages": int(max_pages), "limit": int(limit), "status": "open"})
        out["markets_scanned"] = int(len(markets))
        if not markets.empty:
            text_cols = [c for c in ("title", "subtitle", "ticker", "event_ticker", "series_ticker", "yes_sub_title")
                         if c in markets.columns]
            mask = None
            for c in text_cols:
                m = markets[c].astype(str).str.contains("world cup|fifa|wcup|world-cup", case=False, na=False)
                mask = m if mask is None else (mask | m)
            wc = markets[mask] if mask is not None else markets.iloc[0:0]
            out["world_cup_markets_found"] = int(len(wc))
            keep = [c for c in ("ticker", "title", "yes_bid", "yes_ask", "last_price", "close_time") if c in wc.columns]
            out["samples"] = wc[keep].head(20).to_dict(orient="records") if keep else []
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    (root / "data" / "reports" / "world_cup_kalshi_markets.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run_world_cup_collection(
    config: dict[str, Any],
    project_root: str | Path,
    *,
    dry_run: bool = False,
    max_events: int | None = None,
    env: dict[str, str] | None = None,
    fetch_json: Callable[[str], Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one safe World Cup collection: list events (free), guard, then pull odds.

    dry_run=True performs ONLY the free /events listing (or nothing if no fetcher)
    and never spends credits. Returns a summary dict (also the watcher uses this).
    """
    root = Path(project_root)
    env = os.environ if env is None else env
    now = now or datetime.now(timezone.utc)
    out_cfg = config.get("output", {})
    quota_cfg = config.get("quota", {})
    src_cfg = config.get("source", {})
    min_remaining = float(quota_cfg.get("min_remaining_requests", 100) or 0)
    cap_events = max_events if max_events is not None else int(config.get("defaults", {}).get("max_events_per_run", 1))

    api_key_env = src_cfg.get("api_key_env", "ODDS_API_KEY")
    api_key = src_cfg.get("api_key") or env.get(api_key_env, "")

    summary: dict[str, Any] = {
        "report": "world_cup_collection_summary",
        "generated_at_utc": now.isoformat(),
        "research_only": True,
        "approved": False,
        "dry_run": bool(dry_run),
        "source": "odds_api",
        "sport_key": _sport_key(config),
        "key_detected": bool(api_key),
        "status": "unknown",
        "events_found": 0,
        "events_processed": 0,
        "markets_found": [],
        "bookmakers_found": [],
        "player_props_found": False,
        "game_markets_found": False,
        "rows_normalized": 0,
        "snapshots_added": 0,
        "duplicates_removed": 0,
        "estimated_credit_cost": estimate_credit_cost(config),
        "credits_remaining_before": None,
        "credits_remaining_after": None,
        "raw_file": None,
        "errors": [],
        "blockers": [],
    }

    if not src_cfg.get("enabled", True):
        summary["status"] = "disabled"
        return summary
    if not api_key:
        summary["status"] = "no_key"
        summary["blockers"].append(f"{api_key_env} not set in environment")
        return summary

    if fetch_json is None:
        fetch_json = make_default_fetch_json()

    # --- Step 1: list events (FREE, 0 credits) -----------------------------
    try:
        events = fetch_world_cup_events(api_key, config, fetch_json)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "events_error"
        summary["errors"].append(f"events: {exc!r}")
        return summary
    summary["events_found"] = len(events)
    summary["credits_remaining_before"] = getattr(fetch_json, "quota_remaining", None)
    summary["events"] = events[:10]

    if not events:
        summary["status"] = "no_events"
        summary["blockers"].append("No World Cup events returned (off-window or season inactive).")
        return summary

    if dry_run:
        summary["status"] = "dry_run_ok"
        return summary

    # --- Quota guard BEFORE any credit-spending request --------------------
    remaining = getattr(fetch_json, "quota_remaining", None)
    if remaining is not None and remaining < min_remaining:
        summary["status"] = "skipped_quota"
        summary["blockers"].append(
            f"Odds API credits remaining {remaining} < World Cup floor {min_remaining}; "
            "skipping odds pull to protect NBA closing capture."
        )
        return summary

    # --- Step 2: pull game-market odds (capped) ----------------------------
    try:
        payload = fetch_world_cup_odds(api_key, config, fetch_json)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "odds_error"
        summary["errors"].append(f"odds: {exc!r}")
        summary["credits_remaining_after"] = getattr(fetch_json, "quota_remaining", None)
        return summary
    summary["credits_remaining_after"] = getattr(fetch_json, "quota_remaining", None)

    # Save raw (immutable) before normalizing.
    raw_dir = root / out_cfg.get("raw_dir", "data/raw/world_cup")
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(now).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"{stamp}__odds.json"
    raw_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    summary["raw_file"] = str(raw_path.relative_to(root)).replace("\\", "/")

    frame = normalize_world_cup_odds(
        payload, config, run_time=now, raw_source_file=summary["raw_file"], max_events=cap_events,
    )
    summary["rows_normalized"] = int(len(frame))
    summary["events_processed"] = int(frame["event_id"].nunique()) if not frame.empty else 0
    summary["markets_found"] = sorted(frame["market_type"].dropna().unique().tolist()) if not frame.empty else []
    summary["bookmakers_found"] = sorted(frame["bookmaker"].dropna().unique().tolist()) if not frame.empty else []
    summary["game_markets_found"] = bool(summary["markets_found"])
    summary["player_props_found"] = bool(set(frame.columns) & set(PLAYER_EXTRA_COLUMNS)) if not frame.empty else False

    # Append to the normalized store.
    processed_path = root / out_cfg.get("processed_path", "data/processed/world_cup_odds_snapshots_normalized.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        pd.read_csv(processed_path, low_memory=False) if processed_path.exists() else pd.DataFrame()
    )
    combined, dups = append_snapshots(existing, frame)
    combined.to_csv(processed_path, index=False)
    summary["snapshots_added"] = int(len(frame) - 0)
    summary["duplicates_removed"] = int(dups)
    summary["processed_path"] = str(processed_path.relative_to(root)).replace("\\", "/")
    summary["status"] = "collected" if summary["rows_normalized"] else "no_rows"

    # Run summary + history.
    reports = root / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if out_cfg.get("run_summary_path"):
        (root / out_cfg["run_summary_path"]).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if out_cfg.get("run_history_path"):
        with (root / out_cfg["run_history_path"]).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "run_id": stamp, "run_time_utc": now.isoformat(), "status": summary["status"],
                "events_found": summary["events_found"], "rows": summary["rows_normalized"],
                "credits_after": summary["credits_remaining_after"],
            }, default=str) + "\n")
    return summary
