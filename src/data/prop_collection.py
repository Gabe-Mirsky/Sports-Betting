"""Multi-sport daily player-prop odds collection (research-only).

Loads ``config/prop_collection.yaml``, pulls current player-prop odds for every
enabled league from the enabled sources (The Odds API today; other sources are
listed in the config but skipped gracefully until implemented), saves the raw
responses with timestamped filenames, normalizes them into the player-prop
snapshot schema (``src/data/player_prop_schema.py``), and appends them to the
historical normalized CSV with exact-duplicate protection. Early and closing
snapshots are both kept so line movement stays observable.

Collection plumbing only: no models, no recommendations, no proof-gate or
betting changes. Approved bets and approved parlays remain blocked.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .canonical_games import build_canonical_game_key
from .nba_collection_planner import minutes_until_next_nba_game
from .player_prop_schema import (
    PLAYER_PROP_SNAPSHOT_COLUMNS,
    validate_player_prop_snapshots,
)


GAME_LOCAL_TIMEZONE = "America/New_York"

# Columns that identify one snapshot; rows matching on all of these are exact
# duplicates (re-pull of an unchanged line) and are dropped on append.
DEDUP_COLUMNS = [
    "snapshot_time",
    "sport",
    "league",
    "game_date",
    "canonical_game_key",
    "player_name",
    "team",
    "prop_type",
    "line",
    "over_price",
    "under_price",
    "bookmaker",
    "source",
    "market_id",
]

_OVER_TOKENS = {"over", "yes"}
_UNDER_TOKENS = {"under", "no"}

# Leagues whose season label spans two calendar years (e.g. 2025-26).
_CROSS_YEAR_LEAGUES = {"NBA", "NHL", "NCAAB", "EPL", "LA_LIGA", "SERIE_A", "BUNDESLIGA", "LIGUE_1"}
# Leagues labeled by the season's starting year even though play crosses Jan 1.
_START_YEAR_LEAGUES = {"NFL"}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS = {
    "event_horizon_hours": 36,
    "closing_window_minutes": 60,
    "max_events_per_league_per_run": 6,
    # 0 means "no cap": collect every enabled league. When set, leagues are
    # collected in priority order and the rest are skipped (quota protection).
    "max_leagues_per_run": 0,
}

_QUOTA_DEFAULTS = {
    # Skip the remaining leagues once the Odds API reports fewer credits than
    # this. 0 disables the guard. Quota is read from the x-requests-remaining
    # response header when available; when the header is absent the guard
    # cannot trigger (quota data unknown).
    "min_remaining_requests": 0,
    # Stricter floor for collect-only (non-modeling-priority) leagues: when
    # quota looks limited, low-priority leagues are skipped first so the
    # modeling-priority leagues (NBA) keep collecting. 0 falls back to
    # min_remaining_requests.
    "low_priority_min_remaining": 0,
}

# NBA near-tip prioritization thresholds (minutes to the next known NBA tip).
NBA_NEAR_TIP_MINUTES = 120.0
NBA_HIGH_PRIORITY_MINUTES = 60.0

_CATCH_UP_DEFAULTS = {
    # Warn when the previous run finished more than this many hours ago.
    "max_gap_hours": 24,
}

# Leagues without an explicit priority sort after every prioritized league.
_DEFAULT_LEAGUE_PRIORITY = 1000

_OUTPUT_DEFAULTS = {
    "raw_dir": "data/raw/prop_odds",
    "processed_path": "data/processed/player_prop_snapshots_normalized.csv",
    "run_summary_path": "data/reports/player_prop_collection_run_summary.json",
    "run_log_dir": "data/logs/prop_collection_runs",
    # Append-only one-line-JSON history of every run (the run summary above is
    # overwritten each run; health reporting needs the full history).
    "run_history_path": "data/reports/prop_collection_run_history.jsonl",
}


def load_prop_collection_config(path: str | Path) -> dict[str, Any]:
    """Load + lightly validate the prop-collection YAML config."""

    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Prop collection config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Prop collection config must be a mapping: {config_path}")

    defaults = dict(_CONFIG_DEFAULTS)
    defaults.update(data.get("defaults") or {})
    data["defaults"] = defaults

    output = dict(_OUTPUT_DEFAULTS)
    output.update(data.get("output") or {})
    data["output"] = output

    closing = data.get("closing_snapshot") or {}
    closing.setdefault("window_minutes", defaults["closing_window_minutes"])
    data["closing_snapshot"] = closing

    quota = dict(_QUOTA_DEFAULTS)
    quota.update(data.get("quota") or {})
    data["quota"] = quota

    catch_up = dict(_CATCH_UP_DEFAULTS)
    catch_up.update(data.get("catch_up") or {})
    data["catch_up"] = catch_up

    data.setdefault("sources", {})
    leagues = data.get("leagues") or {}
    if not isinstance(leagues, dict):
        raise ValueError("Config 'leagues' must be a mapping of league -> settings.")
    for league, league_cfg in leagues.items():
        if not isinstance(league_cfg, dict) or not league_cfg.get("sport"):
            raise ValueError(f"League '{league}' must define a 'sport'.")
        league_cfg.setdefault("enabled", False)
        league_cfg.setdefault("modeling_priority", False)
        league_cfg.setdefault("collect_only", not league_cfg["modeling_priority"])
        league_cfg.setdefault("priority", _DEFAULT_LEAGUE_PRIORITY)
        league_cfg.setdefault("sources", {})
    data["leagues"] = leagues
    return data


def enabled_leagues(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return only the leagues with enabled=true, preserving config order."""

    return {
        league: cfg
        for league, cfg in (config.get("leagues") or {}).items()
        if cfg.get("enabled")
    }


def leagues_in_priority_order(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """All configured leagues sorted for collection order.

    ``modeling_priority`` leagues (NBA) always come first regardless of their
    numeric priority, then ascending ``priority``, then config order. Quota
    protection collects high-priority leagues first so that when the league
    cap or the quota guard kicks in, the modeling-priority leagues have
    already been collected.
    """

    items = list((config.get("leagues") or {}).items())
    indexed = list(enumerate(items))

    def sort_key(entry: tuple[int, tuple[str, dict[str, Any]]]) -> tuple[int, float, int]:
        index, (_, cfg) = entry
        try:
            priority = float(cfg.get("priority", _DEFAULT_LEAGUE_PRIORITY))
        except (TypeError, ValueError):
            priority = float(_DEFAULT_LEAGUE_PRIORITY)
        modeling_first = 0 if cfg.get("modeling_priority") else 1
        return (modeling_first, priority, index)

    return [item for _, item in sorted(indexed, key=sort_key)]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def infer_season_label(league: str, game_date: object) -> str:
    """Best-effort season label per league family (collection metadata only)."""

    parsed = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(parsed):
        return ""
    year, month = parsed.year, parsed.month
    league_norm = str(league).strip().upper()
    if league_norm in _CROSS_YEAR_LEAGUES:
        start = year if month >= 8 else year - 1
        return f"{start}-{(start + 1) % 100:02d}"
    if league_norm in _START_YEAR_LEAGUES:
        return str(year if month >= 8 else year - 1)
    return str(year)


def _iso_utc(value: object) -> str:
    """Parse any timestamp-ish value to a normalized ISO-8601 UTC string."""

    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).isoformat()


def _game_local_date(start_time_utc: object) -> str:
    parsed = pd.to_datetime(start_time_utc, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).tz_convert(GAME_LOCAL_TIMEZONE).date().isoformat()


def _safe_game_key(sport: str, league: str, game_date: str, home: object, away: object) -> str:
    try:
        return build_canonical_game_key(sport, league, game_date, home, away)
    except ValueError:
        return ""


def make_default_fetch_json(timeout: float = 30.0) -> Callable[[str], Any]:
    """Build the default JSON fetcher, which tracks Odds API quota headers.

    After every call ``fetch.quota_remaining`` holds the latest
    ``x-requests-remaining`` header value (None until a response includes it).
    Test fakes can set the same attribute to drive the quota guard.
    """

    def fetch(url: str) -> Any:
        request = urllib.request.Request(url, headers={"User-Agent": "nba-kalshi-predictor/prop-collection"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            remaining = response.headers.get("x-requests-remaining")
            if remaining is not None:
                try:
                    fetch.quota_remaining = float(remaining)  # type: ignore[attr-defined]
                except (TypeError, ValueError):
                    pass
            return json.loads(response.read().decode("utf-8"))

    fetch.quota_remaining = None  # type: ignore[attr-defined]
    return fetch


def _dedup_keys(frame: pd.DataFrame) -> pd.Series:
    """Build a stable text key per row so dedup survives CSV round-trips."""

    if frame.empty:
        return pd.Series(dtype="object")
    parts: list[pd.Series] = []
    for column in DEDUP_COLUMNS:
        series = frame[column] if column in frame.columns else pd.Series(pd.NA, index=frame.index)
        if column in {"line", "over_price", "under_price"}:
            numeric = pd.to_numeric(series, errors="coerce")
            text = numeric.map(lambda v: "" if pd.isna(v) else repr(float(v)))
        else:
            text = series.map(lambda v: "" if pd.isna(v) else str(v).strip())
        parts.append(text.astype(str))
    keys = parts[0]
    for part in parts[1:]:
        keys = keys + "\x1f" + part
    return keys


def apply_closing_flags(frame: pd.DataFrame, window_minutes: float) -> pd.DataFrame:
    """Recompute minutes_to_game_start and is_closing_snapshot where possible.

    Rows without a parseable game_start_time/snapshot_time keep their existing
    values. Non-closing snapshots are never dropped: early lines matter for
    line-movement research.
    """

    if frame.empty:
        return frame
    out = frame.copy()
    snap = pd.to_datetime(out.get("snapshot_time"), errors="coerce", utc=True)
    start = pd.to_datetime(out.get("game_start_time"), errors="coerce", utc=True)
    minutes = (start - snap).dt.total_seconds() / 60.0
    computable = minutes.notna()
    existing_minutes = pd.to_numeric(out.get("minutes_to_game_start"), errors="coerce")
    out["minutes_to_game_start"] = existing_minutes.where(~computable, minutes.round(2))
    closing = minutes.ge(0) & minutes.le(float(window_minutes))
    existing_flag = out.get("is_closing_snapshot")
    if existing_flag is None:
        existing_flag = pd.Series(False, index=out.index)
    existing_bool = existing_flag.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})
    out["is_closing_snapshot"] = existing_bool.where(~computable, closing)
    return out


# ---------------------------------------------------------------------------
# The Odds API: fetch + normalize
# ---------------------------------------------------------------------------

def _odds_api_url(base_url: str, path: str, params: dict[str, Any]) -> str:
    return f"{base_url.rstrip('/')}{path}?{urllib.parse.urlencode(params)}"


def normalize_odds_api_event_payload(
    payload: dict[str, Any],
    *,
    sport: str,
    league: str,
    market_map: dict[str, str],
    raw_source_file: str,
    run_time: datetime,
) -> pd.DataFrame:
    """Normalize one Odds API event-odds payload into snapshot-schema rows.

    Over/Under (and Yes/No) outcome pairs are pivoted into one row per
    (bookmaker, market, player, line). Player team/opponent/home_away stay null:
    The Odds API does not publish player-team membership, and weaker ID mapping
    must not fail the run.
    """

    event_id = str(payload.get("id") or "")
    start_iso = _iso_utc(payload.get("commence_time"))
    game_date = _game_local_date(payload.get("commence_time"))
    home = payload.get("home_team")
    away = payload.get("away_team")
    game_key = _safe_game_key(sport, league, game_date, home, away) if game_date else ""
    season = infer_season_label(league, game_date)
    fallback_snapshot = _iso_utc(run_time)

    grouped: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    for bookmaker in payload.get("bookmakers") or []:
        book = str(bookmaker.get("key") or "")
        book_update = bookmaker.get("last_update")
        for market in bookmaker.get("markets") or []:
            market_key = str(market.get("key") or "")
            snapshot_iso = _iso_utc(market.get("last_update") or book_update) or fallback_snapshot
            prop_type = market_map.get(market_key, market_key.removeprefix("player_"))
            for outcome in market.get("outcomes") or []:
                player = str(outcome.get("description") or "").strip()
                point = pd.to_numeric(outcome.get("point"), errors="coerce")
                if not player or pd.isna(point):
                    continue
                side = str(outcome.get("name") or "").strip().lower()
                slot = grouped.setdefault(
                    (book, market_key, player, float(point)),
                    {
                        "snapshot_time": snapshot_iso,
                        "prop_type": prop_type,
                        "over_price": pd.NA,
                        "under_price": pd.NA,
                    },
                )
                price = pd.to_numeric(outcome.get("price"), errors="coerce")
                if side in _OVER_TOKENS:
                    slot["over_price"] = float(price) if not pd.isna(price) else pd.NA
                elif side in _UNDER_TOKENS:
                    slot["under_price"] = float(price) if not pd.isna(price) else pd.NA

    rows: list[dict[str, Any]] = []
    for (book, market_key, player, point), slot in grouped.items():
        rows.append(
            {
                "snapshot_time": slot["snapshot_time"],
                "sport": sport,
                "league": league,
                "season": season,
                "game_date": game_date,
                "game_start_time": start_iso,
                "canonical_game_key": game_key,
                "player_name": player,
                "player_id": pd.NA,
                "team": pd.NA,
                "opponent": pd.NA,
                "home_away": pd.NA,
                "prop_type": slot["prop_type"],
                "line": float(point),
                "over_price": slot["over_price"],
                "under_price": slot["under_price"],
                "bookmaker": book,
                "source": "odds_api",
                "market_id": f"{event_id}:{market_key}",
                "is_closing_snapshot": False,
                "minutes_to_game_start": pd.NA,
                "has_result": False,
                "actual_stat_value": pd.NA,
                "over_won": pd.NA,
                "under_won": pd.NA,
                "raw_source_file": raw_source_file,
            }
        )
    frame = pd.DataFrame(rows, columns=list(PLAYER_PROP_SNAPSHOT_COLUMNS))
    return frame


def _collect_odds_api_league(
    league: str,
    league_cfg: dict[str, Any],
    source_cfg: dict[str, Any],
    api_key: str,
    *,
    raw_dir: Path,
    project_root: Path,
    run_id: str,
    now: datetime,
    horizon_hours: float,
    max_events: int,
    fetch_json: Callable[[str], Any],
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Collect player props for one league from The Odds API."""

    league_source_cfg = (league_cfg.get("sources") or {}).get("odds_api") or {}
    sport_key = league_source_cfg.get("sport_key")
    if not sport_key:
        return {"status": "skipped_no_sport_key", "events": 0, "raw_files": [], "frames": []}
    market_map = dict(league_source_cfg.get("markets") or {})
    if not market_map:
        return {"status": "skipped_no_markets", "events": 0, "raw_files": [], "frames": []}

    base_url = source_cfg.get("base_url", "https://api.the-odds-api.com/v4")
    regions = source_cfg.get("regions", "us")
    odds_format = source_cfg.get("odds_format", "decimal")
    sport = str(league_cfg.get("sport", "")).strip().lower()

    horizon_end = now + timedelta(hours=float(horizon_hours))
    time_fmt = "%Y-%m-%dT%H:%M:%SZ"
    events_url = _odds_api_url(
        base_url,
        f"/sports/{sport_key}/events",
        {
            "apiKey": api_key,
            "dateFormat": "iso",
            "commenceTimeFrom": now.strftime(time_fmt),
            "commenceTimeTo": horizon_end.strftime(time_fmt),
        },
    )
    events = fetch_json(events_url)
    if not isinstance(events, list):
        events = []
    log(f"{league}: odds_api listed {len(events)} upcoming events (horizon {horizon_hours}h)")

    league_raw_dir = raw_dir / league / "odds_api"
    league_raw_dir.mkdir(parents=True, exist_ok=True)
    events_file = league_raw_dir / f"{run_id}__events.json"
    events_file.write_text(json.dumps(events, indent=2, default=str), encoding="utf-8")
    raw_files = [events_file]

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    used_events = 0
    markets_param = ",".join(sorted(market_map))
    for event in events[: max(0, int(max_events))]:
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        odds_url = _odds_api_url(
            base_url,
            f"/sports/{sport_key}/events/{event_id}/odds",
            {
                "apiKey": api_key,
                "regions": regions,
                "markets": markets_param,
                "oddsFormat": odds_format,
                "dateFormat": "iso",
            },
        )
        try:
            payload = fetch_json(odds_url)
        except Exception as exc:  # one bad event must not sink the league
            errors.append(f"{event_id}: {exc}")
            log(f"{league}: event {event_id} fetch failed ({exc}); continuing")
            continue
        raw_path = league_raw_dir / f"{run_id}__{event_id}.json"
        raw_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        raw_files.append(raw_path)
        used_events += 1
        frames.append(
            normalize_odds_api_event_payload(
                payload if isinstance(payload, dict) else {},
                sport=sport,
                league=league,
                market_map=market_map,
                raw_source_file=raw_path.relative_to(project_root).as_posix(),
                run_time=now,
            )
        )

    return {
        "status": "collected",
        "events": used_events,
        "raw_files": raw_files,
        "frames": frames,
        "event_errors": errors,
    }


# ---------------------------------------------------------------------------
# Append + dedup
# ---------------------------------------------------------------------------

def load_existing_snapshots(path: Path) -> pd.DataFrame:
    """Load the historical normalized CSV, reindexed onto the current schema."""

    if not path.exists():
        return pd.DataFrame(columns=list(PLAYER_PROP_SNAPSHOT_COLUMNS))
    frame = pd.read_csv(path, low_memory=False)
    for column in PLAYER_PROP_SNAPSHOT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[list(PLAYER_PROP_SNAPSHOT_COLUMNS)]


def append_snapshots(existing: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Append new snapshots, dropping exact duplicates; history is preserved.

    Returns ``(combined, duplicates_removed)``. Existing rows always win, so
    re-running the collector never rewrites or loses historical snapshots.
    """

    if new.empty:
        return existing.reset_index(drop=True), 0
    combined = pd.concat([existing, new], ignore_index=True)
    keys = _dedup_keys(combined)
    keep = ~keys.duplicated(keep="first")
    duplicates_removed = int((~keep).sum())
    return combined[keep].reset_index(drop=True), duplicates_removed


# ---------------------------------------------------------------------------
# Run history (append-only JSONL; the run summary JSON is overwritten each run)
# ---------------------------------------------------------------------------

def run_outcome(league_results: list[dict[str, Any]]) -> str:
    """Classify a run: failed (any league error) > success (any collected) > skipped."""

    statuses = [str(r.get("status", "")) for r in league_results]
    if any(s == "error" for s in statuses):
        return "failed"
    if any(s == "collected" for s in statuses):
        return "success"
    return "skipped"


def load_run_history(path: str | Path) -> list[dict[str, Any]]:
    """Load the run-history JSONL; unparseable lines are skipped, not fatal."""

    history_path = Path(path)
    if not history_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def detect_collection_gap(
    history: list[dict[str, Any]],
    now: datetime,
    max_gap_hours: float,
) -> dict[str, Any] | None:
    """Return a missed-collection warning when the last run is too old.

    Honest by design: The Odds API does not serve historical odds on the
    current plan, so snapshots from the gap are unrecoverable — the warning
    says so instead of pretending a catch-up backfill happened.
    """

    if not history:
        return None
    last_times = [
        pd.to_datetime(record.get("run_time_utc"), errors="coerce", utc=True)
        for record in history
    ]
    last_times = [t for t in last_times if not pd.isna(t)]
    if not last_times:
        return None
    last_run = max(last_times)
    gap_hours = (pd.Timestamp(now) - last_run).total_seconds() / 3600.0
    if gap_hours <= float(max_gap_hours):
        return None
    return {
        "type": "missed_collection_window",
        "last_run_utc": last_run.isoformat(),
        "gap_hours": round(gap_hours, 2),
        "max_gap_hours": float(max_gap_hours),
        "message": (
            f"Last collection run was {gap_hours:.1f}h ago (limit {float(max_gap_hours):.0f}h). "
            "Odds snapshots from the gap are NOT recoverable: The Odds API does not provide "
            "historical odds on the current plan. Collection continues normally from now."
        ),
    }


def append_run_history(path: str | Path, record: dict[str, Any]) -> None:
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------

def run_prop_collection(
    config: dict[str, Any],
    project_root: str | Path,
    *,
    now: datetime | None = None,
    fetch_json: Callable[[str], Any] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one multi-sport prop-collection pass and write all outputs.

    ``fetch_json`` and ``env`` are injectable for offline tests. Missing API
    keys, disabled sports, and unimplemented sources are skipped gracefully —
    one weak league/source never fails the whole run.
    """

    root = Path(project_root)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    env = os.environ if env is None else env
    fetch = fetch_json or make_default_fetch_json()
    run_id = now.strftime("%Y%m%dT%H%M%SZ")

    output_cfg = config.get("output", _OUTPUT_DEFAULTS)
    raw_dir = root / output_cfg.get("raw_dir", _OUTPUT_DEFAULTS["raw_dir"])
    processed_path = root / output_cfg.get("processed_path", _OUTPUT_DEFAULTS["processed_path"])
    summary_path = root / output_cfg.get("run_summary_path", _OUTPUT_DEFAULTS["run_summary_path"])
    log_dir = root / output_cfg.get("run_log_dir", _OUTPUT_DEFAULTS["run_log_dir"])
    history_path = root / output_cfg.get("run_history_path", _OUTPUT_DEFAULTS["run_history_path"])

    defaults = config.get("defaults", _CONFIG_DEFAULTS)
    horizon_hours = float(defaults.get("event_horizon_hours", 36))
    max_events = int(defaults.get("max_events_per_league_per_run", 6))
    max_leagues = int(defaults.get("max_leagues_per_run", 0) or 0)
    window_minutes = float(
        (config.get("closing_snapshot") or {}).get(
            "window_minutes", defaults.get("closing_window_minutes", 60)
        )
    )
    quota_cfg = config.get("quota") or {}
    min_remaining = float(quota_cfg.get("min_remaining_requests", 0) or 0)
    low_priority_min = float(quota_cfg.get("low_priority_min_remaining", 0) or 0)
    if low_priority_min and low_priority_min < min_remaining:
        low_priority_min = min_remaining
    catch_up_cfg = config.get("catch_up") or {}
    max_gap_hours = float(catch_up_cfg.get("max_gap_hours", _CATCH_UP_DEFAULTS["max_gap_hours"]))

    log_lines: list[str] = []

    def log(message: str) -> None:
        log_lines.append(f"{datetime.now(timezone.utc).isoformat()} {message}")

    log(f"run {run_id}: starting multi-sport prop collection (research-only)")

    # Catch-up check: warn (honestly) when collection was missed, then proceed.
    warnings: list[dict[str, Any]] = []
    gap_warning = detect_collection_gap(load_run_history(history_path), now, max_gap_hours)
    if gap_warning is not None:
        warnings.append(gap_warning)
        log(f"WARNING: {gap_warning['message']}")

    sources_cfg = config.get("sources") or {}
    odds_api_cfg = sources_cfg.get("odds_api") or {}
    odds_api_key_env = odds_api_cfg.get("api_key_env", "ODDS_API_KEY")
    api_key_detected = bool(odds_api_cfg.get("api_key") or env.get(odds_api_key_env, ""))

    # NBA near-tip awareness: how close is the next known NBA tip (from the
    # snapshots already on disk)? Near-tip runs protect NBA quota by skipping
    # low-priority leagues first; modeling-priority leagues always collect
    # first in the league ordering regardless.
    existing = load_existing_snapshots(processed_path)
    nba_minutes = minutes_until_next_nba_game(existing, now)
    nba_within_2h = nba_minutes is not None and 0 < nba_minutes <= NBA_NEAR_TIP_MINUTES
    nba_within_60m = nba_minutes is not None and 0 < nba_minutes <= NBA_HIGH_PRIORITY_MINUTES
    if nba_within_60m:
        log(
            f"NBA tip in {nba_minutes:.0f} minutes: HIGH PRIORITY run "
            "(NBA closing collection comes first; low-priority leagues yield quota)"
        )
    elif nba_within_2h:
        log(f"NBA tip in {nba_minutes:.0f} minutes: NBA closing collection prioritized")

    league_results: list[dict[str, Any]] = []
    new_frames: list[pd.DataFrame] = []
    raw_files_total = 0
    leagues_collected = 0

    def quota_remaining() -> float | None:
        value = getattr(fetch, "quota_remaining", None)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    for league, league_cfg in leagues_in_priority_order(config):
        sport = str(league_cfg.get("sport", "")).strip().lower()
        base_record = {
            "league": league,
            "sport": sport,
            "modeling_priority": bool(league_cfg.get("modeling_priority")),
            "collect_only": bool(league_cfg.get("collect_only")),
        }
        if not league_cfg.get("enabled"):
            league_results.append({**base_record, "source": None, "status": "skipped_disabled"})
            log(f"{league}: skipped (disabled in config)")
            continue

        league_sources = league_cfg.get("sources") or {}
        if not league_sources:
            league_results.append({**base_record, "source": None, "status": "skipped_no_sources"})
            log(f"{league}: skipped (no sources configured)")
            continue

        for source_name in league_sources:
            record = {**base_record, "source": source_name}
            source_cfg = sources_cfg.get(source_name) or {}
            if not source_cfg.get("enabled"):
                record["status"] = "skipped_source_disabled"
                log(f"{league}/{source_name}: skipped (source disabled)")
            elif source_name != "odds_api":
                record["status"] = "skipped_not_implemented"
                log(f"{league}/{source_name}: skipped (collector not implemented yet)")
            else:
                api_key_env = source_cfg.get("api_key_env", "ODDS_API_KEY")
                api_key = source_cfg.get("api_key") or env.get(api_key_env, "")
                remaining = quota_remaining()
                is_low_priority = not league_cfg.get("modeling_priority")
                # Low-priority leagues face a stricter quota floor; the floor
                # tightens further when an NBA game is near tip so the closing
                # collection cannot be starved by collect-only leagues.
                low_floor = low_priority_min if low_priority_min > 0 else min_remaining
                if nba_within_2h and is_low_priority:
                    low_floor = max(low_floor, min_remaining * 2 if min_remaining else low_floor)
                if not api_key:
                    record["status"] = "skipped_no_api_key"
                    record["detail"] = f"set {api_key_env} to enable this source"
                    log(f"{league}/odds_api: skipped ({api_key_env} not set)")
                elif max_leagues > 0 and leagues_collected >= max_leagues:
                    record["status"] = "skipped_league_cap"
                    record["detail"] = f"max_leagues_per_run={max_leagues} already collected"
                    log(f"{league}/odds_api: skipped (league cap {max_leagues} reached)")
                elif min_remaining > 0 and remaining is not None and remaining < min_remaining:
                    record["status"] = "skipped_quota_low"
                    record["detail"] = (
                        f"quota remaining {remaining:.0f} < min_remaining_requests {min_remaining:.0f}"
                    )
                    log(f"{league}/odds_api: skipped (quota low: {remaining:.0f} remaining)")
                elif (
                    is_low_priority
                    and low_floor > 0
                    and remaining is not None
                    and remaining < low_floor
                ):
                    record["status"] = "skipped_quota_low_priority"
                    record["detail"] = (
                        f"quota remaining {remaining:.0f} < low-priority floor {low_floor:.0f}; "
                        "skipped to protect NBA (modeling-priority) collection"
                    )
                    log(
                        f"{league}/odds_api: skipped (low-priority quota floor {low_floor:.0f}, "
                        f"{remaining:.0f} remaining)"
                    )
                else:
                    league_max_events = int(league_cfg.get("max_events_per_run", max_events))
                    try:
                        result = _collect_odds_api_league(
                            league,
                            league_cfg,
                            source_cfg,
                            api_key,
                            raw_dir=raw_dir,
                            project_root=root,
                            run_id=run_id,
                            now=now,
                            horizon_hours=horizon_hours,
                            max_events=league_max_events,
                            fetch_json=fetch,
                            log=log,
                        )
                    except Exception as exc:  # a league failure must not sink the run
                        record["status"] = "error"
                        record["detail"] = str(exc)
                        log(f"{league}/odds_api: ERROR {exc}")
                        league_results.append(record)
                        continue
                    frames = result.pop("frames", [])
                    raw_files = result.pop("raw_files", [])
                    snapshots = int(sum(len(f) for f in frames))
                    new_frames.extend(frames)
                    raw_files_total += len(raw_files)
                    if result.get("status") == "collected":
                        leagues_collected += 1
                    record.update(result)
                    record["raw_files"] = [p.relative_to(root).as_posix() for p in raw_files]
                    record["snapshots_collected"] = snapshots
                    log(f"{league}/odds_api: {record['status']} ({snapshots} snapshots from {record.get('events', 0)} events)")
            league_results.append(record)

    new = (
        pd.concat(new_frames, ignore_index=True)
        if new_frames
        else pd.DataFrame(columns=list(PLAYER_PROP_SNAPSHOT_COLUMNS))
    )
    new = apply_closing_flags(new, window_minutes)

    rows_before = int(len(existing))
    combined, duplicates_removed = append_snapshots(existing, new)
    combined = apply_closing_flags(combined, window_minutes)
    snapshots_added = int(len(combined)) - rows_before

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(processed_path, index=False)

    validation = validate_player_prop_snapshots(combined)
    closing_mask = combined.get("is_closing_snapshot", pd.Series(dtype="object")).map(
        lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"}
    )

    final_remaining = quota_remaining()
    quota_error_tokens = ("401", "402", "429", "quota", "usage limit", "out of usage")

    def _looks_like_quota_error(text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in quota_error_tokens)

    quota_error_seen = any(
        _looks_like_quota_error(str(record.get("detail", "")))
        or any(_looks_like_quota_error(str(err)) for err in record.get("event_errors", []) or [])
        for record in league_results
    )
    likely_quota_issue = bool(
        quota_error_seen
        or any(
            record.get("status") in {"skipped_quota_low", "skipped_quota_low_priority"}
            for record in league_results
        )
        or (min_remaining > 0 and final_remaining is not None and final_remaining < min_remaining)
    )

    collected_leagues = [r["league"] for r in league_results if r.get("status") == "collected"]
    skipped_leagues = [
        {"league": r["league"], "status": r.get("status"), "reason": r.get("detail", "")}
        for r in league_results
        if r.get("status") not in {"collected", "error"}
    ]
    nba_closing_priority = {
        "minutes_until_next_nba_game": nba_minutes,
        "nba_within_2h": nba_within_2h,
        "nba_within_60m": nba_within_60m,
        "high_priority_run": nba_within_60m,
        "nba_prioritized": bool(
            collected_leagues
            and any(
                r["league"] == collected_leagues[0] and r.get("modeling_priority")
                for r in league_results
            )
        ),
        "note": (
            "Modeling-priority leagues (NBA) always collect first; low-priority leagues "
            "are skipped first when quota looks limited."
        ),
    }

    summary = {
        "report": "player_prop_collection_run_summary",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "outcome": run_outcome(league_results),
        "closing_window_minutes": window_minutes,
        "event_horizon_hours": horizon_hours,
        "api_key_detected": api_key_detected,
        "quota": {
            "remaining_requests": final_remaining,
            "min_remaining_requests": min_remaining,
            "low_priority_min_remaining": low_priority_min,
            "likely_quota_issue": likely_quota_issue,
        },
        "nba_closing_priority": nba_closing_priority,
        "leagues_collected": collected_leagues,
        "leagues_skipped": skipped_leagues,
        "warnings": warnings,
        "leagues": league_results,
        "totals": {
            "snapshots_collected_this_run": int(len(new)),
            "snapshots_added": max(0, snapshots_added),
            "duplicates_removed": duplicates_removed,
            "snapshots_total": int(len(combined)),
            "closing_snapshots_total": int(closing_mask.sum()) if not combined.empty else 0,
            "raw_files_saved": raw_files_total,
        },
        "validation": validation,
        "outputs": {
            "processed_path": processed_path.relative_to(root).as_posix(),
            "raw_dir": raw_dir.relative_to(root).as_posix(),
            "run_log": "",
        },
        "research_only": True,
        "approved": False,
        "notes": [
            "Collection only: no models, no recommendations.",
            "Approved bets and approved parlays remain blocked.",
            "Early and closing snapshots are both kept (line movement matters).",
        ],
    }

    log(
        f"run {run_id}: done — {summary['totals']['snapshots_added']} added, "
        f"{duplicates_removed} duplicates removed, {len(combined)} total snapshots"
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{run_id}.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    summary["outputs"]["run_log"] = log_path.relative_to(root).as_posix()

    snapshots_by_league: dict[str, int] = {}
    snapshots_by_sport: dict[str, int] = {}
    for record in league_results:
        count = int(record.get("snapshots_collected", 0) or 0)
        if record.get("status") == "collected":
            snapshots_by_league[record["league"]] = snapshots_by_league.get(record["league"], 0) + count
            sport_name = str(record.get("sport") or "unknown")
            snapshots_by_sport[sport_name] = snapshots_by_sport.get(sport_name, 0) + count

    history_record = {
        "run_id": run_id,
        "run_time_utc": now.isoformat(),
        "outcome": summary["outcome"],
        "api_key_detected": api_key_detected,
        "quota_remaining_requests": final_remaining,
        "likely_quota_issue": likely_quota_issue,
        "nba_minutes_to_next_tip": nba_minutes,
        "nba_high_priority_run": nba_within_60m,
        "snapshots_collected": int(len(new)),
        "snapshots_added": summary["totals"]["snapshots_added"],
        "duplicates_removed": duplicates_removed,
        "snapshots_by_league": snapshots_by_league,
        "snapshots_by_sport": snapshots_by_sport,
        "league_statuses": {
            f"{r['league']}/{r.get('source') or '-'}": r.get("status") for r in league_results
        },
        "errors": [
            f"{r['league']}: {r.get('detail', '')}" for r in league_results if r.get("status") == "error"
        ],
        "warnings": [w.get("message", "") for w in warnings],
        "run_log": summary["outputs"]["run_log"],
    }
    append_run_history(history_path, history_record)
    summary["outputs"]["run_history"] = history_path.relative_to(root).as_posix()

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
