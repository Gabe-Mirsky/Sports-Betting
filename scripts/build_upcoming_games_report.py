"""Build a unified upcoming-games collection report (research-only).

The dashboard needs one stable report for upcoming games across sports.  This
script uses the NBA closing-collection plan when present, then supplements it
with future games found in the normalized player-prop snapshot store.  Missing
inputs produce a valid empty report instead of a crash.

Outputs:
    data/reports/upcoming_games.json
    data/reports/upcoming_games.csv
    data/reports/upcoming_games.md
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SNAPSHOTS_PATH = PROJECT_ROOT / "data" / "processed" / "player_prop_snapshots_normalized.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "prop_collection.yaml"
NBA_PLAN_PATH = REPORTS_DIR / "nba_prop_closing_collection_plan.json"

OUTPUT_COLUMNS = [
    "game_id",
    "game_datetime_utc",
    "sport",
    "league",
    "away_team",
    "home_team",
    "source",
    "props_expected",
    "props_collected",
    "latest_snapshot_time_utc",
    "minutes_until_game",
    "closing_window_status",
    "recommended_collection_action",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_config() -> dict[str, Any]:
    try:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _split_game_key(key: Any) -> tuple[str, str, str, str, str]:
    parts = str(key or "").split("|")
    if len(parts) >= 5:
        return parts[0], parts[1], parts[2], parts[3], parts[4]
    return "", "", "", "", ""


def closing_window_status(minutes_until_game: float | None) -> str:
    if minutes_until_game is None:
        return "not available"
    if minutes_until_game <= 0:
        return "closed/started"
    if minutes_until_game <= 10:
        return "10m window"
    if minutes_until_game <= 30:
        return "30m window"
    if minutes_until_game <= 60:
        return "60m window"
    if minutes_until_game <= 120:
        return "2h window"
    if minutes_until_game <= 360:
        return "6h window"
    if minutes_until_game <= 1440:
        return "24h window"
    return "too early"


def recommended_action(status: str, collect_now: bool, next_time: Any, props_collected: bool) -> str:
    if collect_now:
        return "collect props now"
    if status == "closed/started":
        return "no action - game closed/started"
    if next_time:
        return f"wait until {next_time}"
    if not props_collected and status in {"24h window", "6h window", "2h window", "60m window", "30m window", "10m window"}:
        return "collect props in this window"
    return "standard cadence"


def _league_props_expected(config: dict[str, Any], league: str) -> bool:
    league_cfg = ((config.get("leagues") or {}).get(league) or {})
    markets = (((league_cfg.get("sources") or {}).get("odds_api") or {}).get("markets") or {})
    return bool(league_cfg.get("enabled") and markets)


def _snapshot_latest_by_game() -> pd.DataFrame:
    if not SNAPSHOTS_PATH.exists():
        return pd.DataFrame()
    usecols = [
        "sport",
        "league",
        "game_start_time",
        "canonical_game_key",
        "snapshot_time",
        "source",
    ]
    frame = pd.read_csv(SNAPSHOTS_PATH, usecols=lambda c: c in usecols, low_memory=False)
    if frame.empty or "canonical_game_key" not in frame.columns:
        return pd.DataFrame()
    rows = []
    grouped = frame.groupby(frame["canonical_game_key"].astype(str), dropna=False)
    for key, group in grouped:
        sport, league, _, home, away = _split_game_key(key)
        if not sport and "sport" in group:
            sport = str(group["sport"].dropna().iloc[0]) if group["sport"].notna().any() else ""
        if not league and "league" in group:
            league = str(group["league"].dropna().iloc[0]) if group["league"].notna().any() else ""
        start = None
        if "game_start_time" in group.columns:
            starts = pd.to_datetime(group["game_start_time"], errors="coerce", utc=True).dropna()
            if not starts.empty:
                start = starts.max().to_pydatetime()
        times = pd.to_datetime(group["snapshot_time"], errors="coerce", utc=True).dropna()
        rows.append(
            {
                "canonical_game_key": key,
                "sport": sport,
                "league": league,
                "home_team": home,
                "away_team": away,
                "game_datetime": start,
                "latest_snapshot_time_utc": times.max().isoformat() if not times.empty else None,
                "source": ",".join(sorted(group.get("source", pd.Series(dtype=str)).dropna().astype(str).unique())),
                "props_collected": len(group) > 0,
            }
        )
    return pd.DataFrame(rows)


def build_report(now: datetime | None = None) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    config = _read_config()
    plan = _read_json(NBA_PLAN_PATH)
    snapshots = _snapshot_latest_by_game()
    latest_by_key = {
        str(row["canonical_game_key"]): row
        for _, row in snapshots.iterrows()
    } if not snapshots.empty else {}

    rows_by_key: dict[str, dict[str, Any]] = {}

    for game in plan.get("games", []) if isinstance(plan, dict) else []:
        key = str(game.get("canonical_game_key") or "")
        sport, league, _, home, away = _split_game_key(key)
        start = _parse_dt(game.get("game_start_time"))
        minutes = game.get("minutes_until_game")
        if minutes is None and start is not None:
            minutes = round((start - now).total_seconds() / 60.0, 2)
        try:
            minutes_float = round(float(minutes), 2)
        except (TypeError, ValueError):
            minutes_float = None
        latest = latest_by_key.get(key)
        status = closing_window_status(minutes_float)
        collected = bool(game.get("snapshots_total", 0)) or bool(latest is not None and latest.get("props_collected"))
        latest_snapshot_time = latest.get("latest_snapshot_time_utc") if latest is not None else None
        rows_by_key[key] = {
            "game_id": key,
            "game_datetime_utc": start.isoformat() if start else game.get("game_start_time"),
            "sport": sport or "basketball",
            "league": league or "NBA",
            "away_team": away,
            "home_team": home,
            "source": "nba_prop_closing_collection_plan",
            "props_expected": _league_props_expected(config, league or "NBA"),
            "props_collected": collected,
            "latest_snapshot_time_utc": latest_snapshot_time,
            "minutes_until_game": minutes_float,
            "closing_window_status": status,
            "recommended_collection_action": recommended_action(
                status,
                bool(game.get("collection_needed_now")),
                game.get("next_recommended_collection_time_utc"),
                collected,
            ),
        }

    if not snapshots.empty:
        for _, row in snapshots.iterrows():
            start = row.get("game_datetime")
            if pd.isna(start) or start is None:
                continue
            if start <= now:
                continue
            key = str(row.get("canonical_game_key") or "")
            if key in rows_by_key:
                continue
            minutes_float = round((start - now).total_seconds() / 60.0, 2)
            status = closing_window_status(minutes_float)
            league = str(row.get("league") or "")
            rows_by_key[key] = {
                "game_id": key,
                "game_datetime_utc": start.isoformat(),
                "sport": row.get("sport") or "",
                "league": league,
                "away_team": row.get("away_team") or "",
                "home_team": row.get("home_team") or "",
                "source": row.get("source") or "player_prop_snapshots_normalized",
                "props_expected": _league_props_expected(config, league),
                "props_collected": bool(row.get("props_collected")),
                "latest_snapshot_time_utc": row.get("latest_snapshot_time_utc"),
                "minutes_until_game": minutes_float,
                "closing_window_status": status,
                "recommended_collection_action": recommended_action(status, False, None, True),
            }

    rows = sorted(
        rows_by_key.values(),
        key=lambda r: (r.get("game_datetime_utc") or "9999", r.get("league") or "", r.get("game_id") or ""),
    )
    warnings: list[str] = []
    if not rows:
        warnings.append("No upcoming games found in the NBA plan or normalized prop snapshots.")
    for warning in plan.get("warnings", []) if isinstance(plan, dict) else []:
        warnings.append(str(warning))

    return {
        "report": "upcoming_games",
        "generated_at_utc": now.isoformat(),
        "games": rows,
        "games_total": len(rows),
        "games_upcoming": sum(1 for row in rows if (row.get("minutes_until_game") or 0) > 0),
        "collection_needed_now": [
            row for row in rows if row.get("recommended_collection_action") == "collect props now"
        ],
        "warnings": warnings,
        "research_only": True,
        "approved": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Upcoming Games",
        "",
        f"Generated: {report.get('generated_at_utc')}",
        "",
        "_Research-only collection planner. No bets or recommendations._",
        "",
        "| date/time UTC | sport | league | away | home | props expected | props collected | latest snapshot | window | action |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("games", []):
        lines.append(
            f"| {row.get('game_datetime_utc') or ''} | {row.get('sport') or ''} | "
            f"{row.get('league') or ''} | {row.get('away_team') or ''} | {row.get('home_team') or ''} | "
            f"{row.get('props_expected')} | {row.get('props_collected')} | "
            f"{row.get('latest_snapshot_time_utc') or ''} | {row.get('closing_window_status') or ''} | "
            f"{row.get('recommended_collection_action') or ''} |"
        )
    if report.get("warnings"):
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in report["warnings"]]
    lines += ["", "_Approved bets and approved parlays remain blocked._", ""]
    return "\n".join(lines)


def write_report(report: dict[str, Any]) -> dict[str, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "upcoming_games.json"
    csv_path = REPORTS_DIR / "upcoming_games.csv"
    md_path = REPORTS_DIR / "upcoming_games.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in report.get("games", []):
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}


def main() -> int:
    report = build_report()
    outputs = write_report(report)
    print(f"Upcoming games: {report['games_total']} row(s)")
    for name, path in outputs.items():
        print(f"Wrote {name}: {path.relative_to(PROJECT_ROOT)}")
    print("Research-only: no bets, no recommendations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
