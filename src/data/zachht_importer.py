"""Importer for the zachht basketball odds-history Kaggle dataset.

Despite the "wnba-odds-history" slug this is a worldwide basketball odds archive.
This importer keeps only the NBA and WNBA ``*_main_lines`` files, which carry
moneyline/spread/total at a snapshot timestamp, and normalizes them into one
game-odds snapshot table with opening/closing-snapshot detection per game.

This is GAME-LEVEL odds, not player props: ``market_type = game_market``.
Inspection / ingestion only: no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from .team_aliases import normalize_team_abbr


SOURCE = "zachht"
SOURCE_ROLE = "game_odds_market_source"
SPORT = "basketball"
MARKET_TYPE = "game_market"

# main_lines file -> (league, season_type). Only NBA/WNBA basketball is kept.
MAIN_LINE_FILES = {
    "nba_main_lines.csv": ("NBA", "regular"),
    "nba_main_lines_history.csv": ("NBA", "regular"),
    "nba_preseason_main_lines.csv": ("NBA", "preseason"),
    "wnba_main_lines.csv": ("WNBA", "regular"),
    "wnba_main_lines_history.csv": ("WNBA", "regular"),
}

_ODDS_COLUMNS = [
    "team1_moneyline", "team2_moneyline",
    "team1_spread", "team1_spread_odds", "team2_spread", "team2_spread_odds",
    "over_total", "over_total_odds", "under_total", "under_total_odds",
]


def _book_from_link(link: Any) -> str:
    netloc = urlparse(str(link)).netloc.lower()
    if not netloc:
        return ""
    if "pinnacle" in netloc:
        return "Pinnacle"
    return netloc.replace("www.", "")


def _game_ref(link: Any) -> str:
    return str(link).rstrip("/").split("/")[-1]


def normalize_main_lines_file(path: Path, league: str, season_type: str) -> pd.DataFrame:
    """Normalize one ``*_main_lines`` file into snapshot rows."""

    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["source"] = [SOURCE] * len(raw)
    out["source_role"] = SOURCE_ROLE
    out["sport"] = SPORT
    out["league"] = league
    out["season_type"] = season_type
    out["market_type"] = MARKET_TYPE
    out["team1_name"] = raw.get("team1")
    out["team2_name"] = raw.get("team2")
    out["team1_abbr"] = raw.get("team1").map(normalize_team_abbr) if "team1" in raw else ""
    out["team2_abbr"] = raw.get("team2").map(normalize_team_abbr) if "team2" in raw else ""
    out["game_link"] = raw.get("game_link")
    out["game_ref"] = raw.get("game_link").map(_game_ref) if "game_link" in raw else ""
    out["sportsbook"] = raw.get("game_link").map(_book_from_link) if "game_link" in raw else ""
    out["snapshot_time"] = pd.to_datetime(raw.get("timestamp"), errors="coerce")
    for column in _ODDS_COLUMNS:
        out[column] = pd.to_numeric(raw.get(column), errors="coerce")
    out["odds_format"] = "decimal"
    return out


def _add_open_close_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank snapshots per game by time and flag opening/closing snapshots."""

    if frame.empty:
        for column in ["snapshot_rank", "n_snapshots_for_game", "is_opening_snapshot", "is_closing_snapshot"]:
            frame[column] = pd.Series(dtype="object")
        return frame
    out = frame.copy()
    out["_game_uid"] = out["league"].astype(str) + "|" + out["game_ref"].astype(str)
    out = out.sort_values(["_game_uid", "snapshot_time"], na_position="last")
    grouped = out.groupby("_game_uid", sort=False)
    out["snapshot_rank"] = grouped.cumcount() + 1
    out["n_snapshots_for_game"] = grouped["snapshot_time"].transform("size")
    out["is_opening_snapshot"] = out["snapshot_rank"].eq(1)
    out["is_closing_snapshot"] = out["snapshot_rank"].eq(out["n_snapshots_for_game"])
    return out.drop(columns="_game_uid").reset_index(drop=True)


def normalize_basketball_odds(dataset_dir: str | Path) -> pd.DataFrame:
    """Normalize all NBA/WNBA main-line files into one snapshot table."""

    dataset_dir = Path(dataset_dir)
    frames: list[pd.DataFrame] = []
    for filename, (league, season_type) in MAIN_LINE_FILES.items():
        path = dataset_dir / filename
        if path.exists():
            frames.append(normalize_main_lines_file(path, league, season_type))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return _add_open_close_flags(combined)


def import_zachht_odds(
    dataset_dir: str | Path,
    processed_dir: str | Path,
    reports_dir: str | Path,
) -> dict[str, Any]:
    """Run the full zachht odds import and write the normalized CSV + summary JSON."""

    processed = Path(processed_dir)
    reports = Path(reports_dir)
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    snapshots = normalize_basketball_odds(dataset_dir)
    out_path = processed / "basketball_odds_snapshots_normalized.csv"
    snapshots.to_csv(out_path, index=False)

    summary = _build_summary(dataset_dir, snapshots)
    (reports / "zachht_odds_import_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_summary(dataset_dir: str | Path, snapshots: pd.DataFrame) -> dict[str, Any]:
    if snapshots.empty:
        per_game = pd.Series(dtype=int)
    else:
        per_game = snapshots.groupby(["league", "game_ref"]).size()
    snap_time = pd.to_datetime(snapshots.get("snapshot_time"), errors="coerce") if not snapshots.empty else pd.Series(dtype="datetime64[ns]")
    return {
        "source": SOURCE,
        "source_role": SOURCE_ROLE,
        "market_type": MARKET_TYPE,
        "has_player_props": False,
        "dataset_dir": str(dataset_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_rows": int(len(snapshots)),
        "by_league": (
            {str(k): int(v) for k, v in snapshots["league"].value_counts().items()} if not snapshots.empty else {}
        ),
        "by_season_type": (
            {str(k): int(v) for k, v in snapshots["season_type"].value_counts().items()} if not snapshots.empty else {}
        ),
        "unique_games": int(per_game.size),
        "snapshots_per_game": {
            "min": int(per_game.min()) if per_game.size else 0,
            "median": float(per_game.median()) if per_game.size else 0.0,
            "max": int(per_game.max()) if per_game.size else 0,
        },
        "games_with_multiple_snapshots": int((per_game > 1).sum()) if per_game.size else 0,
        "opening_snapshots": int(snapshots["is_opening_snapshot"].sum()) if not snapshots.empty else 0,
        "closing_snapshots": int(snapshots["is_closing_snapshot"].sum()) if not snapshots.empty else 0,
        "sportsbooks": (
            sorted(str(b) for b in snapshots["sportsbook"].dropna().unique() if str(b)) if not snapshots.empty else []
        ),
        "snapshot_time_range": {
            "min": _iso(snap_time.min()) if len(snap_time) else None,
            "max": _iso(snap_time.max()) if len(snap_time) else None,
        },
        "notes": [
            "Game-level odds only (moneyline/spread/total) - NOT player props.",
            "Odds are decimal; all snapshots come from Pinnacle in this archive.",
            "Opening/closing flags are derived per game by snapshot timestamp order.",
            "game_ref comes from the Pinnacle game_link and does NOT match NBA-stats game_id.",
        ],
        "research_only": True,
        "approved": False,
    }


def _iso(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).isoformat()
