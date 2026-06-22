"""Audit NBA player data coverage and player-derived model features."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.injury_availability import AVAILABILITY_BASE_FEATURE_COLUMNS
from features.player_features import PLAYER_DIFF_FEATURE_COLUMNS
from models.train_model import AVAILABILITY_FEATURE_COLUMNS, PLAYER_ROTATION_FEATURE_COLUMNS


def _date_range(frame: pd.DataFrame, column: str) -> dict[str, str]:
    if frame.empty or column not in frame.columns:
        return {"start": "n/a", "end": "n/a"}
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return {"start": "n/a", "end": "n/a"}
    return {"start": dates.min().date().isoformat(), "end": dates.max().date().isoformat()}


def _safe_count_unique(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].dropna().astype(str).nunique())


def _player_season_count(player_logs: pd.DataFrame) -> int:
    if player_logs.empty:
        return 0
    seasons: set[str] = set()
    if "source_file" in player_logs.columns:
        for value in player_logs["source_file"].dropna().astype(str):
            match = re.search(r"player_game_log_(\d{4})", value)
            if match:
                seasons.add(match.group(1))
        if seasons:
            return len(seasons)
    for column in ["season_start_year", "nba_season", "SEASON_ID"]:
        if column in player_logs.columns:
            values = player_logs[column].dropna().astype(str).str.strip()
            for value in values:
                digits = re.sub(r"\D", "", value)
                if len(digits) >= 5 and digits.startswith("2"):
                    seasons.add(digits[-4:])
                else:
                    match = re.search(r"(\d{4})", value)
                    if match:
                        seasons.add(match.group(1))
    return len(seasons)


def _non_null_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(frame[column].notna().mean())


def _feature_coverage_rows(modeling: pd.DataFrame, columns: list[str], feature_group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        rows.append(
            {
                "feature_group": feature_group,
                "feature": column,
                "present": bool(column in modeling.columns),
                "non_null_rate": _non_null_rate(modeling, column),
                "non_null_rows": int(modeling[column].notna().sum()) if column in modeling.columns else 0,
            }
        )
    return rows


def build_player_data_audit(
    games: pd.DataFrame,
    modeling: pd.DataFrame,
    player_logs: pd.DataFrame,
    availability_reports: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return player data summary, feature coverage, and season coverage tables."""

    availability_reports = availability_reports if availability_reports is not None else pd.DataFrame()
    expected_player_columns = [column for column in PLAYER_ROTATION_FEATURE_COLUMNS if column in PLAYER_DIFF_FEATURE_COLUMNS]
    expected_availability_columns = [
        column
        for column in AVAILABILITY_FEATURE_COLUMNS
        if column.endswith("_diff") or column.startswith(("home_availability_", "away_availability_"))
    ]
    feature_rows = _feature_coverage_rows(modeling, expected_player_columns, "player_rotation")
    feature_rows.extend(_feature_coverage_rows(modeling, expected_availability_columns, "availability"))
    feature_coverage = pd.DataFrame(feature_rows)

    player_columns_present = [column for column in expected_player_columns if column in modeling.columns]
    availability_columns_present = [column for column in expected_availability_columns if column in modeling.columns]
    if player_columns_present:
        player_feature_present_mask = modeling[player_columns_present].notna().any(axis=1)
        player_feature_complete_mask = modeling[player_columns_present].notna().all(axis=1)
    else:
        player_feature_present_mask = pd.Series(False, index=modeling.index)
        player_feature_complete_mask = pd.Series(False, index=modeling.index)

    if availability_columns_present:
        availability_feature_present_mask = modeling[availability_columns_present].notna().any(axis=1)
    else:
        availability_feature_present_mask = pd.Series(False, index=modeling.index)

    season_rows: list[dict[str, Any]] = []
    if "season" in modeling.columns:
        frame = modeling.copy()
        frame["_has_player_features"] = player_feature_present_mask
        frame["_has_availability_features"] = availability_feature_present_mask
        for season, group in frame.groupby("season", dropna=False):
            rows = int(len(group))
            season_rows.append(
                {
                    "season": season,
                    "modeling_games": rows,
                    "games_with_player_features": int(group["_has_player_features"].sum()),
                    "player_feature_coverage": float(group["_has_player_features"].mean()) if rows else 0.0,
                    "games_with_availability_features": int(group["_has_availability_features"].sum()),
                    "availability_feature_coverage": float(group["_has_availability_features"].mean()) if rows else 0.0,
                }
            )
    season_coverage = pd.DataFrame(season_rows)

    player_date_range = _date_range(player_logs, "GAME_DATE")
    game_date_range = _date_range(games, "game_date")
    player_feature_coverage = float(player_feature_present_mask.mean()) if len(modeling) else 0.0
    player_feature_complete_coverage = float(player_feature_complete_mask.mean()) if len(modeling) else 0.0
    availability_feature_coverage = float(availability_feature_present_mask.mean()) if len(modeling) else 0.0

    warnings: list[str] = []
    if player_logs.empty:
        warnings.append("no_raw_player_logs")
    if not player_columns_present:
        warnings.append("no_player_features_in_modeling_dataset")
    missing_player_columns = sorted(set(expected_player_columns) - set(player_columns_present))
    if missing_player_columns:
        warnings.append("missing_expected_player_feature_columns")
    if player_feature_coverage < 0.80:
        warnings.append("low_player_feature_coverage")
    if availability_reports.empty:
        warnings.append("no_availability_reports")

    if player_logs.empty or not player_columns_present:
        status = "not_ready"
    elif player_feature_coverage >= 0.80:
        status = "ready"
    else:
        status = "watchlist"

    summary = {
        "status": status,
        "raw_player_log_rows": int(len(player_logs)),
        "raw_player_log_files": _safe_count_unique(player_logs, "source_file"),
        "raw_player_seasons": _player_season_count(player_logs),
        "raw_player_date_start": player_date_range["start"],
        "raw_player_date_end": player_date_range["end"],
        "raw_players": _safe_count_unique(player_logs, "PLAYER_ID"),
        "raw_player_teams": _safe_count_unique(player_logs, "TEAM_ID"),
        "game_rows": int(len(games)),
        "game_date_start": game_date_range["start"],
        "game_date_end": game_date_range["end"],
        "modeling_rows": int(len(modeling)),
        "expected_player_feature_columns": int(len(expected_player_columns)),
        "player_feature_columns_present": int(len(player_columns_present)),
        "player_feature_row_coverage": player_feature_coverage,
        "player_feature_complete_row_coverage": player_feature_complete_coverage,
        "availability_report_rows": int(len(availability_reports)),
        "expected_availability_feature_columns": int(len(expected_availability_columns)),
        "availability_feature_columns_present": int(len(availability_columns_present)),
        "availability_feature_row_coverage": availability_feature_coverage,
        "season_rows": int(len(season_coverage)),
        "warnings": warnings,
        "next_step": (
            "Use player-aware features in model research and compare CLV/backtest movement against team-only features."
            if status == "ready"
            else "Run player data download/build features, then inspect missing feature coverage before model changes."
        ),
    }
    return summary, feature_coverage, season_coverage


def save_player_data_audit(
    summary: dict[str, Any],
    feature_coverage: pd.DataFrame,
    season_coverage: pd.DataFrame,
    summary_path: str | Path,
    feature_coverage_path: str | Path,
    season_coverage_path: str | Path,
) -> None:
    """Write player data audit artifacts."""

    summary_output = Path(summary_path)
    feature_output = Path(feature_coverage_path)
    season_output = Path(season_coverage_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    feature_output.parent.mkdir(parents=True, exist_ok=True)
    season_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    feature_coverage.replace({np.nan: None}).to_csv(feature_output, index=False)
    season_coverage.replace({np.nan: None}).to_csv(season_output, index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value
