"""Validate historical NBA sportsbook moneyline odds before model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from data.seasons import (  # noqa: E402
    MIN_TRAINING_SPORTSBOOK_MATCH_RATE,
    TRAIN_END_SEASON,
    TRAIN_START_SEASON,
    build_free_odds_split_plan,
    nba_season_display_label,
    season_start_year_from_dates,
)
from data.sportsbook_odds import (  # noqa: E402
    add_sportsbook_game_keys,
    normalize_sportsbook_odds,
    select_closing_odds,
    sportsbook_match_report_by_season,
)
from data.team_aliases import CURRENT_TEAM_ABBRS, normalize_team_abbr  # noqa: E402


REQUIRED_COLUMNS = ["game_date", "home_team", "away_team", "home_moneyline", "away_moneyline"]
OPTIONAL_COLUMNS = ["sportsbook", "timestamp", "is_closing", "spread", "total"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate data/raw/sportsbook/nba_moneyline_odds.csv.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--games-path", default=None)
    parser.add_argument("--report-path", default=str(PROJECT_ROOT / "outputs" / "sportsbook_match_report.csv"))
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def _read_games(path: Path | None) -> tuple[pd.DataFrame, Path | None]:
    candidates = [path] if path else []
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "reports" / "all_game_predictions.csv",
            PROJECT_ROOT / "data" / "interim" / "nba_games.csv",
            PROJECT_ROOT / "data" / "processed" / "modeling_dataset.parquet",
            PROJECT_ROOT / "data" / "interim" / "nba_games.parquet",
        ]
    )
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".parquet":
                frame = pd.read_parquet(candidate)
            else:
                frame = pd.read_csv(candidate, low_memory=False)
        except Exception as exc:
            print(f"WARNING: Could not load NBA games from {candidate}: {exc}")
            continue
        if {"game_date", "home_team_abbr", "away_team_abbr"}.issubset(frame.columns):
            return frame, candidate
    return pd.DataFrame(), None


def _duplicate_game_count(frame: pd.DataFrame) -> int:
    keyed = add_sportsbook_game_keys(frame)
    return int(keyed.duplicated("game_key").sum())


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    input_path = Path(args.input_path) if args.input_path else resolve_project_path(config.data.sportsbook_odds_path)
    report_path = Path(args.report_path)
    games_path = Path(args.games_path) if args.games_path else None

    errors: list[str] = []
    warnings: list[str] = []

    print(f"Expected sportsbook schema: {PROJECT_ROOT / 'data' / 'raw' / 'sportsbook' / 'nba_moneyline_odds.schema.csv'}")
    print(f"Validating sportsbook odds file: {input_path}")

    if not input_path.exists():
        errors.append(f"File does not exist: {input_path}")
        _print_header("Validation failed")
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    raw = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded sportsbook rows: {len(raw):,}")
    print(f"Columns: {list(raw.columns)}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    for column in OPTIONAL_COLUMNS:
        if column not in raw.columns:
            warnings.append(f"Optional column missing: {column}")

    if missing_columns:
        _print_header("Validation failed")
        for warning in warnings:
            print(f"WARNING: {warning}")
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    parsed_dates = pd.to_datetime(raw["game_date"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates:
        errors.append(f"{invalid_dates:,} rows have unparseable game_date values.")

    for column in ["home_moneyline", "away_moneyline"]:
        numeric = pd.to_numeric(raw[column], errors="coerce")
        invalid = int(numeric.isna().sum())
        if invalid:
            errors.append(f"{invalid:,} rows have nonnumeric {column} values.")

    for column in ["home_team", "away_team"]:
        missing = int(raw[column].isna().sum() + raw[column].astype(str).str.strip().eq("").sum())
        if missing:
            errors.append(f"{missing:,} rows have missing or empty {column} values.")

    home_abbr = raw["home_team"].map(normalize_team_abbr)
    away_abbr = raw["away_team"].map(normalize_team_abbr)
    unknown_teams = sorted(
        set(home_abbr[~home_abbr.isin(CURRENT_TEAM_ABBRS)].dropna())
        | set(away_abbr[~away_abbr.isin(CURRENT_TEAM_ABBRS)].dropna())
    )
    if unknown_teams:
        warnings.append(f"Unrecognized normalized team names: {unknown_teams}")

    normalized = pd.DataFrame()
    selected = pd.DataFrame()
    try:
        normalized = normalize_sportsbook_odds(raw)
        selected = select_closing_odds(normalized)
    except Exception as exc:
        errors.append(f"Could not normalize sportsbook odds: {exc}")

    if not normalized.empty:
        duplicate_count = _duplicate_game_count(normalized)
        if duplicate_count:
            warnings.append(
                f"{duplicate_count:,} duplicate game rows found before closing-odds selection. "
                "This is acceptable for multiple books/timestamps, but selected rows should be reviewed."
            )

        season_counts = normalized.groupby("season")["game_key"].count().astype(int).to_dict()
        _print_header("Sportsbook season counts")
        for season, count in season_counts.items():
            print(f"{nba_season_display_label(int(season))}: {count:,} sportsbook rows")

    games, loaded_games_path = _read_games(games_path)
    if games.empty:
        warnings.append("Could not load an NBA game dataset for team/date matching.")
        report = pd.DataFrame()
    else:
        print(f"\nLoaded NBA games from: {loaded_games_path}")
        nba_games = add_sportsbook_game_keys(games)
        nba_keys = set(nba_games["game_key"].dropna())
        sportsbook_keys = set(add_sportsbook_game_keys(selected)["game_key"].dropna()) if not selected.empty else set()
        unmatched_sportsbook = sorted(sportsbook_keys - nba_keys)
        if unmatched_sportsbook:
            warnings.append(f"{len(unmatched_sportsbook):,} selected sportsbook games do not match the NBA dataset.")
        report = sportsbook_match_report_by_season(nba_games, selected)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
        print(f"Saved sportsbook match report to: {report_path}")
        if not report.empty:
            _print_header("Match report")
            print(report.to_string(index=False))

            split_plan = build_free_odds_split_plan(report, mode=config.data.free_odds_split_mode)
            train_seasons = [int(season) for season in split_plan["train_seasons"]]
            validation_season = split_plan["validation_season"]
            print(f"\nFree odds split mode: {config.data.free_odds_split_mode}")
            print(
                "Training seasons: "
                + (
                    f"{nba_season_display_label(min(train_seasons))} through {nba_season_display_label(max(train_seasons))}"
                    if train_seasons
                    else "none"
                )
            )
            print(
                "Validation season: "
                + (nba_season_display_label(int(validation_season)) if validation_season is not None else "none")
            )
            if split_plan["partial_validation_warning"]:
                warnings.append(str(split_plan["partial_validation_warning"]))
            adjusted_train_start = min(train_seasons) if train_seasons else TRAIN_START_SEASON
            adjusted_train_end = max(train_seasons) if train_seasons else TRAIN_END_SEASON
            train_report = report[
                report["season"].between(adjusted_train_start, adjusted_train_end, inclusive="both")
            ]
            low_training = train_report[train_report["match_rate"].lt(MIN_TRAINING_SPORTSBOOK_MATCH_RATE)]
            if not low_training.empty:
                warnings.append(
                    "Historical sportsbook coverage is too low for reliable model training. "
                    f"Low seasons: {', '.join(low_training['season_label'].astype(str))}"
                )

    missing_required_values = {
        column: int(raw[column].isna().sum())
        for column in REQUIRED_COLUMNS
        if column in raw.columns and int(raw[column].isna().sum()) > 0
    }
    if missing_required_values:
        errors.append(f"Missing values in required columns: {missing_required_values}")

    _print_header("Validation result")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print("Sportsbook odds validation failed.")
        return 1
    print("Sportsbook odds validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
