"""Predict upcoming NBA games from the free NBA scoreboard endpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from data.injury_availability import load_availability_reports  # noqa: E402
from data.loaders import load_game_level_dataset  # noqa: E402
from data.player_client import load_raw_player_logs  # noqa: E402
from data.scoreboard import (  # noqa: E402
    fetch_upcoming_games,
    fill_team_abbreviations_from_history,
)
from features.team_features import build_upcoming_modeling_dataset  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from models.predict import predict_game_probabilities  # noqa: E402


OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "season_type",
    "home_team_abbr",
    "away_team_abbr",
    "model_home_win_prob",
    "model_away_win_prob",
    "upcoming_status",
    "game_status_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict upcoming NBA games.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--season-type", default=None, choices=["Regular Season", "Playoffs"])
    parser.add_argument("--games-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _write_empty_predictions(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    games_path = Path(args.games_path) if args.games_path else PROJECT_ROOT / "data" / "interim" / "nba_games.parquet"
    model_path = Path(args.model_path) if args.model_path else PROJECT_ROOT / "data" / "models" / "home_win_model.joblib"
    output_path = Path(args.output_path) if args.output_path else PROJECT_ROOT / "data" / "reports" / "upcoming_predictions.csv"

    if not games_path.exists():
        raise FileNotFoundError(f"Missing game-level data: {games_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    historical_games = load_game_level_dataset(games_path)
    player_logs = load_raw_player_logs(resolve_project_path(config.data.player_cache_dir))
    availability_reports = load_availability_reports(resolve_project_path(config.data.availability_report_path))
    upcoming_games = fetch_upcoming_games(
        start_date=args.start_date,
        days=args.days,
        season_type=args.season_type,
    )
    upcoming_games = fill_team_abbreviations_from_history(upcoming_games, historical_games)

    if upcoming_games.empty:
        _write_empty_predictions(output_path)
        print("No upcoming NBA games found in the selected date range.")
        print(f"Saved empty upcoming prediction file to: {output_path}")
        return

    missing_abbr = upcoming_games["home_team_abbr"].isna() | upcoming_games["away_team_abbr"].isna()
    if missing_abbr.any():
        missing_count = int(missing_abbr.sum())
        upcoming_games = upcoming_games[~missing_abbr].copy()
        print(f"Skipped {missing_count} upcoming games because team abbreviations were unavailable.")

    if upcoming_games.empty:
        _write_empty_predictions(output_path)
        print("No upcoming games could be matched to team abbreviations.")
        print(f"Saved empty upcoming prediction file to: {output_path}")
        return

    feature_rows = build_upcoming_modeling_dataset(
        historical_games,
        upcoming_games,
        player_logs=player_logs,
        availability_reports=availability_reports,
    )
    model_bundle = joblib.load(model_path)
    predictions = predict_game_probabilities(model_bundle, feature_rows)
    metadata = feature_rows[
        [
            column
            for column in [
                "game_id",
                "upcoming_status",
                "game_status_id",
            ]
            if column in feature_rows.columns
        ]
    ].copy()
    predictions = predictions.merge(metadata, on="game_id", how="left", validate="one_to_one")
    predictions = predictions[[column for column in OUTPUT_COLUMNS if column in predictions.columns]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Saved {len(predictions):,} upcoming game predictions to: {output_path}")


if __name__ == "__main__":
    main()
