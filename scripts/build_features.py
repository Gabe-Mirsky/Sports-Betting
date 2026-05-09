"""Build the first cleaned NBA game-level dataset from cached raw logs."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from data.loaders import (  # noqa: E402
    build_game_level_dataset,
    load_raw_team_logs,
    save_game_level_dataset,
)
from data.injury_availability import load_availability_reports  # noqa: E402
from data.player_client import load_raw_player_logs  # noqa: E402
from data.validation import validate_game_level_dataset  # noqa: E402
from features.team_features import build_modeling_dataset, save_modeling_dataset  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def main() -> None:
    setup_logging()
    config = load_config()

    raw_dir = resolve_project_path(config.data.cache_dir)
    player_raw_dir = resolve_project_path(config.data.player_cache_dir)
    availability_path = resolve_project_path(config.data.availability_report_path)
    interim_path = PROJECT_ROOT / "data" / "interim" / "nba_games.parquet"
    processed_path = resolve_project_path(config.data.processed_dir) / "modeling_dataset.parquet"

    team_logs = load_raw_team_logs(raw_dir)
    player_logs = load_raw_player_logs(player_raw_dir)
    availability_reports = load_availability_reports(availability_path)
    games = build_game_level_dataset(team_logs)
    issues = validate_game_level_dataset(games)
    if issues:
        issue_text = "\n".join(f"- {issue}" for issue in issues)
        raise SystemExit(f"Game-level validation failed:\n{issue_text}")

    save_game_level_dataset(games, interim_path)
    modeling = build_modeling_dataset(
        games,
        player_logs=player_logs,
        availability_reports=availability_reports,
    )
    save_modeling_dataset(modeling, processed_path)

    print(f"Built {len(games):,} game rows.")
    print(f"Saved game-level data to: {interim_path}")
    if player_logs.empty:
        print("Player logs: none found yet, so model features are team-level only.")
    else:
        player_feature_count = len([column for column in modeling.columns if column.startswith("player_")])
        print(f"Player logs: loaded {len(player_logs):,} rows and added {player_feature_count:,} feature columns.")
    if availability_reports.empty:
        print(f"Availability reports: none found at {availability_path}.")
    else:
        availability_feature_count = len([column for column in modeling.columns if "availability_" in column])
        print(
            "Availability reports: "
            f"loaded {len(availability_reports):,} rows and added {availability_feature_count:,} feature columns."
        )
    print(f"Built {len(modeling):,} modeling rows.")
    print(f"Saved modeling data to: {processed_path}")


if __name__ == "__main__":
    main()
