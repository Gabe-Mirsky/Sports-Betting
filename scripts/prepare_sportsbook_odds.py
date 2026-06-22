"""Normalize historical sportsbook odds for NBA market-proxy backtests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from data.seasons import (  # noqa: E402
    build_free_odds_split_plan,
    nba_season_display_label,
)
from data.sportsbook_odds import (  # noqa: E402
    load_sportsbook_odds,
    select_closing_odds,
    sportsbook_coverage_by_season,
    sportsbook_match_report_by_season,
)
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare historical sportsbook odds for NBA model evaluation.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--games-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--coverage-output-path", default=None)
    parser.add_argument("--match-report-path", default=None)
    parser.add_argument("--split-output-path", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _read_games(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    input_path = Path(args.input_path) if args.input_path else resolve_project_path(config.data.sportsbook_odds_path)
    output_path = Path(args.output_path) if args.output_path else PROJECT_ROOT / "data" / "processed" / "sportsbook_odds.csv"
    coverage_output_path = (
        Path(args.coverage_output_path)
        if args.coverage_output_path
        else PROJECT_ROOT / "data" / "reports" / "sportsbook_coverage_by_season.csv"
    )
    match_report_path = (
        Path(args.match_report_path)
        if args.match_report_path
        else PROJECT_ROOT / "outputs" / "sportsbook_match_report.csv"
    )
    split_output_path = (
        Path(args.split_output_path)
        if args.split_output_path
        else PROJECT_ROOT / "data" / "processed" / "sportsbook_split_config.json"
    )
    games_path = Path(args.games_path) if args.games_path else PROJECT_ROOT / "data" / "reports" / "all_game_predictions.csv"

    odds = select_closing_odds(load_sportsbook_odds(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(output_path, index=False)

    games = _read_games(games_path)
    coverage = sportsbook_coverage_by_season(games, odds)
    coverage_output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_output_path, index=False)
    match_report = sportsbook_match_report_by_season(games, odds)
    match_report_path.parent.mkdir(parents=True, exist_ok=True)
    match_report.to_csv(match_report_path, index=False)

    split_plan = build_free_odds_split_plan(match_report, mode=config.data.free_odds_split_mode)
    train_seasons = [int(season) for season in split_plan["train_seasons"]]
    validation_season = split_plan["validation_season"]
    split_config = {
        "free_odds_split_mode": config.data.free_odds_split_mode,
        "train_seasons": train_seasons,
        "train_start_season": int(min(train_seasons)) if train_seasons else None,
        "train_end_season": int(max(train_seasons)) if train_seasons else None,
        "validation_season": int(validation_season) if validation_season is not None else None,
        "test_season": int(split_plan["test_season"]),
        "validation_match_rate": split_plan["validation_match_rate"],
        "excluded_due_to_missing_odds": split_plan["excluded_due_to_missing_odds"],
        "partial_validation_warning": split_plan["partial_validation_warning"],
        "season_splits": {str(key): value for key, value in split_plan["season_splits"].items()},
        "reason": "Adjusted from actual free Kaggle sportsbook match coverage; seasons with no usable odds are excluded from train/validation.",
    }
    split_output_path.parent.mkdir(parents=True, exist_ok=True)
    split_output_path.write_text(json.dumps(split_config, indent=2), encoding="utf-8")

    print(f"Loaded sportsbook odds from: {input_path}")
    print(f"Prepared {len(odds):,} selected sportsbook game rows.")
    print(f"Saved normalized odds to: {output_path}")
    print(f"Saved sportsbook season coverage to: {coverage_output_path}")
    print(f"Saved sportsbook match report to: {match_report_path}")
    print(
        "Free-odds split mode: "
        f"{config.data.free_odds_split_mode}"
    )
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
        print(f"WARNING: {split_plan['partial_validation_warning']}")
    print(f"Saved split config to: {split_output_path}")


if __name__ == "__main__":
    main()
