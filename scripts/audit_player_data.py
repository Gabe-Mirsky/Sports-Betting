"""Audit cached NBA player data and player-derived model feature coverage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from config import load_config, resolve_project_path  # noqa: E402
from data.injury_availability import load_availability_reports  # noqa: E402
from data.player_client import load_raw_player_logs  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from reports.player_data_audit import build_player_data_audit, save_player_data_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit player data coverage in the current NBA model dataset.")
    parser.add_argument("--games-path", default=None)
    parser.add_argument("--modeling-path", default=None)
    parser.add_argument("--player-cache-dir", default=None)
    parser.add_argument("--availability-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--output-feature-coverage-path", default=None)
    parser.add_argument("--output-season-coverage-path", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    games_path = Path(args.games_path) if args.games_path else PROJECT_ROOT / "data" / "interim" / "nba_games.parquet"
    modeling_path = (
        Path(args.modeling_path)
        if args.modeling_path
        else resolve_project_path(config.data.processed_dir) / "modeling_dataset.parquet"
    )
    player_cache_dir = (
        Path(args.player_cache_dir) if args.player_cache_dir else resolve_project_path(config.data.player_cache_dir)
    )
    availability_path = (
        Path(args.availability_path)
        if args.availability_path
        else resolve_project_path(config.data.availability_report_path)
    )
    reports_dir = PROJECT_ROOT / "data" / "reports"
    summary_path = (
        Path(args.output_summary_path) if args.output_summary_path else reports_dir / "player_data_summary.json"
    )
    feature_coverage_path = (
        Path(args.output_feature_coverage_path)
        if args.output_feature_coverage_path
        else reports_dir / "player_feature_coverage.csv"
    )
    season_coverage_path = (
        Path(args.output_season_coverage_path)
        if args.output_season_coverage_path
        else reports_dir / "player_data_season_coverage.csv"
    )

    games = _read_frame(games_path)
    modeling = _read_frame(modeling_path)
    player_logs = load_raw_player_logs(player_cache_dir)
    availability_reports = load_availability_reports(availability_path)

    summary, feature_coverage, season_coverage = build_player_data_audit(
        games=games,
        modeling=modeling,
        player_logs=player_logs,
        availability_reports=availability_reports,
    )
    save_player_data_audit(
        summary,
        feature_coverage,
        season_coverage,
        summary_path,
        feature_coverage_path,
        season_coverage_path,
    )

    print(f"Player data status: {summary['status']}")
    print(f"Raw player rows: {summary['raw_player_log_rows']:,}")
    print(
        "Player feature coverage: "
        f"{summary['player_feature_row_coverage']:.1%} "
        f"({summary['player_feature_columns_present']}/{summary['expected_player_feature_columns']} columns)"
    )
    if summary["warnings"]:
        print("Warnings: " + ", ".join(summary["warnings"]))
    print(f"Saved player data summary to: {summary_path}")
    print(f"Saved feature coverage to: {feature_coverage_path}")
    print(f"Saved season coverage to: {season_coverage_path}")


if __name__ == "__main__":
    main()
