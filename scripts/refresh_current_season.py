"""Refresh the current NBA season and rebuild local reports."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.seasons import current_nba_season_start_year, nba_season_display_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh current NBA season data and reports.")
    parser.add_argument("--season-start-year", type=int, default=None)
    parser.add_argument("--regular-season-only", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    season_start_year = args.season_start_year or current_nba_season_start_year()
    label = nba_season_display_label(season_start_year)

    print(f"Refreshing NBA season {label}.")
    print(f"Season start year {season_start_year} means games from {label}.")

    season_types = ["Regular Season"]
    if not args.regular_season_only:
        season_types.append("Playoffs")

    for season_type in season_types:
        print(f"Downloading {season_type} data for {label}.")
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "download_nba_data.py"),
                "--force",
                "--start-season",
                str(season_start_year),
                "--end-season",
                str(season_start_year),
                "--season-type",
                season_type,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_full_pipeline.py"),
    ]
    if args.skip_backtest:
        command.append("--skip-backtest")
        command.append("--skip-sweep")
    if args.skip_dashboard:
        command.append("--skip-dashboard")

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
