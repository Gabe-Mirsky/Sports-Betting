"""Import the ehallmar NBA stats + betting dataset into normalized project tables.

Writes:
    data/processed/nba_games_normalized.csv
    data/processed/nba_player_game_logs_normalized.csv
    data/processed/nba_game_odds_normalized.csv
    data/reports/ehallmar_import_summary.json

This dataset has NO player-prop lines; it is a player_actuals_source. Ingestion
only - no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.ehallmar_importer import import_ehallmar  # noqa: E402
from data.kaggle_dataset_profiler import resolve_dataset_path  # noqa: E402


DEFAULT_SLUG = "ehallmar/nba-historical-stats-and-betting-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the ehallmar NBA dataset.")
    parser.add_argument("--path", default=None, help="Local dataset folder. Overrides the slug.")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--download", action="store_true", help="Download via kagglehub if not local.")
    parser.add_argument("--min-season", type=int, default=None, help="Keep only seasons >= this start year.")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir) if args.processed_dir else PROJECT_ROOT / "data" / "processed"
    reports_dir = Path(args.reports_dir) if args.reports_dir else PROJECT_ROOT / "data" / "reports"

    if args.path:
        dataset_dir = args.path
    else:
        dataset_dir, _, detail = resolve_dataset_path(args.slug, download=args.download)
        if not dataset_dir:
            raise SystemExit(f"Dataset not available ({detail}). Pass --path or --download.")

    summary = import_ehallmar(dataset_dir, processed_dir, reports_dir, min_season=args.min_season)
    print(f"ehallmar import ({summary['source_role']}); has_player_prop_lines={summary['has_player_prop_lines']}")
    print(f"  games: {summary['games']['rows']:,} | player logs: {summary['player_game_logs']['rows']:,} | odds rows: {summary['game_odds']['rows']:,}")
    print(f"  games date range: {summary['games']['date_range']}")
    print(f"  wrote summary: {reports_dir / 'ehallmar_import_summary.json'}")


if __name__ == "__main__":
    main()
