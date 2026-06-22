"""Import the zachht basketball odds-history dataset into normalized snapshots.

Writes:
    data/processed/basketball_odds_snapshots_normalized.csv
    data/reports/zachht_odds_import_summary.json

Game-level odds only (market_type=game_market), NOT player props. Ingestion only -
no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kaggle_dataset_profiler import resolve_dataset_path  # noqa: E402
from data.zachht_importer import import_zachht_odds  # noqa: E402


DEFAULT_SLUG = "zachht/wnba-odds-history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the zachht basketball odds dataset.")
    parser.add_argument("--path", default=None, help="Local dataset folder. Overrides the slug.")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--download", action="store_true", help="Download via kagglehub if not local.")
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

    summary = import_zachht_odds(dataset_dir, processed_dir, reports_dir)
    print(f"zachht import ({summary['source_role']}); market_type={summary['market_type']} has_player_props={summary['has_player_props']}")
    print(f"  snapshots: {summary['snapshot_rows']:,} | unique games: {summary['unique_games']:,} | by league: {summary['by_league']}")
    print(f"  snapshot time range: {summary['snapshot_time_range']}")
    print(f"  wrote summary: {reports_dir / 'zachht_odds_import_summary.json'}")


if __name__ == "__main__":
    main()
