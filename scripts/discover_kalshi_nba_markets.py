"""Discover likely NBA Kalshi markets across all market series."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_client import KalshiAPIClient  # noqa: E402
from data.kalshi_discovery import (  # noqa: E402
    write_broad_nba_market_discovery,
    write_cached_broad_nba_market_taxonomy,
)
from data.kalshi_taxonomy import write_market_taxonomy_outputs  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_end = date.today()
    default_start = default_end - timedelta(days=30)
    parser = argparse.ArgumentParser(
        description=(
            "Search Kalshi markets across all series, keep likely NBA rows, "
            "and classify them into winner/spread/total/team-total/prop categories."
        )
    )
    parser.add_argument("--start-date", default=default_start.isoformat())
    parser.add_argument("--end-date", default=default_end.isoformat())
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--taxonomy-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--skip-historical", action="store_true", help="Only search the current/recent market route.")
    parser.add_argument("--replace-cache", action="store_true", help="Replace the broad discovery cache.")
    parser.add_argument("--cached-only", action="store_true", help="Only rebuild taxonomy from the local broad cache.")
    parser.add_argument(
        "--skip-main-taxonomy",
        action="store_true",
        help="Do not rebuild the combined market taxonomy after broad discovery.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    if args.cached_only:
        taxonomy, summary = write_cached_broad_nba_market_taxonomy(
            input_path=args.output_path,
            taxonomy_path=args.taxonomy_path,
            summary_path=args.summary_path,
        )
        print(f"Rebuilt cached broad taxonomy rows: {len(taxonomy):,}")
        print(f"Broad category counts: {summary.get('category_counts', {})}")
        print("No real trades were placed.")
        return

    client = KalshiAPIClient.from_env(timeout=args.timeout)
    cached, taxonomy, summary = write_broad_nba_market_discovery(
        start_date=args.start_date,
        end_date=args.end_date,
        client=client,
        max_pages=args.max_pages,
        output_path=args.output_path,
        taxonomy_path=args.taxonomy_path,
        summary_path=args.summary_path,
        replace_cache=args.replace_cache,
        include_historical=not args.skip_historical,
    )

    print(f"Discovery window: {args.start_date} to {args.end_date}")
    print(f"Likely NBA rows found this run: {summary.get('raw_discovered_rows_this_run', 0):,}")
    print(f"Recent-route rows this run: {summary.get('raw_recent_rows_this_run', 0):,}")
    print(f"Historical-route rows this run: {summary.get('raw_historical_rows_this_run', 0):,}")
    print(f"Cached broad NBA markets: {len(cached):,}")
    print(f"Broad taxonomy rows: {len(taxonomy):,}")
    print(f"Broad category counts: {summary.get('category_counts', {})}")
    print(f"Player names available for prop discovery: {summary.get('player_names_loaded', 0):,}")

    if not args.skip_main_taxonomy:
        combined_taxonomy, combined_summary = write_market_taxonomy_outputs()
        print(f"Combined taxonomy rows: {len(combined_taxonomy):,}")
        print(f"Combined category counts: {combined_summary.get('category_counts', {})}")

    print("No real trades were placed.")


if __name__ == "__main__":
    main()
