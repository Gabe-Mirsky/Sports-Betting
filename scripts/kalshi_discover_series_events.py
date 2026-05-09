"""Discover NBA Kalshi series through series and events endpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_client import KalshiAPIClient  # noqa: E402
from data.kalshi_event_discovery import discover_and_backfill_nba_series_from_series_and_events  # noqa: E402
from data.kalshi_taxonomy import write_market_taxonomy_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover NBA Kalshi series from series/events endpoints.")
    parser.add_argument("--event-max-pages", type=int, default=50)
    parser.add_argument("--market-max-pages", type=int, default=100)
    parser.add_argument("--include-all-events-scan", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = discover_and_backfill_nba_series_from_series_and_events(
        client=KalshiAPIClient.from_env(timeout=args.timeout),
        event_max_pages=args.event_max_pages,
        market_max_pages=args.market_max_pages,
        include_all_events_scan=args.include_all_events_scan,
    )
    taxonomy, taxonomy_summary = write_market_taxonomy_outputs()

    print(f"Series candidates: {summary['series_list'].get('candidate_rows', 0):,}")
    print(f"Event candidates: {summary['events_by_candidate_series'].get('candidate_event_rows', 0):,}")
    if summary.get("events_all_scan"):
        print(f"All-events NBA candidates: {summary['events_all_scan'].get('candidate_event_rows', 0):,}")
    print(f"Series crawled: {len(summary.get('series_to_crawl', [])):,}")
    print(f"Historical markets cached: {summary.get('cached_markets', 0):,}")
    print(f"Possible game-winner markets: {summary.get('possible_game_winner_markets', 0):,}")
    print(f"Taxonomy rows: {len(taxonomy):,}")
    print(f"Taxonomy category counts: {taxonomy_summary.get('category_counts', {})}")
    print(f"Summary: {PROJECT_ROOT / 'data' / 'reports' / 'kalshi_series_event_discovery_summary.json'}")
    print("No real trades were placed.")


if __name__ == "__main__":
    main()
