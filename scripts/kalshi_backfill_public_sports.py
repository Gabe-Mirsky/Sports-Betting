"""Backfill raw public Kalshi Sports/NBA series, events, and markets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_client import KalshiAPIClient  # noqa: E402
from data.kalshi_public_backfill import backfill_public_sports_nba_markets  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw-first public Kalshi Sports/NBA market backfill.")
    parser.add_argument("--category", default="Sports")
    parser.add_argument("--event-status", default="settled", help="Use all/any/omit to omit the status filter.")
    parser.add_argument("--market-status", default="settled", help="Use all/any/omit to omit the status filter.")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--series-ticker", action="append", default=None)
    parser.add_argument("--max-events-per-series", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-historical-markets", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    result = backfill_public_sports_nba_markets(
        client=KalshiAPIClient.from_env(timeout=args.timeout),
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        category=args.category,
        event_status=args.event_status,
        market_status=args.market_status,
        include_historical_markets=not args.skip_historical_markets,
        force=args.force,
        max_pages=args.max_pages,
        series_tickers=args.series_ticker,
        max_events_per_series=args.max_events_per_series,
        sleep_seconds=args.sleep_seconds,
    )
    summary = result["summary"]
    print(f"Sports series: {summary['sports_series']:,}")
    print(f"NBA candidate series: {summary['nba_series']:,}")
    print(f"NBA events: {summary['nba_events']:,}")
    print(f"NBA markets: {summary['nba_markets']:,}")
    print(f"Possible NBA markets: {summary['possible_nba_markets']:,}")
    print(f"Raw JSON cache: {summary['raw_json_dir']}")
    print("Output paths:")
    print(json.dumps(summary["paths"], indent=2))


if __name__ == "__main__":
    main()
