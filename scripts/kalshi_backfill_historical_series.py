"""Crawl old Kalshi NBA markets by historical series ticker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_client import KalshiAPIClient  # noqa: E402
from data.kalshi_series_backfill import NBA_SERIES_TICKERS, crawl_kalshi_series_markets  # noqa: E402
from data.kalshi_taxonomy import write_market_taxonomy_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical NBA Kalshi markets by series ticker.")
    parser.add_argument(
        "--series-tickers",
        default=",".join(NBA_SERIES_TICKERS),
        help="Comma-separated Kalshi series tickers to crawl.",
    )
    parser.add_argument("--include-recent", action="store_true")
    parser.add_argument("--historical-only", action="store_true")
    parser.add_argument("--replace-cache", action="store_true")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series_tickers = [item.strip().upper() for item in args.series_tickers.split(",") if item.strip()]
    include_historical = True
    include_recent = bool(args.include_recent and not args.historical_only)
    cached, possible, summary = crawl_kalshi_series_markets(
        series_tickers=series_tickers,
        client=KalshiAPIClient.from_env(timeout=args.timeout),
        include_historical=include_historical,
        include_recent=include_recent,
        max_pages=args.max_pages,
        limit=args.limit,
        append=not args.replace_cache,
    )
    taxonomy, taxonomy_summary = write_market_taxonomy_outputs()

    print(f"Historical series rows this run: {summary.get('rows_this_run', 0):,}")
    print(f"Historical series unique markets this run: {summary.get('unique_markets_this_run', 0):,}")
    print(f"Cached series markets: {summary.get('cached_unique_markets', 0):,}")
    print(f"Possible game-winner markets from series cache: {len(possible):,}")
    print(f"Taxonomy rows after crawl: {len(taxonomy):,}")
    print(f"Taxonomy category counts: {taxonomy_summary.get('category_counts', {})}")
    for row in summary.get("run_by_series", []):
        print(
            f"- {row['series_ticker']} {row['route']}: "
            f"{row['unique_markets_this_run']:,} markets"
            + (f" ({row['error']})" if row.get("error") else "")
        )
    print(f"Raw series cache: {summary.get('raw_cache_path')}")
    print(f"Possible markets cache: {summary.get('possible_markets_path')}")
    print(f"Summary: {PROJECT_ROOT / 'data' / 'reports' / 'kalshi_historical_series_backfill_summary.json'}")
    print("No real trades were placed.")


if __name__ == "__main__":
    main()
