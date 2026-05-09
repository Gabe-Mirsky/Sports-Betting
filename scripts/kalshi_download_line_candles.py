"""Download Kalshi candles for direct NBA spread and total markets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_client import KalshiAPIClient  # noqa: E402
from data.line_market_candles import download_line_market_candles_from_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pregame candles for direct NBA line markets.")
    parser.add_argument("--markets-path", default=None)
    parser.add_argument("--taxonomy-path", default=None)
    parser.add_argument("--prices-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--candle-dir", default=None)
    parser.add_argument(
        "--categories",
        default="spread_handicap,total_points_over_under",
        help="Comma-separated market categories to download.",
    )
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    client = KalshiAPIClient.from_env(timeout=args.timeout)
    prices, summary = download_line_market_candles_from_files(
        markets_path=args.markets_path or PROJECT_ROOT / "data" / "raw" / "kalshi" / "underlying_nba_leg_markets.csv",
        taxonomy_path=args.taxonomy_path or PROJECT_ROOT / "data" / "processed" / "kalshi_market_taxonomy.csv",
        client=client,
        categories=categories,
        max_markets=args.max_markets,
        force=args.force,
        candle_dir=args.candle_dir or PROJECT_ROOT / "data" / "raw" / "kalshi" / "line_candles",
        prices_path=args.prices_path or PROJECT_ROOT / "data" / "processed" / "kalshi_line_pregame_prices.csv",
        summary_path=args.summary_path or PROJECT_ROOT / "data" / "reports" / "kalshi_line_candle_summary.json",
    )

    print(f"Line markets considered: {summary.get('candidate_markets', 0):,}")
    print(f"Candle files already cached: {summary.get('cached_before', 0):,}")
    print(f"Candle files downloaded: {summary.get('downloaded', 0):,}")
    print(f"Candle downloads failed: {summary.get('failed', 0):,}")
    print(f"Usable 60-minute line prices: {summary.get('usable_60m_rows', 0):,}")
    print(f"Snapshot rows saved: {len(prices):,}")
    print(f"Saved line pregame prices to: {summary.get('price_path')}")


if __name__ == "__main__":
    main()
