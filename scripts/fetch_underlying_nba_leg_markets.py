"""Fetch direct Kalshi market rows for NBA legs found inside combo markets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_client import KalshiAPIClient  # noqa: E402
from data.underlying_leg_markets import fetch_underlying_leg_markets_from_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch direct market rows for NBA combo legs.")
    parser.add_argument("--legs-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--requests-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--include-game-winners", action="store_true")
    parser.add_argument("--max-tickers", type=int, default=25)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Stop early after this many consecutive missing or failed ticker fetches.",
    )
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legs_path = (
        Path(args.legs_path)
        if args.legs_path
        else PROJECT_ROOT / "data" / "processed" / "kalshi_multivariate_nba_legs.csv"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else PROJECT_ROOT / "data" / "raw" / "kalshi" / "underlying_nba_leg_markets.csv"
    )
    requests_path = (
        Path(args.requests_path)
        if args.requests_path
        else PROJECT_ROOT / "data" / "reports" / "underlying_nba_leg_market_requests.csv"
    )
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else PROJECT_ROOT / "data" / "reports" / "underlying_nba_leg_market_summary.json"
    )
    client = KalshiAPIClient.from_env(timeout=args.timeout)
    markets, requests, summary = fetch_underlying_leg_markets_from_files(
        legs_path=legs_path,
        output_path=output_path,
        requests_path=requests_path,
        summary_path=summary_path,
        client=client,
        include_game_winners=args.include_game_winners,
        max_tickers=args.max_tickers,
        max_consecutive_failures=args.max_consecutive_failures,
    )

    print(f"Candidate direct leg tickers attempted: {summary.get('candidate_tickers', 0):,}")
    print(f"Actual ticker requests made: {summary.get('attempted_tickers', 0):,}")
    print(f"Fetched direct market rows this run: {summary.get('fetched_rows', 0):,}")
    print(f"Cached direct market rows: {len(markets):,}")
    print(f"Missing tickers: {summary.get('missing_tickers', 0):,}")
    if summary.get("stopped_after_consecutive_failures"):
        print(
            "Stopped early after consecutive failures. "
            "This usually means the local environment blocked network calls or Kalshi has no direct rows for those legs."
        )
    print(f"Saved direct market cache to: {output_path}")
    print(f"Saved request log to: {requests_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
