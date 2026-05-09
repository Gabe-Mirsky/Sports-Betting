"""Backfill possible Kalshi NBA markets for a historical date range."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_backfill import backfill_all_markets  # noqa: E402
from data.kalshi_client import KalshiAPIClient  # noqa: E402
from data.kalshi_taxonomy import write_market_taxonomy_outputs  # noqa: E402
from data.loaders import load_game_level_dataset  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Kalshi markets and keep likely NBA team-win rows.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--series-ticker", default="KXNBAGAME")
    parser.add_argument("--games-path", default=None)
    parser.add_argument("--no-targeted", action="store_true", help="Skip expected-ticker backfill from NBA games.")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_games_for_targeting(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"game_id": str})
    return load_game_level_dataset(path)


def load_default_game_sources() -> pd.DataFrame:
    candidates = [
        PROJECT_ROOT / "data" / "interim" / "nba_games.parquet",
        PROJECT_ROOT / "data" / "reports" / "all_game_predictions.csv",
        PROJECT_ROOT / "data" / "reports" / "upcoming_predictions.csv",
    ]
    frames: list[pd.DataFrame] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            frame = load_games_for_targeting(candidate)
        except RuntimeError as exc:
            print(f"Skipping unreadable game source {candidate}: {exc}")
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    games = pd.concat(frames, ignore_index=True)
    if "game_id" in games.columns:
        games = games.drop_duplicates(subset=["game_id"], keep="last")
    return games.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    client = KalshiAPIClient.from_env(timeout=args.timeout)
    extra_params = {"max_pages": args.max_pages}
    if args.series_ticker:
        extra_params["series_ticker"] = args.series_ticker

    nba_games = None
    if not args.no_targeted:
        if args.games_path:
            games_path = Path(args.games_path)
            if games_path.exists():
                nba_games = load_games_for_targeting(games_path)
            else:
                print(f"NBA games file not found, skipping targeted ticker backfill: {games_path}")
        else:
            nba_games = load_default_game_sources()
            if nba_games.empty:
                print("No readable NBA game source found, skipping targeted ticker backfill.")

    possible = backfill_all_markets(
        start_date=args.start_date,
        end_date=args.end_date,
        client=client,
        extra_params=extra_params,
        nba_games_df=nba_games,
        use_targeted_ticker_backfill=not args.no_targeted,
    )
    taxonomy, taxonomy_summary = write_market_taxonomy_outputs()

    live_path = PROJECT_ROOT / "data" / "raw" / "kalshi" / "live_markets.parquet"
    historical_path = PROJECT_ROOT / "data" / "raw" / "kalshi" / "historical_markets.parquet"
    possible_path = PROJECT_ROOT / "data" / "processed" / "kalshi_possible_nba_markets.parquet"
    targeted_path = PROJECT_ROOT / "data" / "raw" / "kalshi" / "targeted_nba_game_markets.parquet"
    missing_path = PROJECT_ROOT / "data" / "reports" / "kalshi_expected_markets_missing.csv"
    print(f"Possible NBA markets saved: {len(possible):,}")
    print(f"Taxonomy markets classified: {len(taxonomy):,}")
    print(f"Taxonomy category counts: {taxonomy_summary.get('category_counts', {})}")
    print(f"Raw recent markets cache: {live_path}")
    print(f"Raw historical markets cache: {historical_path}")
    if not args.no_targeted:
        print(f"Targeted ticker backfill cache: {targeted_path}")
        print(f"Expected tickers still missing: {missing_path}")
    print(f"Filtered possible NBA markets: {possible_path}")
    print("No real trades were placed.")


if __name__ == "__main__":
    main()
