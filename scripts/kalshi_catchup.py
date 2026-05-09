"""Catch up Kalshi markets, matches, and candlesticks after manual gaps."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from data.game_times import add_game_start_times, download_game_start_times_for_games  # noqa: E402
from data.kalshi_backfill import backfill_all_markets  # noqa: E402
from data.kalshi_candles import download_candles_for_matches  # noqa: E402
from data.kalshi_client import KalshiAPIClient  # noqa: E402
from data.kalshi_discovery import write_broad_nba_market_discovery  # noqa: E402
from data.kalshi_matcher import match_games_to_kalshi_markets, save_match_outputs  # noqa: E402
from data.kalshi_taxonomy import write_market_taxonomy_outputs  # noqa: E402
from data.loaders import load_game_level_dataset  # noqa: E402
from data.state import load_sync_state, mark_successful_run, save_sync_state  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from reports.coverage import (  # noqa: E402
    build_kalshi_coverage_report,
    build_kalshi_gap_report,
    save_kalshi_coverage_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Catch up Kalshi market and candle data without always-on polling.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--series-ticker", default="KXNBAGAME")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--broad-max-pages", type=int, default=20)
    parser.add_argument("--skip-broad-discovery", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--fetch-game-times", action="store_true")
    parser.add_argument("--force-candles", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _date_from_state_or_default(state: dict[str, object]) -> str:
    last_market_backfill = state.get("last_market_backfill_ts")
    if last_market_backfill:
        timestamp = pd.to_datetime(last_market_backfill, errors="coerce", utc=True)
        if pd.notna(timestamp):
            return timestamp.date().isoformat()
    return "2023-10-01"


def _price_counts(prices: pd.DataFrame) -> tuple[int, int]:
    if prices.empty:
        return 0, 0
    usable = int((prices["price_quality"] != "missing").sum())
    missing = int((prices["price_quality"] == "missing").sum())
    return usable, missing


def _load_games(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        games = pd.read_csv(path, dtype={"game_id": str})
        if "actual_home_win" in games.columns and "home_win" not in games.columns:
            games["home_win"] = games["actual_home_win"]
        return games
    return load_game_level_dataset(path)


def _load_default_game_sources() -> pd.DataFrame:
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
            frames.append(_load_games(candidate))
        except RuntimeError as exc:
            print(f"Skipping unreadable game source {candidate}: {exc}")
    if not frames:
        return pd.DataFrame()
    games = pd.concat(frames, ignore_index=True)
    if "game_id" in games.columns:
        games = games.drop_duplicates(subset=["game_id"], keep="last")
    return games.reset_index(drop=True)


def _add_available_game_times(games: pd.DataFrame, fetch_missing: bool = False) -> pd.DataFrame:
    game_times_path = PROJECT_ROOT / "data" / "interim" / "nba_game_start_times.csv"
    starts = pd.DataFrame()
    if game_times_path.exists():
        starts = pd.read_csv(game_times_path)
    elif fetch_missing:
        starts = download_game_start_times_for_games(games, output_path=game_times_path, sleep_seconds=0.1)
    if starts.empty:
        return games
    enriched = add_game_start_times(games, starts)
    if fetch_missing and "game_start_time" in enriched.columns:
        missing = enriched[enriched["game_start_time"].isna()].copy()
        if not missing.empty:
            refreshed = download_game_start_times_for_games(missing, output_path=game_times_path, sleep_seconds=0.1)
            starts = pd.concat([starts, refreshed], ignore_index=True).drop_duplicates(
                subset=["game_date", "home_team_abbr", "away_team_abbr"],
                keep="last",
            )
            starts.to_csv(game_times_path, index=False)
            enriched = add_game_start_times(games, starts)
    return enriched


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)
    state = load_sync_state()

    start_date = args.start_date or _date_from_state_or_default(state)
    end_date = args.end_date or datetime.now().date().isoformat()
    extra_params = {"max_pages": args.max_pages}
    if args.series_ticker:
        extra_params["series_ticker"] = args.series_ticker

    client = KalshiAPIClient.from_env(timeout=args.timeout)
    games = _load_default_game_sources()
    if games.empty:
        raise SystemExit("No readable NBA game source found. Run scripts/build_features.py or scripts/predict_upcoming.py first.")
    games = _add_available_game_times(games, fetch_missing=args.fetch_game_times)
    possible_markets = backfill_all_markets(
        start_date,
        end_date,
        client=client,
        extra_params=extra_params,
        nba_games_df=games,
        use_targeted_ticker_backfill=True,
    )
    broad_taxonomy = pd.DataFrame()
    broad_summary: dict[str, object] = {}
    if not args.skip_broad_discovery:
        _, broad_taxonomy, broad_summary = write_broad_nba_market_discovery(
            start_date=start_date,
            end_date=end_date,
            client=client,
            max_pages=args.broad_max_pages,
            replace_cache=False,
            include_historical=True,
        )
    taxonomy, taxonomy_summary = write_market_taxonomy_outputs()
    matches = match_games_to_kalshi_markets(
        games,
        possible_markets,
        auto_match_threshold=config.matching.auto_match_threshold,
        review_match_threshold=config.matching.review_match_threshold,
        search_days_before=config.kalshi.market_search_days_before_game,
        search_days_after=config.kalshi.market_search_days_after_game,
    )
    save_match_outputs(matches)

    auto_matches = matches[matches["match_status"] == "auto_matched"].copy()
    prices = download_candles_for_matches(
        auto_matches,
        games,
        client=client,
        force=args.force_candles,
    )
    updated_state = mark_successful_run(state)
    if not possible_markets.empty:
        updated_state["failed_market_tickers"] = []
    if not prices.empty:
        missing_market_tickers = sorted(
            prices.loc[prices["price_quality"] == "missing", "market_ticker"].dropna().astype(str).unique()
        )
        updated_state["failed_candle_tickers"] = missing_market_tickers
    save_sync_state(updated_state)
    coverage_summary, coverage_monthly = build_kalshi_coverage_report(PROJECT_ROOT)
    gap_report = build_kalshi_gap_report(PROJECT_ROOT)
    save_kalshi_coverage_report(coverage_summary, coverage_monthly, gap_report=gap_report)

    auto_count = int((matches["match_status"] == "auto_matched").sum())
    review_count = int((matches["match_status"] == "needs_review").sum())
    no_match_count = int((matches["match_status"] == "no_match").sum())
    usable_prices, missing_prices = _price_counts(prices)

    print(f"Catch-up window: {start_date} to {end_date}")
    print(f"NBA games loaded: {len(games):,}")
    print(f"Possible NBA Kalshi markets: {len(possible_markets):,}")
    if not args.skip_broad_discovery:
        print(f"Broad NBA markets classified: {len(broad_taxonomy):,}")
        print(f"Broad NBA category counts: {broad_summary.get('category_counts', {})}")
    print(f"Taxonomy markets classified: {len(taxonomy):,}")
    print(f"Taxonomy category counts: {taxonomy_summary.get('category_counts', {})}")
    print(f"Auto matches: {auto_count:,}")
    print(f"Needs review: {review_count:,}")
    print(f"No match: {no_match_count:,}")
    print(f"Pregame price rows usable: {usable_prices:,}")
    print(f"Pregame price rows missing: {missing_prices:,}")
    print(f"Sync state updated: {PROJECT_ROOT / 'data' / 'kalshi_sync_state.json'}")
    print(f"Coverage summary updated: {PROJECT_ROOT / 'data' / 'reports' / 'kalshi_coverage_summary.json'}")
    print("Because candles are historical, this can recover prices even when the script was not running live.")
    print("No real trades were placed.")


if __name__ == "__main__":
    main()
