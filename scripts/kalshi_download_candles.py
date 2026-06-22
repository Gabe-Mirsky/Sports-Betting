"""Download Kalshi candlesticks for auto-matched NBA game markets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_candles import download_candles_for_matches  # noqa: E402
from data.kalshi_client import KalshiAPIClient  # noqa: E402
from data.game_times import add_game_start_times, download_game_start_times_for_games  # noqa: E402
from data.loaders import load_game_level_dataset  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download candlesticks and extract pregame Kalshi prices.")
    parser.add_argument("--matches-path", default=None)
    parser.add_argument("--games-path", default=None)
    parser.add_argument("--game-times-path", default=None)
    parser.add_argument("--fetch-game-times", action="store_true")
    parser.add_argument("--refresh-game-times", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _price_summary(prices: pd.DataFrame) -> dict[str, int]:
    if prices.empty:
        return {"usable_60m": 0, "usable_best_le_120m": 0, "fallback_only": 0, "missing": 0}
    usable = prices[prices["price_quality"] != "missing"].copy()
    usable_60m = int(((usable["snapshot_target"] == "pregame_60m") & (usable["period_interval"] == 1)).sum())
    usable_best = int((usable["snapshot_target"] == "pregame_best_le_120m").sum())
    fallback_only = int((usable["period_interval"].astype(str).isin(["60", "1440"])).sum())
    missing = int((prices["price_quality"] == "missing").sum())
    return {"usable_60m": usable_60m, "usable_best_le_120m": usable_best, "fallback_only": fallback_only, "missing": missing}


def _load_games(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"game_id": str})
    return load_game_level_dataset(path)


def _load_default_games() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in [
        PROJECT_ROOT / "data" / "interim" / "nba_games.parquet",
        PROJECT_ROOT / "data" / "reports" / "all_game_predictions.csv",
        PROJECT_ROOT / "data" / "reports" / "upcoming_predictions.csv",
    ]:
        if not path.exists():
            continue
        try:
            frames.append(_load_games(path))
        except RuntimeError as exc:
            print(f"Skipping unreadable game source {path}: {exc}")
    if not frames:
        return pd.DataFrame()
    games = pd.concat(frames, ignore_index=True)
    if "game_id" in games.columns:
        games = games.drop_duplicates(subset=["game_id"], keep="last")
    return games.reset_index(drop=True)


def _add_available_game_times(
    games: pd.DataFrame,
    args: argparse.Namespace,
    fetch_scope_games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    game_times_path = (
        Path(args.game_times_path)
        if args.game_times_path
        else PROJECT_ROOT / "data" / "interim" / "nba_game_start_times.csv"
    )
    starts = pd.DataFrame()
    source_games = fetch_scope_games if fetch_scope_games is not None and not fetch_scope_games.empty else games
    if args.refresh_game_times:
        starts = download_game_start_times_for_games(source_games, output_path=game_times_path, sleep_seconds=0.1)
    elif game_times_path.exists():
        starts = pd.read_csv(game_times_path)
    elif args.fetch_game_times:
        starts = download_game_start_times_for_games(source_games, output_path=game_times_path, sleep_seconds=0.1)
    if starts.empty:
        return games
    if "game_time_source" in starts.columns:
        source_counts = starts["game_time_source"].fillna("unknown").astype(str).value_counts().to_dict()
        print(f"Game start-time sources: {source_counts}")
    enriched = add_game_start_times(games, starts)
    if args.fetch_game_times and "game_start_time" in enriched.columns:
        missing = enriched[enriched["game_start_time"].isna()].copy()
        if not missing.empty:
            missing_scope = missing
            if fetch_scope_games is not None and not fetch_scope_games.empty and "game_id" in missing.columns:
                scope_ids = set(fetch_scope_games["game_id"].astype(str))
                missing_scope = missing[missing["game_id"].astype(str).isin(scope_ids)].copy()
            if missing_scope.empty:
                return enriched
            refreshed = download_game_start_times_for_games(missing_scope, output_path=game_times_path, sleep_seconds=0.1)
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
    matches_path = (
        Path(args.matches_path)
        if args.matches_path
        else PROJECT_ROOT / "data" / "processed" / "kalshi_game_market_matches.csv"
    )
    games_path = Path(args.games_path) if args.games_path else None
    if not matches_path.exists():
        raise SystemExit(f"Matches file not found: {matches_path}. Run scripts/kalshi_match_games.py first.")
    if games_path is not None and not games_path.exists():
        raise SystemExit(f"NBA games file not found: {games_path}. Run scripts/build_features.py first.")

    matches = pd.read_csv(matches_path, dtype={"game_id": str})
    games = _load_games(games_path) if games_path else _load_default_games()
    if games.empty:
        raise SystemExit("No readable NBA game source found. Run scripts/build_features.py or scripts/predict_upcoming.py first.")
    auto_matches = matches[matches["match_status"] == "auto_matched"].copy()
    matched_game_ids = set(auto_matches["game_id"].dropna().astype(str))
    fetch_scope_games = (
        games[games["game_id"].astype(str).isin(matched_game_ids)].copy()
        if "game_id" in games.columns and matched_game_ids
        else games
    )
    games = _add_available_game_times(games, args, fetch_scope_games=fetch_scope_games)
    candle_dir = PROJECT_ROOT / "data" / "raw" / "kalshi" / "candles"
    cached_before = {
        str(path.stem)
        for pattern in ["*.parquet", "*.csv"]
        for path in candle_dir.glob(pattern)
    } if candle_dir.exists() else set()

    prices = download_candles_for_matches(
        auto_matches,
        games,
        client=KalshiAPIClient.from_env(timeout=args.timeout),
        force=args.force,
    )
    cached_after = {
        str(path.stem)
        for pattern in ["*.parquet", "*.csv"]
        for path in candle_dir.glob(pattern)
    } if candle_dir.exists() else set()
    matched_tickers = set(auto_matches["market_ticker"].dropna().astype(str))
    downloaded_count = len((cached_after - cached_before) & matched_tickers)
    cached_count = len(cached_before & matched_tickers)
    failed_count = len(matched_tickers - cached_after)
    summary = _price_summary(prices)

    print(f"Matched markets: {len(auto_matches):,}")
    print(f"Candle files already cached: {cached_count:,}")
    print(f"Downloaded: {downloaded_count:,}")
    print(f"Failed: {failed_count:,}")
    print(f"Usable 60-minute pregame price: {summary['usable_60m']:,}")
    print(f"Usable best <=120-minute pregame price: {summary['usable_best_le_120m']:,}")
    print(f"Only fallback price: {summary['fallback_only']:,}")
    print(f"Missing price: {summary['missing']:,}")
    print(f"Saved pregame prices to: {PROJECT_ROOT / 'data' / 'processed' / 'kalshi_pregame_prices.csv'}")


if __name__ == "__main__":
    main()
