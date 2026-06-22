"""Match NBA games to filtered Kalshi NBA markets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from data.kalshi_backfill import filter_possible_nba_markets  # noqa: E402
from data.kalshi_matcher import match_games_to_kalshi_markets, save_match_outputs  # noqa: E402
from data.loaders import load_game_level_dataset  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match NBA games to possible Kalshi NBA markets.")
    parser.add_argument("--games-path", default=None)
    parser.add_argument("--markets-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--review-output-path", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_markets(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except (ImportError, ValueError, RuntimeError):
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise


def _default_markets_path() -> Path:
    csv_path = PROJECT_ROOT / "data" / "processed" / "kalshi_possible_nba_markets.csv"
    parquet_path = PROJECT_ROOT / "data" / "processed" / "kalshi_possible_nba_markets.parquet"
    public_path = PROJECT_ROOT / "data" / "processed" / "kalshi_public_possible_nba_markets.csv"
    if public_path.exists():
        return public_path
    if csv_path.exists():
        return csv_path
    if parquet_path.exists():
        return parquet_path
    return public_path


def _load_games(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        games = pd.read_csv(path, dtype={"game_id": str})
        if "actual_home_win" in games.columns and "home_win" not in games.columns:
            games["home_win"] = games["actual_home_win"]
        return games
    return load_game_level_dataset(path)


def _load_default_games() -> pd.DataFrame:
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


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    games_path = Path(args.games_path) if args.games_path else None
    markets_path = (
        Path(args.markets_path)
        if args.markets_path
        else _default_markets_path()
    )
    if games_path is not None and not games_path.exists():
        raise SystemExit(f"NBA games file not found: {games_path}. Run scripts/build_features.py first.")
    if not markets_path.exists():
        raise SystemExit(
            f"Kalshi markets file not found: {markets_path}. Run scripts/kalshi_backfill_markets.py first."
        )

    games = _load_games(games_path) if games_path is not None else _load_default_games()
    if games.empty:
        raise SystemExit("No readable NBA game source found. Run scripts/build_features.py or scripts/predict_upcoming.py first.")
    loaded_markets = _load_markets(markets_path)
    possible_markets = filter_possible_nba_markets(loaded_markets)
    matches = match_games_to_kalshi_markets(
        games,
        possible_markets,
        auto_match_threshold=config.matching.auto_match_threshold,
        review_match_threshold=config.matching.review_match_threshold,
        search_days_before=config.kalshi.market_search_days_before_game,
        search_days_after=config.kalshi.market_search_days_after_game,
    )
    save_match_outputs(
        matches,
        matches_path=Path(args.output_path) if args.output_path else None,
        review_path=Path(args.review_output_path) if args.review_output_path else None,
    )

    auto_count = int((matches["match_status"] == "auto_matched").sum())
    review_count = int((matches["match_status"] == "needs_review").sum())
    no_match_count = int((matches["match_status"] == "no_match").sum())

    print(f"NBA games loaded: {len(games):,}")
    print(f"Kalshi markets loaded: {len(loaded_markets):,}")
    print(f"Possible NBA markets: {len(possible_markets):,}")
    print(f"Auto matches: {auto_count:,}")
    print(f"Needs review: {review_count:,}")
    print(f"No match: {no_match_count:,}")
    print(
        "Saved matches to: "
        f"{Path(args.output_path) if args.output_path else PROJECT_ROOT / 'data' / 'processed' / 'kalshi_game_market_matches.csv'}"
    )
    print(
        "Saved review file to: "
        f"{Path(args.review_output_path) if args.review_output_path else PROJECT_ROOT / 'data' / 'processed' / 'kalshi_matches_needs_review.csv'}"
    )


if __name__ == "__main__":
    main()
