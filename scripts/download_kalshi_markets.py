"""Download public Kalshi NBA game market snapshots without trading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from data.kalshi_client import (  # noqa: E402
    fetch_public_nba_game_markets,
    match_games_to_markets,
)
from data.market_quality import analyze_market_data_quality, save_market_quality_report  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from strategy.signal import add_yes_signals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch public Kalshi NBA game market prices and match them to model predictions."
    )
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        choices=["open", "settled"],
        help="Market status to fetch. Repeat to fetch more than one. Defaults to open.",
    )
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--suggestions-path", default=None)
    parser.add_argument("--quality-path", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _default_predictions_path() -> Path:
    upcoming = PROJECT_ROOT / "data" / "reports" / "upcoming_predictions.csv"
    all_games = PROJECT_ROOT / "data" / "reports" / "all_game_predictions.csv"
    walk_forward = PROJECT_ROOT / "data" / "reports" / "walk_forward_predictions.csv"
    if upcoming.exists():
        return upcoming
    if all_games.exists():
        return all_games
    return walk_forward


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    statuses = args.status or ["open"]
    output_path = Path(args.output_path) if args.output_path else PROJECT_ROOT / "data" / "kalshi" / "markets_live.csv"
    predictions_path = Path(args.predictions_path) if args.predictions_path else _default_predictions_path()
    suggestions_path = (
        Path(args.suggestions_path)
        if args.suggestions_path
        else PROJECT_ROOT / "data" / "reports" / "upcoming_market_suggestions.csv"
    )
    quality_path = (
        Path(args.quality_path)
        if args.quality_path
        else PROJECT_ROOT / "data" / "reports" / "live_market_quality_report.json"
    )

    markets = fetch_public_nba_game_markets(
        statuses=statuses,
        max_pages=args.max_pages,
        timeout=args.timeout,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markets.to_csv(output_path, index=False)

    print(f"Downloaded {len(markets):,} public Kalshi NBA game market rows.")
    print(f"Saved market snapshot to: {output_path}")

    if markets.empty:
        return
    if not predictions_path.exists():
        print(f"No prediction file found to match against: {predictions_path}")
        return

    predictions = pd.read_csv(predictions_path, dtype={"game_id": str})
    matched = match_games_to_markets(predictions, markets)
    quality_report = analyze_market_data_quality(markets, matched)
    save_market_quality_report(quality_report, quality_path)

    if matched.empty:
        suggestions_path.parent.mkdir(parents=True, exist_ok=True)
        matched.to_csv(suggestions_path, index=False)
        print("No public Kalshi markets matched the selected prediction file.")
        print(f"Saved empty suggestions file to: {suggestions_path}")
        print(f"Saved live market quality report to: {quality_path}")
        return

    suggestions = add_yes_signals(
        matched,
        edge_threshold=config.strategy.edge_threshold,
        min_market_price=config.strategy.min_market_price,
        max_market_price=config.strategy.max_market_price,
    )
    suggestions_path.parent.mkdir(parents=True, exist_ok=True)
    suggestions.to_csv(suggestions_path, index=False)

    paper_picks = int(suggestions["trade"].astype(bool).sum()) if "trade" in suggestions.columns else 0
    print(f"Matched {len(matched):,} market rows to predictions from: {predictions_path}")
    print(f"Paper-pick candidates using automatic rules: {paper_picks:,}")
    print(f"Saved matched suggestions to: {suggestions_path}")
    print(f"Saved live market quality report to: {quality_path}")
    print("No real trades were placed.")


if __name__ == "__main__":
    main()
