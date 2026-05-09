"""Print today's paper-trade suggestions without placing real trades."""

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
from data.kalshi_client import (  # noqa: E402
    fetch_public_nba_game_markets,
    load_mock_kalshi_markets,
    match_games_to_markets,
    validate_kalshi_markets,
)
from data.market_quality import analyze_market_data_quality, save_market_quality_report  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from strategy.signal import add_yes_signals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print manual paper-trade suggestions.")
    parser.add_argument("--markets-path", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--edge-threshold", type=float, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    markets_path = (
        Path(args.markets_path)
        if args.markets_path
        else PROJECT_ROOT / "data" / "kalshi" / "markets_today.csv"
    )
    default_upcoming_path = PROJECT_ROOT / "data" / "reports" / "upcoming_predictions.csv"
    default_walk_forward_path = PROJECT_ROOT / "data" / "reports" / "walk_forward_predictions.csv"
    default_single_split_path = PROJECT_ROOT / "data" / "reports" / "model_predictions.csv"
    predictions_path = (
        Path(args.predictions_path)
        if args.predictions_path
        else default_upcoming_path
        if default_upcoming_path.exists()
        else default_walk_forward_path
        if default_walk_forward_path.exists()
        else default_single_split_path
    )
    edge_threshold = args.edge_threshold if args.edge_threshold is not None else config.strategy.edge_threshold

    if not predictions_path.exists():
        print("No prediction file found. Run scripts\\walk_forward.py or scripts\\train.py first.")
        print(f"Expected prediction file: {predictions_path}")
        print("No real trades were placed.")
        return

    predictions = pd.read_csv(predictions_path, dtype={"game_id": str})
    today = pd.Timestamp(datetime.now().date())
    if "game_date" in predictions.columns:
        predictions["game_date"] = pd.to_datetime(predictions["game_date"], errors="coerce").dt.normalize()
        today_predictions = predictions[predictions["game_date"] == today].copy()
        if not today_predictions.empty:
            predictions = today_predictions

    if markets_path.exists():
        markets = load_mock_kalshi_markets(markets_path)
    else:
        print("No manual markets_today.csv found. Trying current public Kalshi NBA markets.")
        try:
            markets = fetch_public_nba_game_markets(statuses=["open"])
        except Exception as exc:
            print(f"Could not fetch current Kalshi markets: {exc}")
            print("You can still use mock/manual CSV mode by creating data\\kalshi\\markets_today.csv.")
            print("No real trades were placed.")
            return
        if markets.empty:
            print("No current public Kalshi NBA game markets were found.")
            print("Run scripts\\predict_upcoming.py first if you need fresh model predictions.")
            print("No real trades were placed.")
            return

    validation = validate_kalshi_markets(markets, predictions)
    if validation["missing_columns"]:
        print(f"Manual markets file is missing columns: {validation['missing_columns']}")
        print("Use data\\kalshi\\markets_template.csv or scripts\\export_market_template.py.")
        print("No real trades were placed.")
        return
    if validation["invalid_price_rows"]:
        print(f"Rows with missing or invalid YES prices: {validation['invalid_price_rows']}")
        print("Fill yes_mid_cents, or yes_bid_cents and yes_ask_cents.")

    matched = match_games_to_markets(predictions, markets)
    if matched.empty:
        print("No manual markets matched the prediction file.")
        print("Check game_date, home_team_abbr, away_team_abbr, and yes_team_abbr.")
        print("No real trades were placed.")
        return
    quality_report = analyze_market_data_quality(markets, matched)

    suggestions = add_yes_signals(
        matched,
        edge_threshold=edge_threshold,
        min_market_price=config.strategy.min_market_price,
        max_market_price=config.strategy.max_market_price,
    )
    trades = suggestions[suggestions["trade"]].copy()
    output_path = PROJECT_ROOT / "data" / "reports" / "paper_trade_suggestions.csv"
    snapshot_path = PROJECT_ROOT / "data" / "processed" / "live_snapshots.csv"
    quality_path = PROJECT_ROOT / "data" / "reports" / "market_data_quality_report.json"
    suggestions.to_csv(output_path, index=False)
    save_market_quality_report(quality_report, quality_path)
    snapshot = suggestions.copy()
    snapshot["timestamp"] = datetime.now().isoformat()
    snapshot["yes_bid"] = snapshot.get("yes_bid_cents", pd.Series(index=snapshot.index, dtype="float64"))
    snapshot["yes_ask"] = snapshot.get("yes_ask_cents", pd.Series(index=snapshot.index, dtype="float64"))
    snapshot["mid_price"] = snapshot.get("yes_mid_cents", pd.Series(index=snapshot.index, dtype="float64"))
    snapshot["edge"] = snapshot.get("edge", pd.Series(index=snapshot.index, dtype="float64"))
    snapshot["trade_signal"] = snapshot.get("reason", pd.Series(index=snapshot.index, dtype="object"))
    snapshot_columns = [
        "timestamp",
        "game_id",
        "market_ticker",
        "home_team_abbr",
        "away_team_abbr",
        "yes_team_abbr",
        "model_yes_prob",
        "yes_bid",
        "yes_ask",
        "mid_price",
        "edge",
        "trade_signal",
    ]
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot[[column for column in snapshot_columns if column in snapshot.columns]].to_csv(snapshot_path, index=False)

    if trades.empty:
        print("No paper trades met the edge threshold.")
        print(f"Saved evaluated suggestions to: {output_path}")
        print(f"Saved live snapshot to: {snapshot_path}")
        if quality_report["warnings"]:
            print("Market data quality warnings:")
            for warning in quality_report["warnings"]:
                print(f"- {warning}")
        print(f"Saved market quality report to: {quality_path}")
        print("No real trades were placed.")
        return

    display_columns = [
        column
        for column in [
            "game_date",
            "home_team_abbr",
            "away_team_abbr",
            "yes_team_abbr",
            "model_prob",
            "market_prob",
            "edge",
            "price_cents",
            "reason",
        ]
        if column in trades.columns
    ]
    print(trades[display_columns].to_string(index=False))
    print(f"Saved evaluated suggestions to: {output_path}")
    print(f"Saved live snapshot to: {snapshot_path}")
    if quality_report["warnings"]:
        print("Market data quality warnings:")
        for warning in quality_report["warnings"]:
            print(f"- {warning}")
    print(f"Saved market quality report to: {quality_path}")
    print("Paper suggestions only. No real trades were placed.")


if __name__ == "__main__":
    main()
