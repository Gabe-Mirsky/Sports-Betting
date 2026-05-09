"""Run a fake-bankroll backtest across multiple edge thresholds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from data.kalshi_client import load_mock_kalshi_markets, match_games_to_markets  # noqa: E402
from data.market_quality import analyze_market_data_quality, save_market_quality_report  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from reports.plots import save_threshold_sweep_plot  # noqa: E402
from strategy.threshold_sweep import parse_thresholds, run_threshold_sweep  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep backtest edge thresholds.")
    parser.add_argument("--thresholds", default="0.00,0.02,0.05,0.08,0.10,0.12,0.15")
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--max-bet-fraction", type=float, default=None)
    parser.add_argument("--markets-path", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    thresholds = parse_thresholds(args.thresholds)
    starting_bankroll = args.bankroll or config.strategy.starting_bankroll
    max_bet_fraction = (
        args.max_bet_fraction
        if args.max_bet_fraction is not None
        else config.strategy.max_bet_fraction
    )

    default_walk_forward_path = PROJECT_ROOT / "data" / "reports" / "walk_forward_predictions.csv"
    default_single_split_path = PROJECT_ROOT / "data" / "reports" / "model_predictions.csv"
    predictions_path = (
        Path(args.predictions_path)
        if args.predictions_path
        else default_walk_forward_path
        if default_walk_forward_path.exists()
        else default_single_split_path
    )
    markets_path = (
        Path(args.markets_path)
        if args.markets_path
        else PROJECT_ROOT / "data" / "kalshi" / "markets_mock.csv"
    )

    predictions = pd.read_csv(predictions_path, dtype={"game_id": str})
    markets = load_mock_kalshi_markets(markets_path)
    matched = match_games_to_markets(predictions, markets)
    if matched.empty:
        raise SystemExit("No markets matched model predictions.")
    quality_report = analyze_market_data_quality(markets, matched)

    sweep = run_threshold_sweep(
        matched,
        thresholds=thresholds,
        starting_bankroll=starting_bankroll,
        max_bet_fraction=max_bet_fraction,
        min_market_price=config.strategy.min_market_price,
        max_market_price=config.strategy.max_market_price,
    )

    sweep_path = PROJECT_ROOT / "data" / "reports" / "threshold_sweep.csv"
    plot_path = PROJECT_ROOT / "data" / "reports" / "threshold_sweep.png"
    quality_path = PROJECT_ROOT / "data" / "reports" / "market_data_quality_report.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(sweep_path, index=False)
    save_threshold_sweep_plot(sweep, plot_path)
    save_market_quality_report(quality_report, quality_path)

    display = sweep[
        [
            "edge_threshold",
            "num_trades",
            "trade_timeline",
            "ending_bankroll",
            "total_return_pct",
            "win_rate",
            "max_drawdown",
            "roi_on_amount_risked",
        ]
    ].copy()
    for column in ["total_return_pct", "win_rate", "max_drawdown", "roi_on_amount_risked"]:
        display[column] = display[column] * 100.0

    print(display.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    timeline = sweep["trade_timeline"].dropna().astype(str).unique().tolist() if "trade_timeline" in sweep.columns else []
    if timeline:
        print(f"Trade counts cover: {', '.join(timeline)}")
    print(f"Saved sweep data to: {sweep_path}")
    print(f"Saved sweep plot to: {plot_path}")
    if quality_report["warnings"]:
        print("Market data quality warnings:")
        for warning in quality_report["warnings"]:
            print(f"- {warning}")
    print(f"Saved market quality report to: {quality_path}")


if __name__ == "__main__":
    main()
