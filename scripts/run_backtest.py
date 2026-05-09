"""Run a fake-bankroll backtest against matched Kalshi-style market data."""

from __future__ import annotations

import argparse
import json
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
from reports.summary import format_backtest_results  # noqa: E402
from strategy.backtest import (  # noqa: E402
    prepare_candlestick_backtest_markets,
    run_backtest,
    save_backtest_outputs,
    summarize_backtest,
)

try:
    from reports.plots import save_edge_distribution, save_equity_curve  # noqa: E402

    PLOTS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional local matplotlib install
    PLOTS_AVAILABLE = False

    def save_edge_distribution(*args: object, **kwargs: object) -> None:
        return None

    def save_equity_curve(*args: object, **kwargs: object) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fake-bankroll backtest.")
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--edge-threshold", type=float, default=None)
    parser.add_argument("--max-bet-fraction", type=float, default=None)
    parser.add_argument("--markets-path", default=None)
    parser.add_argument("--matches-path", default=None)
    parser.add_argument("--prices-path", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--allow-low-quality-prices", action="store_true")
    parser.add_argument("--no-require-bid-ask", action="store_true")
    parser.add_argument("--max-candle-interval-minutes", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _split_quality_config(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    starting_bankroll = args.bankroll or config.strategy.starting_bankroll
    edge_threshold = args.edge_threshold if args.edge_threshold is not None else config.strategy.edge_threshold
    max_bet_fraction = (
        args.max_bet_fraction
        if args.max_bet_fraction is not None
        else config.strategy.max_bet_fraction
    )
    min_volume = args.min_volume if args.min_volume is not None else config.backtest.min_volume
    allowed_price_qualities = None if args.allow_low_quality_prices else _split_quality_config(
        config.backtest.allowed_price_qualities
    )
    require_bid_ask = config.backtest.require_bid_ask and not args.no_require_bid_ask
    max_candle_interval_minutes = (
        args.max_candle_interval_minutes
        if args.max_candle_interval_minutes is not None
        else config.backtest.max_candle_interval_minutes
    )

    predictions_path = (
        Path(args.predictions_path)
        if args.predictions_path
        else PROJECT_ROOT / "data" / "reports" / "model_predictions.csv"
    )
    markets_path = (
        Path(args.markets_path)
        if args.markets_path
        else PROJECT_ROOT / "data" / "kalshi" / "markets_mock.csv"
    )
    matches_path = (
        Path(args.matches_path)
        if args.matches_path
        else PROJECT_ROOT / "data" / "processed" / "kalshi_game_market_matches.csv"
    )
    prices_path = (
        Path(args.prices_path)
        if args.prices_path
        else PROJECT_ROOT / "data" / "processed" / "kalshi_pregame_prices.csv"
    )

    predictions = pd.read_csv(predictions_path, dtype={"game_id": str})
    diagnostics = None
    quality_report = {"warnings": []}
    unmatched_tickers: list[str] = []
    matched_path = PROJECT_ROOT / "data" / "reports" / "matched_markets.csv"

    use_candlestick_mode = args.markets_path is None and matches_path.exists() and prices_path.exists()
    if use_candlestick_mode:
        matches = pd.read_csv(matches_path, dtype={"game_id": str})
        prices = pd.read_csv(prices_path, dtype={"game_id": str})
        matched, diagnostics = prepare_candlestick_backtest_markets(
            predictions,
            matches,
            prices,
            min_volume=min_volume,
            allowed_price_qualities=allowed_price_qualities,
            require_bid_ask=require_bid_ask,
            max_candle_interval_minutes=max_candle_interval_minutes,
        )
        if matched.empty:
            raise SystemExit("No auto-matched Kalshi markets had usable pregame candle prices.")
    else:
        markets = load_mock_kalshi_markets(markets_path)
        matched = match_games_to_markets(predictions, markets)
        if matched.empty:
            raise SystemExit("No mock/manual Kalshi markets matched model predictions.")
        quality_report = analyze_market_data_quality(markets, matched)
        unmatched_tickers = sorted(
            set(markets["market_ticker"].astype(str)) - set(matched["market_ticker"].astype(str))
        )

    trades = run_backtest(
        matched,
        starting_bankroll=starting_bankroll,
        edge_threshold=edge_threshold,
        max_bet_fraction=max_bet_fraction,
        min_market_price=config.strategy.min_market_price,
        max_market_price=config.strategy.max_market_price,
    )
    summary = summarize_backtest(trades, starting_bankroll=starting_bankroll)
    if diagnostics is not None:
        summary.update(diagnostics)

    suffix = args.output_suffix
    trades_path = PROJECT_ROOT / "data" / "reports" / f"backtest_trades{suffix}.csv"
    summary_path = PROJECT_ROOT / "data" / "reports" / f"backtest_summary{suffix}.json"
    equity_path = PROJECT_ROOT / "data" / "reports" / f"equity_curve{suffix}.png"
    matching_report_path = PROJECT_ROOT / "data" / "reports" / f"market_matching_report{suffix}.json"
    quality_report_path = PROJECT_ROOT / "data" / "reports" / f"market_data_quality_report{suffix}.json"
    edge_distribution_path = PROJECT_ROOT / "data" / "reports" / f"edge_distribution{suffix}.png"
    save_backtest_outputs(trades, summary, trades_path, summary_path)
    save_equity_curve(trades, equity_path)
    save_edge_distribution(trades, edge_distribution_path)
    if suffix:
        matched_path = PROJECT_ROOT / "data" / "reports" / f"matched_markets{suffix}.csv"
    matched.to_csv(matched_path, index=False)
    if diagnostics is None:
        matching_payload = {
            "markets_loaded": int(len(markets)),
            "markets_matched": int(len(matched)),
            "markets_unmatched": int(len(unmatched_tickers)),
            "unmatched_market_tickers": unmatched_tickers,
        }
    else:
        matching_payload = diagnostics
    matching_report_path.write_text(
        json.dumps(matching_payload, indent=2),
        encoding="utf-8",
    )
    if diagnostics is None:
        save_market_quality_report(quality_report, quality_report_path)
    else:
        quality_report = {
            "warnings": [],
            "mode": "candlestick_backtest",
            "diagnostics": diagnostics,
            "filters": {
                "allowed_price_qualities": allowed_price_qualities,
                "require_bid_ask": require_bid_ask,
                "min_volume": min_volume,
                "max_candle_interval_minutes": max_candle_interval_minutes,
            },
        }
        quality_report_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")

    print(format_backtest_results(summary))
    if diagnostics is not None:
        print(f"Games available: {diagnostics['games_available']:,}")
        print(f"Games with matched Kalshi market: {diagnostics['games_with_matched_kalshi_market']:,}")
        print(f"Games with usable pregame price: {diagnostics['games_with_usable_pregame_price']:,}")
        print(f"Trades made ({summary.get('trade_timeline', 'n/a')}): {summary['num_trades']:,}")
        print(f"Final bankroll: ${summary['ending_bankroll']:.2f}")
    print(f"Matched markets: {len(matched):,}")
    print(f"Saved trades to: {trades_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved equity curve to: {equity_path}")
    print(f"Saved matched markets to: {matched_path}")
    print(f"Saved matching report to: {matching_report_path}")
    if quality_report["warnings"]:
        print("Market data quality warnings:")
        for warning in quality_report["warnings"]:
            print(f"- {warning}")
    if not PLOTS_AVAILABLE:
        print("Plot generation skipped because matplotlib is not installed in this Python environment.")
    print(f"Saved market quality report to: {quality_report_path}")


if __name__ == "__main__":
    main()
