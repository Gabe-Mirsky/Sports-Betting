"""Compare NBA model probabilities with Kalshi market prices and an expanding blend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from models.market_blend import add_market_blended_probabilities, save_market_blend_outputs  # noqa: E402
from strategy.backtest import prepare_candlestick_backtest_markets, run_backtest, save_backtest_outputs, summarize_backtest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a market-aware probability blend and paper backtest it.")
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--max-bet-fraction", type=float, default=0.03)
    parser.add_argument("--min-train-rows", type=int, default=250)
    parser.add_argument("--min-volume", type=float, default=10)
    parser.add_argument("--no-playoff-features", action="store_true")
    parser.add_argument("--allow-low-quality-prices", action="store_true")
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = PROJECT_ROOT / "data" / "reports"
    prediction_frames = []
    all_game_path = reports_dir / "all_game_predictions.csv"
    walk_forward_path = reports_dir / "walk_forward_predictions.csv"
    upcoming_path = reports_dir / "upcoming_predictions.csv"
    if all_game_path.exists():
        prediction_frames.append(pd.read_csv(all_game_path, dtype={"game_id": str}))
    if walk_forward_path.exists():
        prediction_frames.append(pd.read_csv(walk_forward_path, dtype={"game_id": str}))
    if upcoming_path.exists():
        prediction_frames.append(pd.read_csv(upcoming_path, dtype={"game_id": str}))
    if not prediction_frames:
        raise SystemExit("No prediction files found. Run scripts/train.py or scripts/walk_forward.py first.")
    predictions = pd.concat(prediction_frames, ignore_index=True).drop_duplicates(
        subset=["game_id"],
        keep="last",
    )
    matches = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "kalshi_game_market_matches.csv", dtype={"game_id": str})
    prices = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "kalshi_pregame_prices.csv", dtype={"game_id": str})
    matched, diagnostics = prepare_candlestick_backtest_markets(
        predictions,
        matches,
        prices,
        min_volume=args.min_volume,
        allowed_price_qualities=None if args.allow_low_quality_prices else ("bid_ask_available",),
        require_bid_ask=not args.allow_low_quality_prices,
        max_candle_interval_minutes=None if args.allow_low_quality_prices else 60,
    )
    blended, metrics = add_market_blended_probabilities(
        matched,
        min_train_rows=args.min_train_rows,
        use_playoff_features=not args.no_playoff_features,
    )
    metrics["backtest_input_diagnostics"] = diagnostics
    suffix = args.output_suffix
    save_market_blend_outputs(
        blended,
        metrics,
        PROJECT_ROOT / "data" / "reports" / f"market_blended_predictions{suffix}.csv",
        PROJECT_ROOT / "data" / "reports" / f"market_blend_metrics{suffix}.json",
    )

    backtest_input = blended.copy()
    backtest_input["model_yes_prob"] = backtest_input["blended_yes_prob"]
    trades = run_backtest(
        backtest_input,
        starting_bankroll=args.bankroll,
        edge_threshold=args.edge_threshold,
        max_bet_fraction=args.max_bet_fraction,
    )
    summary = summarize_backtest(trades, starting_bankroll=args.bankroll)
    save_backtest_outputs(
        trades,
        summary,
        PROJECT_ROOT / "data" / "reports" / f"backtest_trades_market_blend{suffix}.csv",
        PROJECT_ROOT / "data" / "reports" / f"backtest_summary_market_blend{suffix}.json",
    )

    print("Probability metrics:")
    for name in ["model", "market", "market_blend"]:
        item = metrics[name]
        print(f"- {name}: log_loss={item['log_loss']:.4f}, brier={item['brier_score']:.4f}, accuracy={item['accuracy']:.4f}")
    if metrics.get("playoffs"):
        item = metrics["playoffs"]["market_blend"]
        print(f"- playoff blend: log_loss={item['log_loss']:.4f}, brier={item['brier_score']:.4f}, accuracy={item['accuracy']:.4f}")
    print(f"Blend paper trades ({summary.get('trade_timeline', 'n/a')}): {summary['num_trades']:,}")
    print(f"Blend ending bankroll: ${summary['ending_bankroll']:.2f}")
    print(f"Saved blend predictions to: {PROJECT_ROOT / 'data' / 'reports' / f'market_blended_predictions{suffix}.csv'}")
    print(f"Saved blend metrics to: {PROJECT_ROOT / 'data' / 'reports' / f'market_blend_metrics{suffix}.json'}")


if __name__ == "__main__":
    main()
