"""Select an optimized slate of individual paper bets before parlays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from strategy.portfolio import optimize_individual_bet_slate, save_portfolio_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize daily slates of individual paper bets.")
    parser.add_argument("--trades-path", default=None)
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--min-edge", type=float, default=None)
    parser.add_argument("--min-expected-roi", type=float, default=0.0)
    parser.add_argument("--use-calibrated-edges", action="store_true")
    parser.add_argument("--trade-column", default="trade")
    parser.add_argument("--expected-roi-column", default=None)
    parser.add_argument("--max-bet-fraction", type=float, default=None)
    parser.add_argument("--max-slate-fraction", type=float, default=0.12)
    parser.add_argument("--max-trades-per-slate", type=int, default=5)
    parser.add_argument("--max-markets-per-game", type=int, default=1)
    parser.add_argument("--max-markets-per-team", type=int, default=2)
    parser.add_argument("--output-trades-path", default=None)
    parser.add_argument("--output-slates-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    if args.use_calibrated_edges:
        trades_path = Path(args.trades_path) if args.trades_path else reports_dir / "edge_calibrated_trades.csv"
        trade_column = "calibrated_trade" if args.trade_column == "trade" else args.trade_column
        expected_roi_column = args.expected_roi_column or "calibrated_expected_roi"
        default_trades_output = reports_dir / "portfolio_trades_calibrated.csv"
        default_slates_output = reports_dir / "portfolio_slates_calibrated.csv"
        default_summary_output = reports_dir / "portfolio_summary_calibrated.json"
    else:
        trades_path = Path(args.trades_path) if args.trades_path else reports_dir / "backtest_trades.csv"
        trade_column = args.trade_column
        expected_roi_column = args.expected_roi_column
        default_trades_output = reports_dir / "portfolio_trades.csv"
        default_slates_output = reports_dir / "portfolio_slates.csv"
        default_summary_output = reports_dir / "portfolio_summary.json"
    output_trades_path = (
        Path(args.output_trades_path)
        if args.output_trades_path
        else default_trades_output
    )
    output_summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else default_summary_output
    )
    output_slates_path = (
        Path(args.output_slates_path)
        if args.output_slates_path
        else default_slates_output
    )
    starting_bankroll = args.bankroll if args.bankroll is not None else config.strategy.starting_bankroll
    if args.min_edge is not None:
        min_edge = args.min_edge
    elif args.use_calibrated_edges:
        min_edge = -1.0
    else:
        min_edge = config.strategy.edge_threshold
    max_bet_fraction = (
        args.max_bet_fraction
        if args.max_bet_fraction is not None
        else config.strategy.max_bet_fraction
    )

    trades = pd.read_csv(trades_path, dtype={"game_id": str})
    selected, slates, summary = optimize_individual_bet_slate(
        trades,
        starting_bankroll=starting_bankroll,
        min_edge=min_edge,
        min_expected_roi=args.min_expected_roi,
        trade_column=trade_column,
        expected_roi_column=expected_roi_column,
        max_bet_fraction=max_bet_fraction,
        max_slate_fraction=args.max_slate_fraction,
        max_trades_per_slate=args.max_trades_per_slate,
        max_markets_per_game=args.max_markets_per_game,
        max_markets_per_team=args.max_markets_per_team,
    )
    save_portfolio_outputs(selected, slates, summary, output_trades_path, output_slates_path, output_summary_path)

    print(
        f"Portfolio selected trades ({summary.get('trade_timeline', 'n/a')}): "
        f"{summary.get('num_selected_trades', 0):,}"
    )
    print(f"Trade filter column: {trade_column}")
    print(f"Candidate bets: {summary.get('num_candidate_bets', 0):,}")
    print(f"Slates: {summary.get('num_slates', 0):,}")
    print(f"Rejected by team cap: {summary.get('rejected_by_team_cap', 0):,}")
    print(f"Ending bankroll: ${summary.get('ending_bankroll', 0):.2f}")
    print(f"Saved portfolio trades to: {output_trades_path}")
    print(f"Saved portfolio slates to: {output_slates_path}")
    print(f"Saved portfolio summary to: {output_summary_path}")


if __name__ == "__main__":
    main()
