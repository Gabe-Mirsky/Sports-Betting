"""Score paper-trading strategy readiness across calibrated signal families."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.readiness import build_strategy_readiness_report, save_strategy_readiness_outputs  # noqa: E402


def _default_specs(reports_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "name": "raw_calibrated",
            "input_path": str(reports_dir / "edge_calibrated_trades.csv"),
            "signal_column": "calibrated_trade",
            "expected_roi_column": "calibrated_expected_roi",
            "portfolio_summary_path": str(reports_dir / "portfolio_summary_calibrated.json"),
        },
        {
            "name": "clv_filtered_calibrated",
            "input_path": str(reports_dir / "clv_filtered_trades.csv"),
            "signal_column": "clv_filtered_trade",
            "expected_roi_column": "calibrated_expected_roi",
            "portfolio_summary_path": str(reports_dir / "portfolio_summary_clv_filtered.json"),
        },
        {
            "name": "defensive_clv_filtered",
            "input_path": str(reports_dir / "defensive_filtered_trades.csv"),
            "signal_column": "defensive_trade",
            "expected_roi_column": "calibrated_expected_roi",
            "portfolio_summary_path": str(reports_dir / "portfolio_summary_defensive.json"),
        },
        {
            "name": "market_blend_calibrated",
            "input_path": str(reports_dir / "edge_calibrated_trades_market_blend.csv"),
            "signal_column": "calibrated_trade",
            "expected_roi_column": "calibrated_expected_roi",
            "portfolio_summary_path": str(reports_dir / "portfolio_summary_market_blend_calibrated.json"),
        },
        {
            "name": "consensus_calibrated",
            "input_path": str(reports_dir / "edge_consensus_calibrated_trades.csv"),
            "signal_column": "consensus_trade",
            "expected_roi_column": "consensus_expected_roi",
            "portfolio_summary_path": str(reports_dir / "portfolio_summary_consensus_calibrated.json"),
        },
        {
            "name": "robust_consensus",
            "input_path": str(reports_dir / "edge_robust_consensus_trades.csv"),
            "signal_column": "robust_calibrated_trade",
            "expected_roi_column": "robust_expected_roi",
            "portfolio_summary_path": str(reports_dir / "portfolio_summary_robust_consensus.json"),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score calibrated strategy readiness.")
    parser.add_argument("--min-signals", type=int, default=100)
    parser.add_argument("--min-months", type=int, default=6)
    parser.add_argument("--min-positive-month-share", type=float, default=0.60)
    parser.add_argument("--min-avg-profit-per-share", type=float, default=0.0)
    parser.add_argument("--min-ending-bankroll", type=float, default=100.0)
    parser.add_argument("--max-drawdown-floor", type=float, default=-0.60)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-monthly-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    readiness_path = Path(args.output_path) if args.output_path else reports_dir / "strategy_readiness.csv"
    monthly_path = (
        Path(args.output_monthly_path)
        if args.output_monthly_path
        else reports_dir / "strategy_readiness_monthly.csv"
    )
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "strategy_readiness_summary.json"
    )

    readiness, monthly, summary = build_strategy_readiness_report(
        _default_specs(reports_dir),
        min_signals=args.min_signals,
        min_months=args.min_months,
        min_positive_month_share=args.min_positive_month_share,
        min_avg_profit_per_share=args.min_avg_profit_per_share,
        min_ending_bankroll=args.min_ending_bankroll,
        max_drawdown_floor=args.max_drawdown_floor,
    )
    save_strategy_readiness_outputs(readiness, monthly, summary, readiness_path, monthly_path, summary_path)

    print(f"Strategies evaluated: {summary.get('strategies_evaluated', 0):,}")
    print(f"Paper-trade candidates: {summary.get('paper_trade_candidates', 0):,}")
    print(f"Watchlist: {summary.get('watchlist', 0):,}")
    print(f"Not ready: {summary.get('not_ready', 0):,}")
    print(f"Parlay-ready: {summary.get('parlay_ready', 0):,}")
    print(f"Saved readiness table to: {readiness_path}")
    print(f"Saved readiness summary to: {summary_path}")


if __name__ == "__main__":
    main()
