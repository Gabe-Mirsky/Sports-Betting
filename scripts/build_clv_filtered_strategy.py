"""Build a side-specific CLV-filtered calibrated strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.clv_filter import add_expanding_clv_filter, save_clv_filter_outputs  # noqa: E402


def _default_input_path(reports_dir: Path) -> Path:
    best_price_aware = reports_dir / "edge_calibration_price_aware_best_trades.csv"
    if best_price_aware.exists():
        return best_price_aware
    return reports_dir / "edge_calibrated_trades.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply side-specific expanding CLV gates to calibrated signals.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="calibrated_trade")
    parser.add_argument("--yes-min-history-rows", type=int, default=50)
    parser.add_argument("--yes-min-avg-clv-cents", type=float, default=0.0)
    parser.add_argument("--yes-min-positive-clv-rate", type=float, default=0.40)
    parser.add_argument("--yes-min-avg-profit-per-share", type=float, default=0.0)
    parser.add_argument("--no-min-history-rows", type=int, default=50)
    parser.add_argument("--no-min-avg-clv-cents", type=float, default=0.25)
    parser.add_argument("--no-min-positive-clv-rate", type=float, default=0.50)
    parser.add_argument("--no-min-avg-profit-per-share", type=float, default=0.0)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-side-audit-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else _default_input_path(reports_dir)
    output_path = Path(args.output_path) if args.output_path else reports_dir / "clv_filtered_trades.csv"
    side_audit_path = (
        Path(args.output_side_audit_path)
        if args.output_side_audit_path
        else reports_dir / "clv_filtered_side_audit.csv"
    )
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "clv_filtered_summary.json"
    )
    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    filtered, side_audit, summary = add_expanding_clv_filter(
        rows,
        signal_column=args.signal_column,
        side_rules={
            "YES": {
                "min_history_rows": args.yes_min_history_rows,
                "min_avg_clv_cents": args.yes_min_avg_clv_cents,
                "min_positive_clv_rate": args.yes_min_positive_clv_rate,
                "min_avg_profit_per_share": args.yes_min_avg_profit_per_share,
            },
            "NO": {
                "min_history_rows": args.no_min_history_rows,
                "min_avg_clv_cents": args.no_min_avg_clv_cents,
                "min_positive_clv_rate": args.no_min_positive_clv_rate,
                "min_avg_profit_per_share": args.no_min_avg_profit_per_share,
            },
        },
    )
    save_clv_filter_outputs(filtered, side_audit, summary, output_path, side_audit_path, summary_path)
    print(f"Base calibrated signals: {summary.get('base_signals', 0):,}")
    print(f"CLV-filtered trades: {summary.get('clv_filtered_trades', 0):,}")
    print(f"YES filtered trades: {summary.get('yes_clv_filtered_trades', 0):,}")
    print(f"NO filtered trades: {summary.get('no_clv_filtered_trades', 0):,}")
    print(f"Average CLV: {summary.get('avg_clv_cents', 0.0):.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Saved filtered trades to: {output_path}")
    print(f"Saved side audit to: {side_audit_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
