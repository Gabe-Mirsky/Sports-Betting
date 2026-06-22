"""Audit calibrated NO probabilities against CLV and settlement outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.no_calibration_audit import build_no_calibration_audit, save_no_calibration_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit calibrated NO probability quality.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="calibrated_trade")
    parser.add_argument("--min-segment-rows", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="no_calibration")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = (
        Path(args.input_path)
        if args.input_path
        else reports_dir / "edge_calibration_price_aware_best_trades.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    reports, summary = build_no_calibration_audit(
        rows,
        signal_column=args.signal_column,
        min_segment_rows=args.min_segment_rows,
    )
    save_no_calibration_audit(reports, summary, output_dir, prefix=args.prefix)
    print(f"NO calibration rows: {summary.get('selected_no_rows', 0):,}")
    print(f"Status: {summary.get('status', 'n/a')}")
    print(f"Forecast win rate: {summary.get('avg_forecast_win_rate', 0.0):.1%}")
    print(f"Actual win rate: {summary.get('actual_win_rate', 0.0):.1%}")
    print(f"Calibration error: {summary.get('calibration_error', 0.0):+.1%}")
    print(f"Average CLV: {summary.get('avg_clv_cents', 0.0):+.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Saved NO calibration audit to: {output_dir}")


if __name__ == "__main__":
    main()
