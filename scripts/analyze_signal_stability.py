"""Analyze month-by-month stability for a calibrated signal column."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.stability import save_signal_stability_outputs, summarize_signal_stability  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze signal stability by month.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="consensus_trade")
    parser.add_argument("--expected-roi-column", default="consensus_expected_roi")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "edge_consensus_calibrated_trades.csv"
    output_path = Path(args.output_path) if args.output_path else reports_dir / "signal_stability_consensus.csv"
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "signal_stability_consensus_summary.json"
    )

    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    monthly, summary = summarize_signal_stability(
        rows,
        signal_column=args.signal_column,
        expected_roi_column=args.expected_roi_column,
    )
    save_signal_stability_outputs(monthly, summary, output_path, summary_path)

    print(f"Signals ({summary.get('timeline', 'n/a')}): {summary.get('signals', 0):,}")
    print(f"Positive months: {summary.get('positive_months', 0):,}/{summary.get('months', 0):,}")
    print(f"Worst month: {summary.get('worst_month', 'n/a')}")
    print(f"Saved stability table to: {output_path}")
    print(f"Saved stability summary to: {summary_path}")


if __name__ == "__main__":
    main()
