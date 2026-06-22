"""Audit selected NO-side signals for CLV/profit consistency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.no_side_audit import build_no_side_audit, save_no_side_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit selected NO-side CLV signals.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="clv_filtered_trade")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="no_side_audit")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "clv_filtered_trades.csv"
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    reports, summary = build_no_side_audit(rows, signal_column=args.signal_column)
    save_no_side_audit(reports, summary, output_dir, prefix=args.prefix)
    print(f"Selected NO rows: {summary.get('selected_no_rows', 0):,}")
    print(f"Status: {summary.get('status', 'n/a')}")
    print(f"Average CLV: {summary.get('avg_clv_cents', 0.0):+.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Win rate: {summary.get('win_rate', 0.0):.1%}")
    print(f"Profit/share: {summary.get('avg_profit_per_share', 0.0):+.3f}")
    print(f"Positive-CLV losses: {summary.get('positive_clv_loss_count', 0):,}")
    print(f"Profit math mismatches: {summary.get('profit_math_mismatch_count', 0):,}")
    print(f"Saved NO-side audit to: {output_dir}")


if __name__ == "__main__":
    main()
