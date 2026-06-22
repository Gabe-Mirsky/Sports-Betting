"""Audit model/calibration residuals against outcome, profit, and CLV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.residual_audit import build_residual_audit, save_residual_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit model-vs-market residuals.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="calibrated_trade")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="residual")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "edge_calibrated_trades.csv"
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    reports, summary = build_residual_audit(rows, signal_column=args.signal_column)
    save_residual_audit(reports, summary, output_dir, prefix=args.prefix)
    print(f"Signals: {summary.get('signals', 0):,}")
    print(f"Avg calibrated residual: {summary.get('avg_calibrated_residual', 0.0):+.3f}")
    print(f"Realized win rate: {summary.get('realized_win_rate', 0.0):.1%}")
    print(f"Avg calibrated win rate: {summary.get('avg_calibrated_win_rate', 0.0):.1%}")
    print(f"Avg profit/share: {summary.get('avg_profit_per_share', 0.0):+.3f}")
    print(f"Avg CLV: {summary.get('avg_clv_cents', 0.0):+.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Saved residual audit to: {output_dir}")


if __name__ == "__main__":
    main()
