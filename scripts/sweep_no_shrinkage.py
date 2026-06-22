"""Sweep research-only conservative NO probability shrinkage settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.no_shrinkage import run_no_shrinkage_research, save_no_shrinkage_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep calibrated NO probability shrinkage.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="calibrated_trade")
    parser.add_argument("--min-train-months", type=int, default=2)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="no_shrinkage")
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
    descriptive, validated, folds, summary = run_no_shrinkage_research(
        rows,
        signal_column=args.signal_column,
        min_train_months=args.min_train_months,
        min_rows=args.min_rows,
    )
    save_no_shrinkage_outputs(descriptive, validated, folds, summary, output_dir, prefix=args.prefix)
    print(f"NO shrinkage status: {summary.get('status', 'n/a')}")
    print(f"Descriptive best policy: {summary.get('descriptive_best_policy', 'n/a')}")
    print(f"Walk-forward signals: {summary.get('signals', 0):,}")
    print(f"Average CLV: {summary.get('avg_clv_cents', 0.0):+.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Average profit/share: {summary.get('avg_profit_per_share', 0.0):+.3f}")
    print(f"Saved NO shrinkage research to: {output_dir}")


if __name__ == "__main__":
    main()
