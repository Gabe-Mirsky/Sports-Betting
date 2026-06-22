"""Build a compact diagnosis of why calibrated edges fail proof gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.edge_failure import build_edge_failure_diagnosis, save_edge_failure_diagnosis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose CLV/profit failures in calibrated edge signals.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="calibrated_trade")
    parser.add_argument("--min-segment-rows", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="edge_failure")
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
    reports, summary = build_edge_failure_diagnosis(
        rows,
        signal_column=args.signal_column,
        min_segment_rows=args.min_segment_rows,
    )
    save_edge_failure_diagnosis(reports, summary, output_dir, prefix=args.prefix)
    print(f"Edge failure status: {summary.get('status', 'n/a')}")
    print(f"Signals: {summary.get('signals', 0):,}")
    print(f"Average CLV: {summary.get('avg_clv_cents', 0.0):+.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Average profit/share: {summary.get('avg_profit_per_share', 0.0):+.3f}")
    print(f"Worst segment source: {summary.get('worst_segment', 'n/a')}")
    print(f"Saved edge failure diagnosis to: {output_dir}")


if __name__ == "__main__":
    main()
