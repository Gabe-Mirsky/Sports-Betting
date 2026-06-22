"""Audit whether the best pregame snapshot policy has broad CLV support."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.snapshot_clv_audit import build_snapshot_clv_audit, save_snapshot_clv_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CLV distribution for a pregame snapshot policy.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="trade")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="snapshot_clv_best_le_120m")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = (
        Path(args.input_path)
        if args.input_path
        else reports_dir / "pregame_snapshot_entry_best_le_120m_trades.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir

    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    reports, summary = build_snapshot_clv_audit(rows, signal_column=args.signal_column)
    save_snapshot_clv_audit(reports, summary, output_dir, prefix=args.prefix)

    print(f"Snapshot CLV status: {summary.get('status', 'n/a')}")
    print(f"Signals: {summary.get('signals', 0):,}")
    print(f"Average CLV: {summary.get('avg_clv_cents', 0.0):+.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Flat CLV rate: {summary.get('flat_clv_rate', 0.0):.1%}")
    print(f"Top positive CLV share: {summary.get('top_positive_clv_share', 0.0):.1%}")
    print(f"Saved snapshot CLV audit to: {output_dir}")


if __name__ == "__main__":
    main()
