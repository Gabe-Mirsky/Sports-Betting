"""Audit the failure month in defensive walk-forward signals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.defensive_failure_audit import build_defensive_failure_audit, save_defensive_failure_audit  # noqa: E402


EMPTY_INPUT_COLUMNS = [
    "date",
    "game_id",
    "market_ticker",
    "walk_forward_defensive_signal",
    "price_cents",
    "clv_cents",
    "realized_profit_per_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit why defensive walk-forward failed in one month.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="walk_forward_defensive_signal")
    parser.add_argument("--failure-month", default="2026-03")
    parser.add_argument("--schedule-context-path", default=None)
    parser.add_argument("--min-segment-rows", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="defensive_failure")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = (
        Path(args.input_path)
        if args.input_path
        else reports_dir / "defensive_walk_forward_trades.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    try:
        rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    except EmptyDataError:
        rows = pd.DataFrame(columns=EMPTY_INPUT_COLUMNS)
    reports, summary = build_defensive_failure_audit(
        rows,
        signal_column=args.signal_column,
        failure_month=args.failure_month,
        schedule_context_path=args.schedule_context_path,
        min_segment_rows=args.min_segment_rows,
    )
    save_defensive_failure_audit(reports, summary, output_dir, prefix=args.prefix)
    print(f"Failure month: {summary.get('failure_month', 'n/a')}")
    print(f"Failure month rows: {summary.get('failure_month_rows', 0):,}")
    print(f"Failure month profit/share: {summary.get('failure_month_avg_profit_per_share', 0.0):+.3f}")
    print(f"Failure month positive CLV: {summary.get('failure_month_positive_clv_rate', 0.0):.1%}")
    print(f"Schedule context: {summary.get('schedule_context_status', 'n/a')}")
    print(f"Saved defensive failure audit to: {output_dir}")


if __name__ == "__main__":
    main()
