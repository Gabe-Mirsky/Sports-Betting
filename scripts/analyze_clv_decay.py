"""Audit why walk-forward CLV hit rate decays over time."""

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
from strategy.clv_decay import build_clv_decay_audit, save_clv_decay_audit  # noqa: E402


EMPTY_INPUT_COLUMNS = [
    "date",
    "game_id",
    "market_ticker",
    "walk_forward_clv_price_signal",
    "clv_cents",
    "realized_profit_per_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CLV decay in walk-forward selected rows.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="walk_forward_clv_price_signal")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="clv_decay")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = (
        Path(args.input_path)
        if args.input_path
        else reports_dir / "clv_price_month_walk_forward_trades.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    try:
        rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    except EmptyDataError:
        rows = pd.DataFrame(columns=EMPTY_INPUT_COLUMNS)
    reports, summary = build_clv_decay_audit(rows, signal_column=args.signal_column)
    save_clv_decay_audit(reports, summary, output_dir, prefix=args.prefix)
    print(f"CLV decay status: {summary.get('status', 'n/a')}")
    print(f"Rows: {summary.get('rows', 0):,}")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Positive CLV rate change: {summary.get('positive_clv_rate_change', 0.0):+.1%}")
    print(f"Average CLV change: {summary.get('avg_clv_cents_change', 0.0):+.2f} cents")
    print(f"Saved CLV decay reports to: {output_dir}")


if __name__ == "__main__":
    main()
