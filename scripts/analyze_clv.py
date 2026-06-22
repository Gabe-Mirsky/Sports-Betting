"""Analyze closing-line value for historical single-game paper trades."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from strategy.clv import build_clv_reports, save_clv_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLV reports from a backtest trade log.")
    parser.add_argument("--trades-path", default=str(PROJECT_ROOT / "data" / "reports" / "backtest_trades.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    parser.add_argument("--prefix", default="clv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trades = pd.read_csv(args.trades_path, dtype={"game_id": str})
    reports, summary = build_clv_reports(trades)
    save_clv_reports(reports, summary, args.output_dir, prefix=args.prefix)

    print(f"Trades with CLV: {summary.get('trades_with_clv', 0):,}")
    print(f"Average CLV: {summary.get('avg_clv_cents', 0.0):.2f} cents")
    print(f"Median CLV: {summary.get('median_clv_cents', 0.0):.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0) * 100:.1f}%")
    print(f"Saved CLV reports to: {args.output_dir}")


if __name__ == "__main__":
    main()
