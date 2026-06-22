"""Sweep price-aware calibration settings against historical calibrated signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.edge_calibration import sweep_price_aware_calibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep side + price + edge calibration settings.")
    parser.add_argument("--trades-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    trades_path = Path(args.trades_path) if args.trades_path else reports_dir / "backtest_trades.csv"
    output_path = (
        Path(args.output_path)
        if args.output_path
        else reports_dir / "price_aware_calibration_sweep.csv"
    )
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "price_aware_calibration_sweep_summary.json"
    )
    trades = pd.read_csv(trades_path, dtype={"game_id": str, "market_ticker": str})
    rules, summary = sweep_price_aware_calibration(trades)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rules.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Rules tested: {summary.get('rules_tested', 0):,}")
    print(f"Candidates: {summary.get('candidates', 0):,}")
    print(f"Watchlist rules: {summary.get('watchlist_rules', 0):,}")
    print(f"Best status: {summary.get('best_status', 'n/a')}")
    print(f"Best signals: {summary.get('best_signals', 0):,}")
    print(f"Best positive CLV rate: {summary.get('best_positive_clv_rate', 0.0):.1%}")
    print(f"Saved sweep to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
