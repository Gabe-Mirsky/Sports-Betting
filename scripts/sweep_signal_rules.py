"""Sweep stricter calibrated signal rules for pre-parlay research."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.rule_sweep import run_signal_rule_sweep, save_signal_rule_sweep_outputs  # noqa: E402


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep stricter calibrated signal rules.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="consensus_trade")
    parser.add_argument("--expected-roi-column", default="consensus_expected_roi")
    parser.add_argument("--history-column", default="edge_bin_history_rows")
    parser.add_argument("--secondary-history-column", default="edge_bin_history_rows_blend")
    parser.add_argument("--min-edges", default="-0.10,-0.05,0.00,0.02,0.05")
    parser.add_argument("--min-expected-rois", default="0.00,0.10,0.25,0.50,0.75")
    parser.add_argument("--min-history-rows", default="0,50,100,150,200")
    parser.add_argument("--min-price-cents", default="1,5,10")
    parser.add_argument("--max-price-cents", default="90,95,99")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-monthly-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "edge_consensus_calibrated_trades.csv"
    output_path = Path(args.output_path) if args.output_path else reports_dir / "signal_rule_sweep.csv"
    monthly_path = (
        Path(args.output_monthly_path)
        if args.output_monthly_path
        else reports_dir / "signal_rule_sweep_best_monthly.csv"
    )
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "signal_rule_sweep_summary.json"
    )

    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    rules, best_monthly, summary = run_signal_rule_sweep(
        rows,
        signal_column=args.signal_column,
        expected_roi_column=args.expected_roi_column,
        min_edges=_parse_float_list(args.min_edges),
        min_expected_rois=_parse_float_list(args.min_expected_rois),
        min_history_rows=_parse_int_list(args.min_history_rows),
        min_price_cents=_parse_float_list(args.min_price_cents),
        max_price_cents=_parse_float_list(args.max_price_cents),
        history_column=args.history_column,
        secondary_history_column=args.secondary_history_column or None,
    )
    save_signal_rule_sweep_outputs(rules, best_monthly, summary, output_path, monthly_path, summary_path)

    print(f"Rules tested: {summary.get('rules_tested', 0):,}")
    print(f"Exploratory candidates: {summary.get('exploratory_candidates', 0):,}")
    print(f"Watchlist rules: {summary.get('watchlist_rules', 0):,}")
    print(f"Best rule: {summary.get('best_rule', 'n/a')}")
    print(
        "Best rule signals "
        f"({summary.get('best_rule_timeline', 'n/a')}): {summary.get('best_rule_signals', 0):,}"
    )
    print(
        "Best rule positive months: "
        f"{summary.get('best_rule_positive_months', 0):,}/{summary.get('best_rule_months', 0):,}"
    )
    print(f"Saved rule sweep to: {output_path}")
    print(f"Saved best-rule monthly table to: {monthly_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
