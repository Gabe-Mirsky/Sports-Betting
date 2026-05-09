"""Validate signal-rule selection with nested walk-forward monthly folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from strategy.rule_sweep import (  # noqa: E402
    run_walk_forward_signal_rule_validation,
    save_walk_forward_rule_validation_outputs,
)


def _float_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested walk-forward validation for calibrated signal rules.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="consensus_trade")
    parser.add_argument("--expected-roi-column", default="consensus_expected_roi")
    parser.add_argument("--min-edges", default=None)
    parser.add_argument("--min-expected-rois", default=None)
    parser.add_argument("--min-history-rows", default=None)
    parser.add_argument("--min-price-cents", default=None)
    parser.add_argument("--max-price-cents", default=None)
    parser.add_argument("--min-train-rows", type=int, default=50)
    parser.add_argument("--min-train-months", type=int, default=2)
    parser.add_argument("--output-validated-path", default=None)
    parser.add_argument("--output-folds-path", default=None)
    parser.add_argument("--output-monthly-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "edge_consensus_calibrated_trades.csv"
    validated_path = (
        Path(args.output_validated_path)
        if args.output_validated_path
        else reports_dir / "signal_rule_walk_forward_trades.csv"
    )
    folds_path = (
        Path(args.output_folds_path)
        if args.output_folds_path
        else reports_dir / "signal_rule_walk_forward_folds.csv"
    )
    monthly_path = (
        Path(args.output_monthly_path)
        if args.output_monthly_path
        else reports_dir / "signal_rule_walk_forward_monthly.csv"
    )
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "signal_rule_walk_forward_summary.json"
    )

    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    validated, folds, monthly, summary = run_walk_forward_signal_rule_validation(
        rows,
        signal_column=args.signal_column,
        expected_roi_column=args.expected_roi_column,
        min_edges=_float_list(args.min_edges),
        min_expected_rois=_float_list(args.min_expected_rois),
        min_history_rows=_int_list(args.min_history_rows),
        min_price_cents=_float_list(args.min_price_cents),
        max_price_cents=_float_list(args.max_price_cents),
        min_train_rows=args.min_train_rows,
        min_train_months=args.min_train_months,
    )
    save_walk_forward_rule_validation_outputs(
        validated,
        folds,
        monthly,
        summary,
        validated_path,
        folds_path,
        monthly_path,
        summary_path,
    )

    print(f"Walk-forward rule status: {summary.get('status', 'n/a')}")
    print(f"Evaluated months: {summary.get('evaluated_months', 0):,}")
    print(f"Skipped months: {summary.get('skipped_months', 0):,}")
    print(f"Walk-forward rule signals ({summary.get('timeline', 'n/a')}): {summary.get('signals', 0):,}")
    print(
        "Positive months: "
        f"{summary.get('positive_months', 0):,}/{summary.get('months', 0):,}"
    )
    print(f"Saved validated rows to: {validated_path}")
    print(f"Saved fold table to: {folds_path}")
    print(f"Saved monthly table to: {monthly_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
