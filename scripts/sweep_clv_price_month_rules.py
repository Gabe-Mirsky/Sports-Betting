"""Sweep CLV-filtered YES-only rules by price range and monthly stability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.clv_concentration import (  # noqa: E402
    run_clv_price_month_sweep,
    run_walk_forward_clv_price_month_validation,
    save_clv_price_month_sweep_outputs,
    save_walk_forward_clv_price_month_outputs,
)


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep CLV-filtered price ranges against monthly stability.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="clv_filtered_trade")
    parser.add_argument("--side", default="YES")
    parser.add_argument("--price-breaks", default="0,25,40,55,70,85,100")
    parser.add_argument("--min-rows", type=int, default=25)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--min-train-months", type=int, default=2)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-monthly-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--output-folds-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "clv_filtered_trades.csv"
    output_path = Path(args.output_path) if args.output_path else reports_dir / "clv_price_month_sweep.csv"
    monthly_path = (
        Path(args.output_monthly_path)
        if args.output_monthly_path
        else reports_dir / "clv_price_month_sweep_monthly.csv"
    )
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "clv_price_month_sweep_summary.json"
    )
    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    if args.walk_forward:
        output_path = (
            Path(args.output_path)
            if args.output_path
            else reports_dir / "clv_price_month_walk_forward_trades.csv"
        )
        folds_path = (
            Path(args.output_folds_path)
            if args.output_folds_path
            else reports_dir / "clv_price_month_walk_forward_folds.csv"
        )
        monthly_path = (
            Path(args.output_monthly_path)
            if args.output_monthly_path
            else reports_dir / "clv_price_month_walk_forward_monthly.csv"
        )
        summary_path = (
            Path(args.output_summary_path)
            if args.output_summary_path
            else reports_dir / "clv_price_month_walk_forward_summary.json"
        )
        validated, folds, monthly, summary = run_walk_forward_clv_price_month_validation(
            rows,
            signal_column=args.signal_column,
            side=args.side,
            price_breaks=_parse_float_list(args.price_breaks),
            min_rows=args.min_rows,
            min_train_months=args.min_train_months,
        )
        save_walk_forward_clv_price_month_outputs(
            validated,
            folds,
            monthly,
            summary,
            output_path,
            folds_path,
            monthly_path,
            summary_path,
        )
        print(f"Walk-forward status: {summary.get('status', 'n/a')}")
        print(f"Evaluated months: {summary.get('evaluated_months', 0):,}")
        print(f"Signals: {summary.get('signals', 0):,}")
        print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
        print(f"Positive month share: {summary.get('positive_month_share', 0.0):.1%}")
        print(f"Saved validated rows to: {output_path}")
        print(f"Saved folds to: {folds_path}")
        print(f"Saved monthly table to: {monthly_path}")
        print(f"Saved summary to: {summary_path}")
        return

    rules, monthly, summary = run_clv_price_month_sweep(
        rows,
        signal_column=args.signal_column,
        side=args.side,
        price_breaks=_parse_float_list(args.price_breaks),
        min_rows=args.min_rows,
    )
    save_clv_price_month_sweep_outputs(rules, monthly, summary, output_path, monthly_path, summary_path)
    print(f"Rules tested: {summary.get('rules_tested', 0):,}")
    print(f"Stability candidates: {summary.get('stability_candidates', 0):,}")
    print(f"Watchlist rules: {summary.get('watchlist_rules', 0):,}")
    print(f"Best rule: {summary.get('best_rule', 'n/a')}")
    print(f"Best status: {summary.get('best_status', 'n/a')}")
    print(f"Best positive CLV rate: {summary.get('best_rule_positive_clv_rate', 0.0):.1%}")
    print(f"Best positive month share: {summary.get('best_rule_positive_month_share', 0.0):.1%}")
    print(f"Saved sweep to: {output_path}")
    print(f"Saved monthly sweep table to: {monthly_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
