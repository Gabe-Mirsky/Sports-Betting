"""Build a defensive version of the CLV-filtered strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.defensive_filter import (  # noqa: E402
    add_defensive_filters,
    run_defensive_sample_expansion,
    run_defensive_rule_sweep,
    run_walk_forward_defensive_validation,
    save_defensive_sample_expansion_outputs,
    save_defensive_filter_outputs,
    save_defensive_rule_sweep_outputs,
    save_walk_forward_defensive_outputs,
)


def _parse_float_list(value: str) -> list[float]:
    parsed: list[float] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        parsed.append(float("inf") if stripped.lower() in {"inf", "infinity"} else float(stripped))
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply defensive price/ROI/liquidity filters.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="clv_filtered_trade")
    parser.add_argument("--min-price-cents", type=float, default=10.0)
    parser.add_argument("--max-price-cents", type=float, default=100.0)
    parser.add_argument("--min-calibrated-expected-roi", type=float, default=0.0)
    parser.add_argument("--max-calibrated-expected-roi", type=float, default=3.0)
    parser.add_argument("--min-volume", type=float, default=10.0)
    parser.add_argument("--max-volume", type=float, default=1000.0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--sample-expansion", action="store_true")
    parser.add_argument("--min-price-values", default="5,10,12.5,15")
    parser.add_argument("--max-price-values", default="40,45,50,55,100")
    parser.add_argument("--min-roi-values", default="0,0.25,0.5")
    parser.add_argument("--max-roi-values", default="2,3,5,10")
    parser.add_argument("--max-volume-values", default="100,1000,10000,inf")
    parser.add_argument("--min-rows", type=int, default=25)
    parser.add_argument("--min-train-months", type=int, default=2)
    parser.add_argument("--target-min-signals", type=int, default=100)
    parser.add_argument("--target-max-signals", type=int, default=150)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-audit-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--output-monthly-path", default=None)
    parser.add_argument("--output-folds-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "clv_filtered_trades.csv"
    output_path = Path(args.output_path) if args.output_path else reports_dir / "defensive_filtered_trades.csv"
    audit_path = (
        Path(args.output_audit_path)
        if args.output_audit_path
        else reports_dir / "defensive_filter_audit.csv"
    )
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "defensive_filter_summary.json"
    )
    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    if args.sample_expansion:
        output_path = (
            Path(args.output_path)
            if args.output_path
            else reports_dir / "defensive_sample_expansion.csv"
        )
        monthly_path = (
            Path(args.output_monthly_path)
            if args.output_monthly_path
            else reports_dir / "defensive_sample_expansion_monthly.csv"
        )
        summary_path = (
            Path(args.output_summary_path)
            if args.output_summary_path
            else reports_dir / "defensive_sample_expansion_summary.json"
        )
        candidates, monthly, summary = run_defensive_sample_expansion(
            rows,
            signal_column=args.signal_column,
            min_price_values=_parse_float_list(args.min_price_values),
            max_price_values=_parse_float_list(args.max_price_values),
            min_roi_values=_parse_float_list(args.min_roi_values),
            max_roi_values=_parse_float_list(args.max_roi_values),
            max_volume_values=_parse_float_list(args.max_volume_values),
            min_train_months=args.min_train_months,
            target_min_signals=args.target_min_signals,
            target_max_signals=args.target_max_signals,
        )
        save_defensive_sample_expansion_outputs(candidates, monthly, summary, output_path, monthly_path, summary_path)
        print(f"Sample expansion status: {summary.get('status', 'n/a')}")
        print(f"Rules tested: {summary.get('rules_tested', 0):,}")
        print(f"Expanded candidates: {summary.get('expanded_sample_candidates', 0):,}")
        print(f"Best rule: {summary.get('best_rule', 'n/a')}")
        print(f"Best signals: {summary.get('best_rule_signals', 0):,}")
        print(f"Best positive CLV rate: {summary.get('best_rule_positive_clv_rate', 0.0):.1%}")
        print(f"Saved candidates to: {output_path}")
        print(f"Saved monthly table to: {monthly_path}")
        print(f"Saved summary to: {summary_path}")
        return

    if args.walk_forward:
        output_path = (
            Path(args.output_path)
            if args.output_path
            else reports_dir / "defensive_walk_forward_trades.csv"
        )
        folds_path = (
            Path(args.output_folds_path)
            if args.output_folds_path
            else reports_dir / "defensive_walk_forward_folds.csv"
        )
        monthly_path = (
            Path(args.output_monthly_path)
            if args.output_monthly_path
            else reports_dir / "defensive_walk_forward_monthly.csv"
        )
        summary_path = (
            Path(args.output_summary_path)
            if args.output_summary_path
            else reports_dir / "defensive_walk_forward_summary.json"
        )
        validated, folds, monthly, summary = run_walk_forward_defensive_validation(
            rows,
            signal_column=args.signal_column,
            min_price_values=_parse_float_list(args.min_price_values),
            max_price_values=_parse_float_list(args.max_price_values),
            min_roi_values=_parse_float_list(args.min_roi_values),
            max_roi_values=_parse_float_list(args.max_roi_values),
            max_volume_values=_parse_float_list(args.max_volume_values),
            min_rows=args.min_rows,
            min_train_months=args.min_train_months,
        )
        save_walk_forward_defensive_outputs(
            validated,
            folds,
            monthly,
            summary,
            output_path,
            folds_path,
            monthly_path,
            summary_path,
        )
        print(f"Walk-forward defensive status: {summary.get('status', 'n/a')}")
        print(f"Evaluated months: {summary.get('evaluated_months', 0):,}")
        print(f"Signals: {summary.get('signals', 0):,}")
        print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
        print(f"Positive month share: {summary.get('positive_month_share', 0.0):.1%}")
        print(f"Saved validated rows to: {output_path}")
        print(f"Saved folds to: {folds_path}")
        print(f"Saved monthly table to: {monthly_path}")
        print(f"Saved summary to: {summary_path}")
        return

    if args.sweep:
        output_path = (
            Path(args.output_path)
            if args.output_path
            else reports_dir / "defensive_rule_sweep.csv"
        )
        monthly_path = (
            Path(args.output_monthly_path)
            if args.output_monthly_path
            else reports_dir / "defensive_rule_sweep_monthly.csv"
        )
        summary_path = (
            Path(args.output_summary_path)
            if args.output_summary_path
            else reports_dir / "defensive_rule_sweep_summary.json"
        )
        rules, monthly, summary = run_defensive_rule_sweep(
            rows,
            signal_column=args.signal_column,
            min_price_values=_parse_float_list(args.min_price_values),
            max_price_values=_parse_float_list(args.max_price_values),
            min_roi_values=_parse_float_list(args.min_roi_values),
            max_roi_values=_parse_float_list(args.max_roi_values),
            max_volume_values=_parse_float_list(args.max_volume_values),
            min_rows=args.min_rows,
        )
        save_defensive_rule_sweep_outputs(rules, monthly, summary, output_path, monthly_path, summary_path)
        print(f"Rules tested: {summary.get('rules_tested', 0):,}")
        print(f"Defensive candidates: {summary.get('defensive_candidates', 0):,}")
        print(f"Best rule: {summary.get('best_rule', 'n/a')}")
        print(f"Best status: {summary.get('best_status', 'n/a')}")
        print(f"Best positive CLV rate: {summary.get('best_rule_positive_clv_rate', 0.0):.1%}")
        print(f"Saved rule sweep to: {output_path}")
        print(f"Saved monthly table to: {monthly_path}")
        print(f"Saved summary to: {summary_path}")
        return

    filtered, audit, summary = add_defensive_filters(
        rows,
        signal_column=args.signal_column,
        min_price_cents=args.min_price_cents,
        max_price_cents=args.max_price_cents,
        min_calibrated_expected_roi=args.min_calibrated_expected_roi,
        max_calibrated_expected_roi=args.max_calibrated_expected_roi,
        min_volume=args.min_volume,
        max_volume=args.max_volume,
    )
    save_defensive_filter_outputs(filtered, audit, summary, output_path, audit_path, summary_path)
    print(f"Base signals: {summary.get('base_signals', 0):,}")
    print(f"Defensive trades: {summary.get('defensive_trades', 0):,}")
    print(f"Blocked trades: {summary.get('blocked_trades', 0):,}")
    print(f"Average CLV: {summary.get('avg_clv_cents', 0.0):.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Saved defensive trades to: {output_path}")
    print(f"Saved defensive audit to: {audit_path}")
    print(f"Saved defensive summary to: {summary_path}")


if __name__ == "__main__":
    main()
