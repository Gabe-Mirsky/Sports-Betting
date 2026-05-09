"""Calibrate model-vs-market edge estimates from historical paper results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.edge_calibration import calibrate_edges_and_save  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build expanding-window edge calibration artifacts.")
    parser.add_argument("--trades-path", default=None)
    parser.add_argument("--min-history-rows", type=int, default=75)
    parser.add_argument("--min-calibrated-profit-per-share", type=float, default=0.0)
    parser.add_argument("--min-calibrated-roi", type=float, default=None)
    parser.add_argument("--min-observed-yes-rate", type=float, default=None)
    parser.add_argument("--shrinkage-rows", type=int, default=150)
    parser.add_argument("--output-calibrated-path", default=None)
    parser.add_argument("--output-bins-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--output-audit-path", default=None)
    parser.add_argument("--output-negative-edge-path", default=None)
    parser.add_argument("--output-audit-summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    trades_path = Path(args.trades_path) if args.trades_path else reports_dir / "backtest_trades.csv"
    calibrated_path = (
        Path(args.output_calibrated_path)
        if args.output_calibrated_path
        else reports_dir / "edge_calibrated_trades.csv"
    )
    bins_path = Path(args.output_bins_path) if args.output_bins_path else reports_dir / "edge_calibration_bins.csv"
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "edge_calibration_summary.json"
    )
    audit_path = (
        Path(args.output_audit_path)
        if args.output_audit_path
        else reports_dir / "edge_calibration_audit.csv"
    )
    negative_edge_path = (
        Path(args.output_negative_edge_path)
        if args.output_negative_edge_path
        else reports_dir / "edge_calibration_negative_edge_signals.csv"
    )
    audit_summary_path = (
        Path(args.output_audit_summary_path)
        if args.output_audit_summary_path
        else reports_dir / "edge_calibration_audit_summary.json"
    )

    calibrated, bins, summary = calibrate_edges_and_save(
        trades_path=trades_path,
        calibrated_output_path=calibrated_path,
        bins_output_path=bins_path,
        summary_output_path=summary_path,
        audit_output_path=audit_path,
        negative_edge_output_path=negative_edge_path,
        audit_summary_output_path=audit_summary_path,
        min_history_rows=args.min_history_rows,
        min_calibrated_profit_per_share=args.min_calibrated_profit_per_share,
        min_calibrated_roi=args.min_calibrated_roi,
        min_observed_yes_rate=args.min_observed_yes_rate,
        shrinkage_rows=args.shrinkage_rows,
    )
    print(f"Calibrated rows: {len(calibrated):,}")
    print(f"Edge bins: {len(bins):,}")
    print(f"Calibrated trades ({summary.get('trade_timeline', 'n/a')}): {summary.get('calibrated_trades', 0):,}")
    print(f"Saved calibrated trades to: {calibrated_path}")
    print(f"Saved edge bins to: {bins_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved edge audit to: {audit_path}")
    print(f"Saved negative-edge signal audit to: {negative_edge_path}")


if __name__ == "__main__":
    main()
