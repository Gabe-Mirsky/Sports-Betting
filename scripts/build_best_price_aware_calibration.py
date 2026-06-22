"""Materialize the top price-aware calibration sweep rule."""

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
from strategy.edge_calibration import price_aware_calibrate_edges_and_save  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calibrated trades from the top price-aware sweep rule.")
    parser.add_argument("--trades-path", default=None)
    parser.add_argument("--sweep-path", default=None)
    parser.add_argument("--output-prefix", default="edge_calibration_price_aware_best")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    trades_path = Path(args.trades_path) if args.trades_path else reports_dir / "backtest_trades.csv"
    sweep_path = Path(args.sweep_path) if args.sweep_path else reports_dir / "price_aware_calibration_sweep.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(f"Price-aware calibration sweep file not found: {sweep_path}")
    sweep = pd.read_csv(sweep_path)
    if sweep.empty:
        raise ValueError(f"Price-aware calibration sweep file is empty: {sweep_path}")
    best = sweep.iloc[0]
    prefix = args.output_prefix
    calibrated, bins, summary = price_aware_calibrate_edges_and_save(
        trades_path=trades_path,
        calibrated_output_path=reports_dir / f"{prefix}_trades.csv",
        bins_output_path=reports_dir / f"{prefix}_bins.csv",
        summary_output_path=reports_dir / f"{prefix}_summary.json",
        audit_output_path=reports_dir / f"{prefix}_audit.csv",
        negative_edge_output_path=reports_dir / f"{prefix}_negative_edge_signals.csv",
        audit_summary_output_path=reports_dir / f"{prefix}_audit_summary.json",
        min_history_rows=int(best["min_history_rows"]),
        min_price_history_rows=int(best["min_price_history_rows"]),
        min_calibrated_profit_per_share=float(best["min_calibrated_profit_per_share"]),
        shrinkage_rows=int(best["shrinkage_rows"]),
    )
    summary = {
        **summary,
        "selected_sweep_rule": {
            "status": str(best.get("status", "n/a")),
            "signals": int(best.get("signals", 0)),
            "avg_profit_per_share": float(best.get("avg_profit_per_share", 0.0)),
            "avg_clv_cents": float(best.get("avg_clv_cents", 0.0)),
            "positive_clv_rate": float(best.get("positive_clv_rate", 0.0)),
        },
    }
    (reports_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best price-aware rule status: {best.get('status', 'n/a')}")
    print(f"Best price-aware calibrated rows: {len(calibrated):,}")
    print(f"Best price-aware bins: {len(bins):,}")
    print(f"Best price-aware calibrated trades: {summary.get('calibrated_trades', 0):,}")
    print(f"Saved best price-aware trades to: {reports_dir / f'{prefix}_trades.csv'}")


if __name__ == "__main__":
    main()
