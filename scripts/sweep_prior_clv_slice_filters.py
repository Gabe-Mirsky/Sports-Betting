"""Walk-forward sweep of prior-month CLV slice filters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.prior_clv_slice_filter import run_prior_clv_slice_filter, save_prior_clv_slice_filter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep prior-month CLV slice filters.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="calibrated_trade")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="prior_clv_slice_filter")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = (
        Path(args.input_path)
        if args.input_path
        else reports_dir / "edge_calibration_price_aware_best_trades.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir

    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    selected, policies, folds, summary = run_prior_clv_slice_filter(rows, signal_column=args.signal_column)
    save_prior_clv_slice_filter(selected, policies, folds, summary, output_dir, prefix=args.prefix)

    print(f"Prior CLV slice status: {summary.get('status', 'n/a')}")
    print(f"Policies tested: {summary.get('policies_tested', 0):,}")
    print(f"Candidate policies: {summary.get('candidate_policies', 0):,}")
    print(f"Best policy: {summary.get('best_policy', 'n/a')}")
    print(f"Best signals: {summary.get('best_signals', 0):,}")
    print(f"Best average CLV: {summary.get('best_avg_clv_cents', 0.0):+.2f} cents")
    print(f"Best positive CLV rate: {summary.get('best_positive_clv_rate', 0.0):.1%}")
    print(f"Best profit/share: {summary.get('best_avg_profit_per_share', 0.0):+.3f}")
    print(f"Saved prior CLV slice filter reports to: {output_dir}")


if __name__ == "__main__":
    main()
