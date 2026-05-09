"""Apply a lower-confidence-bound screen to calibrated edge rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.robustness import add_confidence_screen, save_confidence_screen_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen calibrated edges with a confidence lower bound.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="consensus_trade")
    parser.add_argument("--expected-roi-column", default="consensus_expected_roi")
    parser.add_argument("--probability-column", default="calibrated_yes_rate")
    parser.add_argument("--sample-size-column", default="edge_bin_history_rows")
    parser.add_argument("--blend-probability-column", default="calibrated_yes_rate_blend")
    parser.add_argument("--blend-sample-size-column", default="edge_bin_history_rows_blend")
    parser.add_argument("--min-history-rows", type=int, default=100)
    parser.add_argument("--confidence-z", type=float, default=0.75)
    parser.add_argument("--min-lower-profit-per-share", type=float, default=0.0)
    parser.add_argument("--min-expected-roi", type=float, default=0.0)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "edge_consensus_calibrated_trades.csv"
    output_path = Path(args.output_path) if args.output_path else reports_dir / "edge_robust_consensus_trades.csv"
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "edge_robust_consensus_summary.json"
    )

    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    screened, summary = add_confidence_screen(
        rows,
        signal_column=args.signal_column,
        expected_roi_column=args.expected_roi_column,
        probability_column=args.probability_column,
        sample_size_column=args.sample_size_column,
        blend_probability_column=args.blend_probability_column,
        blend_sample_size_column=args.blend_sample_size_column,
        min_history_rows=args.min_history_rows,
        confidence_z=args.confidence_z,
        min_lower_profit_per_share=args.min_lower_profit_per_share,
        min_expected_roi=args.min_expected_roi,
    )
    save_confidence_screen_outputs(screened, summary, output_path, summary_path)

    print(f"Robust signals ({summary.get('trade_timeline', 'n/a')}): {summary.get('robust_signals', 0):,}")
    print(f"Base signals: {summary.get('base_signals', 0):,}")
    print(f"Confidence source: {summary.get('confidence_source', 'n/a')}")
    print(f"Saved robust rows to: {output_path}")
    print(f"Saved robust summary to: {summary_path}")


if __name__ == "__main__":
    main()
