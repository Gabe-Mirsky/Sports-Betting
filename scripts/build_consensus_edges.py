"""Build consensus calibrated edge rows from raw and market-blend calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.consensus import build_consensus_calibrated_edges, save_consensus_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a consensus calibrated edge file.")
    parser.add_argument("--raw-calibrated-path", default=None)
    parser.add_argument("--market-blend-calibrated-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    raw_path = (
        Path(args.raw_calibrated_path)
        if args.raw_calibrated_path
        else reports_dir / "edge_calibrated_trades.csv"
    )
    blend_path = (
        Path(args.market_blend_calibrated_path)
        if args.market_blend_calibrated_path
        else reports_dir / "edge_calibrated_trades_market_blend.csv"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else reports_dir / "edge_consensus_calibrated_trades.csv"
    )
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "edge_consensus_summary.json"
    )

    raw = pd.read_csv(raw_path, dtype={"game_id": str, "market_ticker": str})
    blend = pd.read_csv(blend_path, dtype={"game_id": str, "market_ticker": str})
    consensus, summary = build_consensus_calibrated_edges(raw, blend)
    save_consensus_outputs(consensus, summary, output_path, summary_path)

    print(f"Consensus trades ({summary.get('trade_timeline', 'n/a')}): {summary.get('consensus_trades', 0):,}")
    print(f"Raw calibrated trades: {summary.get('raw_calibrated_trades', 0):,}")
    print(f"Market-blend calibrated trades: {summary.get('market_blend_calibrated_trades', 0):,}")
    print(f"Saved consensus rows to: {output_path}")
    print(f"Saved consensus summary to: {summary_path}")


if __name__ == "__main__":
    main()
