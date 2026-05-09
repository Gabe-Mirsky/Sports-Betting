"""Evaluate direct spread/total Kalshi line markets against model probabilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from strategy.line_market_eval import (  # noqa: E402
    prepare_line_market_model_eval,
    save_line_market_model_eval,
    summarize_line_market_model_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct Kalshi spread/total markets.")
    parser.add_argument("--line-prices-path", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--snapshot-target", default="pregame_60m")
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = PROJECT_ROOT / "data" / "reports"
    line_prices_path = (
        Path(args.line_prices_path)
        if args.line_prices_path
        else PROJECT_ROOT / "data" / "processed" / "kalshi_line_pregame_prices.csv"
    )
    predictions_path = Path(args.predictions_path) if args.predictions_path else reports_dir / "market_type_predictions.csv"
    output_path = Path(args.output_path) if args.output_path else reports_dir / "line_market_model_eval.csv"
    summary_path = (
        Path(args.summary_path) if args.summary_path else reports_dir / "line_market_model_eval_summary.json"
    )

    line_prices = pd.read_csv(line_prices_path) if line_prices_path.exists() else pd.DataFrame()
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    eval_rows = prepare_line_market_model_eval(
        line_prices,
        predictions,
        snapshot_target=args.snapshot_target,
        edge_threshold=args.edge_threshold,
    )
    summary = summarize_line_market_model_eval(eval_rows, edge_threshold=args.edge_threshold)
    save_line_market_model_eval(eval_rows, summary, output_path, summary_path)

    print(f"Line market eval rows ({summary.get('timeline', 'n/a')}): {summary.get('rows', 0):,}")
    print(f"Edge signals at {args.edge_threshold:.2%}: {summary.get('signals', 0):,}")
    print(f"Readiness status: {summary.get('status', 'n/a')}")
    print(f"Saved evaluation rows to: {output_path}")
    print(f"Saved evaluation summary to: {summary_path}")


if __name__ == "__main__":
    main()
