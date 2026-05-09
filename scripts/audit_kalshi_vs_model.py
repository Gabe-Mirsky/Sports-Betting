"""Audit where Kalshi pregame prices beat our model probabilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from reports.market_gap_audit import write_market_gap_audit_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model probabilities with Kalshi pregame probabilities.")
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--modeling-path", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = PROJECT_ROOT / "data" / "reports"
    predictions_path = Path(args.predictions_path) if args.predictions_path else reports_dir / "market_blended_predictions.csv"
    modeling_path = (
        Path(args.modeling_path)
        if args.modeling_path
        else PROJECT_ROOT / "data" / "processed" / "modeling_dataset.parquet"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir

    predictions = pd.read_csv(predictions_path, dtype={"game_id": str, "market_ticker": str})
    modeling = pd.read_parquet(modeling_path) if modeling_path.exists() else pd.DataFrame()
    detail, segments, summary = write_market_gap_audit_outputs(predictions, modeling, output_dir)

    print(f"Audited rows: {len(detail):,}")
    print(f"Timeline: {summary.get('timeline', 'n/a')}")
    print(f"Kalshi closer than model: {summary.get('market_beats_model_rows', 0):,} rows")
    print(f"Model closer than Kalshi: {summary.get('model_beats_market_rows', 0):,} rows")
    print(f"Model log loss: {summary.get('model_log_loss', float('nan')):.4f}")
    print(f"Kalshi log loss: {summary.get('market_log_loss', float('nan')):.4f}")
    if not segments.empty:
        top = segments[segments["rows"] >= 25].head(1)
        if not top.empty:
            row = top.iloc[0]
            print(
                "Largest Kalshi advantage segment: "
                f"{row['segment']}={row['value']} "
                f"({int(row['rows']):,} rows, avg advantage {row['avg_kalshi_edge_over_model']:.4f})"
            )
    print(f"Saved detail: {output_dir / 'kalshi_model_gap_audit.csv'}")
    print(f"Saved segments: {output_dir / 'kalshi_model_gap_segments.csv'}")
    print(f"Saved summary: {output_dir / 'kalshi_model_gap_summary.json'}")


if __name__ == "__main__":
    main()
