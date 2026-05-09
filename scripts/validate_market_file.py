"""Validate a local Kalshi-style market CSV against saved predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_client import load_mock_kalshi_markets, validate_kalshi_markets  # noqa: E402
from data.market_quality import analyze_market_data_quality, save_market_quality_report  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local market CSV.")
    parser.add_argument("--markets-path", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    markets_path = (
        Path(args.markets_path)
        if args.markets_path
        else PROJECT_ROOT / "data" / "kalshi" / "markets_mock.csv"
    )
    default_walk_forward_path = PROJECT_ROOT / "data" / "reports" / "walk_forward_predictions.csv"
    default_single_split_path = PROJECT_ROOT / "data" / "reports" / "model_predictions.csv"
    predictions_path = (
        Path(args.predictions_path)
        if args.predictions_path
        else default_walk_forward_path
        if default_walk_forward_path.exists()
        else default_single_split_path
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else PROJECT_ROOT / "data" / "reports" / "market_validation_report.json"
    )

    markets = load_mock_kalshi_markets(markets_path)
    predictions = pd.read_csv(predictions_path, dtype={"game_id": str}) if predictions_path.exists() else None
    report = validate_kalshi_markets(markets, predictions)
    quality_report = analyze_market_data_quality(markets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    quality_path = output_path.parent / "market_data_quality_report.json"
    save_market_quality_report(quality_report, quality_path)

    print(f"Validated {report['rows']:,} market rows.")
    if report["issues"]:
        print("Issues:")
        for issue in report["issues"]:
            print(f"- {issue}")
    else:
        print("No validation issues found.")
    if report["matched_rows"] is not None:
        print(f"Matched rows: {report['matched_rows']}")
        print(f"Unmatched rows: {report['unmatched_rows']}")
    print(f"Saved report to: {output_path}")
    if quality_report["warnings"]:
        print("Quality warnings:")
        for warning in quality_report["warnings"]:
            print(f"- {warning}")
    else:
        print("No market data quality warnings found.")
    print(f"Saved quality report to: {quality_path}")


if __name__ == "__main__":
    main()
